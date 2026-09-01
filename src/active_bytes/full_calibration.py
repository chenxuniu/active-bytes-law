"""Execute one frozen full FP8 KV-only calibration and save a candidate checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping

from .decode_doctor import _atomic_write_json
from .kv_calibration import (
    _file_sha256,
    _package_versions,
    run_kv_calibration_doctor,
    validate_revision,
)


def contract_sha256(contract: Mapping[str, Any]) -> str:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_full_contract(
    contract: Mapping[str, Any],
    *,
    calibration_image_id: str | None,
    evidence_paths: Mapping[str, Path],
) -> dict[str, Any]:
    reasons: list[str] = []
    if contract.get("status") != "frozen":
        reasons.append("calibration contract is not frozen")
    expected_image = contract.get("runtime", {}).get("calibration_image_id")
    if not calibration_image_id or calibration_image_id != expected_image:
        reasons.append("running calibration image ID does not match the contract")
    current_packages = _package_versions()
    expected_packages = contract.get("runtime", {}).get("packages", {})
    if current_packages != expected_packages:
        reasons.append("calibration package versions do not match the contract")

    expected_evidence = contract.get("qualification_evidence", {})
    evidence: dict[str, Any] = {}
    for label in ("doctor_r01", "doctor_r02", "repeat_comparison"):
        path = evidence_paths.get(label)
        expected_hash = expected_evidence.get(label, {}).get("sha256")
        if path is None or not path.is_file():
            reasons.append(f"qualification evidence {label} is missing")
            continue
        observed_hash = _file_sha256(path)
        evidence[label] = {"path": str(path), "sha256": observed_hash}
        if observed_hash != expected_hash:
            reasons.append(f"qualification evidence {label} hash does not match")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reasons.append(f"qualification evidence {label} is not valid JSON")
            continue
        if not value.get("qc_pass"):
            reasons.append(f"qualification evidence {label} did not pass QC")

    return {
        "qc_pass": not reasons,
        "qc_reasons": reasons,
        "current_packages": current_packages,
        "calibration_image_id": calibration_image_id,
        "evidence": evidence,
    }


def run_full_calibration(
    *,
    contract_path: Path,
    checkpoint_output_dir: Path,
    source_revision: str,
    calibration_image_id: str | None,
    evidence_paths: Mapping[str, Path],
) -> dict[str, Any]:
    validate_revision(source_revision, label="source revision")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validation = validate_full_contract(
        contract,
        calibration_image_id=calibration_image_id,
        evidence_paths=evidence_paths,
    )
    if not validation["qc_pass"]:
        raise ValueError("; ".join(validation["qc_reasons"]))

    model = contract["model"]
    dataset = contract["dataset"]
    report = run_kv_calibration_doctor(
        model_id=model["id"],
        model_revision=model["revision"],
        dataset_id=dataset["id"],
        dataset_revision=dataset["revision"],
        dataset_split=dataset["split"],
        num_calibration_samples=dataset["num_calibration_samples"],
        max_sequence_length=dataset["max_sequence_length"],
        seed=dataset["seed"],
        checkpoint_output_dir=checkpoint_output_dir,
        checkpoint_metadata=contract,
    )
    report["contract"] = {
        "id": contract["contract_id"],
        "path": str(contract_path),
        "canonical_sha256": contract_sha256(contract),
        "source_revision": source_revision,
        "qualification_validation": validation,
    }
    expected = contract["expected_model_invariants"]
    reasons = list(report["qc_reasons"])
    before = report["baseline_parameters_before"]
    if before["parameter_count"] != expected["parameter_count"]:
        reasons.append("baseline parameter count does not match the frozen invariant")
    if before["logical_nbytes"] != expected["logical_parameter_nbytes"]:
        reasons.append("baseline parameter bytes do not match the frozen invariant")
    if report["kv_scales"]["complete_layer_count"] != expected["attention_layers"]:
        reasons.append("calibrated attention-layer count does not match the invariant")
    if not report.get("checkpoint_saved"):
        reasons.append("the calibrated checkpoint was not saved")
    report["qc_reasons"] = reasons
    report["qc_pass"] = not reasons
    report["checkpoint_candidate"] = report["qc_pass"]
    report["requires_frozen_inference_load_audit"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--checkpoint-output-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--doctor-r01", required=True, type=Path)
    parser.add_argument("--doctor-r02", required=True, type=Path)
    parser.add_argument("--repeat-comparison", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_full_calibration(
            contract_path=args.contract,
            checkpoint_output_dir=args.checkpoint_output_dir,
            source_revision=args.source_revision,
            calibration_image_id=os.environ.get("TEL_CALIBRATION_IMAGE_ID"),
            evidence_paths={
                "doctor_r01": args.doctor_r01,
                "doctor_r02": args.doctor_r02,
                "repeat_comparison": args.repeat_comparison,
            },
        )
    except Exception as error:
        traceback.print_exc()
        report = {
            "schema_version": 1,
            "measurement": "offline-fp8-kv-full-calibration",
            "non_paper_measurement": True,
            "may_enter_paper_outcomes": False,
            "checkpoint_saved": False,
            "checkpoint_candidate": False,
            "qc_pass": False,
            "qc_reasons": [f"{type(error).__name__}: {error}"],
        }
    _atomic_write_json(args.output_json, report)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"kv_scales", "added_state_entries", "checkpoint"}
    }
    if "kv_scales" in report:
        summary["kv_scales"] = {
            key: value
            for key, value in report["kv_scales"].items()
            if key != "layers"
        }
    if "checkpoint" in report:
        summary["checkpoint"] = {
            key: value
            for key, value in report["checkpoint"].items()
            if key != "files"
        }
    summary["output_json"] = str(args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
