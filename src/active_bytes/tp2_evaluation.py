"""Evaluate the released two-GH200 TP=2 holdout without refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_replication_evaluation import evaluate_replication_rows
from .primary_identification import collect_identification_rows, sha256_file
from .tp2_identification import TP2_OUTCOME
from .tp2_release import OFFICIAL_RELEASE_SHA256, verify_tp2_release


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _alignment_bundle_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["order"])):
        digest.update(str(row["run_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["alignment_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_tp2_holdout(
    campaign_lock_path: Path,
    release_record_path: Path,
    identification_freeze_dir: Path,
    identification_lock_path: Path,
    execution_addendum_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply the frozen TP=2 coefficients to every sealed holdout run."""

    if OFFICIAL_RELEASE_SHA256 is None:
        release_verification = {
            "issues": [
                "TP=2 holdout is sealed: no official release digest is compiled"
            ],
            "qc_pass": False,
        }
    else:
        release_verification = verify_tp2_release(
            release_record_path,
            identification_freeze_dir,
            identification_lock_path,
            campaign_lock_path,
            execution_addendum_path,
            expected_release_sha256=OFFICIAL_RELEASE_SHA256,
        )
    lock, rows, issues = collect_identification_rows(
        campaign_lock_path,
        results_root,
        result_domain="tp2-holdout",
    )
    if release_verification.get("qc_pass") is not True:
        issues.extend(
            f"release verification: {issue}"
            for issue in release_verification.get("issues", [])
        )
    if len(rows) != int(lock["run_count"]):
        issues.append(
            f"expected {lock['run_count']} accepted holdout runs; found {len(rows)}"
        )

    addendum = json.loads(execution_addendum_path.read_text(encoding="utf-8"))
    expected_weight = int(
        addendum["source_evidence"]["audited_full_model_weight_storage_bytes"]
    )
    expected_kv = int(
        addendum["source_evidence"][
            "audited_logical_kv_bytes_per_attended_token"
        ]
    )
    for row in rows:
        alignment = json.loads(
            (results_root / row["alignment_path_relative_to_results"]).read_text(
                encoding="utf-8"
            )
        )
        if alignment.get("measurement") != "aligned-frozen-tp2-decode-repeat":
            issues.append(f"{row['run_id']}: alignment is not TP=2")
        if alignment.get("paper_candidate_measurement") is not True:
            issues.append(f"{row['run_id']}: alignment is not paper-candidate")
        if alignment.get("tensor_parallel_size") != 2:
            issues.append(f"{row['run_id']}: tensor-parallel size differs from two")
        if alignment.get("host_gpu_indices") != [0, 1]:
            issues.append(f"{row['run_id']}: host GPU pair differs from [0, 1]")
        if int(alignment["weights"]["unique_storage_bytes"]) != expected_weight:
            issues.append(f"{row['run_id']}: full-model weight coordinate changed")
        if int(row["kv_write_bytes_per_token"]) != expected_kv:
            issues.append(f"{row['run_id']}: logical KV coordinate changed")
        row[TP2_OUTCOME] = row["gross_gpu_joules_per_token"]

    if issues:
        report = {
            "schema_version": 1,
            "measurement": "gh200-tp2-held-out-form-evaluation",
            "accepted_run_count": len(rows),
            "issues": issues,
            "qc_pass": False,
        }
        _atomic_json(output_dir / "tp2-holdout-summary.json", report)
        return report

    release = json.loads(release_record_path.read_text(encoding="utf-8"))
    coefficient = json.loads(
        (identification_freeze_dir / "coefficient-artifact.json").read_text(
            encoding="utf-8"
        )
    )
    envelope = json.loads(
        (identification_freeze_dir / "discrepancy-envelope.json").read_text(
            encoding="utf-8"
        )
    )
    analysis = evaluate_replication_rows(
        rows,
        coefficient["fit"]["coefficients"],
        envelope["envelope"]["common_residual_range_joules_per_token"],
        release["primary_gates"],
        expected_cells=int(release["holdout_campaign"]["cell_count"]),
        expected_repeats=int(
            release["holdout_campaign"]["repetitions_per_cell"]
        ),
        form_replication_gate_name="tp2_functional_form_pass",
    )
    analysis["prediction_contract"].update(
        {
            "outcome": TP2_OUTCOME,
            "equation": release["analysis_equation"],
            "both_device_energies_are_summed": True,
            "nvlink_energy_is_not_separately_attributed": True,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "tp2-holdout-runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(analysis["runs"])
    scalar_cells = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in analysis["cells"]
    ]
    cells_path = output_dir / "tp2-holdout-cells.csv"
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_cells[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_cells)

    report = {
        "schema_version": 1,
        "measurement": "gh200-tp2-held-out-form-evaluation",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "alignment_bundle_sha256": _alignment_bundle_sha(rows),
        "release_record_sha256": sha256_file(release_record_path),
        "coefficient_artifact_sha256": sha256_file(
            identification_freeze_dir / "coefficient-artifact.json"
        ),
        "discrepancy_envelope_sha256": sha256_file(
            identification_freeze_dir / "discrepancy-envelope.json"
        ),
        "analysis": analysis,
        "tables": {
            "runs": {"path": runs_path.name, "sha256": sha256_file(runs_path)},
            "cells": {"path": cells_path.name, "sha256": sha256_file(cells_path)},
        },
        "claim_boundary": {
            "tp2_functional_form_supported_in_frozen_stratum": analysis["gates"][
                "tp2_functional_form_pass"
            ],
            "single_gpu_coefficients_transfer_to_tp2_supported": False,
            "nvlink_energy_separately_attributed": False,
            "other_tensor_parallel_degrees_supported": False,
            "cross_node_or_cross_sku_supported": False,
            "dvfs_conclusion_supported": False,
        },
        "issues": [],
        "qc_pass": True,
    }
    report_path = output_dir / "tp2-holdout-evaluation.json"
    _atomic_json(report_path, report)
    summary = {
        "schema_version": 1,
        "measurement": report["measurement"],
        "campaign_lock_sha256": report["campaign_lock_sha256"],
        "alignment_bundle_sha256": report["alignment_bundle_sha256"],
        "release_record_sha256": report["release_record_sha256"],
        "accepted_run_count": analysis["run_count"],
        "cell_count": analysis["cell_count"],
        "summary": analysis["summary"],
        "gates": analysis["gates"],
        "cells": analysis["cells"],
        "claim_boundary": report["claim_boundary"],
        "evaluation_json": {
            "path": report_path.name,
            "sha256": sha256_file(report_path),
        },
        "qc_pass": True,
    }
    _atomic_json(output_dir / "tp2-holdout-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--release-record", required=True, type=Path)
    parser.add_argument("--identification-freeze-dir", required=True, type=Path)
    parser.add_argument("--identification-lock", required=True, type=Path)
    parser.add_argument("--execution-addendum", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate_tp2_holdout(
        args.campaign_lock,
        args.release_record,
        args.identification_freeze_dir,
        args.identification_lock,
        args.execution_addendum,
        args.results_root,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
