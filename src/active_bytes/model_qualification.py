"""Fail-closed evaluation of a non-paper model/geometry qualification run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .campaign import sha256_json
from .decode_doctor import _atomic_write_json


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_model_qualification(
    *,
    contract: Mapping[str, Any],
    campaign_lock: Mapping[str, Any],
    doctor: Mapping[str, Any],
    runtime_audit: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if contract.get("schema_version") != 1:
        reasons.append("qualification contract schema_version is not 1")
    if contract.get("may_enter_paper_outcomes") is not False:
        reasons.append("qualification contract does not prohibit paper outcomes")

    lock_without_digest = dict(campaign_lock)
    observed_lock_digest = lock_without_digest.pop("lock_sha256", None)
    if observed_lock_digest != sha256_json(lock_without_digest):
        reasons.append("campaign lock self-digest is invalid")
    runs = campaign_lock.get("run_order", [])
    if len(runs) != 1:
        reasons.append("qualification campaign must contain exactly one run")
        run: Mapping[str, Any] = {}
    else:
        run = runs[0]

    expected = contract["expected"]
    expected_model = expected["model"]
    expected_geometry = expected["geometry"]
    parameters = run.get("parameters", {})
    for name, value in (
        ("model", expected_model["name"]),
        ("model_revision", expected_model["revision"]),
        ("target_mean_attended_history_tokens", expected_geometry["target_mean_attended_history_tokens"]),
        ("target_batch", expected_geometry["target_batch"]),
        ("metered_decode_tokens_per_request", expected_geometry["metered_decode_tokens_per_request"]),
        ("weight_dtype", "bf16"),
        ("kv_cache_dtype", "bf16"),
        ("attention_backend", expected["runtime"]["attention_backend"]),
    ):
        if parameters.get(name) != value:
            reasons.append(f"locked {name} does not match the qualification contract")

    if doctor.get("qc_pass") is not True:
        reasons.append("batch doctor did not pass")
    if doctor.get("non_paper_measurement") is not True:
        reasons.append("batch doctor is not marked non-paper")
    doctor_runtime = doctor.get("runtime", {})
    doctor_geometry = doctor.get("geometry", {})
    if doctor_runtime.get("model") != expected_model["name"]:
        reasons.append("batch doctor loaded the wrong model")
    if doctor_runtime.get("model_revision") != expected_model["revision"]:
        reasons.append("batch doctor loaded the wrong model revision")
    if doctor_geometry.get("batch") != expected_geometry["target_batch"]:
        reasons.append("batch doctor used the wrong batch")
    if doctor_geometry.get("target_mean_attended_history_tokens") != expected_geometry[
        "target_mean_attended_history_tokens"
    ]:
        reasons.append("batch doctor used the wrong attended-history target")
    if doctor_geometry.get("metered_decode_tokens_per_request") != expected_geometry[
        "metered_decode_tokens_per_request"
    ]:
        reasons.append("batch doctor used the wrong decode-token count")
    if not math.isclose(
        float(doctor_geometry.get("mean_attended_history_tokens", -1)),
        float(expected_geometry["target_mean_attended_history_tokens"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append("batch doctor did not realize the exact attended-history mean")
    decode_seconds = doctor_geometry.get("decode_seconds")
    if not isinstance(decode_seconds, (int, float)) or decode_seconds < expected_geometry[
        "minimum_decode_seconds"
    ]:
        reasons.append("qualification decode interval is too short for the energy campaign")

    if runtime_audit.get("qc_pass") is not True:
        reasons.append("runtime audit did not pass")
    if runtime_audit.get("non_paper_measurement") is not True:
        reasons.append("runtime audit is not marked non-paper")
    if runtime_audit.get("frozen_run_execution") is not True:
        reasons.append("runtime audit did not execute the frozen qualification run")
    if runtime_audit.get("campaign_lock_sha256") != observed_lock_digest:
        reasons.append("runtime audit used a different campaign lock")

    runtime = runtime_audit.get("runtime", {})
    expected_runtime = expected["runtime"]
    for name, value in (
        ("model", expected_model["name"]),
        ("model_revision", expected_model["revision"]),
        ("attention_backend", expected_runtime["attention_backend"]),
        ("weight_dtype", "bfloat16"),
        ("requested_kv_cache_dtype", "auto"),
    ):
        if runtime.get(name) != value:
            reasons.append(f"runtime audit {name} does not match the contract")
    if runtime_audit.get("cache_contract", {}).get("qc_pass") is not True:
        reasons.append("resolved KV cache dtype contract failed")

    observed_model_geometry = runtime_audit.get("model_geometry", {})
    for name, value in expected_model["config"].items():
        if observed_model_geometry.get(name) != value:
            reasons.append(f"loaded model geometry has unexpected {name}")

    weight_bytes = runtime_audit.get("weights", {}).get("unique_storage_bytes")
    bounds = expected_model["unique_storage_bytes_bounds"]
    if not isinstance(weight_bytes, int) or not bounds[0] <= weight_bytes <= bounds[1]:
        reasons.append("loaded unique weight storage is outside the qualification bounds")
    cache = runtime_audit.get("cache", {})
    if not cache.get("tensor_count") or cache.get("gpu_tensor_count") != cache.get(
        "tensor_count"
    ):
        reasons.append("runtime audit did not discover an all-GPU KV cache")

    logical_kv_bytes = observed_model_geometry.get(
        "logical_kv_bytes_per_attended_token"
    )
    active_byte_coordinates = None
    if (
        isinstance(weight_bytes, int)
        and isinstance(logical_kv_bytes, int)
        and weight_bytes > 0
        and logical_kv_bytes > 0
    ):
        weight_read = weight_bytes / expected_geometry["target_batch"]
        kv_read = (
            logical_kv_bytes
            * expected_geometry["target_mean_attended_history_tokens"]
        )
        active_byte_coordinates = {
            "weight_read_obligation_bytes_per_useful_token": weight_read,
            "kv_read_obligation_bytes_per_useful_token": kv_read,
            "kv_to_weight_obligation_ratio": kv_read / weight_read,
        }

    return {
        "schema_version": 1,
        "measurement": contract.get(
            "measurement", "gh200-qwen2p5-14b-model-qualification"
        ),
        "qualification_id": contract.get("qualification_id"),
        "energy_measurement": False,
        "may_enter_paper_outcomes": False,
        "qualified_for_campaign_design": not reasons,
        "qc_pass": not reasons,
        "qc_reasons": reasons,
        "campaign_lock_sha256": observed_lock_digest,
        "observed": {
            "decode_seconds": decode_seconds,
            "unique_weight_storage_bytes": weight_bytes,
            "logical_kv_bytes_per_attended_token": observed_model_geometry.get(
                "logical_kv_bytes_per_attended_token"
            ),
            "active_byte_coordinates_at_qualification_geometry": (
                active_byte_coordinates
            ),
            "model_geometry": observed_model_geometry,
            "cache_tensor_count": cache.get("tensor_count"),
            "cache_logical_nbytes": cache.get("logical_nbytes"),
        },
        "next_action": (
            "freeze-identification-and-unopened-holdout"
            if not reasons
            else "stop-and-review-qualification-failure"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-contract", required=True, type=Path)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--doctor-json", required=True, type=Path)
    parser.add_argument("--runtime-audit-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {
        "contract": json.loads(args.qualification_contract.read_text(encoding="utf-8")),
        "campaign_lock": json.loads(args.campaign_lock.read_text(encoding="utf-8")),
        "doctor": json.loads(args.doctor_json.read_text(encoding="utf-8")),
        "runtime_audit": json.loads(args.runtime_audit_json.read_text(encoding="utf-8")),
    }
    report = evaluate_model_qualification(**inputs)
    report["artifact_sha256"] = {
        "qualification_contract": _artifact_sha256(args.qualification_contract),
        "campaign_lock": _artifact_sha256(args.campaign_lock),
        "doctor_json": _artifact_sha256(args.doctor_json),
        "runtime_audit_json": _artifact_sha256(args.runtime_audit_json),
    }
    _atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
