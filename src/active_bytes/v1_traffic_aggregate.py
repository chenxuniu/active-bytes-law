"""Audit and aggregate a complete GH200 V1 traffic-anchor campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .decode_doctor import _atomic_write_json


T_95_DF = {1: 12.706204736, 2: 4.30265273, 3: 3.182446305,
           4: 2.776445105, 5: 2.570581836, 6: 2.446911851,
           7: 2.364624252, 8: 2.306004135, 9: 2.262157163,
           10: 2.228138852}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if count > 1 else 0.0
    t_value = T_95_DF.get(count - 1, 1.959963985) if count > 1 else 0.0
    half_width = t_value * sample_sd / math.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "median": statistics.median(values),
        "sample_sd": sample_sd,
        "coefficient_of_variation": sample_sd / mean if mean else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def _fit_two_component_law(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xw = [float(row["weight_bytes_per_token"]) for row in rows]
    xk = [float(row["kv_read_bytes_per_token"]) for row in rows]
    y = [float(row["observed_read_bytes_per_token"]) for row in rows]
    aa = sum(value * value for value in xw)
    ab = sum(a * b for a, b in zip(xw, xk, strict=True))
    bb = sum(value * value for value in xk)
    ay = sum(a * value for a, value in zip(xw, y, strict=True))
    by = sum(b * value for b, value in zip(xk, y, strict=True))
    determinant = aa * bb - ab * ab
    if determinant <= 0:
        raise ValueError("traffic-law design matrix is singular")
    rho_weight = (ay * bb - by * ab) / determinant
    rho_kv = (by * aa - ay * ab) / determinant
    predicted = [rho_weight * a + rho_kv * b
                 for a, b in zip(xw, xk, strict=True)]
    residuals = [value - estimate
                 for value, estimate in zip(y, predicted, strict=True)]
    mean_y = statistics.fmean(y)
    sse = sum(value * value for value in residuals)
    sst = sum((value - mean_y) ** 2 for value in y)
    ape = [abs(residual) / value
           for residual, value in zip(residuals, y, strict=True) if value]
    return {
        "equation": "observed_read_bytes_per_token = rho_weight * weight_bytes_per_token + rho_kv * kv_read_bytes_per_token",
        "fit_intercept": False,
        "observation_count": len(rows),
        "rho_weight": rho_weight,
        "rho_kv": rho_kv,
        "r_squared": 1.0 - sse / sst if sst else 1.0,
        "rmse_bytes_per_token": math.sqrt(sse / len(rows)),
        "mean_absolute_percentage_error": statistics.fmean(ape),
        "maximum_absolute_percentage_error": max(ape),
    }


def _validated_report(path: Path, expected: dict[str, Any],
                      lock_sha: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read traffic report: {exc}"]
    if report.get("measurement") != "gh200-v1-application-range-replay-traffic-anchor":
        reasons.append("unexpected measurement type")
    if not report.get("qc_pass"):
        reasons.append("traffic report QC did not pass")
    if report.get("energy_measurement") is not False:
        reasons.append("traffic report is not explicitly non-energy")
    if report.get("campaign_lock_sha256") != lock_sha:
        reasons.append("campaign lock SHA does not match")
    run = report.get("run", {})
    for key in ("run_id", "cell_id", "order", "repeat", "split"):
        if run.get(key) != expected.get(key):
            reasons.append(f"run field {key} does not match lock")
    parameters = expected["parameters"]
    geometry = report.get("geometry", {})
    if geometry.get("batch") != parameters["target_batch"]:
        reasons.append("observed batch does not match lock")
    if geometry.get("target_mean_attended_history_tokens") != parameters[
        "target_mean_attended_history_tokens"
    ]:
        reasons.append("observed history length does not match lock")
    hashes = report.get("artifact_sha256", {})
    for filename, key in (("anchor.json", "anchor_json"),
                          ("ncu.csv", "ncu_csv")):
        artifact = path.parent / filename
        if not artifact.is_file():
            reasons.append(f"missing artifact {filename}")
        elif _sha256(artifact) != hashes.get(key):
            reasons.append(f"artifact hash mismatch for {filename}")
    return report, reasons


def aggregate_v1_traffic(campaign_lock_path: Path,
                         results_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = json.loads(campaign_lock_path.read_text(encoding="utf-8"))
    lock_sha = campaign["lock_sha256"]
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for expected in campaign["run_order"]:
        run_id = expected["run_id"]
        candidates = sorted((results_root / run_id).glob("attempt-*/traffic.json"))
        valid: list[tuple[Path, dict[str, Any]]] = []
        invalid: list[dict[str, Any]] = []
        for candidate in candidates:
            report, reasons = _validated_report(candidate, expected, lock_sha)
            if reasons:
                invalid.append({"path": str(candidate), "reasons": reasons})
            elif report is not None:
                valid.append((candidate, report))
        if len(valid) != 1:
            issues.append({"run_id": run_id, "expected_valid_reports": 1,
                           "observed_valid_reports": len(valid),
                           "candidate_count": len(candidates),
                           "invalid_candidates": invalid})
            continue
        path, report = valid[0]
        observed = report["observed_hbm"]
        active = report["active_bytes"]
        rows.append({
            "order": expected["order"], "run_id": run_id,
            "cell_id": expected["cell_id"], "repeat": expected["repeat"],
            "batch": report["geometry"]["batch"],
            "target_mean_attended_history_tokens": report["geometry"]["target_mean_attended_history_tokens"],
            "weight_bytes_per_token": active["weight_bytes_per_token"],
            "kv_read_bytes_per_token": active["kv_read_bytes_per_token"],
            "accounted_read_bytes_per_token": active["active_bytes_read"],
            "observed_read_bytes_per_token": observed["read_bytes_per_useful_token"],
            "observed_write_bytes_per_token": observed["write_bytes_per_useful_token"],
            "observed_read_write_bytes_per_token": observed["read_write_bytes_per_useful_token"],
            "observed_read_over_accounted_read": report["descriptive_uncorrected_ratios"]["observed_read_over_accounted_read"],
            "traffic_json": str(path), "traffic_json_sha256": _sha256(path),
        })
    rows.sort(key=lambda row: int(row["order"]))
    cells: list[dict[str, Any]] = []
    for cell in campaign["cells"]:
        selected = [row for row in rows if row["cell_id"] == cell["cell_id"]]
        if len(selected) != int(cell["repetitions"]):
            continue
        cells.append({
            "cell_id": cell["cell_id"],
            "batch": cell["parameters"]["target_batch"],
            "target_mean_attended_history_tokens": cell["parameters"]["target_mean_attended_history_tokens"],
            "expected_repetitions": cell["repetitions"],
            "observed_repetitions": len(selected),
            "repeat_ids": sorted(int(row["repeat"]) for row in selected),
            "weight_bytes_per_token": selected[0]["weight_bytes_per_token"],
            "kv_read_bytes_per_token": selected[0]["kv_read_bytes_per_token"],
            "observed_read_bytes_per_token": _summary([float(row["observed_read_bytes_per_token"]) for row in selected]),
            "observed_read_write_bytes_per_token": _summary([float(row["observed_read_write_bytes_per_token"]) for row in selected]),
            "observed_read_over_accounted_read": _summary([float(row["observed_read_over_accounted_read"]) for row in selected]),
        })
    qc_pass = not issues and len(rows) == int(campaign["run_count"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "measurement": "gh200-v1-traffic-campaign-aggregate",
        "campaign_id": campaign["campaign_id"],
        "campaign_lock_sha256": lock_sha,
        "campaign_lock_file_sha256": _sha256(campaign_lock_path),
        "expected_run_count": campaign["run_count"],
        "accepted_run_count": len(rows),
        "expected_cell_count": campaign["cell_count"],
        "complete_cell_count": len(cells),
        "issues": issues,
        "cells": cells,
        "raw_observed_traffic_law": _fit_two_component_law(rows) if qc_pass else None,
        "formal_cache_credit_applied": False,
        "formal_v1_decision_eligible": False,
        "formal_v1_note": "This aggregate audits raw observed HBM traffic. Apply only the separately frozen cache/residency correction and simultaneous interval procedure before a formal V1 decision.",
        "qc_pass": qc_pass,
    }
    return report, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)
    report, rows = aggregate_v1_traffic(args.campaign_lock, args.results_root)
    _atomic_write_json(args.output_json, report)
    _write_csv(args.output_csv, rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
