"""Freeze the prespecified Qwen2.5-14B duration-model identification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .primary_identification import (
    CALIBRATION_SPLIT,
    FIT_SPLIT,
    OUTCOME,
    _alignment_bundle_sha,
    _atomic_json,
    _invert,
    _matvec,
    _quadratic,
    collect_identification_rows,
    sha256_file,
    student_t_quantile,
)


PREDICTOR_NAMES = (
    "intercept_joules_per_token",
    "alpha_weight_joules_per_decimal_gb",
    "beta_kv_joules_per_decimal_gb",
    "p_time_watts",
)


def _duration_design_row(row: Mapping[str, Any]) -> list[float]:
    useful_tokens = int(row["metered_useful_tokens"])
    if useful_tokens <= 0:
        raise ValueError("metered useful tokens must be positive")
    return [
        1.0,
        float(row["weight_gb_per_token"]),
        float(row["kv_rw_gb_per_token"]),
        float(row["decode_seconds"]) / useful_tokens,
    ]


def _prediction(row: Mapping[str, Any], coefficients: Mapping[str, Any]) -> float:
    return sum(
        value * float(coefficients[name])
        for value, name in zip(_duration_design_row(row), PREDICTOR_NAMES)
    )


def fit_duration_ols_hc3(
    rows: Sequence[Mapping[str, Any]],
    analysis_id: str = "qwen14b-duration-identification-v1",
) -> dict[str, Any]:
    parameter_count = len(PREDICTOR_NAMES)
    if len(rows) <= parameter_count:
        raise ValueError("the four-parameter fit requires more than four runs")
    design = [_duration_design_row(row) for row in rows]
    outcome = [float(row["gross_gpu_joules_per_token"]) for row in rows]
    xtx = [
        [
            sum(row[left] * row[right] for row in design)
            for right in range(parameter_count)
        ]
        for left in range(parameter_count)
    ]
    xty = [
        sum(row[column] * value for row, value in zip(design, outcome))
        for column in range(parameter_count)
    ]
    xtx_inverse = _invert(xtx)
    coefficients = _matvec(xtx_inverse, xty)
    predictions = [
        sum(value * coefficient for value, coefficient in zip(row, coefficients))
        for row in design
    ]
    residuals = [value - prediction for value, prediction in zip(outcome, predictions)]
    leverages = [_quadratic(row, xtx_inverse) for row in design]
    if any(value >= 1.0 - 1e-12 for value in leverages):
        raise ValueError("one or more HC3 leverages are numerically one")

    meat = [[0.0] * parameter_count for _ in range(parameter_count)]
    for row, residual, leverage in zip(design, residuals, leverages):
        scale = (residual / (1.0 - leverage)) ** 2
        for left in range(parameter_count):
            for right in range(parameter_count):
                meat[left][right] += scale * row[left] * row[right]
    intermediate = [
        [
            sum(xtx_inverse[row][k] * meat[k][column] for k in range(parameter_count))
            for column in range(parameter_count)
        ]
        for row in range(parameter_count)
    ]
    covariance = [
        [
            sum(intermediate[row][k] * xtx_inverse[k][column] for k in range(parameter_count))
            for column in range(parameter_count)
        ]
        for row in range(parameter_count)
    ]
    standard_errors = [
        math.sqrt(max(covariance[index][index], 0.0))
        for index in range(parameter_count)
    ]
    degrees_of_freedom = len(rows) - parameter_count
    traffic_quantile = student_t_quantile(1.0 - 0.05 / 2.0, degrees_of_freedom)
    traffic_lowers = {
        PREDICTOR_NAMES[index]: coefficients[index]
        - traffic_quantile * standard_errors[index]
        for index in (1, 2)
    }
    coefficient_map = dict(zip(PREDICTOR_NAMES, coefficients))
    standard_error_map = dict(zip(PREDICTOR_NAMES, standard_errors))
    centered_total = sum((value - statistics.fmean(outcome)) ** 2 for value in outcome)
    residual_sum = sum(value * value for value in residuals)
    relative_errors = [
        abs(residual) / max(abs(value), 1e-12)
        for residual, value in zip(residuals, outcome)
    ]
    return {
        "analysis_id": analysis_id,
        "estimator": "unweighted run-level OLS with HC3 covariance",
        "outcome": OUTCOME,
        "predictor_names": list(PREDICTOR_NAMES),
        "observations": len(rows),
        "parameters": parameter_count,
        "degrees_of_freedom": degrees_of_freedom,
        "coefficients": coefficient_map,
        "standard_errors": standard_error_map,
        "hc3_covariance": covariance,
        "traffic_slope_positivity": {
            "familywise_alpha": 0.05,
            "bonferroni_one_sided_quantile": traffic_quantile,
            "lower_bounds": traffic_lowers,
            "qc_pass": all(value > 0.0 for value in traffic_lowers.values()),
        },
        "time_term": {
            "point_estimate_watts": coefficient_map["p_time_watts"],
            "finite_nonnegative_qc_pass": math.isfinite(
                coefficient_map["p_time_watts"]
            )
            and coefficient_map["p_time_watts"] >= 0.0,
            "causal_idle_power_interpretation_authorized": False,
        },
        "centered_r_squared": (
            1.0 - residual_sum / centered_total if centered_total > 0 else None
        ),
        "mean_absolute_relative_error": statistics.fmean(relative_errors),
        "median_absolute_relative_error": statistics.median(relative_errors),
        "maximum_absolute_relative_error": max(relative_errors),
        "run_diagnostics": [
            {
                "order": row["order"],
                "run_id": row["run_id"],
                "cell_id": row["cell_id"],
                "decode_seconds_per_useful_token": design_row[3],
                "observed_joules_per_token": observed,
                "predicted_joules_per_token": predicted,
                "residual_joules_per_token": residual,
                "absolute_relative_error": relative_error,
                "leverage": leverage,
            }
            for row, design_row, observed, predicted, residual, relative_error, leverage in zip(
                rows,
                design,
                outcome,
                predictions,
                residuals,
                relative_errors,
                leverages,
            )
        ],
    }


def duration_calibration_envelope(
    rows: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]
) -> dict[str, Any]:
    coefficients = fit["coefficients"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["cell_id"]), []).append(dict(row))
    if not groups:
        raise ValueError("calibration split contains no cells")
    gamma = 0.05
    reports: list[dict[str, Any]] = []
    for cell_id, cell_rows in sorted(groups.items()):
        residuals = [
            float(row["gross_gpu_joules_per_token"])
            - _prediction(row, coefficients)
            for row in cell_rows
        ]
        if len(residuals) < 2:
            raise ValueError(f"{cell_id}: calibration interval needs at least two runs")
        degrees = len(residuals) - 1
        quantile = student_t_quantile(1.0 - gamma / (2.0 * len(groups)), degrees)
        mean = statistics.fmean(residuals)
        standard_error = statistics.stdev(residuals) / math.sqrt(len(residuals))
        reports.append(
            {
                "cell_id": cell_id,
                "runs": len(residuals),
                "mean_residual_joules_per_token": mean,
                "sample_standard_deviation_joules_per_token": statistics.stdev(
                    residuals
                ),
                "student_t_quantile": quantile,
                "interval_joules_per_token": [
                    mean - quantile * standard_error,
                    mean + quantile * standard_error,
                ],
                "run_residuals": [
                    {
                        "run_id": row["run_id"],
                        "residual_joules_per_token": residual,
                    }
                    for row, residual in zip(cell_rows, residuals)
                ],
            }
        )
    return {
        "familywise_gamma": gamma,
        "cell_count": len(reports),
        "method": "Bonferroni Student-t intervals for prespecified cell-mean residuals",
        "common_residual_range_joules_per_token": [
            min(row["interval_joules_per_token"][0] for row in reports),
            max(row["interval_joules_per_token"][1] for row in reports),
        ],
        "cells": reports,
    }


def freeze_model_replication_identification(
    campaign_lock_path: Path,
    addendum_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    runtime = addendum.get("replication_runtime", {})
    result_domain = str(runtime.get("result_domain", "qwen14b-identification"))
    measurement_prefix = str(
        runtime.get("measurement_prefix", "gh200-qwen2p5-14b")
    )
    analysis_id = str(
        runtime.get(
            "analysis_id",
            addendum.get("analysis_policy", {}).get(
                "analysis_id", "qwen14b-duration-identification-v1"
            ),
        )
    )
    lock, rows, issues = collect_identification_rows(
        campaign_lock_path,
        results_root,
        result_domain=result_domain,
    )
    addendum_sha = sha256_file(addendum_path)
    expected_addendum_shas = {
        str(run["parameters"].get("execution_addendum_sha256"))
        for run in lock["run_order"]
    }
    if expected_addendum_shas != {addendum_sha}:
        issues.append("form-replication addendum digest does not match every run")
    evidence = addendum["qualification_evidence"]
    expected_qualification_shas = {
        str(run["parameters"].get("qualification_summary_sha256"))
        for run in lock["run_order"]
    }
    if expected_qualification_shas != {evidence["qualification_summary_sha256"]}:
        issues.append("qualification summary digest does not match every run")

    design = addendum["identification_design"]
    expected_counts = {
        FIT_SPLIT: int(design["coefficient_fit"]["run_count"]),
        CALIBRATION_SPLIT: int(design["residual_calibration"]["run_count"]),
    }
    observed_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in expected_counts
    }
    for split, expected in expected_counts.items():
        if observed_counts[split] != expected:
            issues.append(f"{split}: expected {expected} runs; found {observed_counts[split]}")
    if len(rows) != int(lock["run_count"]):
        issues.append(f"expected {lock['run_count']} accepted runs; found {len(rows)}")
    for row in rows:
        if round(
            float(row["weight_bytes_per_token"]) * int(row["target_batch"])
        ) != int(
            evidence["observed_unique_weight_storage_bytes"]
        ):
            issues.append(f"{row['run_id']}: weight storage differs from qualification")
        if int(row["kv_write_bytes_per_token"]) != int(
            evidence["observed_logical_kv_bytes_per_attended_token"]
        ):
            issues.append(f"{row['run_id']}: KV geometry differs from qualification")
    if issues:
        report = {
            "schema_version": 1,
            "measurement": f"{measurement_prefix}-identification-freeze",
            "accepted_run_count": len(rows),
            "issues": issues,
            "qc_pass": False,
        }
        _atomic_json(output_dir / "identification-freeze-summary.json", report)
        return report

    fit_rows = [row for row in rows if row["split"] == FIT_SPLIT]
    calibration_rows = [row for row in rows if row["split"] == CALIBRATION_SPLIT]
    fit = fit_duration_ols_hc3(fit_rows, analysis_id=analysis_id)
    bundle_sha = _alignment_bundle_sha(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_artifact = {
        "schema_version": 1,
        "measurement": f"{measurement_prefix}-duration-coefficient-artifact",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "campaign_lock_file_sha256": sha256_file(campaign_lock_path),
        "form_replication_addendum_sha256": addendum_sha,
        "qualification_summary_sha256": evidence["qualification_summary_sha256"],
        "alignment_bundle_sha256": bundle_sha,
        "outcome_contract": {
            "estimand": OUTCOME,
            "scope": "GPU-board NVML scope-0 instantaneous-power integral",
            "idle_correction_applied": False,
        },
        "fit": fit,
        "qc_pass": True,
    }
    coefficient_path = output_dir / "coefficient-artifact.json"
    _atomic_json(coefficient_path, coefficient_artifact)
    coefficient_sha = sha256_file(coefficient_path)

    envelope = duration_calibration_envelope(calibration_rows, fit)
    envelope_artifact = {
        "schema_version": 1,
        "measurement": f"{measurement_prefix}-duration-discrepancy-envelope",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "alignment_bundle_sha256": bundle_sha,
        "coefficient_artifact_sha256": coefficient_sha,
        "calibration_run_count": len(calibration_rows),
        "calibration_cell_count": len({row["cell_id"] for row in calibration_rows}),
        "envelope": envelope,
        "qc_pass": True,
    }
    envelope_path = output_dir / "discrepancy-envelope.json"
    _atomic_json(envelope_path, envelope_artifact)

    table_path = output_dir / "accepted-runs.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    traffic_gate = fit["traffic_slope_positivity"]["qc_pass"]
    time_gate = fit["time_term"]["finite_nonnegative_qc_pass"]
    scientific_gate = traffic_gate and time_gate
    summary = {
        "schema_version": 1,
        "measurement": f"{measurement_prefix}-identification-freeze",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "accepted_run_count": len(rows),
        "fit_run_count": len(fit_rows),
        "calibration_run_count": len(calibration_rows),
        "coefficient_artifact": {
            "path": coefficient_path.name,
            "sha256": coefficient_sha,
        },
        "discrepancy_envelope": {
            "path": envelope_path.name,
            "sha256": sha256_file(envelope_path),
        },
        "accepted_run_table": {
            "path": table_path.name,
            "sha256": sha256_file(table_path),
        },
        "scientific_decisions": {
            "traffic_slope_positivity_pass": traffic_gate,
            "time_term_finite_nonnegative_pass": time_gate,
            "identification_gate_pass": scientific_gate,
            "cross_model_form_confirmed": False,
        },
        "holdout_release_candidate": scientific_gate,
        "holdout_release_note": (
            "A separate content-addressed release must bind these artifacts "
            "before any sealed holdout run may execute."
        ),
        "issues": [],
        "qc_pass": True,
    }
    _atomic_json(output_dir / "identification-freeze-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--form-replication-addendum", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = freeze_model_replication_identification(
        args.campaign_lock,
        args.form_replication_addendum,
        args.results_root,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
