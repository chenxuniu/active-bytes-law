"""Evaluate the sealed GH200 cells without refitting the frozen law."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation_release import sha256_file, verify_evaluation_release
from .primary_identification import (
    _invert,
    _matvec,
    _quadratic,
    collect_identification_rows,
    student_t_quantile,
)


DENOMINATOR_FLOOR_JOULES_PER_TOKEN = 0.05
MEDIAN_RELATIVE_ERROR_LIMIT = 0.15
MEDIAN_RELATIVE_HALF_WIDTH_LIMIT = 0.15
RESIDUAL_TREND_EQUIVALENCE_LIMIT = 0.05


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _alignment_bundle_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["order"]):
        digest.update(str(row["run_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["alignment_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hc3_trends(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(cell_rows) <= 3:
        raise ValueError("residual-trend regression requires more than three cells")
    log_batch = [math.log2(float(row["target_batch"])) for row in cell_rows]
    log_context = [
        math.log2(float(row["target_mean_attended_history_tokens"]) + 1.0)
        for row in cell_rows
    ]
    mean_batch = statistics.fmean(log_batch)
    mean_context = statistics.fmean(log_context)
    design = [
        [1.0, batch - mean_batch, context - mean_context]
        for batch, context in zip(log_batch, log_context)
    ]
    outcome = [float(row["signed_relative_residual"]) for row in cell_rows]
    parameter_count = 3
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
        raise ValueError("one or more residual-trend HC3 leverages are numerically one")
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
        math.sqrt(max(covariance[index][index], 0.0)) for index in range(parameter_count)
    ]
    degrees = len(cell_rows) - parameter_count
    # One required stratum, two slopes, familywise 90%: per-tail 0.10 / 4.
    quantile = student_t_quantile(1.0 - 0.10 / 4.0, degrees)
    names = ("intercept", "log2_batch", "log2_context_plus_one")
    estimates = dict(zip(names, coefficients))
    intervals = {
        names[index]: [
            coefficients[index] - quantile * standard_errors[index],
            coefficients[index] + quantile * standard_errors[index],
        ]
        for index in range(parameter_count)
    }
    slope_pass = {
        name: (
            intervals[name][0] >= -RESIDUAL_TREND_EQUIVALENCE_LIMIT
            and intervals[name][1] <= RESIDUAL_TREND_EQUIVALENCE_LIMIT
        )
        for name in ("log2_batch", "log2_context_plus_one")
    }
    return {
        "estimator": "cell-mean OLS with HC3 covariance",
        "observations": len(cell_rows),
        "degrees_of_freedom": degrees,
        "predictors": ["centered_log2_batch", "centered_log2_context_plus_one"],
        "centers": {
            "log2_batch": mean_batch,
            "log2_context_plus_one": mean_context,
        },
        "coefficients": estimates,
        "standard_errors": dict(zip(names, standard_errors)),
        "hc3_covariance": covariance,
        "familywise_confidence": 0.90,
        "per_tail_probability": 0.10 / 4.0,
        "student_t_quantile": quantile,
        "intervals": intervals,
        "equivalence_bounds_residual_fraction_per_doubling": [
            -RESIDUAL_TREND_EQUIVALENCE_LIMIT,
            RESIDUAL_TREND_EQUIVALENCE_LIMIT,
        ],
        "slope_qc_pass": slope_pass,
        "qc_pass": all(slope_pass.values()),
    }


def evaluate_held_out_rows(
    rows: Sequence[Mapping[str, Any]],
    coefficients: Mapping[str, Any],
    residual_range: Sequence[float],
    *,
    slope_positivity_pass: bool,
    coefficient_equivalence_pass: bool,
) -> dict[str, Any]:
    if len(residual_range) != 2 or residual_range[0] > residual_range[1]:
        raise ValueError("residual range must contain ordered lower and upper bounds")
    c = float(coefficients["c_joules_per_token"])
    alpha = float(coefficients["alpha_joules_per_decimal_gb"])
    beta = float(coefficients["beta_joules_per_decimal_gb"])
    lower_residual, upper_residual = map(float, residual_range)
    groups: dict[str, list[dict[str, Any]]] = {}
    run_reports: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        prediction = (
            c
            + alpha * float(row["weight_gb_per_token"])
            + beta * float(row["kv_rw_gb_per_token"])
        )
        observed = float(row["gross_gpu_joules_per_token"])
        residual = observed - prediction
        row.update(
            {
                "predicted_gpu_joules_per_token": prediction,
                "residual_joules_per_token": residual,
            }
        )
        run_reports.append(row)
        groups.setdefault(str(row["cell_id"]), []).append(row)
    if len(groups) != 6:
        raise ValueError(f"expected six evaluation cells; found {len(groups)}")

    cell_reports: list[dict[str, Any]] = []
    for cell_id, cell_rows in sorted(
        groups.items(),
        key=lambda item: (
            int(item[1][0]["target_mean_attended_history_tokens"]),
            int(item[1][0]["target_batch"]),
        ),
    ):
        if len(cell_rows) != 5:
            raise ValueError(f"{cell_id}: expected five accepted repeats; found {len(cell_rows)}")
        contexts = {int(row["target_mean_attended_history_tokens"]) for row in cell_rows}
        batches = {int(row["target_batch"]) for row in cell_rows}
        if len(contexts) != 1 or len(batches) != 1:
            raise ValueError(f"{cell_id}: geometry changes across repeats")
        observed_values = [float(row["gross_gpu_joules_per_token"]) for row in cell_rows]
        prediction_values = [
            float(row["predicted_gpu_joules_per_token"]) for row in cell_rows
        ]
        observed = statistics.fmean(observed_values)
        prediction = statistics.fmean(prediction_values)
        residual = observed - prediction
        lower = prediction + lower_residual
        upper = prediction + upper_residual
        t_quantile = student_t_quantile(0.975, len(observed_values) - 1)
        observed_half_width = (
            t_quantile
            * statistics.stdev(observed_values)
            / math.sqrt(len(observed_values))
        )
        prediction_denominator = max(
            abs(prediction), DENOMINATOR_FLOOR_JOULES_PER_TOKEN
        )
        observed_denominator = max(
            abs(observed), DENOMINATOR_FLOOR_JOULES_PER_TOKEN
        )
        cell_reports.append(
            {
                "cell_id": cell_id,
                "target_mean_attended_history_tokens": next(iter(contexts)),
                "target_batch": next(iter(batches)),
                "repeat_count": len(cell_rows),
                "run_ids": [str(row["run_id"]) for row in cell_rows],
                "observed_mean_joules_per_token": observed,
                "observed_sample_standard_deviation_joules_per_token": statistics.stdev(
                    observed_values
                ),
                "observed_mean_95_percent_interval_joules_per_token": [
                    observed - observed_half_width,
                    observed + observed_half_width,
                ],
                "frozen_prediction_joules_per_token": prediction,
                "frozen_envelope_joules_per_token": [lower, upper],
                "mean_residual_joules_per_token": residual,
                "covered": lower <= observed <= upper,
                "relative_half_width": (upper - lower)
                / (2.0 * prediction_denominator),
                "absolute_relative_error": abs(residual) / observed_denominator,
                "signed_relative_residual": residual / observed_denominator,
                "prediction_denominator_floor_active": abs(prediction)
                < DENOMINATOR_FLOOR_JOULES_PER_TOKEN,
                "observed_denominator_floor_active": abs(observed)
                < DENOMINATOR_FLOOR_JOULES_PER_TOKEN,
            }
        )

    trend = _hc3_trends(cell_reports)
    coverage_count = sum(bool(row["covered"]) for row in cell_reports)
    relative_widths = [float(row["relative_half_width"]) for row in cell_reports]
    absolute_errors = [float(row["absolute_relative_error"]) for row in cell_reports]
    coverage_pass = coverage_count >= math.ceil(0.9 * len(cell_reports))
    width_pass = statistics.median(relative_widths) <= MEDIAN_RELATIVE_HALF_WIDTH_LIMIT
    error_pass = statistics.median(absolute_errors) <= MEDIAN_RELATIVE_ERROR_LIMIT
    v2_pass = (
        slope_positivity_pass
        and coverage_pass
        and width_pass
        and error_pass
        and trend["qc_pass"]
    )
    return {
        "prediction_contract": {
            "model": "c + alpha * weight_decimal_GB_per_token + beta * KV_read_write_decimal_GB_per_token",
            "coefficients": {
                "c_joules_per_token": c,
                "alpha_joules_per_decimal_gb": alpha,
                "beta_joules_per_decimal_gb": beta,
            },
            "residual_range_joules_per_token": [lower_residual, upper_residual],
            "coefficient_refit_performed": False,
            "envelope_recalibration_performed": False,
        },
        "denominator_floor_joules_per_token": DENOMINATOR_FLOOR_JOULES_PER_TOKEN,
        "run_count": len(run_reports),
        "cell_count": len(cell_reports),
        "runs": run_reports,
        "cells": cell_reports,
        "residual_trends": trend,
        "summary": {
            "coverage_count": coverage_count,
            "coverage_total": len(cell_reports),
            "coverage_fraction": coverage_count / len(cell_reports),
            "required_coverage_count": math.ceil(0.9 * len(cell_reports)),
            "median_relative_half_width": statistics.median(relative_widths),
            "median_absolute_relative_error": statistics.median(absolute_errors),
            "maximum_absolute_relative_error": max(absolute_errors),
            "prediction_floor_active_cell_count": sum(
                bool(row["prediction_denominator_floor_active"])
                for row in cell_reports
            ),
            "observed_floor_active_cell_count": sum(
                bool(row["observed_denominator_floor_active"])
                for row in cell_reports
            ),
        },
        "gates": {
            "p2_slope_positivity_pass": slope_positivity_pass,
            "evaluation_coverage_pass": coverage_pass,
            "evaluation_median_relative_half_width_pass": width_pass,
            "evaluation_median_absolute_relative_error_pass": error_pass,
            "evaluation_residual_trends_pass": trend["qc_pass"],
            "v2_two_coefficient_law_pass": v2_pass,
            "p3_single_coefficient_equivalence_pass": coefficient_equivalence_pass,
            "highest_supported_claim": "P2" if v2_pass else "P1",
        },
    }


def evaluate_primary_campaign(
    evaluation_lock_path: Path,
    results_root: Path,
    release_record_path: Path,
    identification_freeze_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    release_verification = verify_evaluation_release(
        release_record_path, identification_freeze_dir, evaluation_lock_path
    )
    issues = list(release_verification["issues"])
    lock, rows, collection_issues = collect_identification_rows(
        evaluation_lock_path,
        results_root,
        result_domain="primary-evaluation",
    )
    issues.extend(collection_issues)
    if len(rows) != 30:
        issues.append(f"expected 30 accepted evaluation runs; found {len(rows)}")
    if any(row["split"] != "evaluation" for row in rows):
        issues.append("a non-evaluation run entered the held-out analysis")
    if issues:
        return {
            "schema_version": 1,
            "measurement": "gh200-primary-held-out-evaluation",
            "accepted_run_count": len(rows),
            "issues": issues,
            "qc_pass": False,
        }

    coefficient_path = identification_freeze_dir / "coefficient-artifact.json"
    envelope_path = identification_freeze_dir / "discrepancy-envelope.json"
    summary_path = identification_freeze_dir / "identification-freeze-summary.json"
    coefficient = json.loads(coefficient_path.read_text(encoding="utf-8"))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    identification_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = evaluate_held_out_rows(
        rows,
        coefficient["fit"]["coefficients_scaled"],
        envelope["envelope"]["common_residual_range_joules_per_token"],
        slope_positivity_pass=bool(
            identification_summary["scientific_decisions"]["p2_slope_positivity_pass"]
        ),
        coefficient_equivalence_pass=bool(
            identification_summary["scientific_decisions"][
                "p3_coefficient_equivalence_pass"
            ]
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "evaluation-runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(analysis["runs"])
    cells_path = output_dir / "evaluation-cells.csv"
    scalar_cells = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
        for row in analysis["cells"]
    ]
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_cells[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_cells)

    report = {
        "schema_version": 1,
        "measurement": "gh200-primary-held-out-evaluation",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "evaluation_alignment_bundle_sha256": _alignment_bundle_sha(rows),
        "release_verification": release_verification,
        "frozen_artifacts": {
            "coefficient_artifact_sha256": sha256_file(coefficient_path),
            "discrepancy_envelope_sha256": sha256_file(envelope_path),
            "identification_summary_sha256": sha256_file(summary_path),
        },
        "analysis": analysis,
        "tables": {
            "evaluation_runs_csv": {
                "path": runs_path.name,
                "sha256": sha256_file(runs_path),
            },
            "evaluation_cells_csv": {
                "path": cells_path.name,
                "sha256": sha256_file(cells_path),
            },
        },
        "issues": [],
        "qc_pass": True,
    }
    report_path = output_dir / "held-out-evaluation.json"
    _atomic_json(report_path, report)
    result = {
        "schema_version": 1,
        "measurement": report["measurement"],
        "campaign_lock_sha256": report["campaign_lock_sha256"],
        "evaluation_alignment_bundle_sha256": report[
            "evaluation_alignment_bundle_sha256"
        ],
        "accepted_run_count": analysis["run_count"],
        "cell_count": analysis["cell_count"],
        "summary": analysis["summary"],
        "gates": analysis["gates"],
        "residual_trends": analysis["residual_trends"],
        "cells": analysis["cells"],
        "held_out_evaluation_json": {
            "path": report_path.name,
            "sha256": sha256_file(report_path),
        },
        "qc_pass": True,
    }
    _atomic_json(output_dir / "held-out-evaluation-summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-lock", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--release-record", required=True, type=Path)
    parser.add_argument("--identification-freeze-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate_primary_campaign(
        args.evaluation_lock,
        args.results_root,
        args.release_record,
        args.identification_freeze_dir,
        args.output_dir,
    )
    if not report.get("qc_pass"):
        _atomic_json(args.output_dir / "held-out-evaluation-summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
