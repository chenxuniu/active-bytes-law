"""Freeze the GH200 BF16 identification fit and calibration envelope.

The frozen campaign records gross scope-0 GPU-board joules per useful token as
its primary outcome.  This analysis therefore does not subtract an idle power
estimate.  The separately generated idle-window audit may be bound as
diagnostic provenance, but its samples never change an outcome in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIT_SPLIT = "coefficient-fit"
CALIBRATION_SPLIT = "residual-calibration"
OUTCOME = "gross_gpu_board_joules_per_useful_token"
BYTES_PER_DECIMAL_GB = 1_000_000_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _betacf(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x
        denominator = (qam + m2) * (a + m2)
        aa = numerator / denominator
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x
        aa /= (a + m2) * (qap + m2)
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("invalid regularized-beta arguments")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_beta(x, degrees_of_freedom / 2.0, 0.5)
    return 1.0 - tail if value > 0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_quantile(1.0 - probability, degrees_of_freedom)
    low = 0.0
    high = 1.0
    while student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
    for _ in range(120):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _invert(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    augmented = [
        [float(value) for value in row]
        + [1.0 if index == column else 0.0 for column in range(size)]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(left * right for left, right in zip(row, vector)) for row in matrix]


def _quadratic(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> float:
    return sum(
        vector[row] * matrix[row][column] * vector[column]
        for row in range(len(vector))
        for column in range(len(vector))
    )


def fit_ols_hc3(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) <= 3:
        raise ValueError("the three-parameter fit requires more than three runs")
    design = [
        [1.0, float(row["weight_gb_per_token"]), float(row["kv_rw_gb_per_token"])]
        for row in rows
    ]
    outcome = [float(row["gross_gpu_joules_per_token"]) for row in rows]
    parameter_count = 3
    xtx = [
        [sum(row[left] * row[right] for row in design) for right in range(parameter_count)]
        for left in range(parameter_count)
    ]
    xty = [
        sum(row[column] * value for row, value in zip(design, outcome))
        for column in range(parameter_count)
    ]
    xtx_inverse = _invert(xtx)
    coefficients = _matvec(xtx_inverse, xty)
    predictions = [sum(x * beta for x, beta in zip(row, coefficients)) for row in design]
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
    standard_errors = [math.sqrt(max(covariance[index][index], 0.0)) for index in range(3)]
    degrees_of_freedom = len(rows) - parameter_count
    slope_lower_quantile = student_t_quantile(1.0 - 0.05 / 2.0, degrees_of_freedom)
    slope_lowers = [
        coefficients[index] - slope_lower_quantile * standard_errors[index]
        for index in (1, 2)
    ]
    centered_total = sum((value - statistics.fmean(outcome)) ** 2 for value in outcome)
    residual_sum = sum(value * value for value in residuals)
    r_squared = 1.0 - residual_sum / centered_total if centered_total > 0 else None
    mape = statistics.fmean(
        abs(residual) / max(abs(value), 1e-12)
        for residual, value in zip(residuals, outcome)
    )
    ratio = None
    if coefficients[1] > 0 and coefficients[2] > 0:
        log_ratio = math.log(coefficients[2] / coefficients[1])
        gradient = [0.0, -1.0 / coefficients[1], 1.0 / coefficients[2]]
        ratio_se = math.sqrt(max(_quadratic(gradient, covariance), 0.0))
        tost_quantile = student_t_quantile(0.95, degrees_of_freedom)
        lower_log = log_ratio - tost_quantile * ratio_se
        upper_log = log_ratio + tost_quantile * ratio_se
        ratio = {
            "beta_over_alpha": math.exp(log_ratio),
            "log_beta_over_alpha": log_ratio,
            "delta_method_standard_error": ratio_se,
            "tost_interval_confidence": 0.90,
            "tost_interval_log": [lower_log, upper_log],
            "tost_interval_beta_over_alpha": [math.exp(lower_log), math.exp(upper_log)],
            "equivalence_bounds_beta_over_alpha": [1.0 / 1.2, 1.2],
            "equivalence_qc_pass": lower_log >= -math.log(1.2) and upper_log <= math.log(1.2),
        }
    return {
        "estimator": "unweighted run-level OLS with HC3 covariance",
        "predictor_units": ["intercept", "decimal_GB_weight_per_token", "decimal_GB_KV_read_write_per_token"],
        "outcome": OUTCOME,
        "observations": len(rows),
        "parameters": parameter_count,
        "degrees_of_freedom": degrees_of_freedom,
        "coefficients_scaled": {
            "c_joules_per_token": coefficients[0],
            "alpha_joules_per_decimal_gb": coefficients[1],
            "beta_joules_per_decimal_gb": coefficients[2],
        },
        "coefficients_per_byte": {
            "alpha_joules_per_byte": coefficients[1] / BYTES_PER_DECIMAL_GB,
            "beta_joules_per_byte": coefficients[2] / BYTES_PER_DECIMAL_GB,
        },
        "standard_errors_scaled": {
            "c_joules_per_token": standard_errors[0],
            "alpha_joules_per_decimal_gb": standard_errors[1],
            "beta_joules_per_decimal_gb": standard_errors[2],
        },
        "hc3_covariance_scaled": covariance,
        "slope_positivity": {
            "familywise_alpha": 0.05,
            "bonferroni_one_sided_quantile": slope_lower_quantile,
            "alpha_lower_joules_per_decimal_gb": slope_lowers[0],
            "beta_lower_joules_per_decimal_gb": slope_lowers[1],
            "qc_pass": slope_lowers[0] > 0.0 and slope_lowers[1] > 0.0,
        },
        "coefficient_equivalence": ratio,
        "centered_r_squared": r_squared,
        "mean_absolute_percentage_error": mape,
        "run_diagnostics": [
            {
                "order": row["order"],
                "run_id": row["run_id"],
                "cell_id": row["cell_id"],
                "observed_joules_per_token": observed,
                "predicted_joules_per_token": predicted,
                "residual_joules_per_token": residual,
                "leverage": leverage,
            }
            for row, observed, predicted, residual, leverage in zip(
                rows, outcome, predictions, residuals, leverages
            )
        ],
    }


def _accepted_alignment(
    run_root: Path, run: Mapping[str, Any], campaign_sha: str
) -> tuple[Path | None, list[str]]:
    accepted: list[Path] = []
    issues: list[str] = []
    for path in sorted(run_root.glob("attempt-*/alignment.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            issues.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        observed = value.get("run", {})
        if (
            value.get("qc_pass") is True
            and value.get("campaign_lock_sha256") == campaign_sha
            and all(observed.get(key) == run[key] for key in ("run_id", "cell_id", "repeat", "split", "order"))
        ):
            accepted.append(path)
    if len(accepted) != 1:
        issues.append(
            f"{run['run_id']}: expected exactly one accepted alignment; found {len(accepted)}"
        )
        return None, issues
    return accepted[0], issues


def collect_identification_rows(
    campaign_lock_path: Path,
    results_root: Path,
    *,
    result_domain: str = "primary-identification",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    lock = json.loads(campaign_lock_path.read_text(encoding="utf-8"))
    campaign_sha = lock["lock_sha256"]
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    weight_bytes_values: set[int] = set()
    kv_bytes_values: set[int] = set()
    for run in sorted(lock["run_order"], key=lambda row: row["order"]):
        path, run_issues = _accepted_alignment(
            results_root / result_domain / run["run_id"], run, campaign_sha
        )
        issues.extend(run_issues)
        if path is None:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        totals = value.get("totals", {})
        active = value.get("active_bytes", {})
        weights = value.get("weights", {})
        geometry = value.get("model_geometry", {})
        try:
            gross = float(totals["gpu_joules_per_token"])
            weight_bytes = int(weights["unique_storage_bytes"])
            kv_bytes = int(geometry["kv_bytes_per_historical_token"])
            weight_per_token = float(active["weight_bytes_per_token"])
            kv_read_per_token = float(active["kv_read_bytes_per_token"])
            kv_write_per_token = float(active["kv_write_bytes_per_token"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{run['run_id']}: malformed scientific fields: {exc}")
            continue
        expected_weight = weight_bytes / int(run["parameters"]["target_batch"])
        expected_kv_read = kv_bytes * int(
            run["parameters"]["target_mean_attended_history_tokens"]
        )
        if not math.isclose(weight_per_token, expected_weight, rel_tol=0.0, abs_tol=1e-6):
            issues.append(f"{run['run_id']}: weight-byte coordinate disagrees with lock")
        if not math.isclose(kv_read_per_token, expected_kv_read, rel_tol=0.0, abs_tol=1e-6):
            issues.append(f"{run['run_id']}: KV-read coordinate disagrees with lock")
        if not math.isclose(kv_write_per_token, kv_bytes, rel_tol=0.0, abs_tol=1e-6):
            issues.append(f"{run['run_id']}: KV-write coordinate disagrees with runtime geometry")
        weight_bytes_values.add(weight_bytes)
        kv_bytes_values.add(kv_bytes)
        kv_rw = kv_read_per_token + kv_write_per_token
        rows.append(
            {
                "order": run["order"],
                "run_id": run["run_id"],
                "cell_id": run["cell_id"],
                "split": run["split"],
                "repeat": run["repeat"],
                "target_batch": run["parameters"]["target_batch"],
                "target_mean_attended_history_tokens": run["parameters"][
                    "target_mean_attended_history_tokens"
                ],
                "gross_gpu_joules_per_token": gross,
                "decode_seconds": float(totals["decode_seconds"]),
                "metered_useful_tokens": int(totals["metered_useful_tokens"]),
                "weight_bytes_per_token": weight_per_token,
                "kv_read_bytes_per_token": kv_read_per_token,
                "kv_write_bytes_per_token": kv_write_per_token,
                "kv_rw_bytes_per_token": kv_rw,
                "weight_gb_per_token": weight_per_token / BYTES_PER_DECIMAL_GB,
                "kv_rw_gb_per_token": kv_rw / BYTES_PER_DECIMAL_GB,
                "alignment_sha256": sha256_file(path),
                "alignment_path_relative_to_results": str(path.relative_to(results_root)),
            }
        )
    if len(weight_bytes_values) != 1:
        issues.append(f"weight storage is not invariant: {sorted(weight_bytes_values)}")
    if len(kv_bytes_values) != 1:
        issues.append(f"KV bytes/token is not invariant: {sorted(kv_bytes_values)}")
    return lock, rows, issues


def _alignment_bundle_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["order"]):
        digest.update(str(row["run_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["alignment_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def calibration_envelope(
    rows: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]
) -> dict[str, Any]:
    coefficients = fit["coefficients_scaled"]
    c = float(coefficients["c_joules_per_token"])
    alpha = float(coefficients["alpha_joules_per_decimal_gb"])
    beta = float(coefficients["beta_joules_per_decimal_gb"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["cell_id"]), []).append(dict(row))
    if not groups:
        raise ValueError("calibration split contains no cells")
    gamma = 0.05
    cell_reports: list[dict[str, Any]] = []
    for cell_id, cell_rows in sorted(groups.items()):
        residuals = [
            float(row["gross_gpu_joules_per_token"])
            - (c + alpha * float(row["weight_gb_per_token"]) + beta * float(row["kv_rw_gb_per_token"]))
            for row in cell_rows
        ]
        if len(residuals) < 2:
            raise ValueError(f"{cell_id}: calibration interval needs at least two runs")
        degrees = len(residuals) - 1
        quantile = student_t_quantile(
            1.0 - gamma / (2.0 * len(groups)), degrees
        )
        mean = statistics.fmean(residuals)
        standard_error = statistics.stdev(residuals) / math.sqrt(len(residuals))
        lower = mean - quantile * standard_error
        upper = mean + quantile * standard_error
        cell_reports.append(
            {
                "cell_id": cell_id,
                "runs": len(residuals),
                "mean_residual_joules_per_token": mean,
                "sample_standard_deviation_joules_per_token": statistics.stdev(residuals),
                "student_t_quantile": quantile,
                "interval_joules_per_token": [lower, upper],
                "run_residuals": [
                    {"run_id": row["run_id"], "residual_joules_per_token": residual}
                    for row, residual in zip(cell_rows, residuals)
                ],
            }
        )
    return {
        "familywise_gamma": gamma,
        "cell_count": len(cell_reports),
        "method": "Bonferroni Student-t intervals for prespecified cell-mean residuals",
        "common_residual_range_joules_per_token": [
            min(row["interval_joules_per_token"][0] for row in cell_reports),
            max(row["interval_joules_per_token"][1] for row in cell_reports),
        ],
        "cells": cell_reports,
    }


def freeze_primary_identification(
    campaign_lock_path: Path,
    results_root: Path,
    output_dir: Path,
    *,
    idle_audit_path: Path | None = None,
) -> dict[str, Any]:
    lock, rows, issues = collect_identification_rows(campaign_lock_path, results_root)
    expected_counts = {FIT_SPLIT: 30, CALIBRATION_SPLIT: 15}
    observed_counts = {
        split: sum(row["split"] == split for row in rows) for split in expected_counts
    }
    for split, expected in expected_counts.items():
        if observed_counts[split] != expected:
            issues.append(f"{split}: expected {expected} runs; found {observed_counts[split]}")
    if len(rows) != int(lock["run_count"]):
        issues.append(f"expected {lock['run_count']} accepted runs; found {len(rows)}")
    idle_audit = None
    if idle_audit_path is not None:
        idle_value = json.loads(idle_audit_path.read_text(encoding="utf-8"))
        if idle_value.get("campaign_lock_sha256") != lock["lock_sha256"]:
            issues.append("idle audit campaign lock does not match identification lock")
        if idle_value.get("accepted_run_count") != lock["run_count"]:
            issues.append("idle audit does not cover every identification run")
        idle_audit = {
            "path": str(idle_audit_path),
            "sha256": sha256_file(idle_audit_path),
            "artifact_qc_pass": idle_value.get("artifact_qc_pass"),
            "all_runs_have_bracketing_samples": idle_value.get(
                "all_runs_have_bracketing_samples"
            ),
            "paper_outcome_eligible": idle_value.get(
                "idle_correction_contract", {}
            ).get("paper_outcome_eligible"),
        }
    if issues:
        return {
            "schema_version": 1,
            "measurement": "gh200-primary-identification-freeze",
            "qc_pass": False,
            "issues": issues,
            "accepted_run_count": len(rows),
            "output_dir": str(output_dir),
        }

    fit_rows = [row for row in rows if row["split"] == FIT_SPLIT]
    calibration_rows = [row for row in rows if row["split"] == CALIBRATION_SPLIT]
    fit = fit_ols_hc3(fit_rows)
    bundle_sha = _alignment_bundle_sha(rows)
    coefficient_artifact = {
        "schema_version": 1,
        "measurement": "gh200-primary-gross-energy-coefficient-artifact",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "campaign_lock_file_sha256": sha256_file(campaign_lock_path),
        "alignment_bundle_sha256": bundle_sha,
        "outcome_contract": {
            "estimand": OUTCOME,
            "scope": "GPU-board NVML scope-0 instantaneous-power integral",
            "idle_correction_applied": False,
            "reason": "The pre-outcome campaign lock froze gross board energy as primary.",
        },
        "fit_split": FIT_SPLIT,
        "fit_run_count": len(fit_rows),
        "fit_cell_count": len({row["cell_id"] for row in fit_rows}),
        "fit": fit,
        "idle_window_audit_diagnostic": idle_audit,
        "qc_pass": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_path = output_dir / "coefficient-artifact.json"
    _atomic_json(coefficient_path, coefficient_artifact)
    coefficient_sha = sha256_file(coefficient_path)

    envelope = calibration_envelope(calibration_rows, fit)
    envelope_artifact = {
        "schema_version": 1,
        "measurement": "gh200-primary-gross-energy-discrepancy-envelope",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "alignment_bundle_sha256": bundle_sha,
        "coefficient_artifact_sha256": coefficient_sha,
        "outcome": OUTCOME,
        "calibration_split": CALIBRATION_SPLIT,
        "calibration_run_count": len(calibration_rows),
        "calibration_cell_count": len({row["cell_id"] for row in calibration_rows}),
        "envelope": envelope,
        "qc_pass": True,
    }
    envelope_path = output_dir / "discrepancy-envelope.json"
    _atomic_json(envelope_path, envelope_artifact)
    envelope_sha = sha256_file(envelope_path)

    run_table_path = output_dir / "accepted-runs.csv"
    with run_table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "measurement": "gh200-primary-identification-freeze",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "outcome": OUTCOME,
        "accepted_run_count": len(rows),
        "fit_run_count": len(fit_rows),
        "calibration_run_count": len(calibration_rows),
        "coefficient_artifact": {
            "path": coefficient_path.name,
            "sha256": coefficient_sha,
        },
        "discrepancy_envelope": {
            "path": envelope_path.name,
            "sha256": envelope_sha,
        },
        "accepted_run_table": {
            "path": run_table_path.name,
            "sha256": sha256_file(run_table_path),
        },
        "scientific_decisions": {
            "p2_slope_positivity_pass": fit["slope_positivity"]["qc_pass"],
            "p3_coefficient_equivalence_pass": (
                fit["coefficient_equivalence"] is not None
                and fit["coefficient_equivalence"]["equivalence_qc_pass"]
            ),
        },
        "evaluation_release_candidate": True,
        "evaluation_release_note": (
            "A separate fail-closed release record must bind these two digests "
            "before any evaluation run is opened. Scientific gate failures do "
            "not authorize refitting or relabeling."
        ),
        "issues": [],
        "qc_pass": True,
    }
    _atomic_json(output_dir / "identification-freeze-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--idle-audit-json", type=Path)
    args = parser.parse_args(argv)
    report = freeze_primary_identification(
        args.campaign_lock,
        args.results_root,
        args.output_dir,
        idle_audit_path=args.idle_audit_json,
    )
    if not report.get("qc_pass"):
        _atomic_json(args.output_dir / "identification-freeze-summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
