"""Evaluate the sealed GH200 duration-augmented V2 holdout without refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .primary_identification import collect_identification_rows, sha256_file


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


def evaluate_v2_rows(
    rows: Sequence[Mapping[str, Any]], model_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = model_artifact["frozen_model"]
    coefficients = frozen["coefficients"]
    intercept = float(coefficients["intercept_joules_per_token"])
    alpha = float(coefficients["alpha_weight_joules_per_decimal_gb"])
    beta = float(coefficients["beta_kv_joules_per_decimal_gb"])
    p_time = float(coefficients["p_time_watts"])
    gates = model_artifact["primary_gates"]

    run_reports: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        duration_per_token = float(row["decode_seconds"]) / int(
            row["metered_useful_tokens"]
        )
        prediction = (
            intercept
            + alpha * float(row["weight_gb_per_token"])
            + beta * float(row["kv_rw_gb_per_token"])
            + p_time * duration_per_token
        )
        observed = float(row["gross_gpu_joules_per_token"])
        residual = observed - prediction
        row.update(
            {
                "decode_seconds_per_useful_token": duration_per_token,
                "frozen_prediction_joules_per_token": prediction,
                "residual_joules_per_token": residual,
                "absolute_relative_error": abs(residual) / abs(observed),
            }
        )
        run_reports.append(row)
        groups.setdefault(str(row["cell_id"]), []).append(row)

    expected_cells = int(model_artifact["new_holdout"]["cell_count"])
    expected_repeats = int(model_artifact["new_holdout"]["repetitions_per_cell"])
    if len(groups) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells; found {len(groups)}")

    cell_reports: list[dict[str, Any]] = []
    for cell_id, cell_rows in sorted(
        groups.items(),
        key=lambda item: (
            int(item[1][0]["target_mean_attended_history_tokens"]),
            int(item[1][0]["target_batch"]),
        ),
    ):
        if len(cell_rows) != expected_repeats:
            raise ValueError(
                f"{cell_id}: expected {expected_repeats} repeats; found {len(cell_rows)}"
            )
        observed = statistics.fmean(
            float(row["gross_gpu_joules_per_token"]) for row in cell_rows
        )
        prediction = statistics.fmean(
            float(row["frozen_prediction_joules_per_token"]) for row in cell_rows
        )
        residual = observed - prediction
        absolute_relative_error = abs(residual) / abs(observed)
        cell_reports.append(
            {
                "cell_id": cell_id,
                "target_mean_attended_history_tokens": int(
                    cell_rows[0]["target_mean_attended_history_tokens"]
                ),
                "target_batch": int(cell_rows[0]["target_batch"]),
                "repeat_count": len(cell_rows),
                "run_ids": [str(row["run_id"]) for row in cell_rows],
                "observed_mean_joules_per_token": observed,
                "observed_sample_standard_deviation_joules_per_token": statistics.stdev(
                    float(row["gross_gpu_joules_per_token"]) for row in cell_rows
                ),
                "frozen_prediction_mean_joules_per_token": prediction,
                "mean_residual_joules_per_token": residual,
                "absolute_relative_error": absolute_relative_error,
                "passes_ten_percent_error": absolute_relative_error <= 0.10,
            }
        )

    errors = [float(row["absolute_relative_error"]) for row in cell_reports]
    passing_cells = sum(bool(row["passes_ten_percent_error"]) for row in cell_reports)
    median_error = statistics.median(errors)
    maximum_error = max(errors)
    median_pass = median_error <= float(
        gates["median_absolute_relative_error_maximum"]
    )
    maximum_pass = maximum_error <= float(
        gates["maximum_absolute_relative_error_maximum"]
    )
    count_pass = passing_cells >= int(gates["required_cells_passing_ten_percent_error"])
    scientific_pass = median_pass and maximum_pass and count_pass
    return {
        "prediction_contract": {
            "equation": frozen["equation"],
            "coefficients": coefficients,
            "coefficient_refit_performed": False,
            "duration_is_observed_over_same_decode_interval": True,
        },
        "run_count": len(run_reports),
        "cell_count": len(cell_reports),
        "runs": run_reports,
        "cells": cell_reports,
        "summary": {
            "median_absolute_relative_error": median_error,
            "maximum_absolute_relative_error": maximum_error,
            "cells_passing_ten_percent_error": passing_cells,
            "cells_total": len(cell_reports),
        },
        "gates": {
            "median_absolute_relative_error_pass": median_pass,
            "maximum_absolute_relative_error_pass": maximum_pass,
            "required_cell_count_pass": count_pass,
            "duration_v2_holdout_pass": scientific_pass,
        },
    }


def evaluate_campaign(
    campaign_lock_path: Path,
    model_artifact_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    lock, rows, issues = collect_identification_rows(
        campaign_lock_path,
        results_root,
        result_domain="duration-v2-holdout",
    )
    model_sha = sha256_file(model_artifact_path)
    expected_model_shas = {
        str(run["parameters"].get("v2_model_artifact_sha256"))
        for run in lock["run_order"]
    }
    if expected_model_shas != {model_sha}:
        issues.append("V2 model artifact digest does not match every locked run")
    expected_runs = int(lock["run_count"])
    if len(rows) != expected_runs:
        issues.append(f"expected {expected_runs} accepted runs; found {len(rows)}")
    if issues:
        report = {
            "schema_version": 1,
            "measurement": "gh200-duration-v2-held-out-evaluation",
            "accepted_run_count": len(rows),
            "issues": issues,
            "qc_pass": False,
        }
        _atomic_json(output_dir / "duration-v2-summary.json", report)
        return report

    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    analysis = evaluate_v2_rows(rows, artifact)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "duration-v2-runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(analysis["runs"])
    scalar_cells = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in analysis["cells"]
    ]
    cells_path = output_dir / "duration-v2-cells.csv"
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_cells[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_cells)

    report = {
        "schema_version": 1,
        "measurement": "gh200-duration-v2-held-out-evaluation",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "alignment_bundle_sha256": _alignment_bundle_sha(rows),
        "model_artifact_sha256": model_sha,
        "analysis": analysis,
        "tables": {
            "runs": {"path": runs_path.name, "sha256": sha256_file(runs_path)},
            "cells": {"path": cells_path.name, "sha256": sha256_file(cells_path)},
        },
        "issues": [],
        "qc_pass": True,
    }
    report_path = output_dir / "duration-v2-evaluation.json"
    _atomic_json(report_path, report)
    summary = {
        "schema_version": 1,
        "measurement": report["measurement"],
        "campaign_lock_sha256": report["campaign_lock_sha256"],
        "alignment_bundle_sha256": report["alignment_bundle_sha256"],
        "model_artifact_sha256": model_sha,
        "accepted_run_count": analysis["run_count"],
        "cell_count": analysis["cell_count"],
        "summary": analysis["summary"],
        "gates": analysis["gates"],
        "cells": analysis["cells"],
        "evaluation_json": {
            "path": report_path.name,
            "sha256": sha256_file(report_path),
        },
        "qc_pass": True,
    }
    _atomic_json(output_dir / "duration-v2-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = evaluate_campaign(
        args.campaign_lock, args.model_artifact, args.results_root, args.output_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
