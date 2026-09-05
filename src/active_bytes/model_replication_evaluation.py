"""Evaluate a released model-replication holdout without refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_replication_release import verify_model_replication_release
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


def evaluate_replication_rows(
    rows: Sequence[Mapping[str, Any]],
    coefficients: Mapping[str, Any],
    common_residual_range: Sequence[float],
    gates: Mapping[str, Any],
    *,
    expected_cells: int,
    expected_repeats: int,
    form_replication_gate_name: str = "qwen2p5_14b_form_replication_pass",
) -> dict[str, Any]:
    intercept = float(coefficients["intercept_joules_per_token"])
    alpha = float(coefficients["alpha_weight_joules_per_decimal_gb"])
    beta = float(coefficients["beta_kv_joules_per_decimal_gb"])
    p_time = float(coefficients["p_time_watts"])
    residual_low, residual_high = map(float, common_residual_range)

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
        if observed <= 0.0:
            raise ValueError(f"{row['run_id']}: observed energy must be positive")
        residual = observed - prediction
        row.update(
            {
                "decode_seconds_per_useful_token": duration_per_token,
                "frozen_prediction_joules_per_token": prediction,
                "residual_joules_per_token": residual,
                "absolute_relative_error": abs(residual) / observed,
                "inside_frozen_common_residual_band": (
                    residual_low <= residual <= residual_high
                ),
            }
        )
        run_reports.append(row)
        groups.setdefault(str(row["cell_id"]), []).append(row)

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
        relative_error = abs(residual) / observed
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
                "absolute_relative_error": relative_error,
                "passes_ten_percent_error": relative_error <= 0.10,
                "inside_frozen_common_residual_band": (
                    residual_low <= residual <= residual_high
                ),
            }
        )

    errors = [float(row["absolute_relative_error"]) for row in cell_reports]
    passing_cells = sum(bool(row["passes_ten_percent_error"]) for row in cell_reports)
    covered_cells = sum(
        bool(row["inside_frozen_common_residual_band"]) for row in cell_reports
    )
    median_error = statistics.median(errors)
    maximum_error = max(errors)
    median_pass = median_error <= float(
        gates["median_absolute_relative_error_maximum"]
    )
    maximum_pass = maximum_error <= float(
        gates["maximum_absolute_relative_error_maximum"]
    )
    count_pass = passing_cells >= int(
        gates["required_cells_passing_ten_percent_error"]
    )
    scientific_pass = median_pass and maximum_pass and count_pass
    return {
        "prediction_contract": {
            "equation": (
                "E_token = intercept + alpha_weight * weight_decimal_GB_per_token "
                "+ beta_kv * kv_read_write_decimal_GB_per_token "
                "+ p_time_W * decode_seconds_per_useful_token"
            ),
            "coefficients": dict(coefficients),
            "coefficient_refit_performed": False,
            "duration_is_observed_over_same_decode_interval": True,
            "duration_term_is_retrospective_and_noncausal": True,
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
            "descriptive_frozen_residual_band_coverage_count": covered_cells,
            "descriptive_frozen_residual_band_coverage_fraction": (
                covered_cells / len(cell_reports)
            ),
        },
        "gates": {
            "median_absolute_relative_error_pass": median_pass,
            "maximum_absolute_relative_error_pass": maximum_pass,
            "required_cell_count_pass": count_pass,
            form_replication_gate_name: scientific_pass,
            "residual_band_coverage_is_a_primary_gate": False,
        },
    }


def evaluate_campaign(
    campaign_lock_path: Path,
    release_record_path: Path,
    identification_freeze_dir: Path,
    form_replication_addendum_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    release: Mapping[str, Any] = {}
    try:
        release = json.loads(release_record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    runtime = release.get("replication_runtime", {})
    result_domain = str(runtime.get("result_domain", "qwen14b-holdout"))
    output_prefix = str(runtime.get("output_prefix", "qwen14b-holdout"))
    evaluation_measurement = str(
        runtime.get(
            "evaluation_measurement",
            "gh200-qwen2p5-14b-held-out-form-evaluation",
        )
    )
    form_replication_gate_name = str(
        runtime.get(
            "form_replication_gate_name",
            "qwen2p5_14b_form_replication_pass",
        )
    )
    release_verification = verify_model_replication_release(
        release_record_path,
        identification_freeze_dir,
        campaign_lock_path,
        form_replication_addendum_path,
    )
    lock, rows, issues = collect_identification_rows(
        campaign_lock_path,
        results_root,
        result_domain=result_domain,
    )
    if release_verification.get("qc_pass") is not True:
        issues.extend(
            f"release verification: {issue}"
            for issue in release_verification.get("issues", [])
        )
    if len(rows) != int(lock["run_count"]):
        issues.append(f"expected {lock['run_count']} accepted runs; found {len(rows)}")

    addendum = json.loads(form_replication_addendum_path.read_text(encoding="utf-8"))
    evidence = addendum["qualification_evidence"]
    for row in rows:
        if round(
            float(row["weight_bytes_per_token"]) * int(row["target_batch"])
        ) != int(evidence["observed_unique_weight_storage_bytes"]):
            issues.append(f"{row['run_id']}: weight storage differs from qualification")
        if int(row["kv_write_bytes_per_token"]) != int(
            evidence["observed_logical_kv_bytes_per_attended_token"]
        ):
            issues.append(f"{row['run_id']}: KV geometry differs from qualification")
    if issues:
        report = {
            "schema_version": 1,
            "measurement": evaluation_measurement,
            "accepted_run_count": len(rows),
            "issues": issues,
            "qc_pass": False,
        }
        _atomic_json(output_dir / f"{output_prefix}-summary.json", report)
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
        form_replication_gate_name=form_replication_gate_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / f"{output_prefix}-runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(analysis["runs"])
    scalar_cells = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in analysis["cells"]
    ]
    cells_path = output_dir / f"{output_prefix}-cells.csv"
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_cells[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_cells)

    report = {
        "schema_version": 1,
        "measurement": evaluation_measurement,
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
            "cross_model_form_supported": analysis["gates"][
                form_replication_gate_name
            ],
            "universal_coefficients_supported": False,
            "cross_hardware_generalization_supported": False,
            "dvfs_conclusion_supported": False,
        },
        "issues": [],
        "qc_pass": True,
    }
    report_path = output_dir / f"{output_prefix}-evaluation.json"
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
    _atomic_json(output_dir / f"{output_prefix}-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--release-record", required=True, type=Path)
    parser.add_argument("--identification-freeze-dir", required=True, type=Path)
    parser.add_argument("--form-replication-addendum", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = evaluate_campaign(
        args.campaign_lock,
        args.release_record,
        args.identification_freeze_dir,
        args.form_replication_addendum,
        args.results_root,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
