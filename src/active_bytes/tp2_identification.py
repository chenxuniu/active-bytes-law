"""Freeze the prespecified two-GH200 TP=2 duration-model identification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .model_replication import (
    duration_calibration_envelope,
    fit_duration_ols_hc3,
)
from .primary_identification import (
    CALIBRATION_SPLIT,
    FIT_SPLIT,
    _alignment_bundle_sha,
    _atomic_json,
    collect_identification_rows,
    sha256_file,
)


TP2_OUTCOME = "sum_two_gpu_boards_joules_per_useful_token"


def _qualification_bundle(
    qualification_lock_path: Path, results_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    lock = json.loads(qualification_lock_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for run in lock["run_order"]:
        accepted: list[Path] = []
        for path in sorted(
            (
                results_root
                / "tp2-qualification"
                / run["run_id"]
            ).glob("attempt-*/alignment.json")
        ):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                value.get("qc_pass") is True
                and value.get("paper_candidate_measurement") is False
                and value.get("measurement")
                == "aligned-frozen-tp2-decode-repeat"
                and value.get("campaign_lock_sha256") == lock["lock_sha256"]
                and value.get("run", {}).get("run_id") == run["run_id"]
                and value.get("tensor_parallel_size") == 2
                and value.get("host_gpu_indices") == [0, 1]
            ):
                accepted.append(path)
        if len(accepted) != 1:
            issues.append(
                f"{run['run_id']}: expected exactly one accepted qualification; "
                f"found {len(accepted)}"
            )
            continue
        rows.append(
            {
                "order": run["order"],
                "run_id": run["run_id"],
                "alignment_sha256": sha256_file(accepted[0]),
                "alignment_path_relative_to_results": str(
                    accepted[0].relative_to(results_root)
                ),
            }
        )
    return rows, issues


def freeze_tp2_identification(
    campaign_lock_path: Path,
    qualification_lock_path: Path,
    addendum_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    lock, rows, issues = collect_identification_rows(
        campaign_lock_path,
        results_root,
        result_domain="tp2-identification",
    )
    qualification_rows, qualification_issues = _qualification_bundle(
        qualification_lock_path, results_root
    )
    issues.extend(qualification_issues)

    addendum_sha = sha256_file(addendum_path)
    identification_addendum_shas = {
        str(run.get("parameters", {}).get("execution_addendum_sha256"))
        for run in lock.get("run_order", [])
    }
    if identification_addendum_shas != {addendum_sha}:
        issues.append("identification lock does not bind the exact TP=2 addendum")
    qualification_lock = json.loads(
        qualification_lock_path.read_text(encoding="utf-8")
    )
    qualification_addendum_shas = {
        str(run.get("parameters", {}).get("execution_addendum_sha256"))
        for run in qualification_lock.get("run_order", [])
    }
    if qualification_addendum_shas != {addendum_sha}:
        issues.append("qualification lock does not bind the exact TP=2 addendum")
    expected_qualification_runs = int(
        addendum["qualification_design"]["required_accepted_runs"]
    )
    if qualification_lock.get("run_count") != expected_qualification_runs:
        issues.append("qualification lock run count differs from the addendum")
    if len(qualification_rows) != expected_qualification_runs:
        issues.append(
            f"expected {expected_qualification_runs} accepted qualification runs; "
            f"found {len(qualification_rows)}"
        )

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
            issues.append(
                f"{split}: expected {expected} accepted runs; "
                f"found {observed_counts[split]}"
            )
    if len(rows) != int(lock["run_count"]):
        issues.append(
            f"expected {lock['run_count']} accepted runs; found {len(rows)}"
        )

    expected_weight = int(
        addendum["source_evidence"]["audited_full_model_weight_storage_bytes"]
    )
    expected_kv = int(
        addendum["source_evidence"][
            "audited_logical_kv_bytes_per_attended_token"
        ]
    )
    for row in rows:
        path = results_root / row["alignment_path_relative_to_results"]
        alignment = json.loads(path.read_text(encoding="utf-8"))
        if alignment.get("measurement") != "aligned-frozen-tp2-decode-repeat":
            issues.append(f"{row['run_id']}: alignment is not TP=2")
        if alignment.get("paper_candidate_measurement") is not True:
            issues.append(f"{row['run_id']}: alignment is not paper-candidate")
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
            "measurement": "gh200-tp2-identification-freeze",
            "accepted_run_count": len(rows),
            "qualification_accepted_run_count": len(qualification_rows),
            "issues": issues,
            "qc_pass": False,
        }
        _atomic_json(output_dir / "identification-freeze-summary.json", report)
        return report

    fit_rows = [row for row in rows if row["split"] == FIT_SPLIT]
    calibration_rows = [
        row for row in rows if row["split"] == CALIBRATION_SPLIT
    ]
    fit = fit_duration_ols_hc3(
        fit_rows, analysis_id=addendum["analysis_policy"]["analysis_id"]
    )
    fit["outcome"] = TP2_OUTCOME
    bundle_sha = _alignment_bundle_sha(rows)
    qualification_bundle_sha = _alignment_bundle_sha(qualification_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficient_artifact = {
        "schema_version": 1,
        "measurement": "gh200-tp2-duration-coefficient-artifact",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "campaign_lock_file_sha256": sha256_file(campaign_lock_path),
        "qualification_campaign_lock_sha256": qualification_lock["lock_sha256"],
        "qualification_campaign_lock_file_sha256": sha256_file(
            qualification_lock_path
        ),
        "qualification_alignment_bundle_sha256": qualification_bundle_sha,
        "execution_addendum_sha256": sha256_file(addendum_path),
        "alignment_bundle_sha256": bundle_sha,
        "outcome_contract": {
            "estimand": TP2_OUTCOME,
            "scope": (
                "sum of two GH200 scope-0 board instantaneous-power integrals "
                "over identical decode boundaries"
            ),
            "idle_correction_applied": False,
            "single_device_energy_is_not_an_outcome": True,
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
        "measurement": "gh200-tp2-duration-discrepancy-envelope",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "alignment_bundle_sha256": bundle_sha,
        "coefficient_artifact_sha256": coefficient_sha,
        "outcome": TP2_OUTCOME,
        "calibration_run_count": len(calibration_rows),
        "calibration_cell_count": len(
            {row["cell_id"] for row in calibration_rows}
        ),
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
        "measurement": "gh200-tp2-identification-freeze",
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": lock["lock_sha256"],
        "outcome": TP2_OUTCOME,
        "accepted_run_count": len(rows),
        "fit_run_count": len(fit_rows),
        "calibration_run_count": len(calibration_rows),
        "qualification_accepted_run_count": len(qualification_rows),
        "qualification_alignment_bundle_sha256": qualification_bundle_sha,
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
            "tp2_functional_form_confirmed": False,
            "nvlink_energy_separately_attributed": False,
        },
        "holdout_release_candidate": scientific_gate,
        "holdout_release_note": (
            "A separate content-addressed release record must bind these "
            "artifacts before any sealed TP=2 holdout run may execute."
        ),
        "issues": [],
        "qc_pass": True,
    }
    _atomic_json(output_dir / "identification-freeze-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--qualification-lock", required=True, type=Path)
    parser.add_argument("--execution-addendum", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = freeze_tp2_identification(
        args.campaign_lock,
        args.qualification_lock,
        args.execution_addendum,
        args.results_root,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("qc_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
