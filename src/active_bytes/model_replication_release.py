"""Fail-closed verification of the Qwen2.5-14B holdout release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .primary_identification import sha256_file


REQUIRED_ARTIFACTS = (
    "coefficient-artifact.json",
    "discrepancy-envelope.json",
    "accepted-runs.csv",
    "identification-freeze-summary.json",
)
OFFICIAL_RELEASE_FILENAME = "gh200-qwen2p5-14b-holdout-release-v1.json"
OFFICIAL_RELEASE_SHA256 = (
    "7b9b9adcdc9a40b5e60e02f176cc18ecfd78242779da6f2a631457094857051d"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_model_replication_release(
    release_record_path: Path,
    identification_freeze_dir: Path,
    holdout_lock_path: Path,
    form_replication_addendum_path: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        release = json.loads(release_record_path.read_text(encoding="utf-8"))
        holdout_lock = json.loads(holdout_lock_path.read_text(encoding="utf-8"))
        addendum = json.loads(
            form_replication_addendum_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1,
            "measurement": "gh200-qwen2p5-14b-holdout-release-verification",
            "qc_pass": False,
            "issues": [f"{type(exc).__name__}: {exc}"],
        }

    release_sha = sha256_file(release_record_path)
    if release_record_path.name != OFFICIAL_RELEASE_FILENAME:
        issues.append("holdout release record does not have the official filename")
    if release_sha != OFFICIAL_RELEASE_SHA256:
        issues.append("official holdout release record digest mismatch")
    addendum_sha = sha256_file(form_replication_addendum_path)
    if addendum_sha != release.get(
        "form_replication_addendum_sha256"
    ):
        issues.append("form-replication addendum digest mismatch")

    policy = release.get("release_policy", {})
    for key in (
        "released",
        "release_requires_exact_artifact_hashes",
        "holdout_may_not_refit_coefficients",
        "holdout_may_not_recalibrate_the_envelope",
        "duration_uses_observed_same_interval_decode_time",
        "accepted_attempts_are_never_overwritten",
        "cross_model_claim_requires_primary_holdout_gate_pass",
    ):
        if policy.get(key) is not True:
            issues.append(f"release policy is missing required true flag: {key}")
    if policy.get("cross_hardware_claim_authorized") is not False:
        issues.append("release must explicitly reject a cross-hardware claim")
    if policy.get("dvfs_claim_authorized") is not False:
        issues.append("release must explicitly reject a DVFS claim")

    expected_holdout = release.get("holdout_campaign", {})
    if holdout_lock.get("lock_sha256") != expected_holdout.get(
        "campaign_lock_sha256"
    ):
        issues.append("holdout campaign lock digest mismatch")
    if sha256_file(holdout_lock_path) != expected_holdout.get(
        "campaign_lock_file_sha256"
    ):
        issues.append("holdout campaign lock file digest mismatch")
    if holdout_lock.get("run_count") != expected_holdout.get("run_count"):
        issues.append("holdout run count mismatch")
    if holdout_lock.get("cell_count") != expected_holdout.get("cell_count"):
        issues.append("holdout cell count mismatch")
    observed_coordinates: set[tuple[int, int]] = set()
    for run in holdout_lock.get("run_order", []):
        parameters = run.get("parameters", {})
        if parameters.get("execution_state") != "sealed-unreleased":
            issues.append(f"{run.get('run_id')}: source lock was not sealed")
        if parameters.get("requires_frozen_identification_release") is not True:
            issues.append(f"{run.get('run_id')}: source lock lacks release requirement")
        if parameters.get("execution_addendum_sha256") != addendum_sha:
            issues.append(f"{run.get('run_id')}: source lock addendum digest mismatch")
        observed_coordinates.add(
            (
                int(parameters.get("target_mean_attended_history_tokens", -1)),
                int(parameters.get("target_batch", -1)),
            )
        )
    expected_coordinates = {
        (int(context), int(batch))
        for context in expected_holdout.get(
            "target_mean_attended_history_tokens", []
        )
        for batch in expected_holdout.get("target_batches", [])
    }
    if observed_coordinates != expected_coordinates:
        issues.append("holdout coordinates differ from the released grid")

    parent_gates = addendum.get("holdout_gates", {})
    released_gates = release.get("primary_gates", {})
    gate_pairs = {
        "median_absolute_relative_error_maximum": (
            parent_gates.get("median_absolute_relative_error_maximum"),
            released_gates.get("median_absolute_relative_error_maximum"),
        ),
        "maximum_absolute_relative_error_maximum": (
            parent_gates.get("maximum_absolute_relative_error_maximum"),
            released_gates.get("maximum_absolute_relative_error_maximum"),
        ),
        "required_cells_passing_ten_percent_error": (
            parent_gates.get("required_cells_passing_ten_percent_error"),
            released_gates.get("required_cells_passing_ten_percent_error"),
        ),
        "coefficient_refit_before_gate_decision": (
            parent_gates.get("coefficient_refit_before_gate_decision"),
            released_gates.get("coefficient_refit_before_gate_decision"),
        ),
        "failed_or_censored_cells_are_not_deleted": (
            parent_gates.get("failed_or_censored_cells_are_not_deleted"),
            released_gates.get("failed_or_censored_cells_are_not_deleted"),
        ),
    }
    for key, (parent_value, released_value) in gate_pairs.items():
        if parent_value != released_value:
            issues.append(f"released primary gate differs from parent addendum: {key}")

    expected_artifacts = release.get("frozen_artifacts", {})
    observed_artifacts: dict[str, str | None] = {}
    for name in REQUIRED_ARTIFACTS:
        path = identification_freeze_dir / name
        if not path.is_file():
            issues.append(f"identification artifact is missing: {name}")
            observed_artifacts[name] = None
            continue
        digest = sha256_file(path)
        observed_artifacts[name] = digest
        if digest != expected_artifacts.get(name):
            issues.append(f"identification artifact digest mismatch: {name}")

    summary: Mapping[str, Any] = {}
    coefficient: Mapping[str, Any] = {}
    envelope: Mapping[str, Any] = {}
    if all(
        observed_artifacts.get(name) == expected_artifacts.get(name)
        for name in REQUIRED_ARTIFACTS
    ):
        try:
            summary = json.loads(
                (identification_freeze_dir / "identification-freeze-summary.json").read_text(
                    encoding="utf-8"
                )
            )
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
        except (OSError, ValueError) as exc:
            issues.append(f"frozen identification JSON is malformed: {exc}")

    identification = release.get("identification_campaign", {})
    if summary:
        if summary.get("qc_pass") is not True:
            issues.append("identification freeze did not pass technical QC")
        if summary.get("holdout_release_candidate") is not True:
            issues.append("identification freeze did not authorize a holdout release")
        for key in ("accepted_run_count", "fit_run_count", "calibration_run_count"):
            if summary.get(key) != identification.get(key):
                issues.append(f"identification summary disagrees on {key}")
        if summary.get("campaign_lock_sha256") != identification.get(
            "campaign_lock_sha256"
        ):
            issues.append("identification campaign digest mismatch")
        if summary.get("coefficient_artifact", {}).get("sha256") != expected_artifacts.get(
            "coefficient-artifact.json"
        ):
            issues.append("identification summary does not bind coefficient artifact")
        if summary.get("discrepancy_envelope", {}).get("sha256") != expected_artifacts.get(
            "discrepancy-envelope.json"
        ):
            issues.append("identification summary does not bind residual envelope")
        if summary.get("accepted_run_table", {}).get("sha256") != expected_artifacts.get(
            "accepted-runs.csv"
        ):
            issues.append("identification summary does not bind accepted-run table")
        decisions = summary.get("scientific_decisions", {})
        released_decisions = release.get("identification_decisions", {})
        for key in (
            "identification_gate_pass",
            "time_term_finite_nonnegative_pass",
            "cross_model_form_confirmed",
        ):
            if decisions.get(key) != released_decisions.get(key):
                issues.append(f"identification decision mismatch: {key}")
        if decisions.get("traffic_slope_positivity_pass") != released_decisions.get(
            "traffic_slope_familywise_positivity_pass"
        ):
            issues.append("identification decision mismatch: traffic slope positivity")

    frozen = release.get("frozen_estimates", {})
    if frozen.get("equation") != addendum.get("analysis_policy", {}).get("equation"):
        issues.append("released equation differs from parent addendum")
    if coefficient:
        if coefficient.get("campaign_lock_sha256") != identification.get(
            "campaign_lock_sha256"
        ):
            issues.append("coefficient artifact campaign digest mismatch")
        if coefficient.get("fit", {}).get("coefficients") != frozen.get("coefficients"):
            issues.append("released coefficient snapshot differs from frozen artifact")
    if envelope:
        if envelope.get("coefficient_artifact_sha256") != expected_artifacts.get(
            "coefficient-artifact.json"
        ):
            issues.append("residual envelope does not bind the coefficient artifact")
        observed_range = envelope.get("envelope", {}).get(
            "common_residual_range_joules_per_token"
        )
        if observed_range != frozen.get("common_residual_range_joules_per_token"):
            issues.append("released residual range differs from frozen artifact")

    return {
        "schema_version": 1,
        "measurement": "gh200-qwen2p5-14b-holdout-release-verification",
        "release_id": release.get("release_id"),
        "release_record_sha256": release_sha,
        "identification_freeze_dir": str(identification_freeze_dir),
        "holdout_campaign_lock_sha256": holdout_lock.get("lock_sha256"),
        "observed_identification_artifacts": observed_artifacts,
        "coefficients_are_immutable": policy.get(
            "holdout_may_not_refit_coefficients"
        ),
        "envelope_is_immutable": policy.get(
            "holdout_may_not_recalibrate_the_envelope"
        ),
        "issues": issues,
        "qc_pass": not issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-record", required=True, type=Path)
    parser.add_argument("--identification-freeze-dir", required=True, type=Path)
    parser.add_argument("--holdout-lock", required=True, type=Path)
    parser.add_argument("--form-replication-addendum", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify_model_replication_release(
        args.release_record,
        args.identification_freeze_dir,
        args.holdout_lock,
        args.form_replication_addendum,
    )
    _atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
