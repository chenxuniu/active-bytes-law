"""Fail-closed verification of a content-addressed TP=2 holdout release."""

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
OFFICIAL_RELEASE_FILENAME = "gh200-tp2-nvlink-holdout-release-v1.json"
# Filled only by the later, reviewed release commit after identification is
# frozen.  Keeping this unset makes the currently sealed holdout fail closed
# even if an untrusted record and adjacent sidecar are placed on the node.
OFFICIAL_RELEASE_SHA256: str | None = None


def verify_tp2_release(
    release_record_path: Path,
    identification_freeze_dir: Path,
    identification_lock_path: Path,
    holdout_lock_path: Path,
    execution_addendum_path: Path,
    *,
    expected_release_sha256: str,
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        release = json.loads(release_record_path.read_text(encoding="utf-8"))
        identification_lock = json.loads(
            identification_lock_path.read_text(encoding="utf-8")
        )
        holdout_lock = json.loads(holdout_lock_path.read_text(encoding="utf-8"))
        addendum = json.loads(execution_addendum_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1,
            "measurement": "gh200-tp2-holdout-release-verification",
            "issues": [f"{type(exc).__name__}: {exc}"],
            "qc_pass": False,
        }

    release_sha = sha256_file(release_record_path)
    if release_sha != expected_release_sha256:
        issues.append("holdout release record digest does not match its sidecar")
    if release.get("execution_addendum_sha256") != sha256_file(
        execution_addendum_path
    ):
        issues.append("execution addendum digest mismatch")

    expected_identification = release.get("identification_campaign", {})
    if identification_lock.get("lock_sha256") != expected_identification.get(
        "campaign_lock_sha256"
    ):
        issues.append("identification lock digest mismatch")
    if sha256_file(identification_lock_path) != expected_identification.get(
        "campaign_lock_file_sha256"
    ):
        issues.append("identification lock file digest mismatch")
    if identification_lock.get("run_count") != 45:
        issues.append("identification lock does not contain 45 runs")

    expected_holdout = release.get("holdout_campaign", {})
    if holdout_lock.get("lock_sha256") != expected_holdout.get(
        "campaign_lock_sha256"
    ):
        issues.append("holdout lock digest mismatch")
    if sha256_file(holdout_lock_path) != expected_holdout.get(
        "campaign_lock_file_sha256"
    ):
        issues.append("holdout lock file digest mismatch")
    if holdout_lock.get("run_count") != expected_holdout.get("run_count"):
        issues.append("holdout run count mismatch")
    if holdout_lock.get("cell_count") != expected_holdout.get("cell_count"):
        issues.append("holdout cell count mismatch")
    if expected_holdout.get("repetitions_per_cell") != 5:
        issues.append("released holdout repetition count differs from five")
    addendum_sha = sha256_file(execution_addendum_path)
    for label, lock in (
        ("identification", identification_lock),
        ("holdout", holdout_lock),
    ):
        observed_addenda = {
            run.get("parameters", {}).get("execution_addendum_sha256")
            for run in lock.get("run_order", [])
        }
        if observed_addenda != {addendum_sha}:
            issues.append(f"{label} lock does not bind the exact execution addendum")
    for run in holdout_lock.get("run_order", []):
        parameters = run.get("parameters", {})
        if parameters.get("execution_state") != "sealed-unreleased":
            issues.append(f"{run.get('run_id')}: holdout source lock is not sealed")
        if parameters.get("requires_frozen_identification_release") is not True:
            issues.append(f"{run.get('run_id')}: release requirement is absent")
        if parameters.get("tensor_parallel_size") != 2:
            issues.append(f"{run.get('run_id')}: tensor parallel size changed")
        if parameters.get("host_gpu_indices") != [0, 1]:
            issues.append(f"{run.get('run_id')}: host GPU pair changed")

    policy = release.get("release_policy", {})
    if policy.get("released") is not True:
        issues.append("release record does not authorize TP=2 holdout execution")
    for key in (
        "coefficients_are_immutable",
        "residual_envelope_is_immutable",
        "holdout_may_not_refit",
        "duration_uses_same_interval_observation",
        "both_device_energies_must_be_summed",
        "failed_attempts_are_preserved",
    ):
        if policy.get(key) is not True:
            issues.append(f"release policy is missing required true flag: {key}")
    for key in (
        "nvlink_energy_separately_attributed",
        "tp1_coefficients_are_tp2_predictions",
        "dvfs_claim_authorized",
        "cross_sku_claim_authorized",
    ):
        if policy.get(key) is not False:
            issues.append(f"release policy must set {key} to false")

    expected_artifacts = release.get("frozen_artifacts", {})
    observed_artifacts: dict[str, str | None] = {}
    for name in REQUIRED_ARTIFACTS:
        path = identification_freeze_dir / name
        if not path.is_file():
            observed_artifacts[name] = None
            issues.append(f"identification artifact is missing: {name}")
            continue
        observed = sha256_file(path)
        observed_artifacts[name] = observed
        if observed != expected_artifacts.get(name):
            issues.append(f"identification artifact digest mismatch: {name}")

    summary: Mapping[str, Any] = {}
    coefficient: Mapping[str, Any] = {}
    if all(observed_artifacts.get(name) for name in REQUIRED_ARTIFACTS):
        try:
            summary = json.loads(
                (
                    identification_freeze_dir
                    / "identification-freeze-summary.json"
                ).read_text(encoding="utf-8")
            )
            coefficient = json.loads(
                (identification_freeze_dir / "coefficient-artifact.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as exc:
            issues.append(f"frozen identification JSON is malformed: {exc}")
    if summary:
        if summary.get("qc_pass") is not True:
            issues.append("identification freeze failed technical QC")
        if summary.get("holdout_release_candidate") is not True:
            issues.append("identification did not authorize a holdout release")
        if summary.get("campaign_lock_sha256") != identification_lock.get(
            "lock_sha256"
        ):
            issues.append("identification summary campaign digest mismatch")
        if summary.get("accepted_run_count") != 45:
            issues.append("identification summary does not contain 45 accepted runs")
        decisions = summary.get("scientific_decisions", {})
        if decisions.get("identification_gate_pass") is not True:
            issues.append("identification scientific gate did not pass")
        if decisions.get("tp2_functional_form_confirmed") is not False:
            issues.append("identification alone must not claim TP=2 confirmation")
    if coefficient:
        if coefficient.get("outcome_contract", {}).get("estimand") != (
            "sum_two_gpu_boards_joules_per_useful_token"
        ):
            issues.append("coefficient artifact has the wrong energy estimand")
        if coefficient.get("fit", {}).get("coefficients") != release.get(
            "frozen_estimates", {}
        ).get("coefficients"):
            issues.append("released coefficients differ from frozen artifact")
    frozen_estimates = release.get("frozen_estimates", {})
    expected_equation = addendum.get("analysis_policy", {}).get("equation")
    if release.get("analysis_equation") != expected_equation:
        issues.append("released equation differs from the preregistered equation")
    if frozen_estimates.get("equation") != expected_equation:
        issues.append("frozen-estimate equation differs from the preregistered equation")
    if release.get("primary_gates") != addendum.get("holdout_gates"):
        issues.append("released primary gates differ from the preregistered gates")
    if all(observed_artifacts.get(name) for name in REQUIRED_ARTIFACTS):
        try:
            envelope = json.loads(
                (identification_freeze_dir / "discrepancy-envelope.json").read_text(
                    encoding="utf-8"
                )
            )
            observed_range = envelope["envelope"][
                "common_residual_range_joules_per_token"
            ]
            if observed_range != frozen_estimates.get(
                "common_residual_range_joules_per_token"
            ):
                issues.append("released residual range differs from frozen artifact")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(f"frozen residual envelope is malformed: {exc}")

    return {
        "schema_version": 1,
        "measurement": "gh200-tp2-holdout-release-verification",
        "release_id": release.get("release_id"),
        "release_record_sha256": release_sha,
        "identification_freeze_dir": str(identification_freeze_dir),
        "identification_campaign_lock_sha256": identification_lock.get(
            "lock_sha256"
        ),
        "holdout_campaign_lock_sha256": holdout_lock.get("lock_sha256"),
        "observed_identification_artifacts": observed_artifacts,
        "coefficients_are_immutable": policy.get("coefficients_are_immutable"),
        "envelope_is_immutable": policy.get(
            "residual_envelope_is_immutable"
        ),
        "issues": issues,
        "qc_pass": not issues,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-record", required=True, type=Path)
    parser.add_argument("--identification-freeze-dir", required=True, type=Path)
    parser.add_argument("--identification-lock", required=True, type=Path)
    parser.add_argument("--holdout-lock", required=True, type=Path)
    parser.add_argument("--execution-addendum", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.release_record.name != OFFICIAL_RELEASE_FILENAME:
        report = {
            "schema_version": 1,
            "measurement": "gh200-tp2-holdout-release-verification",
            "issues": ["release record is not the compiled official filename"],
            "qc_pass": False,
        }
    elif OFFICIAL_RELEASE_SHA256 is None:
        report = {
            "schema_version": 1,
            "measurement": "gh200-tp2-holdout-release-verification",
            "issues": [
                "TP=2 holdout is sealed: no official release digest is compiled"
            ],
            "qc_pass": False,
        }
    else:
        report = verify_tp2_release(
            args.release_record,
            args.identification_freeze_dir,
            args.identification_lock,
            args.holdout_lock,
            args.execution_addendum,
            expected_release_sha256=OFFICIAL_RELEASE_SHA256,
        )
    _atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
