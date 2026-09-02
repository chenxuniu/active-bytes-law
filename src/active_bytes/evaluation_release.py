"""Fail-closed verification of the GH200 evaluation release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


REQUIRED_ARTIFACTS = (
    "coefficient-artifact.json",
    "discrepancy-envelope.json",
    "accepted-runs.csv",
    "identification-freeze-summary.json",
)
OFFICIAL_RELEASE_FILENAME = "gh200-primary-bf16-evaluation-release-v1.json"
OFFICIAL_RELEASE_SHA256 = (
    "264f311880bd5c118eba805a810902f1efd684c31e4fb54221ec5268ff02cb0a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evaluation_release(
    release_record_path: Path,
    identification_freeze_dir: Path,
    evaluation_lock_path: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        release = json.loads(release_record_path.read_text(encoding="utf-8"))
        evaluation_lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1,
            "measurement": "gh200-primary-evaluation-release-verification",
            "qc_pass": False,
            "issues": [f"{type(exc).__name__}: {exc}"],
        }

    policy = release.get("release_policy", {})
    observed_release_sha = sha256_file(release_record_path)
    if (
        release_record_path.name == OFFICIAL_RELEASE_FILENAME
        and observed_release_sha != OFFICIAL_RELEASE_SHA256
    ):
        issues.append("official evaluation release record digest mismatch")
    if policy.get("released") is not True:
        issues.append("release record does not authorize evaluation")
    for key in (
        "release_requires_exact_artifact_hashes",
        "evaluation_may_not_refit_coefficients",
        "evaluation_may_not_recalibrate_the_envelope",
    ):
        if policy.get(key) is not True:
            issues.append(f"release policy is missing required true flag: {key}")

    expected_lock = release.get("evaluation_campaign", {})
    if evaluation_lock.get("lock_sha256") != expected_lock.get("campaign_lock_sha256"):
        issues.append("evaluation campaign lock digest does not match release record")
    observed_lock_file_sha = sha256_file(evaluation_lock_path)
    if observed_lock_file_sha != expected_lock.get("campaign_lock_file_sha256"):
        issues.append("evaluation campaign lock file digest does not match release record")
    if evaluation_lock.get("run_count") != expected_lock.get("run_count"):
        issues.append("evaluation campaign run count does not match release record")

    expected_artifacts = release.get("frozen_artifacts", {})
    observed_artifacts: dict[str, str | None] = {}
    for name in REQUIRED_ARTIFACTS:
        path = identification_freeze_dir / name
        if not path.is_file():
            issues.append(f"identification artifact is missing: {name}")
            observed_artifacts[name] = None
            continue
        observed = sha256_file(path)
        observed_artifacts[name] = observed
        if observed != expected_artifacts.get(name):
            issues.append(f"identification artifact digest mismatch: {name}")

    summary: Mapping[str, Any] = {}
    coefficient: Mapping[str, Any] = {}
    envelope: Mapping[str, Any] = {}
    if all(observed_artifacts.get(name) == expected_artifacts.get(name) for name in REQUIRED_ARTIFACTS):
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
        if summary.get("qc_pass") is not True or summary.get("evaluation_release_candidate") is not True:
            issues.append("identification summary is not an evaluation release candidate")
        for key in ("accepted_run_count", "fit_run_count", "calibration_run_count", "outcome"):
            if summary.get(key) != identification.get(key):
                issues.append(f"identification summary disagrees on {key}")
        if summary.get("campaign_lock_sha256") != identification.get("campaign_lock_sha256"):
            issues.append("identification summary campaign digest mismatch")
        if summary.get("coefficient_artifact", {}).get("sha256") != expected_artifacts.get(
            "coefficient-artifact.json"
        ):
            issues.append("summary does not bind released coefficient artifact")
        if summary.get("discrepancy_envelope", {}).get("sha256") != expected_artifacts.get(
            "discrepancy-envelope.json"
        ):
            issues.append("summary does not bind released discrepancy envelope")

    if coefficient and envelope:
        if coefficient.get("campaign_lock_sha256") != identification.get("campaign_lock_sha256"):
            issues.append("coefficient artifact campaign digest mismatch")
        if envelope.get("coefficient_artifact_sha256") != expected_artifacts.get(
            "coefficient-artifact.json"
        ):
            issues.append("envelope does not bind released coefficient artifact")
        decisions = release.get("identification_decisions", {})
        observed_decisions = summary.get("scientific_decisions", {})
        if observed_decisions.get("p2_slope_positivity_pass") != decisions.get(
            "two_slope_familywise_positivity_pass"
        ):
            issues.append("slope-positivity decision mismatch")
        if observed_decisions.get("p3_coefficient_equivalence_pass") != decisions.get(
            "single_coefficient_equivalence_pass"
        ):
            issues.append("coefficient-equivalence decision mismatch")

    return {
        "schema_version": 1,
        "measurement": "gh200-primary-evaluation-release-verification",
        "release_id": release.get("release_id"),
        "release_record_sha256": observed_release_sha,
        "identification_freeze_dir": str(identification_freeze_dir),
        "evaluation_campaign_lock_sha256": evaluation_lock.get("lock_sha256"),
        "observed_identification_artifacts": observed_artifacts,
        "outcome": identification.get("outcome"),
        "coefficients_are_immutable": policy.get("evaluation_may_not_refit_coefficients"),
        "envelope_is_immutable": policy.get("evaluation_may_not_recalibrate_the_envelope"),
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
    parser.add_argument("--evaluation-lock", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify_evaluation_release(
        args.release_record, args.identification_freeze_dir, args.evaluation_lock
    )
    _atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
