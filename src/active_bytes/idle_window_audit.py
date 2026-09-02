"""Audit the idle telemetry available around frozen GH200 energy repeats.

This module is intentionally diagnostic.  It does not choose an idle-window
duration after outcomes are visible and it never promotes gross board energy
to an idle-corrected paper outcome.  It records whether the already-collected
telemetry contains samples before and after every accepted decode repeat so a
separate, frozen correction policy can be judged without reopening raw runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _window_summary(
    rows: Iterable[Mapping[str, Any]], *, power_field: str
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["monotonic_ns"])
    powers = [float(row[power_field]) for row in ordered]
    span_seconds = (
        (ordered[-1]["monotonic_ns"] - ordered[0]["monotonic_ns"]) / 1e9
        if len(ordered) >= 2
        else 0.0
    )
    return {
        "sample_count": len(ordered),
        "span_seconds": span_seconds,
        "mean_power_w": statistics.fmean(powers) if powers else None,
        "median_power_w": statistics.median(powers) if powers else None,
        "minimum_power_w": min(powers) if powers else None,
        "maximum_power_w": max(powers) if powers else None,
        "has_two_or_more_samples": len(ordered) >= 2,
    }


def _accepted_alignment(
    run_root: Path, *, run: Mapping[str, Any], campaign_lock_sha256: str
) -> tuple[Path | None, list[str]]:
    accepted: list[Path] = []
    malformed: list[str] = []
    for path in sorted(run_root.glob("attempt-*/alignment.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            malformed.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        observed_run = value.get("run", {})
        if (
            value.get("qc_pass") is True
            and value.get("campaign_lock_sha256") == campaign_lock_sha256
            and observed_run.get("run_id") == run["run_id"]
            and observed_run.get("cell_id") == run["cell_id"]
            and observed_run.get("repeat") == run["repeat"]
            and observed_run.get("split") == run["split"]
        ):
            accepted.append(path)
    if len(accepted) != 1:
        malformed.append(
            f"{run['run_id']}: expected one accepted alignment; found {len(accepted)}"
        )
        return None, malformed
    return accepted[0], malformed


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "maximum": max(values) if values else None,
    }


def audit_idle_windows(
    campaign_lock_path: Path,
    results_root: Path,
    *,
    result_domain: str = "primary-identification",
    guard_seconds: float = 0.25,
    gpu_index: int = 0,
) -> dict[str, Any]:
    if guard_seconds < 0:
        raise ValueError("guard_seconds must be non-negative")
    lock = json.loads(campaign_lock_path.read_text(encoding="utf-8"))
    campaign_sha = lock["lock_sha256"]
    issues: list[str] = []
    reports: list[dict[str, Any]] = []
    guard_ns = round(guard_seconds * 1e9)

    for run in sorted(lock["run_order"], key=lambda row: row["order"]):
        run_root = results_root / result_domain / run["run_id"]
        alignment_path, run_issues = _accepted_alignment(
            run_root, run=run, campaign_lock_sha256=campaign_sha
        )
        issues.extend(run_issues)
        if alignment_path is None:
            continue
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        telemetry_path = alignment_path.parent / "telemetry.jsonl"
        if not telemetry_path.is_file():
            issues.append(f"{run['run_id']}: telemetry.jsonl is missing")
            continue
        telemetry = [
            row
            for row in _read_jsonl(telemetry_path)
            if row.get("gpu_index") == gpu_index
        ]
        required = {"monotonic_ns", "gpu_instant_power_w"}
        if not telemetry or any(not required.issubset(row) for row in telemetry):
            issues.append(f"{run['run_id']}: GPU telemetry is empty or malformed")
            continue
        episodes = alignment.get("episodes", [])
        if not episodes:
            issues.append(f"{run['run_id']}: accepted alignment has no episodes")
            continue
        decode_start_ns = min(int(row["start_ns"]) for row in episodes)
        decode_end_ns = max(int(row["end_ns"]) for row in episodes)
        pre = [
            row
            for row in telemetry
            if int(row["monotonic_ns"]) <= decode_start_ns - guard_ns
        ]
        post = [
            row
            for row in telemetry
            if int(row["monotonic_ns"]) >= decode_end_ns + guard_ns
        ]
        pre_summary = _window_summary(pre, power_field="gpu_instant_power_w")
        post_summary = _window_summary(post, power_field="gpu_instant_power_w")
        reports.append(
            {
                "order": run["order"],
                "run_id": run["run_id"],
                "cell_id": run["cell_id"],
                "split": run["split"],
                "repeat": run["repeat"],
                "decode_start_monotonic_ns": decode_start_ns,
                "decode_end_monotonic_ns": decode_end_ns,
                "decode_seconds": alignment["totals"]["decode_seconds"],
                "gross_gpu_joules_per_token": alignment["totals"][
                    "gpu_joules_per_token"
                ],
                "pre_decode": pre_summary,
                "post_decode": post_summary,
                "has_bracketing_samples": (
                    pre_summary["has_two_or_more_samples"]
                    and post_summary["has_two_or_more_samples"]
                ),
                "artifact_sha256": {
                    "alignment_json": _sha256(alignment_path),
                    "telemetry_jsonl": _sha256(telemetry_path),
                },
            }
        )

    pre_spans = [row["pre_decode"]["span_seconds"] for row in reports]
    post_spans = [row["post_decode"]["span_seconds"] for row in reports]
    pre_means = [
        row["pre_decode"]["mean_power_w"]
        for row in reports
        if row["pre_decode"]["mean_power_w"] is not None
    ]
    post_means = [
        row["post_decode"]["mean_power_w"]
        for row in reports
        if row["post_decode"]["mean_power_w"] is not None
    ]
    complete = len(reports) == int(lock["run_count"]) and not issues
    return {
        "schema_version": 1,
        "measurement": "gh200-primary-idle-window-availability-audit",
        "non_paper_measurement": True,
        "campaign_id": lock["campaign_id"],
        "campaign_lock_sha256": campaign_sha,
        "campaign_lock_file_sha256": _sha256(campaign_lock_path),
        "result_domain": result_domain,
        "gpu_index": gpu_index,
        "guard_seconds": guard_seconds,
        "expected_run_count": lock["run_count"],
        "accepted_run_count": len(reports),
        "artifact_qc_pass": complete,
        "issues": issues,
        "all_runs_have_bracketing_samples": (
            complete and all(row["has_bracketing_samples"] for row in reports)
        ),
        "pre_decode_span_seconds": _distribution(pre_spans),
        "post_decode_span_seconds": _distribution(post_spans),
        "pre_decode_mean_power_w": _distribution(pre_means),
        "post_decode_mean_power_w": _distribution(post_means),
        "idle_correction_contract": {
            "policy_frozen": False,
            "paper_outcome_eligible": False,
            "reason": (
                "This audit describes already-collected bracketing telemetry. "
                "It does not choose a post-outcome duration, estimator, or "
                "correction rule."
            ),
        },
        "runs": reports,
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--result-domain", default="primary-identification")
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_idle_windows(
        args.campaign_lock,
        args.results_root,
        result_domain=args.result_domain,
        guard_seconds=args.guard_seconds,
        gpu_index=args.gpu_index,
    )
    _atomic_write(args.output_json, report)
    print(
        json.dumps(
            {
                "measurement": report["measurement"],
                "accepted_run_count": report["accepted_run_count"],
                "artifact_qc_pass": report["artifact_qc_pass"],
                "all_runs_have_bracketing_samples": report[
                    "all_runs_have_bracketing_samples"
                ],
                "pre_decode_span_seconds": report["pre_decode_span_seconds"],
                "post_decode_span_seconds": report["post_decode_span_seconds"],
                "idle_correction_contract": report["idle_correction_contract"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["artifact_qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
