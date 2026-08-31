"""Strict validation for one static decode-only iteration trace."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .accounting import summarize_trace
from .batch_doctor import balanced_prompt_lengths


REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "episode_id",
    "iteration_id",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "active_request_ids",
    "useful_tokens_by_request",
    "metered_useful_output_tokens",
    "attended_length_by_request",
    "live_kv_blocks",
    "allocated_kv_blocks",
    "accepted_tokens",
    "rejected_tokens",
    "speculative_draft_tokens",
    "preemptions",
    "swaps",
    "recomputed_tokens",
    "prefix_cache_hits",
    "offloaded_bytes",
    "scheduler_mode",
    "attention_backend",
    "graph_mode",
    "kv_cache_dtype",
    "weight_dtype",
}

STABLE_FIELDS = (
    "run_id",
    "episode_id",
    "scheduler_mode",
    "attention_backend",
    "graph_mode",
    "kv_cache_dtype",
    "weight_dtype",
)

ZERO_FIELDS = (
    "rejected_tokens",
    "speculative_draft_tokens",
    "preemptions",
    "swaps",
    "recomputed_tokens",
    "prefix_cache_hits",
    "offloaded_bytes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: trace row must be an object")
            rows.append(value)
    return rows


def validate_static_trace(
    rows: list[Mapping[str, Any]],
    *,
    expected_batch: int,
    prompt_tokens: int | None = None,
    target_mean_attended_history_tokens: int | None = None,
    measured_decode_tokens: int = 128,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_batch <= 0 or measured_decode_tokens <= 0:
        raise ValueError("batch and measured tokens must be positive")
    if (prompt_tokens is None) == (target_mean_attended_history_tokens is None):
        raise ValueError("specify exactly one prompt or target-mean geometry")
    if prompt_tokens is not None:
        if prompt_tokens < 0:
            raise ValueError("prompt must be non-negative")
        prompt_lengths = [prompt_tokens] * expected_batch
    else:
        assert target_mean_attended_history_tokens is not None
        prompt_lengths = balanced_prompt_lengths(
            target_mean_attended_history_tokens=target_mean_attended_history_tokens,
            batch=expected_batch,
            measured_decode_tokens=measured_decode_tokens,
        )
    if len(rows) != measured_decode_tokens:
        errors.append(
            f"expected {measured_decode_tokens} decode iterations, observed {len(rows)}"
        )
    if not rows:
        return {"qc_pass": False, "errors": errors or ["trace is empty"]}

    baseline: dict[str, Any] = {}
    expected_requests: set[str] | None = None
    prompt_tokens_by_request: dict[str, int] | None = None
    previous_end: int | None = None
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS.difference(row))
        if missing:
            errors.append(f"row {index}: missing fields {', '.join(missing)}")
            continue
        if row["schema_version"] != 1:
            errors.append(f"row {index}: schema_version must be 1")
        if row["iteration_id"] != index:
            errors.append(f"row {index}: iteration_id is {row['iteration_id']!r}, expected {index}")
        start = row["monotonic_start_ns"]
        end = row["monotonic_end_ns"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            errors.append(f"row {index}: invalid monotonic interval")
        elif previous_end is not None and start < previous_end:
            errors.append(f"row {index}: interval overlaps or moves backward")
        if isinstance(end, int):
            previous_end = end

        active = row["active_request_ids"]
        if not isinstance(active, list) or len(active) != expected_batch or len(set(active)) != len(active):
            errors.append(f"row {index}: active request set is not a unique batch of {expected_batch}")
            active_set: set[str] = set()
        else:
            active_set = set(active)
        if expected_requests is None:
            expected_requests = active_set
        elif active_set != expected_requests:
            errors.append(f"row {index}: request membership changed (late entry or exit)")

        useful = row["useful_tokens_by_request"]
        attended = row["attended_length_by_request"]
        if not isinstance(useful, Mapping) or set(useful) != active_set:
            errors.append(f"row {index}: useful-token keys do not match active requests")
        else:
            bad = [request_id for request_id, count in useful.items() if count != 1]
            if bad:
                errors.append(f"row {index}: each active request must produce exactly one useful token")
        if row["metered_useful_output_tokens"] != expected_batch:
            errors.append(f"row {index}: metered useful-token total is not {expected_batch}")
        if row["accepted_tokens"] != expected_batch:
            errors.append(f"row {index}: accepted_tokens is not {expected_batch}")
        if not isinstance(attended, Mapping) or set(attended) != active_set:
            errors.append(f"row {index}: attended-length keys do not match active requests")
        else:
            if prompt_tokens_by_request is None:
                if sorted(attended.values()) != sorted(prompt_lengths):
                    errors.append(
                        "row 0: initial attended lengths do not match the preregistered balanced prompts"
                    )
                prompt_tokens_by_request = dict(attended)
            else:
                geometry_membership_matches = set(attended) == set(
                    prompt_tokens_by_request
                )
                if geometry_membership_matches and any(
                    value != prompt_tokens_by_request[request_id] + index
                    for request_id, value in attended.items()
                ):
                    errors.append(
                        f"row {index}: canonical history lengths do not match the preregistered prompts before KV write"
                    )
        for field in ZERO_FIELDS:
            if row[field] != 0:
                errors.append(f"row {index}: {field} must be zero in static identification")
        for field in ("live_kv_blocks", "allocated_kv_blocks"):
            if not isinstance(row[field], int) or row[field] < 0:
                errors.append(f"row {index}: {field} must be a non-negative integer")
        for field in STABLE_FIELDS:
            if field not in baseline:
                baseline[field] = row[field]
            elif row[field] != baseline[field]:
                errors.append(f"row {index}: {field} changed within the episode")

    summary: dict[str, Any] = {}
    try:
        summary = summarize_trace(rows)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"trace accounting failed: {exc}")
    if summary:
        expected_useful = expected_batch * measured_decode_tokens
        expected_lbar = sum(prompt_lengths) / expected_batch + (
            measured_decode_tokens - 1
        ) / 2
        if summary["metered_useful_tokens"] != expected_useful:
            errors.append(
                f"metered useful tokens are {summary['metered_useful_tokens']}, expected {expected_useful}"
            )
        if not math.isclose(summary["effective_batch"], expected_batch, abs_tol=1e-12):
            errors.append(f"effective batch is {summary['effective_batch']}, expected {expected_batch}")
        if not math.isclose(summary["mean_attended_context"], expected_lbar, abs_tol=1e-12):
            errors.append(
                f"mean attended context is {summary['mean_attended_context']}, expected {expected_lbar}"
            )

    return {
        "qc_pass": not errors,
        "errors": errors,
        "requested_api_output_tokens_per_request": measured_decode_tokens + 1,
        "unmetered_bootstrap_tokens_per_request": 1,
        "metered_decode_tokens_per_request": measured_decode_tokens,
        "expected_batch": expected_batch,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_by_request": prompt_tokens_by_request,
        "target_mean_attended_history_tokens": (
            target_mean_attended_history_tokens
        ),
        **summary,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=int)
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--prompt-tokens", type=int)
    geometry.add_argument("--target-mean-attended-history-tokens", type=int)
    parser.add_argument("--measured-decode-tokens", type=int, default=128)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = validate_static_trace(
        read_jsonl(args.trace),
        expected_batch=args.batch,
        prompt_tokens=args.prompt_tokens,
        target_mean_attended_history_tokens=(
            args.target_mean_attended_history_tokens
        ),
        measured_decode_tokens=args.measured_decode_tokens,
    )
    if args.report:
        _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
