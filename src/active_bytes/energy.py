"""Aggregate decode-only cumulative-energy episodes into independent repeats."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: energy record must be an object")
            records.append(value)
    return records


def _episode(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    required = (
        "run_id",
        "cell_id",
        "repeat",
        "episode_id",
        "boundary",
        "sensor",
        "counter_start_mj",
        "counter_end_mj",
        "counter_read_start_monotonic_ns",
        "go_monotonic_ns",
        "decode_done_monotonic_ns",
        "counter_read_end_monotonic_ns",
        "metered_useful_tokens",
        "decode_seconds",
        "qc_pass",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"energy row {index} is missing {', '.join(missing)}")
    if record["boundary"] != "decode-only":
        raise ValueError(f"energy row {index} is not decode-only")
    if record["sensor"] != "DCGM_FI_DEV_TOTAL_ENERGY":
        raise ValueError(f"energy row {index} has unsupported sensor")
    start = record["counter_start_mj"]
    end = record["counter_end_mj"]
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
        raise ValueError(f"energy row {index} has an invalid cumulative counter delta")
    useful = record["metered_useful_tokens"]
    seconds = record["decode_seconds"]
    if not isinstance(useful, int) or isinstance(useful, bool) or useful <= 0:
        raise ValueError(f"energy row {index} has invalid useful tokens")
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        raise ValueError(f"energy row {index} has invalid decode_seconds")
    times = [
        record["counter_read_start_monotonic_ns"],
        record["go_monotonic_ns"],
        record["decode_done_monotonic_ns"],
        record["counter_read_end_monotonic_ns"],
    ]
    if any(not isinstance(value, int) or value < 0 for value in times) or times != sorted(times):
        raise ValueError(f"energy row {index} violates start-read <= GO <= DONE <= end-read")
    integrated = record.get("integrated_power_joules")
    if integrated is not None and (
        not isinstance(integrated, (int, float)) or integrated < 0
    ):
        raise ValueError(f"energy row {index} has invalid integrated_power_joules")
    return {
        "run_id": record["run_id"],
        "cell_id": record["cell_id"],
        "repeat": record["repeat"],
        "joules": (end - start) / 1000.0,
        "useful_tokens": useful,
        "decode_seconds": float(seconds),
        "integrated_power_joules": integrated,
        "qc_pass": bool(record["qc_pass"]),
    }


def summarize_energy(
    records: Iterable[Mapping[str, Any]],
    *,
    minimum_repeats: int = 3,
    minimum_decode_seconds: float = 30.0,
    cv_limit: float = 0.03,
    integration_error_limit: float = 0.02,
) -> dict[str, Any]:
    if minimum_repeats <= 0 or minimum_decode_seconds <= 0:
        raise ValueError("minimum repeats and duration must be positive")
    repeat_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    parsing_errors: list[str] = []
    for index, record in enumerate(records):
        try:
            episode = _episode(record, index)
        except ValueError as exc:
            parsing_errors.append(str(exc))
            continue
        repeat_groups[(episode["cell_id"], episode["repeat"])].append(episode)

    repeats_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (cell_id, repeat), episodes in sorted(repeat_groups.items()):
        joules = sum(item["joules"] for item in episodes)
        useful = sum(item["useful_tokens"] for item in episodes)
        seconds = sum(item["decode_seconds"] for item in episodes)
        integrated_values = [item["integrated_power_joules"] for item in episodes]
        integrated = (
            sum(float(value) for value in integrated_values)
            if all(value is not None for value in integrated_values)
            else None
        )
        relative_error = None
        if integrated is not None:
            relative_error = abs(joules - integrated) / max(joules, integrated, 1e-12)
        qc_reasons: list[str] = []
        if not all(item["qc_pass"] for item in episodes):
            qc_reasons.append("one or more episodes failed QC")
        if seconds < minimum_decode_seconds:
            qc_reasons.append(
                f"decode-only duration {seconds:.3f}s is below {minimum_decode_seconds:.3f}s"
            )
        if relative_error is not None and relative_error > integration_error_limit:
            qc_reasons.append(
                f"counter/power disagreement {relative_error:.4%} exceeds {integration_error_limit:.2%}"
            )
        repeats_by_cell[cell_id].append(
            {
                "repeat": repeat,
                "run_ids": sorted({item["run_id"] for item in episodes}),
                "episodes": len(episodes),
                "joules": joules,
                "metered_useful_tokens": useful,
                "decode_seconds": seconds,
                "joules_per_useful_token": joules / useful,
                "integrated_power_joules": integrated,
                "counter_power_relative_error": relative_error,
                "qc_pass": not qc_reasons,
                "qc_reasons": qc_reasons,
            }
        )

    cells: dict[str, Any] = {}
    all_pass = not parsing_errors
    for cell_id, repeats in sorted(repeats_by_cell.items()):
        valid = [item for item in repeats if item["qc_pass"]]
        values = [item["joules_per_useful_token"] for item in valid]
        mean = statistics.fmean(values) if values else None
        stdev = statistics.stdev(values) if len(values) >= 2 else None
        cv = stdev / mean if stdev is not None and mean not in (None, 0) else None
        reasons: list[str] = []
        if len(valid) < minimum_repeats:
            reasons.append(f"only {len(valid)} valid repeats; need {minimum_repeats}")
        if cv is None:
            reasons.append("CV requires at least two valid repeats")
        elif cv > cv_limit:
            reasons.append(f"repeat CV {cv:.4%} exceeds {cv_limit:.2%}")
        cell_pass = not reasons
        all_pass = all_pass and cell_pass
        cells[cell_id] = {
            "qc_pass": cell_pass,
            "qc_reasons": reasons,
            "observed_repeats": len(repeats),
            "valid_repeats": len(valid),
            "mean_joules_per_useful_token": mean,
            "sample_stdev_joules_per_useful_token": stdev,
            "coefficient_of_variation": cv,
            "repeats": repeats,
        }
    if not cells:
        all_pass = False
        parsing_errors.append("no valid energy records were found")
    return {
        "schema_version": 1,
        "qc_pass": all_pass,
        "parsing_errors": parsing_errors,
        "minimum_repeats": minimum_repeats,
        "minimum_decode_seconds": minimum_decode_seconds,
        "cv_limit": cv_limit,
        "integration_error_limit": integration_error_limit,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-repeats", type=int, default=3)
    parser.add_argument("--minimum-decode-seconds", type=float, default=30.0)
    parser.add_argument("--cv-limit", type=float, default=0.03)
    parser.add_argument("--integration-error-limit", type=float, default=0.02)
    args = parser.parse_args(argv)
    report = summarize_energy(
        read_jsonl(args.input),
        minimum_repeats=args.minimum_repeats,
        minimum_decode_seconds=args.minimum_decode_seconds,
        cv_limit=args.cv_limit,
        integration_error_limit=args.integration_error_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; qc_pass={report['qc_pass']}")
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
