"""Boundary-aligned integration of sampled GPU power telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _interpolate(left: tuple[int, float], right: tuple[int, float], target_ns: int) -> float:
    if right[0] == left[0]:
        raise ValueError("telemetry timestamps must be unique")
    fraction = (target_ns - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])


def integrate_power(
    samples: Iterable[Mapping[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
    maximum_gap_seconds: float = 0.25,
) -> dict[str, Any]:
    """Interpolate both boundaries and integrate power with trapezoids."""

    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
        raise ValueError("end_ns must be greater than start_ns")
    if maximum_gap_seconds <= 0:
        raise ValueError("maximum_gap_seconds must be positive")
    points: list[tuple[int, float]] = []
    for index, sample in enumerate(samples):
        try:
            timestamp = sample["monotonic_ns"]
            power = sample["power_w"]
        except KeyError as exc:
            raise ValueError(f"telemetry row {index} is missing {exc.args[0]}") from exc
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError(f"telemetry row {index} has invalid monotonic_ns")
        if not isinstance(power, (int, float)) or power < 0:
            raise ValueError(f"telemetry row {index} has invalid power_w")
        points.append((timestamp, float(power)))
    points.sort()
    if len(points) < 2 or any(points[index][0] >= points[index + 1][0] for index in range(len(points) - 1)):
        raise ValueError("telemetry needs at least two strictly increasing samples")
    if points[0][0] > start_ns or points[-1][0] < end_ns:
        raise ValueError("telemetry samples must bracket both decode boundaries")

    start_left = None
    start_right = None
    end_left = None
    end_right = None
    for left, right in zip(points, points[1:]):
        if left[0] <= start_ns <= right[0]:
            start_left, start_right = left, right
        if left[0] <= end_ns <= right[0]:
            end_left, end_right = left, right
    if start_left is None or end_left is None or start_right is None or end_right is None:
        raise ValueError("could not locate telemetry boundary brackets")

    bounded: list[tuple[int, float]] = [
        (start_ns, _interpolate(start_left, start_right, start_ns))
    ]
    bounded.extend(point for point in points if start_ns < point[0] < end_ns)
    bounded.append((end_ns, _interpolate(end_left, end_right, end_ns)))
    energy_joules = 0.0
    maximum_gap_ns = 0
    for left, right in zip(bounded, bounded[1:]):
        gap_ns = right[0] - left[0]
        maximum_gap_ns = max(maximum_gap_ns, gap_ns)
        energy_joules += ((left[1] + right[1]) / 2) * (gap_ns / 1_000_000_000)
    allowed_gap_ns = maximum_gap_seconds * 1_000_000_000
    return {
        "schema_version": 1,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000,
        "boundary_start_power_w": bounded[0][1],
        "boundary_end_power_w": bounded[-1][1],
        "integration_points": len(bounded),
        "maximum_gap_seconds": maximum_gap_ns / 1_000_000_000,
        "gap_qc_pass": maximum_gap_ns <= allowed_gap_ns,
        "integrated_power_joules": energy_joules,
    }


def select_power_series(
    samples: Iterable[Mapping[str, Any]],
    *,
    power_field: str,
    gpu_index: int | None = None,
) -> list[dict[str, Any]]:
    """Select one scoped power field and normalize it for integration."""

    selected: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if gpu_index is not None and sample.get("gpu_index") != gpu_index:
            continue
        if "monotonic_ns" not in sample:
            raise ValueError(f"telemetry row {index} is missing monotonic_ns")
        if power_field not in sample:
            raise ValueError(f"telemetry row {index} is missing {power_field}")
        selected.append(
            {
                "monotonic_ns": sample["monotonic_ns"],
                "power_w": sample[power_field],
            }
        )
    if not selected:
        raise ValueError("no telemetry rows matched the requested GPU and power field")
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: telemetry row must be an object")
            rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--start-ns", required=True, type=int)
    parser.add_argument("--end-ns", required=True, type=int)
    parser.add_argument("--maximum-gap-ms", type=float, default=250.0)
    parser.add_argument("--power-field", default="power_w")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = integrate_power(
        select_power_series(
            read_jsonl(args.input),
            power_field=args.power_field,
            gpu_index=args.gpu_index,
        ),
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        maximum_gap_seconds=args.maximum_gap_ms / 1000,
    )
    report["power_field"] = args.power_field
    report["gpu_index"] = args.gpu_index
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; gap_qc_pass={report['gap_qc_pass']}")
    return 0 if report["gap_qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
