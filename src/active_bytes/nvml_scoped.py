"""Scoped NVML telemetry and meter-audit support for Grace Hopper systems."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


NVML_FI_DEV_POWER_AVERAGE = 185
NVML_FI_DEV_POWER_INSTANT = 186
NVML_POWER_SCOPE_GPU = 0
NVML_POWER_SCOPE_MODULE = 1


def _trapezoidal_joules(rows: list[Mapping[str, Any]], power_field: str) -> float:
    energy_joules = 0.0
    for left, right in zip(rows, rows[1:]):
        gap_seconds = (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
        energy_joules += (
            (float(left[power_field]) + float(right[power_field])) / 2.0
        ) * gap_seconds
    return energy_joules


def summarize_scoped_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    maximum_gap_seconds: float = 0.25,
    module_counter_error_limit: float = 0.02,
) -> dict[str, Any]:
    """Summarize scoped power and verify the module cumulative-energy boundary."""

    if maximum_gap_seconds <= 0:
        raise ValueError("maximum_gap_seconds must be positive")
    if not 0 < module_counter_error_limit < 1:
        raise ValueError("module_counter_error_limit must be between zero and one")

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    required = (
        "gpu_index",
        "monotonic_ns",
        "gpu_average_power_w",
        "gpu_instant_power_w",
        "module_average_power_w",
        "module_instant_power_w",
        "total_energy_counter_mj",
    )
    for index, sample in enumerate(samples):
        missing = [field for field in required if field not in sample]
        if missing:
            raise ValueError(f"sample {index} is missing {', '.join(missing)}")
        gpu_index = sample["gpu_index"]
        if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
            raise ValueError(f"sample {index} has invalid gpu_index")
        grouped[gpu_index].append(sample)

    devices: dict[str, Any] = {}
    all_pass = bool(grouped)
    for gpu_index, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["monotonic_ns"])
        if len(rows) < 2:
            raise ValueError(f"GPU {gpu_index} needs at least two samples")
        timestamps = [row["monotonic_ns"] for row in rows]
        if any(
            not isinstance(value, int) or value < 0
            for value in timestamps
        ) or any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"GPU {gpu_index} timestamps must be strictly increasing")
        for row_index, row in enumerate(rows):
            for field in required[2:]:
                value = row[field]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"GPU {gpu_index} sample {row_index} has invalid {field}"
                    )

        duration_seconds = (timestamps[-1] - timestamps[0]) / 1e9
        if duration_seconds <= 0:
            raise ValueError(f"GPU {gpu_index} has non-positive duration")
        maximum_gap = max(
            (right - left) / 1e9 for left, right in zip(timestamps, timestamps[1:])
        )
        counter_joules = (
            float(rows[-1]["total_energy_counter_mj"])
            - float(rows[0]["total_energy_counter_mj"])
        ) / 1000.0
        if counter_joules < 0:
            raise ValueError(f"GPU {gpu_index} cumulative energy decreased")

        gpu_average_joules = _trapezoidal_joules(rows, "gpu_average_power_w")
        gpu_instant_joules = _trapezoidal_joules(rows, "gpu_instant_power_w")
        module_average_joules = _trapezoidal_joules(rows, "module_average_power_w")
        module_instant_joules = _trapezoidal_joules(rows, "module_instant_power_w")
        counter_average_power_w = counter_joules / duration_seconds
        counter_changes = [
            (rows[index]["monotonic_ns"], (
                float(rows[index]["total_energy_counter_mj"])
                - float(rows[index - 1]["total_energy_counter_mj"])
            ) / 1000.0)
            for index in range(1, len(rows))
            if rows[index]["total_energy_counter_mj"]
            != rows[index - 1]["total_energy_counter_mj"]
        ]
        counter_update_intervals = [
            (right[0] - left[0]) / 1e9
            for left, right in zip(counter_changes, counter_changes[1:])
        ]
        module_counter_relative_error = abs(module_average_joules - counter_joules) / max(
            module_average_joules, counter_joules, 1e-12
        )
        cross_scope_difference = abs(gpu_average_joules - counter_joules) / max(
            gpu_average_joules, counter_joules, 1e-12
        )
        gap_pass = maximum_gap <= maximum_gap_seconds
        counter_pass = module_counter_relative_error <= module_counter_error_limit
        device_pass = gap_pass and counter_pass
        all_pass = all_pass and device_pass

        devices[str(gpu_index)] = {
            "samples": len(rows),
            "duration_seconds": duration_seconds,
            "maximum_gap_seconds": maximum_gap,
            "sampling_gap_qc_pass": gap_pass,
            "mean_gpu_average_power_w": statistics.fmean(
                float(row["gpu_average_power_w"]) for row in rows
            ),
            "mean_module_average_power_w": statistics.fmean(
                float(row["module_average_power_w"]) for row in rows
            ),
            "counter_average_power_w": counter_average_power_w,
            "gpu_average_integrated_joules": gpu_average_joules,
            "gpu_instant_integrated_joules": gpu_instant_joules,
            "module_average_integrated_joules": module_average_joules,
            "module_instant_integrated_joules": module_instant_joules,
            "module_counter_joules": counter_joules,
            "energy_counter_changed_samples": len(counter_changes),
            "energy_counter_zero_delta_fraction": (
                (len(rows) - 1 - len(counter_changes)) / (len(rows) - 1)
            ),
            "energy_counter_median_update_interval_seconds": (
                statistics.median(counter_update_intervals)
                if counter_update_intervals
                else None
            ),
            "energy_counter_maximum_update_interval_seconds": (
                max(counter_update_intervals) if counter_update_intervals else None
            ),
            "energy_counter_median_update_joules": (
                statistics.median(change[1] for change in counter_changes)
                if counter_changes
                else None
            ),
            "module_counter_relative_error": module_counter_relative_error,
            "module_counter_qc_pass": counter_pass,
            "cross_scope_difference": cross_scope_difference,
            "qc_pass": device_pass,
        }

    return {
        "schema_version": 1,
        "measurement": "nvml-scoped-power-audit",
        "scope_contract": {
            "gpu_power_scope": 0,
            "module_power_scope": 1,
            "cumulative_counter_scope": "module-on-validated-GH200-stack",
            "gpu_and_module_energy_must_not_be_compared_as_like_for_like": True,
        },
        "maximum_gap_seconds": maximum_gap_seconds,
        "module_counter_error_limit": module_counter_error_limit,
        "qc_pass": all_pass,
        "devices": devices,
    }


def _field_power_w(field: Any, *, label: str) -> float:
    if field.nvmlReturn != 0:
        raise RuntimeError(f"NVML failed for {label}: return={field.nvmlReturn}")
    return float(field.value.uiVal) / 1000.0


def collect_scoped_samples(*, duration_seconds: float, interval_seconds: float) -> list[dict[str, Any]]:
    """Collect synchronized GPU- and module-scope power from every visible GPU."""

    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    pynvml = importlib.import_module("pynvml")
    pynvml.nvmlInit()
    try:
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        if not handles:
            raise RuntimeError("NVML reported no visible GPUs")
        requests = [
            (NVML_FI_DEV_POWER_AVERAGE, NVML_POWER_SCOPE_GPU),
            (NVML_FI_DEV_POWER_AVERAGE, NVML_POWER_SCOPE_MODULE),
            (NVML_FI_DEV_POWER_INSTANT, NVML_POWER_SCOPE_GPU),
            (NVML_FI_DEV_POWER_INSTANT, NVML_POWER_SCOPE_MODULE),
        ]
        samples: list[dict[str, Any]] = []
        start = time.monotonic()
        deadline = start + duration_seconds
        next_sample = start
        sequence = 0
        while True:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(next_sample - now)
            if time.monotonic() > deadline and sequence > 0:
                break
            for gpu_index, handle in enumerate(handles):
                read_start_ns = time.monotonic_ns()
                fields = pynvml.nvmlDeviceGetFieldValues(handle, requests)
                counter_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                read_end_ns = time.monotonic_ns()
                by_key = {(field.fieldId, field.scopeId): field for field in fields}
                samples.append(
                    {
                        "schema_version": 1,
                        "sample_sequence": sequence,
                        "gpu_index": gpu_index,
                        "monotonic_ns": (read_start_ns + read_end_ns) // 2,
                        "wall_time_ns": time.time_ns(),
                        "gpu_average_power_w": _field_power_w(
                            by_key[(NVML_FI_DEV_POWER_AVERAGE, NVML_POWER_SCOPE_GPU)],
                            label="GPU average power",
                        ),
                        "module_average_power_w": _field_power_w(
                            by_key[(NVML_FI_DEV_POWER_AVERAGE, NVML_POWER_SCOPE_MODULE)],
                            label="module average power",
                        ),
                        "gpu_instant_power_w": _field_power_w(
                            by_key[(NVML_FI_DEV_POWER_INSTANT, NVML_POWER_SCOPE_GPU)],
                            label="GPU instantaneous power",
                        ),
                        "module_instant_power_w": _field_power_w(
                            by_key[(NVML_FI_DEV_POWER_INSTANT, NVML_POWER_SCOPE_MODULE)],
                            label="module instantaneous power",
                        ),
                        "total_energy_counter_mj": float(counter_mj),
                    }
                )
            sequence += 1
            next_sample = start + sequence * interval_seconds
        return samples
    finally:
        pynvml.nvmlShutdown()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--maximum-gap-ms", type=float, default=250.0)
    parser.add_argument("--module-counter-error-limit", type=float, default=0.02)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args(argv)
    samples = collect_scoped_samples(
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_ms / 1000.0,
    )
    report = summarize_scoped_samples(
        samples,
        maximum_gap_seconds=args.maximum_gap_ms / 1000.0,
        module_counter_error_limit=args.module_counter_error_limit,
    )
    _write_jsonl(args.output_jsonl, samples)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
