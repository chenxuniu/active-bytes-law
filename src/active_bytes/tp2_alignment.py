"""Align one TP=2 decode repeat with simultaneous two-device telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .telemetry import integrate_power, read_jsonl, select_power_series


def _validated_index_pair(values: Sequence[int], *, label: str) -> tuple[int, int]:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != 2 or len(set(normalized)) != 2 or any(
        value < 0 for value in normalized
    ):
        raise ValueError(f"{label} must contain exactly two distinct nonnegative indexes")
    return normalized


def _counter_map(episode: Mapping[str, Any]) -> dict[str, float]:
    value = episode.get("module_counter_joules_by_visible_gpu")
    if not isinstance(value, Mapping):
        raise ValueError("TP=2 episode lacks per-device module energy counters")
    counters: dict[str, float] = {}
    for key, raw in value.items():
        try:
            index = str(int(key))
            joules = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("TP=2 episode has a malformed counter entry") from exc
        if joules <= 0.0:
            raise ValueError(f"visible GPU {index} has a non-positive energy counter delta")
        counters[index] = joules
    return counters


def align_tp2_repeat(
    telemetry: Iterable[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    *,
    visible_gpu_indices: Sequence[int] = (0, 1),
    host_gpu_indices: Sequence[int] = (0, 1),
    maximum_gap_seconds: float = 0.05,
    module_counter_error_limit: float = 0.02,
) -> dict[str, Any]:
    """Integrate both devices over identical decode boundaries and sum energy.

    The primary outcome is the sum of the two scope-0 instantaneous-power
    integrals.  The two scope-1 module counters are secondary, independent QC
    channels.  Device-specific values remain in the artifact so an aggregate
    pass can never hide a failed or missing device.
    """

    visible = _validated_index_pair(visible_gpu_indices, label="visible_gpu_indices")
    host = _validated_index_pair(host_gpu_indices, label="host_gpu_indices")
    if not 0 < module_counter_error_limit < 1:
        raise ValueError("module_counter_error_limit must be between zero and one")
    if not repeat.get("qc_pass"):
        raise ValueError("TP=2 repeat failed its runner-side QC")
    runtime = repeat.get("runtime", {})
    if runtime.get("tensor_parallel_size") != 2:
        raise ValueError("repeat is not a TP=2 execution")
    if runtime.get("visible_gpu_count") != 2:
        raise ValueError("repeat did not resolve exactly two visible devices")
    episodes = repeat.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("TP=2 repeat contains no episodes")

    rows = list(telemetry)
    observed_telemetry_devices = {
        int(row["gpu_index"])
        for row in rows
        if isinstance(row.get("gpu_index"), int)
    }
    if observed_telemetry_devices != set(visible):
        raise ValueError(
            "telemetry device membership differs from the frozen TP=2 pair: "
            f"{sorted(observed_telemetry_devices)}"
        )

    aligned: list[dict[str, Any]] = []
    reasons: list[str] = []
    device_totals = {
        str(index): {
            "gpu_instant_joules": 0.0,
            "module_instant_joules": 0.0,
            "module_counter_joules": 0.0,
        }
        for index in visible
    }
    total_useful_tokens = 0
    total_decode_seconds = 0.0

    for episode in episodes:
        start_ns = int(episode["boundary"]["go_monotonic_ns"])
        end_ns = int(episode["boundary"]["done_monotonic_ns"])
        useful_tokens = int(episode["metered_useful_tokens"])
        if useful_tokens <= 0:
            raise ValueError("metered useful tokens must be positive")
        counters = _counter_map(episode)
        if set(counters) != {str(index) for index in visible}:
            raise ValueError("counter device membership differs from the TP=2 pair")

        devices: list[dict[str, Any]] = []
        episode_reasons: list[str] = []
        for visible_index, host_index in zip(visible, host):
            gpu = integrate_power(
                select_power_series(
                    rows,
                    power_field="gpu_instant_power_w",
                    gpu_index=visible_index,
                ),
                start_ns=start_ns,
                end_ns=end_ns,
                maximum_gap_seconds=maximum_gap_seconds,
            )
            module = integrate_power(
                select_power_series(
                    rows,
                    power_field="module_instant_power_w",
                    gpu_index=visible_index,
                ),
                start_ns=start_ns,
                end_ns=end_ns,
                maximum_gap_seconds=maximum_gap_seconds,
            )
            if not gpu["gap_qc_pass"] or not module["gap_qc_pass"]:
                episode_reasons.append(
                    f"visible GPU {visible_index} telemetry gap exceeded the frozen limit"
                )
            key = str(visible_index)
            counter_joules = counters[key]
            gpu_joules = float(gpu["integrated_power_joules"])
            module_joules = float(module["integrated_power_joules"])
            device_totals[key]["gpu_instant_joules"] += gpu_joules
            device_totals[key]["module_instant_joules"] += module_joules
            device_totals[key]["module_counter_joules"] += counter_joules
            devices.append(
                {
                    "visible_gpu_index": visible_index,
                    "host_gpu_index": host_index,
                    "gpu_instant_integral": gpu,
                    "module_instant_integral": module,
                    "module_counter_joules": counter_joules,
                    "gpu_joules_per_token": gpu_joules / useful_tokens,
                    "module_instant_joules_per_token": (
                        module_joules / useful_tokens
                    ),
                    "module_counter_joules_per_token": (
                        counter_joules / useful_tokens
                    ),
                }
            )

        gpu_sum = sum(
            float(row["gpu_instant_integral"]["integrated_power_joules"])
            for row in devices
        )
        module_sum = sum(
            float(row["module_instant_integral"]["integrated_power_joules"])
            for row in devices
        )
        counter_sum = sum(float(row["module_counter_joules"]) for row in devices)
        aligned.append(
            {
                "episode_id": episode["episode_id"],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "decode_seconds": (end_ns - start_ns) / 1e9,
                "metered_useful_tokens": useful_tokens,
                "devices": devices,
                "gpu_sum_instant_joules": gpu_sum,
                "module_sum_instant_joules": module_sum,
                "module_sum_counter_joules": counter_sum,
                "gpu_sum_joules_per_token": gpu_sum / useful_tokens,
                "module_sum_instant_joules_per_token": module_sum / useful_tokens,
                "module_sum_counter_joules_per_token": counter_sum / useful_tokens,
                "qc_reasons": episode_reasons,
                "qc_pass": not episode_reasons,
            }
        )
        reasons.extend(
            f"episode {episode['episode_id']}: {reason}"
            for reason in episode_reasons
        )
        total_useful_tokens += useful_tokens
        total_decode_seconds += (end_ns - start_ns) / 1e9

    device_reports: dict[str, Any] = {}
    for visible_index, host_index in zip(visible, host):
        key = str(visible_index)
        totals = device_totals[key]
        module_joules = totals["module_instant_joules"]
        counter_joules = totals["module_counter_joules"]
        error = abs(module_joules - counter_joules) / max(
            module_joules, counter_joules, 1e-12
        )
        counter_pass = error <= module_counter_error_limit
        if not counter_pass:
            reasons.append(
                f"visible GPU {visible_index} module integration disagrees with "
                f"its counter by {error:.3%}"
            )
        device_reports[key] = {
            "visible_gpu_index": visible_index,
            "host_gpu_index": host_index,
            **totals,
            "module_instant_counter_relative_error": error,
            "module_counter_qc_pass": counter_pass,
        }

    total_gpu_joules = sum(
        value["gpu_instant_joules"] for value in device_totals.values()
    )
    total_module_joules = sum(
        value["module_instant_joules"] for value in device_totals.values()
    )
    total_counter_joules = sum(
        value["module_counter_joules"] for value in device_totals.values()
    )
    aggregate_error = abs(total_module_joules - total_counter_joules) / max(
        total_module_joules, total_counter_joules, 1e-12
    )
    if aggregate_error > module_counter_error_limit:
        reasons.append(
            "summed module integration disagrees with summed counters by "
            f"{aggregate_error:.3%}"
        )
    if total_decode_seconds < 30.0 and repeat.get("paper_candidate_measurement"):
        reasons.append("aligned paper-candidate repeat contains less than 30 seconds of decode")

    weights = dict(repeat["weights"])
    driver_worker_weight_bytes = int(weights["unique_storage_bytes"])
    accounting_weight_bytes = int(weights["accounting_total_unique_storage_bytes"])
    weights.update(
        {
            "driver_worker_unique_storage_bytes": driver_worker_weight_bytes,
            "unique_storage_bytes": accounting_weight_bytes,
            "inventory_scope": "frozen-full-model-accounting-coordinate",
        }
    )
    return {
        "schema_version": 1,
        "measurement": "aligned-frozen-tp2-decode-repeat",
        "paper_candidate_measurement": bool(
            repeat.get("paper_candidate_measurement", True)
        ),
        "campaign_lock_sha256": repeat["campaign_lock_sha256"],
        "run": repeat["run"],
        "tensor_parallel_size": 2,
        "visible_gpu_indices": list(visible),
        "host_gpu_indices": list(host),
        "scope_contract": {
            "primary": (
                "sum of both GH200 GPU-board NVML scope-0 instantaneous-power "
                "integrals over identical decode boundaries"
            ),
            "secondary": (
                "sum of both GH200 module scope-1 instantaneous-power integrals"
            ),
            "counter": "sum of two independent GH200 module cumulative-energy counters",
            "no_single_device_result_is_the_tp2_outcome": True,
        },
        "episode_count": len(aligned),
        "episodes": aligned,
        "devices": device_reports,
        "totals": {
            "decode_seconds": total_decode_seconds,
            "minimum_episode_decode_seconds": min(
                float(episode["decode_seconds"]) for episode in aligned
            ),
            "metered_useful_tokens": total_useful_tokens,
            "gpu_instant_joules": total_gpu_joules,
            "module_instant_joules": total_module_joules,
            "module_counter_joules": total_counter_joules,
            "gpu_joules_per_token": total_gpu_joules / total_useful_tokens,
            "module_instant_joules_per_token": (
                total_module_joules / total_useful_tokens
            ),
            "module_counter_joules_per_token": (
                total_counter_joules / total_useful_tokens
            ),
            "module_instant_counter_relative_error": aggregate_error,
        },
        "active_bytes": repeat["active_bytes"],
        "model_geometry": repeat["model_geometry"],
        "weights": weights,
        "runtime": repeat["runtime"],
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def _parse_indexes(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indexes must be comma-separated integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-jsonl", required=True, type=Path)
    parser.add_argument("--repeat-json", required=True, type=Path)
    parser.add_argument("--visible-gpu-indices", type=_parse_indexes, default=[0, 1])
    parser.add_argument("--host-gpu-indices", type=_parse_indexes, default=[0, 1])
    parser.add_argument("--maximum-gap-ms", type=float, default=50.0)
    parser.add_argument("--module-counter-error-limit", type=float, default=0.02)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    repeat = json.loads(args.repeat_json.read_text(encoding="utf-8"))
    report = align_tp2_repeat(
        read_jsonl(args.telemetry_jsonl),
        repeat,
        visible_gpu_indices=args.visible_gpu_indices,
        host_gpu_indices=args.host_gpu_indices,
        maximum_gap_seconds=args.maximum_gap_ms / 1000.0,
        module_counter_error_limit=args.module_counter_error_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
