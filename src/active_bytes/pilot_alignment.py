"""Align one frozen pilot repeat with scoped GH200 power telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .telemetry import integrate_power, read_jsonl, select_power_series


def align_pilot_repeat(
    telemetry: Iterable[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    *,
    gpu_index: int = 0,
    host_gpu_index: int | None = None,
    maximum_gap_seconds: float = 0.05,
    module_counter_error_limit: float = 0.02,
) -> dict[str, Any]:
    rows = list(telemetry)
    if not repeat.get("qc_pass"):
        raise ValueError("pilot repeat failed its runner-side QC")
    episodes = repeat.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("pilot repeat contains no episodes")
    aligned: list[dict[str, Any]] = []
    reasons: list[str] = []
    total_gpu_joules = 0.0
    total_module_instant_joules = 0.0
    total_module_counter_joules = 0.0
    total_useful_tokens = 0
    total_decode_seconds = 0.0
    for episode in episodes:
        start_ns = episode["boundary"]["go_monotonic_ns"]
        end_ns = episode["boundary"]["done_monotonic_ns"]
        gpu = integrate_power(
            select_power_series(
                rows, power_field="gpu_instant_power_w", gpu_index=gpu_index
            ),
            start_ns=start_ns,
            end_ns=end_ns,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        module = integrate_power(
            select_power_series(
                rows, power_field="module_instant_power_w", gpu_index=gpu_index
            ),
            start_ns=start_ns,
            end_ns=end_ns,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        episode_reasons: list[str] = []
        if not gpu["gap_qc_pass"] or not module["gap_qc_pass"]:
            episode_reasons.append("telemetry sampling gap exceeded the frozen limit")
        useful = int(episode["metered_useful_tokens"])
        counter_joules = float(episode["module_counter_joules"])
        gpu_joules = float(gpu["integrated_power_joules"])
        module_joules = float(module["integrated_power_joules"])
        aligned.append(
            {
                "episode_id": episode["episode_id"],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "decode_seconds": (end_ns - start_ns) / 1e9,
                "metered_useful_tokens": useful,
                "gpu_instant_integral": gpu,
                "module_instant_integral": module,
                "module_counter_joules": counter_joules,
                "gpu_joules_per_token": gpu_joules / useful,
                "module_instant_joules_per_token": module_joules / useful,
                "module_counter_joules_per_token": counter_joules / useful,
                "qc_reasons": episode_reasons,
                "qc_pass": not episode_reasons,
            }
        )
        reasons.extend(
            f"episode {episode['episode_id']}: {reason}" for reason in episode_reasons
        )
        total_gpu_joules += gpu_joules
        total_module_instant_joules += module_joules
        total_module_counter_joules += counter_joules
        total_useful_tokens += useful
        total_decode_seconds += (end_ns - start_ns) / 1e9
    agreement_error = abs(
        total_module_instant_joules - total_module_counter_joules
    ) / max(total_module_instant_joules, total_module_counter_joules, 1e-12)
    if agreement_error > module_counter_error_limit:
        reasons.append(
            "aggregated module instantaneous integration disagrees with the "
            f"counter by {agreement_error:.3%}"
        )
    minimum_repeat_seconds = min(
        float(episode["decode_seconds"]) for episode in episodes
    )
    if total_decode_seconds < 30.0:
        reasons.append("aligned repeat contains less than 30 seconds of decode")
    return {
        "schema_version": 1,
        "measurement": "aligned-frozen-pilot-repeat",
        "campaign_lock_sha256": repeat["campaign_lock_sha256"],
        "run": repeat["run"],
        "gpu_index": gpu_index,
        "host_gpu_index": host_gpu_index,
        "scope_contract": {
            "primary": "GPU-board scope-0 instantaneous-power integral",
            "secondary": "GH200 module scope-1 instantaneous-power integral",
            "counter": "GH200 module cumulative-energy counter",
        },
        "episode_count": len(aligned),
        "episodes": aligned,
        "totals": {
            "decode_seconds": total_decode_seconds,
            "minimum_episode_decode_seconds": minimum_repeat_seconds,
            "metered_useful_tokens": total_useful_tokens,
            "gpu_instant_joules": total_gpu_joules,
            "module_instant_joules": total_module_instant_joules,
            "module_counter_joules": total_module_counter_joules,
            "gpu_joules_per_token": total_gpu_joules / total_useful_tokens,
            "module_instant_joules_per_token": (
                total_module_instant_joules / total_useful_tokens
            ),
            "module_counter_joules_per_token": (
                total_module_counter_joules / total_useful_tokens
            ),
            "module_instant_counter_relative_error": agreement_error,
        },
        "active_bytes": repeat["active_bytes"],
        "model_geometry": repeat["model_geometry"],
        "weights": repeat["weights"],
        "runtime": repeat["runtime"],
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-jsonl", required=True, type=Path)
    parser.add_argument("--repeat-json", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--host-gpu-index",
        type=int,
        help="Host-visible GPU index; the container-visible telemetry index may differ.",
    )
    parser.add_argument("--maximum-gap-ms", type=float, default=50.0)
    parser.add_argument("--module-counter-error-limit", type=float, default=0.02)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    repeat = json.loads(args.repeat_json.read_text(encoding="utf-8"))
    report = align_pilot_repeat(
        read_jsonl(args.telemetry_jsonl),
        repeat,
        gpu_index=args.gpu_index,
        host_gpu_index=args.host_gpu_index,
        maximum_gap_seconds=args.maximum_gap_ms / 1000.0,
        module_counter_error_limit=args.module_counter_error_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
