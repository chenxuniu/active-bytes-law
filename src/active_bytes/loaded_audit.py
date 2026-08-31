"""Align scoped NVML telemetry with a vLLM loaded-meter doctor interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .telemetry import integrate_power, read_jsonl, select_power_series


POWER_FIELDS = (
    "gpu_average_power_w",
    "gpu_instant_power_w",
    "module_average_power_w",
    "module_instant_power_w",
)


def align_loaded_audit(
    telemetry: Iterable[Mapping[str, Any]],
    doctor: Mapping[str, Any],
    *,
    gpu_index: int = 0,
    maximum_gap_seconds: float = 0.05,
    module_counter_error_limit: float = 0.02,
) -> dict[str, Any]:
    """Integrate all scoped fields on the doctor's exact GO/DONE interval."""

    rows = list(telemetry)
    try:
        start_ns = doctor["boundary"]["go_monotonic_ns"]
        end_ns = doctor["boundary"]["decode_done_monotonic_ns"]
        counter_joules = doctor["energy"]["module_energy_joules"]
        token_boundary_pass = doctor["token_boundary_qc_pass"]
    except KeyError as exc:
        raise ValueError(f"doctor report is missing {exc.args[0]}") from exc
    if not isinstance(counter_joules, (int, float)) or counter_joules <= 0:
        raise ValueError("doctor module energy must be positive")

    integrals: dict[str, Any] = {}
    for field in POWER_FIELDS:
        report = integrate_power(
            select_power_series(rows, power_field=field, gpu_index=gpu_index),
            start_ns=start_ns,
            end_ns=end_ns,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        integrals[field] = report

    module_average_joules = integrals["module_average_power_w"][
        "integrated_power_joules"
    ]
    module_instant_joules = integrals["module_instant_power_w"][
        "integrated_power_joules"
    ]
    average_error = abs(module_average_joules - counter_joules) / max(
        module_average_joules, counter_joules
    )
    instant_error = abs(module_instant_joules - counter_joules) / max(
        module_instant_joules, counter_joules
    )
    gaps_pass = all(report["gap_qc_pass"] for report in integrals.values())
    counter_agreement_pass = min(average_error, instant_error) <= module_counter_error_limit
    reasons: list[str] = []
    if not token_boundary_pass:
        reasons.append("doctor token boundary failed")
    if not gaps_pass:
        reasons.append("one or more telemetry fields exceeded the sampling-gap limit")
    if not counter_agreement_pass:
        reasons.append(
            "neither module average nor module instantaneous integration agrees "
            f"with the counter within {module_counter_error_limit:.2%}"
        )
    return {
        "schema_version": 1,
        "measurement": "loaded-scoped-power-alignment",
        "gpu_index": gpu_index,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1e9,
        "module_counter_joules": counter_joules,
        "integrals": integrals,
        "module_average_counter_relative_error": average_error,
        "module_instant_counter_relative_error": instant_error,
        "module_counter_error_limit": module_counter_error_limit,
        "counter_agreement_qc_pass": counter_agreement_pass,
        "sampling_gap_qc_pass": gaps_pass,
        "token_boundary_qc_pass": bool(token_boundary_pass),
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-jsonl", required=True, type=Path)
    parser.add_argument("--doctor-json", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--maximum-gap-ms", type=float, default=50.0)
    parser.add_argument("--module-counter-error-limit", type=float, default=0.02)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    doctor = json.loads(args.doctor_json.read_text())
    report = align_loaded_audit(
        read_jsonl(args.telemetry_jsonl),
        doctor,
        gpu_index=args.gpu_index,
        maximum_gap_seconds=args.maximum_gap_ms / 1000.0,
        module_counter_error_limit=args.module_counter_error_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
