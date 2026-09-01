"""Compare two non-paper FP8 KV calibration-doctor artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .decode_doctor import _atomic_write_json


def _layer_map(report: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    rows = report.get("kv_scales", {}).get("layers", [])
    result: dict[str, tuple[float, float]] = {}
    for row in rows:
        name = row.get("name")
        k_scale = row.get("k_scale")
        v_scale = row.get("v_scale")
        valid = (
            isinstance(name, str)
            and isinstance(k_scale, (int, float))
            and isinstance(v_scale, (int, float))
        )
        if not valid:
            continue
        result[name] = (float(k_scale), float(v_scale))
    return result


def compare_calibration_doctors(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-8,
) -> dict[str, Any]:
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("scale tolerances cannot be negative")
    reasons: list[str] = []
    if not first.get("qc_pass") or not second.get("qc_pass"):
        reasons.append("both source doctors must pass their own QC")

    contract_paths = (
        ("model",),
        ("dataset", "id"),
        ("dataset", "revision"),
        ("dataset", "split"),
        ("dataset", "seed"),
        ("dataset", "num_calibration_samples"),
        ("dataset", "max_sequence_length"),
        ("dataset", "rendered_text_sha256"),
        ("recipe",),
        ("runtime", "packages"),
        ("runtime", "cuda"),
    )

    def resolve(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    mismatched_contract_fields = [
        ".".join(path)
        for path in contract_paths
        if resolve(first, path) != resolve(second, path)
    ]
    if mismatched_contract_fields:
        reasons.append("the two doctors do not share an identical calibration contract")

    first_probe = first.get("baseline_parameters_before", {}).get("probe_sha256")
    second_probe = second.get("baseline_parameters_before", {}).get("probe_sha256")
    parameter_probe_match = bool(first_probe) and first_probe == second_probe
    if not parameter_probe_match:
        reasons.append("the baseline model parameter probes differ")

    first_layers = _layer_map(first)
    second_layers = _layer_map(second)
    layer_names_match = bool(first_layers) and first_layers.keys() == second_layers.keys()
    if not layer_names_match:
        reasons.append("the discovered scale-bearing layer sets differ or are empty")

    differences: list[dict[str, Any]] = []
    for name in sorted(first_layers.keys() & second_layers.keys()):
        first_k, first_v = first_layers[name]
        second_k, second_v = second_layers[name]
        for kind, left, right in (
            ("k_scale", first_k, second_k),
            ("v_scale", first_v, second_v),
        ):
            denominator = max(abs(left), abs(right), absolute_tolerance)
            relative_difference = abs(left - right) / denominator
            close = math.isclose(
                left,
                right,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
            differences.append(
                {
                    "name": name,
                    "kind": kind,
                    "first": left,
                    "second": right,
                    "absolute_difference": abs(left - right),
                    "relative_difference": relative_difference,
                    "within_tolerance": close,
                }
            )
    out_of_tolerance = [row for row in differences if not row["within_tolerance"]]
    if out_of_tolerance:
        reasons.append("one or more calibrated scales differ beyond tolerance")

    return {
        "schema_version": 1,
        "measurement": "fp8-kv-calibration-doctor-repeat-comparison",
        "non_paper_measurement": True,
        "may_enter_paper_outcomes": False,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "mismatched_contract_fields": mismatched_contract_fields,
        "parameter_probe_match": parameter_probe_match,
        "layer_names_match": layer_names_match,
        "compared_scale_count": len(differences),
        "maximum_absolute_difference": max(
            (row["absolute_difference"] for row in differences), default=None
        ),
        "maximum_relative_difference": max(
            (row["relative_difference"] for row in differences), default=None
        ),
        "out_of_tolerance": out_of_tolerance,
        "qc_reasons": reasons,
        "qc_pass": not reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-json", required=True, type=Path)
    parser.add_argument("--second-json", required=True, type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    first = json.loads(args.first_json.read_text(encoding="utf-8"))
    second = json.loads(args.second_json.read_text(encoding="utf-8"))
    report = compare_calibration_doctors(
        first,
        second,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    report["first_json"] = str(args.first_json)
    report["second_json"] = str(args.second_json)
    _atomic_write_json(args.output_json, report)
    summary = {
        key: value for key, value in report.items() if key != "out_of_tolerance"
    }
    summary["out_of_tolerance_count"] = len(report["out_of_tolerance"])
    summary["output_json"] = str(args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
