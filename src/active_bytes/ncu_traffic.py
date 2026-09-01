"""Parse one application-range Nsight Compute CSV into a V1 traffic record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .decode_doctor import _atomic_write_json


METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ncu_csv(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header_index = None
    for index, row in enumerate(rows):
        if "Metric Name" in row and "Metric Value" in row:
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV has no metric header")
    header = rows[header_index]
    metric_index = header.index("Metric Name")
    value_index = header.index("Metric Value")
    unit_index = header.index("Metric Unit")
    found: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for row in rows[header_index + 1 :]:
        if len(row) <= max(metric_index, value_index, unit_index):
            continue
        metric = row[metric_index]
        if metric not in found:
            continue
        if row[unit_index] != "byte":
            raise ValueError(f"{metric} has non-byte unit {row[unit_index]!r}")
        text = row[value_index].replace(",", "").strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{metric} has invalid value {text!r}") from exc
        if value < 0:
            raise ValueError(f"{metric} is negative")
        found[metric].append(value)
    for metric, values in found.items():
        if len(values) != 1:
            raise ValueError(
                f"expected one range-level value for {metric}; observed {len(values)}"
            )
    return {metric: values[0] for metric, values in found.items()}


def build_traffic_report(anchor_json: Path, ncu_csv: Path) -> dict[str, Any]:
    anchor = json.loads(anchor_json.read_text(encoding="utf-8"))
    if not anchor.get("qc_pass"):
        raise ValueError("anchor runtime QC did not pass")
    observed = parse_ncu_csv(ncu_csv)
    useful_tokens = int(anchor["profile_range"]["metered_useful_tokens"])
    obligations = anchor["uncorrected_obligation_totals"]
    read_bytes = observed["dram__bytes_read.sum"]
    write_bytes = observed["dram__bytes_write.sum"]
    return {
        "schema_version": 1,
        "measurement": "gh200-v1-application-range-replay-traffic-anchor",
        "energy_measurement": False,
        "run": anchor["run"],
        "campaign_lock_sha256": anchor["campaign_lock_sha256"],
        "profile_range": anchor["profile_range"],
        "geometry": anchor["geometry"],
        "active_bytes": anchor["active_bytes"],
        "observed_hbm": {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "read_write_bytes": read_bytes + write_bytes,
            "read_bytes_per_useful_token": read_bytes / useful_tokens,
            "write_bytes_per_useful_token": write_bytes / useful_tokens,
            "read_write_bytes_per_useful_token": (read_bytes + write_bytes)
            / useful_tokens,
        },
        "uncorrected_obligation_totals": obligations,
        "descriptive_uncorrected_ratios": {
            "observed_read_over_accounted_read": read_bytes
            / float(obligations["read_bytes"]),
            "observed_read_write_over_accounted_read_write": (
                read_bytes + write_bytes
            )
            / float(obligations["read_write_bytes"]),
            "observed_write_over_logical_kv_write": write_bytes
            / float(obligations["kv_write_bytes"]),
        },
        "artifact_sha256": {
            "anchor_json": _sha256(anchor_json),
            "ncu_csv": _sha256(ncu_csv),
        },
        "formal_cache_credit_applied": False,
        "formal_v1_decision_eligible": False,
        "formal_v1_note": "Apply the separately frozen cache/residency credit and simultaneous interval only after all repeats are complete.",
        "qc_reasons": [],
        "qc_pass": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-json", required=True, type=Path)
    parser.add_argument("--ncu-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_traffic_report(args.anchor_json, args.ncu_csv)
    _atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
