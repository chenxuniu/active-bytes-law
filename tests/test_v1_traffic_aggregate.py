import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.v1_traffic_aggregate import (  # noqa: E402
    _fit_two_component_law,
    aggregate_v1_traffic,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V1TrafficAggregateTests(unittest.TestCase):
    def test_two_component_fit_recovers_coefficients(self):
        rows = []
        for weight, kv in ((10.0, 1.0), (5.0, 1.0),
                           (10.0, 3.0), (5.0, 3.0)):
            rows.append({
                "weight_bytes_per_token": weight,
                "kv_read_bytes_per_token": kv,
                "observed_read_bytes_per_token": 0.9 * weight + 1.1 * kv,
            })
        fit = _fit_two_component_law(rows)
        self.assertAlmostEqual(fit["rho_weight"], 0.9)
        self.assertAlmostEqual(fit["rho_kv"], 1.1)
        self.assertAlmostEqual(fit["r_squared"], 1.0)

    def test_complete_campaign_is_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "lock.json"
            results = root / "results"
            run_order = []
            cells = []
            for cell_index, batch in enumerate((4, 8)):
                cell_id = f"cell-{cell_index}"
                parameters = {
                    "target_batch": batch,
                    "target_mean_attended_history_tokens": 4096,
                }
                cells.append({"cell_id": cell_id, "repetitions": 2,
                              "parameters": parameters})
                for repeat in (1, 2):
                    order = len(run_order)
                    run_order.append({
                        "run_id": f"run-{order}", "cell_id": cell_id,
                        "order": order, "repeat": repeat,
                        "split": "profiler-anchor", "parameters": parameters,
                    })
            lock = {"campaign_id": "test", "lock_sha256": "lock-sha",
                    "run_count": 4, "cell_count": 2,
                    "run_order": run_order, "cells": cells}
            lock_path.write_text(json.dumps(lock))
            for expected in run_order:
                attempt = results / expected["run_id"] / "attempt-1"
                attempt.mkdir(parents=True)
                anchor, ncu = attempt / "anchor.json", attempt / "ncu.csv"
                anchor.write_text("anchor")
                ncu.write_text("ncu")
                weight = 100.0 / expected["parameters"]["target_batch"]
                kv = 10.0
                observed = 0.9 * weight + 1.1 * kv
                traffic = {
                    "measurement": "gh200-v1-application-range-replay-traffic-anchor",
                    "qc_pass": True, "energy_measurement": False,
                    "campaign_lock_sha256": "lock-sha",
                    "run": {key: expected[key] for key in
                            ("run_id", "cell_id", "order", "repeat", "split")},
                    "geometry": {"batch": expected["parameters"]["target_batch"],
                                 "target_mean_attended_history_tokens": 4096},
                    "active_bytes": {"weight_bytes_per_token": weight,
                                     "kv_read_bytes_per_token": kv,
                                     "active_bytes_read": weight + kv},
                    "observed_hbm": {"read_bytes_per_useful_token": observed,
                                     "write_bytes_per_useful_token": 1.0,
                                     "read_write_bytes_per_useful_token": observed + 1},
                    "descriptive_uncorrected_ratios": {
                        "observed_read_over_accounted_read": observed / (weight + kv)},
                    "artifact_sha256": {"anchor_json": sha256(anchor),
                                        "ncu_csv": sha256(ncu)},
                }
                (attempt / "traffic.json").write_text(json.dumps(traffic))
            report, rows = aggregate_v1_traffic(lock_path, results)
            self.assertTrue(report["qc_pass"])
            self.assertEqual(report["accepted_run_count"], 4)
            self.assertEqual(report["complete_cell_count"], 2)
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(
                report["raw_observed_traffic_law"]["rho_weight"], 0.9)

    def test_missing_run_fails_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parameters = {"target_batch": 4,
                          "target_mean_attended_history_tokens": 4096}
            expected = {"run_id": "missing", "cell_id": "cell", "order": 0,
                        "repeat": 1, "split": "profiler-anchor",
                        "parameters": parameters}
            lock = {"campaign_id": "test", "lock_sha256": "lock",
                    "run_count": 1, "cell_count": 1,
                    "run_order": [expected],
                    "cells": [{"cell_id": "cell", "repetitions": 1,
                               "parameters": parameters}]}
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock))
            report, rows = aggregate_v1_traffic(lock_path, root / "results")
            self.assertFalse(report["qc_pass"])
            self.assertEqual(rows, [])
            self.assertEqual(report["issues"][0]["run_id"], "missing")


if __name__ == "__main__":
    unittest.main()
