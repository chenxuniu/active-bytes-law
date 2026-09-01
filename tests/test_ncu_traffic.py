import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.ncu_traffic import build_traffic_report, parse_ncu_csv  # noqa: E402


HEADER = (
    '"ID","Metric Name","Metric Unit","Metric Value"\n'
    '"0","dram__bytes_read.sum","byte","2,048"\n'
    '"0","dram__bytes_write.sum","byte","512"\n'
)


class NcuTrafficTests(unittest.TestCase):
    def test_parse_single_range_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ncu.csv"
            path.write_text("==PROF== note\n" + HEADER)
            self.assertEqual(
                parse_ncu_csv(path),
                {"dram__bytes_read.sum": 2048.0, "dram__bytes_write.sum": 512.0},
            )

    def test_multiple_range_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ncu.csv"
            path.write_text(HEADER + '"1","dram__bytes_read.sum","byte","3"\n')
            with self.assertRaisesRegex(ValueError, "expected one range-level"):
                parse_ncu_csv(path)

    def test_traffic_report_uses_useful_token_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "ncu.csv"
            csv_path.write_text(HEADER)
            anchor_path = root / "anchor.json"
            anchor_path.write_text(
                json.dumps(
                    {
                        "qc_pass": True,
                        "run": {"run_id": "r"},
                        "campaign_lock_sha256": "lock",
                        "profile_range": {"metered_useful_tokens": 4},
                        "geometry": {"batch": 4},
                        "active_bytes": {},
                        "uncorrected_obligation_totals": {
                            "read_bytes": 1024,
                            "read_write_bytes": 1280,
                            "kv_write_bytes": 256,
                        },
                    }
                )
            )
            report = build_traffic_report(anchor_path, csv_path)
            self.assertEqual(report["observed_hbm"]["read_bytes_per_useful_token"], 512)
            self.assertEqual(
                report["descriptive_uncorrected_ratios"][
                    "observed_read_over_accounted_read"
                ],
                2,
            )


if __name__ == "__main__":
    unittest.main()
