import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.idle_window_audit import audit_idle_windows  # noqa: E402


class IdleWindowAuditTests(unittest.TestCase):
    def test_reports_bracketing_samples_without_promoting_paper_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            results = root / "results"
            run = {
                "order": 0,
                "run_id": "run-1",
                "cell_id": "cell-1",
                "split": "coefficient-fit",
                "repeat": 1,
            }
            lock_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "campaign-1",
                        "lock_sha256": "a" * 64,
                        "run_count": 1,
                        "run_order": [run],
                    }
                )
            )
            attempt = results / "primary-identification" / "run-1" / "attempt-1"
            attempt.mkdir(parents=True)
            alignment = {
                "campaign_lock_sha256": "a" * 64,
                "run": run,
                "qc_pass": True,
                "episodes": [{"start_ns": 1_000_000_000, "end_ns": 2_000_000_000}],
                "totals": {
                    "decode_seconds": 1.0,
                    "gpu_joules_per_token": 0.5,
                },
            }
            (attempt / "alignment.json").write_text(json.dumps(alignment))
            telemetry = [
                (0, 90.0),
                (500_000_000, 91.0),
                (1_000_000_000, 400.0),
                (2_000_000_000, 390.0),
                (2_500_000_000, 92.0),
                (3_000_000_000, 91.0),
            ]
            with (attempt / "telemetry.jsonl").open("w") as handle:
                for timestamp, power in telemetry:
                    handle.write(
                        json.dumps(
                            {
                                "gpu_index": 0,
                                "monotonic_ns": timestamp,
                                "gpu_instant_power_w": power,
                            }
                        )
                        + "\n"
                    )

            report = audit_idle_windows(
                lock_path, results, guard_seconds=0.25, gpu_index=0
            )
            self.assertTrue(report["artifact_qc_pass"])
            self.assertTrue(report["all_runs_have_bracketing_samples"])
            self.assertEqual(report["campaign_lock_sha256"], "a" * 64)
            self.assertFalse(
                report["idle_correction_contract"]["paper_outcome_eligible"]
            )
            self.assertEqual(report["accepted_run_count"], 1)
            self.assertEqual(report["runs"][0]["pre_decode"]["sample_count"], 2)
            self.assertEqual(report["runs"][0]["post_decode"]["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
