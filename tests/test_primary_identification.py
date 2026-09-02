import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.primary_identification import (  # noqa: E402
    freeze_primary_identification,
    student_t_quantile,
)


class PrimaryIdentificationTests(unittest.TestCase):
    def test_student_t_quantile_matches_known_df4_value(self):
        self.assertAlmostEqual(student_t_quantile(0.975, 4), 2.776445105, places=7)

    def test_complete_split_freezes_two_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "output"
            lock_path = root / "lock.json"
            campaign_sha = "a" * 64
            weight_bytes = 15_200_000_000
            kv_bytes = 57_344
            runs = []
            order = 0
            cells = []
            for split, batches in (
                ("coefficient-fit", (4, 32)),
                ("residual-calibration", (16,)),
            ):
                for context in (4096, 8192, 16384):
                    for batch in batches:
                        cells.append((split, context, batch))
            for split, context, batch in cells:
                cell_id = f"cell-l{context}-b{batch}"
                for repeat in range(1, 6):
                    run_id = f"run-{order:02d}"
                    run = {
                        "order": order,
                        "run_id": run_id,
                        "cell_id": cell_id,
                        "split": split,
                        "repeat": repeat,
                        "parameters": {
                            "target_batch": batch,
                            "target_mean_attended_history_tokens": context,
                        },
                    }
                    runs.append(run)
                    weight_per_token = weight_bytes / batch
                    kv_read = kv_bytes * context
                    kv_write = float(kv_bytes)
                    weight_gb = weight_per_token / 1e9
                    kv_gb = (kv_read + kv_write) / 1e9
                    noise = (repeat - 3) * 0.0002
                    calibration_shift = 0.002 * (context / 4096) if split == "residual-calibration" else 0.0
                    outcome = 0.08 + 0.14 * weight_gb + 0.24 * kv_gb + noise + calibration_shift
                    alignment = {
                        "campaign_lock_sha256": campaign_sha,
                        "run": {
                            key: run[key]
                            for key in ("order", "run_id", "cell_id", "split", "repeat")
                        },
                        "qc_pass": True,
                        "totals": {
                            "gpu_joules_per_token": outcome,
                            "decode_seconds": 31.0,
                            "metered_useful_tokens": batch * 1024,
                        },
                        "active_bytes": {
                            "weight_bytes_per_token": weight_per_token,
                            "kv_read_bytes_per_token": kv_read,
                            "kv_write_bytes_per_token": kv_write,
                        },
                        "weights": {"unique_storage_bytes": weight_bytes},
                        "model_geometry": {
                            "kv_bytes_per_historical_token": kv_bytes
                        },
                    }
                    attempt = results / "primary-identification" / run_id / "attempt-1"
                    attempt.mkdir(parents=True)
                    (attempt / "alignment.json").write_text(json.dumps(alignment))
                    order += 1
            lock_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "campaign-1",
                        "lock_sha256": campaign_sha,
                        "run_count": 45,
                        "run_order": runs,
                    }
                )
            )

            summary = freeze_primary_identification(lock_path, results, output)
            self.assertTrue(summary["qc_pass"])
            self.assertEqual(summary["outcome"], "gross_gpu_board_joules_per_useful_token")
            self.assertEqual(summary["accepted_run_count"], 45)
            self.assertEqual(summary["fit_run_count"], 30)
            self.assertEqual(summary["calibration_run_count"], 15)
            self.assertTrue(summary["scientific_decisions"]["p2_slope_positivity_pass"])
            coefficient = json.loads((output / "coefficient-artifact.json").read_text())
            fit = coefficient["fit"]
            self.assertAlmostEqual(
                fit["coefficients_scaled"]["alpha_joules_per_decimal_gb"],
                0.14,
                places=6,
            )
            self.assertAlmostEqual(
                fit["coefficients_scaled"]["beta_joules_per_decimal_gb"],
                0.24,
                places=6,
            )
            self.assertFalse(coefficient["outcome_contract"]["idle_correction_applied"])
            self.assertTrue(summary["evaluation_release_candidate"])
            envelope = json.loads((output / "discrepancy-envelope.json").read_text())
            self.assertEqual(envelope["calibration_cell_count"], 3)
            self.assertEqual(envelope["coefficient_artifact_sha256"], summary["coefficient_artifact"]["sha256"])
            self.assertTrue(math.isfinite(envelope["envelope"]["common_residual_range_joules_per_token"][0]))


if __name__ == "__main__":
    unittest.main()
