import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.primary_evaluation import evaluate_held_out_rows  # noqa: E402


class PrimaryEvaluationTests(unittest.TestCase):
    def rows(self, residual_by_cell=None):
        residual_by_cell = residual_by_cell or {}
        rows = []
        order = 0
        c, alpha, beta = 0.10, 0.20, 0.30
        for context in (4096, 8192, 16384):
            for batch in (8, 24):
                cell = f"l{context}-b{batch}"
                weight = 15.2 / batch
                kv = 0.000057344 * (context + 1)
                prediction = c + alpha * weight + beta * kv
                for repeat in range(1, 6):
                    noise = (repeat - 3) * 0.0001
                    rows.append(
                        {
                            "order": order,
                            "run_id": f"run-{order}",
                            "cell_id": cell,
                            "split": "evaluation",
                            "repeat": repeat,
                            "target_batch": batch,
                            "target_mean_attended_history_tokens": context,
                            "gross_gpu_joules_per_token": prediction
                            + residual_by_cell.get(cell, 0.0)
                            + noise,
                            "weight_gb_per_token": weight,
                            "kv_rw_gb_per_token": kv,
                        }
                    )
                    order += 1
        return rows

    def test_exact_held_out_law_passes_all_cell_gates(self):
        result = evaluate_held_out_rows(
            self.rows(),
            {
                "c_joules_per_token": 0.10,
                "alpha_joules_per_decimal_gb": 0.20,
                "beta_joules_per_decimal_gb": 0.30,
            },
            [-0.02, 0.02],
            slope_positivity_pass=True,
            coefficient_equivalence_pass=False,
        )
        self.assertEqual(result["summary"]["coverage_count"], 6)
        self.assertAlmostEqual(result["summary"]["median_absolute_relative_error"], 0.0)
        self.assertTrue(result["gates"]["v2_two_coefficient_law_pass"])
        self.assertEqual(result["gates"]["highest_supported_claim"], "P2")
        self.assertFalse(result["gates"]["p3_single_coefficient_equivalence_pass"])
        self.assertTrue(result["residual_trends"]["qc_pass"])

    def test_systematic_batch_residual_fails_coverage_and_trend(self):
        residuals = {
            f"l{context}-b{batch}": (-0.12 if batch == 8 else 0.12)
            for context in (4096, 8192, 16384)
            for batch in (8, 24)
        }
        result = evaluate_held_out_rows(
            self.rows(residuals),
            {
                "c_joules_per_token": 0.10,
                "alpha_joules_per_decimal_gb": 0.20,
                "beta_joules_per_decimal_gb": 0.30,
            },
            [-0.02, 0.02],
            slope_positivity_pass=True,
            coefficient_equivalence_pass=False,
        )
        self.assertEqual(result["summary"]["coverage_count"], 0)
        self.assertFalse(result["residual_trends"]["qc_pass"])
        self.assertFalse(result["gates"]["v2_two_coefficient_law_pass"])
        self.assertEqual(result["gates"]["highest_supported_claim"], "P1")


if __name__ == "__main__":
    unittest.main()
