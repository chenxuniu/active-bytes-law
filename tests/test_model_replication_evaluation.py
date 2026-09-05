import unittest

from active_bytes.model_replication_evaluation import evaluate_replication_rows


class ModelReplicationEvaluationTests(unittest.TestCase):
    def test_exact_frozen_predictions_pass_without_refit(self):
        coefficients = {
            "intercept_joules_per_token": 0.01,
            "alpha_weight_joules_per_decimal_gb": 0.1,
            "beta_kv_joules_per_decimal_gb": 0.2,
            "p_time_watts": 100.0,
        }
        gates = {
            "median_absolute_relative_error_maximum": 0.05,
            "maximum_absolute_relative_error_maximum": 0.1,
            "required_cells_passing_ten_percent_error": 6,
        }
        rows = []
        for context in (6144, 12288, 14336):
            for batch in (6, 12):
                for repeat in range(1, 6):
                    duration = 30.0 + repeat
                    tokens = batch * 1024
                    weight = 2.0 / batch
                    kv = context / 10000.0
                    outcome = (
                        0.01
                        + 0.1 * weight
                        + 0.2 * kv
                        + 100.0 * duration / tokens
                    )
                    rows.append(
                        {
                            "order": len(rows),
                            "run_id": f"r-{context}-{batch}-{repeat}",
                            "cell_id": f"c-{context}-{batch}",
                            "target_batch": batch,
                            "target_mean_attended_history_tokens": context,
                            "gross_gpu_joules_per_token": outcome,
                            "decode_seconds": duration,
                            "metered_useful_tokens": tokens,
                            "weight_gb_per_token": weight,
                            "kv_rw_gb_per_token": kv,
                        }
                    )
        result = evaluate_replication_rows(
            rows,
            coefficients,
            [-0.02, 0.01],
            gates,
            expected_cells=6,
            expected_repeats=5,
        )
        self.assertTrue(result["gates"]["qwen2p5_14b_form_replication_pass"])
        self.assertFalse(
            result["prediction_contract"]["coefficient_refit_performed"]
        )
        self.assertFalse(
            result["gates"]["residual_band_coverage_is_a_primary_gate"]
        )
        self.assertAlmostEqual(result["summary"]["maximum_absolute_relative_error"], 0.0)

    def test_replication_gate_name_can_be_bound_by_release(self):
        coefficients = {
            "intercept_joules_per_token": 0.01,
            "alpha_weight_joules_per_decimal_gb": 0.1,
            "beta_kv_joules_per_decimal_gb": 0.2,
            "p_time_watts": 100.0,
        }
        gates = {
            "median_absolute_relative_error_maximum": 0.05,
            "maximum_absolute_relative_error_maximum": 0.1,
            "required_cells_passing_ten_percent_error": 1,
        }
        row = {
            "order": 0,
            "run_id": "m7-r1",
            "cell_id": "m7-c1",
            "target_batch": 1,
            "target_mean_attended_history_tokens": 1,
            "gross_gpu_joules_per_token": 0.41,
            "decode_seconds": 1.0,
            "metered_useful_tokens": 1000,
            "weight_gb_per_token": 1.0,
            "kv_rw_gb_per_token": 1.0,
        }
        rows = [
            {**row, "order": order, "run_id": f"m7-r{order + 1}"}
            for order in range(2)
        ]
        result = evaluate_replication_rows(
            rows,
            coefficients,
            [-0.02, 0.01],
            gates,
            expected_cells=1,
            expected_repeats=2,
            form_replication_gate_name="mistral7b_form_replication_pass",
        )
        self.assertTrue(result["gates"]["mistral7b_form_replication_pass"])
        self.assertNotIn("qwen2p5_14b_form_replication_pass", result["gates"])


if __name__ == "__main__":
    unittest.main()
