import unittest

from active_bytes.duration_v2_evaluation import evaluate_v2_rows


class DurationV2EvaluationTests(unittest.TestCase):
    def test_exact_frozen_predictions_pass(self):
        artifact = {
            "frozen_model": {
                "equation": "test",
                "coefficients": {
                    "intercept_joules_per_token": 0.01,
                    "alpha_weight_joules_per_decimal_gb": 0.1,
                    "beta_kv_joules_per_decimal_gb": 0.2,
                    "p_time_watts": 100.0,
                },
            },
            "new_holdout": {"cell_count": 9, "repetitions_per_cell": 5},
            "primary_gates": {
                "median_absolute_relative_error_maximum": 0.05,
                "maximum_absolute_relative_error_maximum": 0.1,
                "required_cells_passing_ten_percent_error": 9,
            },
        }
        rows = []
        for context in (6144, 10240, 14336):
            for batch in (12, 20, 28):
                for repeat in range(1, 6):
                    duration = 30.0 + repeat
                    tokens = 10000
                    weight = 1.0 / batch
                    kv = context / 10000.0
                    outcome = 0.01 + 0.1 * weight + 0.2 * kv + 100 * duration / tokens
                    rows.append(
                        {
                            "order": len(rows),
                            "run_id": f"r-{context}-{batch}-{repeat}",
                            "cell_id": f"c-{context}-{batch}",
                            "repeat": repeat,
                            "target_batch": batch,
                            "target_mean_attended_history_tokens": context,
                            "gross_gpu_joules_per_token": outcome,
                            "decode_seconds": duration,
                            "metered_useful_tokens": tokens,
                            "weight_gb_per_token": weight,
                            "kv_rw_gb_per_token": kv,
                        }
                    )
        result = evaluate_v2_rows(rows, artifact)
        self.assertTrue(result["gates"]["duration_v2_holdout_pass"])
        self.assertAlmostEqual(result["summary"]["maximum_absolute_relative_error"], 0.0)

    def test_gate_name_can_identify_same_sku_transfer(self):
        artifact = {
            "frozen_model": {
                "equation": "test",
                "coefficients": {
                    "intercept_joules_per_token": 1.0,
                    "alpha_weight_joules_per_decimal_gb": 0.0,
                    "beta_kv_joules_per_decimal_gb": 0.0,
                    "p_time_watts": 0.0,
                },
            },
            "new_holdout": {"cell_count": 1, "repetitions_per_cell": 2},
            "primary_gates": {
                "median_absolute_relative_error_maximum": 0.05,
                "maximum_absolute_relative_error_maximum": 0.1,
                "required_cells_passing_ten_percent_error": 1,
            },
        }
        rows = [
            {
                "order": repeat,
                "run_id": f"r-{repeat}",
                "cell_id": "target-device-cell",
                "repeat": repeat + 1,
                "target_batch": 1,
                "target_mean_attended_history_tokens": 1,
                "gross_gpu_joules_per_token": 1.0,
                "decode_seconds": 1.0,
                "metered_useful_tokens": 1,
                "weight_gb_per_token": 0.0,
                "kv_rw_gb_per_token": 0.0,
            }
            for repeat in range(2)
        ]
        result = evaluate_v2_rows(
            rows, artifact, scientific_gate_name="same_sku_device_transfer_pass"
        )
        self.assertTrue(result["gates"]["same_sku_device_transfer_pass"])
        self.assertNotIn("duration_v2_holdout_pass", result["gates"])


if __name__ == "__main__":
    unittest.main()
