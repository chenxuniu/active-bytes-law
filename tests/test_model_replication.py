import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.model_replication import (  # noqa: E402
    duration_calibration_envelope,
    fit_duration_ols_hc3,
)


def synthetic_rows(*, time_coefficient: float = 140.0):
    rows = []
    order = 0
    weight_bytes = 29_548_455_936
    kv_bytes = 196_608
    for context in (4096, 10240, 16384):
        for batch in (4, 16):
            for repeat in range(1, 6):
                useful_tokens = batch * 1024
                weight_gb = weight_bytes / batch / 1e9
                kv_gb = kv_bytes * (context + 1) / 1e9
                seconds_per_token = (
                    0.0006
                    + context * 2.0e-8
                    + 0.002 / batch
                    + context * 1.0e-7 / batch
                    + repeat * 1.0e-6
                )
                outcome = (
                    0.025
                    + 0.12 * weight_gb
                    + 0.21 * kv_gb
                    + time_coefficient * seconds_per_token
                )
                rows.append(
                    {
                        "order": order,
                        "run_id": f"fit-{order:02d}",
                        "cell_id": f"fit-l{context}-b{batch}",
                        "weight_gb_per_token": weight_gb,
                        "kv_rw_gb_per_token": kv_gb,
                        "decode_seconds": seconds_per_token * useful_tokens,
                        "metered_useful_tokens": useful_tokens,
                        "gross_gpu_joules_per_token": outcome,
                    }
                )
                order += 1
    return rows


class ModelReplicationTests(unittest.TestCase):
    def test_duration_fit_recovers_prespecified_coefficients(self):
        fit = fit_duration_ols_hc3(synthetic_rows())
        coefficients = fit["coefficients"]
        self.assertAlmostEqual(coefficients["intercept_joules_per_token"], 0.025, places=7)
        self.assertAlmostEqual(
            coefficients["alpha_weight_joules_per_decimal_gb"], 0.12, places=7
        )
        self.assertAlmostEqual(
            coefficients["beta_kv_joules_per_decimal_gb"], 0.21, places=7
        )
        self.assertAlmostEqual(coefficients["p_time_watts"], 140.0, places=5)
        self.assertTrue(fit["traffic_slope_positivity"]["qc_pass"])
        self.assertTrue(fit["time_term"]["finite_nonnegative_qc_pass"])
        self.assertFalse(
            fit["time_term"]["causal_idle_power_interpretation_authorized"]
        )

    def test_negative_time_term_fails_its_prespecified_gate(self):
        fit = fit_duration_ols_hc3(synthetic_rows(time_coefficient=-20.0))
        self.assertLess(fit["coefficients"]["p_time_watts"], 0.0)
        self.assertFalse(fit["time_term"]["finite_nonnegative_qc_pass"])

    def test_calibration_envelope_has_one_interval_per_cell(self):
        fit = fit_duration_ols_hc3(synthetic_rows())
        rows = []
        coefficients = fit["coefficients"]
        for context in (4096, 10240, 16384):
            for repeat, residual in enumerate((-0.002, -0.001, 0.0, 0.001, 0.002), 1):
                batch = 8
                useful_tokens = batch * 1024
                weight_gb = 29_548_455_936 / batch / 1e9
                kv_gb = 196_608 * (context + 1) / 1e9
                seconds_per_token = 0.001 + context * 3.0e-8 + repeat * 1.0e-6
                prediction = (
                    coefficients["intercept_joules_per_token"]
                    + coefficients["alpha_weight_joules_per_decimal_gb"] * weight_gb
                    + coefficients["beta_kv_joules_per_decimal_gb"] * kv_gb
                    + coefficients["p_time_watts"] * seconds_per_token
                )
                rows.append(
                    {
                        "run_id": f"cal-l{context}-r{repeat}",
                        "cell_id": f"cal-l{context}-b8",
                        "weight_gb_per_token": weight_gb,
                        "kv_rw_gb_per_token": kv_gb,
                        "decode_seconds": seconds_per_token * useful_tokens,
                        "metered_useful_tokens": useful_tokens,
                        "gross_gpu_joules_per_token": prediction + residual,
                    }
                )
        envelope = duration_calibration_envelope(rows, fit)
        self.assertEqual(envelope["cell_count"], 3)
        self.assertEqual(len(envelope["cells"]), 3)
        self.assertTrue(
            all(
                math.isfinite(bound)
                for bound in envelope["common_residual_range_joules_per_token"]
            )
        )


if __name__ == "__main__":
    unittest.main()
