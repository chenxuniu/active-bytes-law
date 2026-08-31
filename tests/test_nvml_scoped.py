from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.nvml_scoped import summarize_scoped_samples  # noqa: E402


def constant_samples(*, gpu_watts: float, module_watts: float):
    rows = []
    for second in range(11):
        rows.append(
            {
                "gpu_index": 0,
                "monotonic_ns": second * 1_000_000_000,
                "gpu_average_power_w": gpu_watts,
                "gpu_instant_power_w": gpu_watts,
                "module_average_power_w": module_watts,
                "module_instant_power_w": module_watts,
                "total_energy_counter_mj": second * module_watts * 1000,
            }
        )
    return rows


class NvmlScopedTests(unittest.TestCase):
    def test_module_counter_is_compared_only_with_module_scope(self):
        report = summarize_scoped_samples(
            constant_samples(gpu_watts=100.0, module_watts=150.0),
            maximum_gap_seconds=1.1,
        )
        device = report["devices"]["0"]
        self.assertTrue(report["qc_pass"])
        self.assertAlmostEqual(device["module_counter_relative_error"], 0.0)
        self.assertAlmostEqual(device["cross_scope_difference"], 1 / 3)

    def test_wrong_module_counter_fails_qc(self):
        rows = constant_samples(gpu_watts=100.0, module_watts=150.0)
        for second, row in enumerate(rows):
            row["total_energy_counter_mj"] = second * 100.0 * 1000
        report = summarize_scoped_samples(rows, maximum_gap_seconds=1.1)
        self.assertFalse(report["qc_pass"])
        self.assertFalse(report["devices"]["0"]["module_counter_qc_pass"])

    def test_sampling_gap_fails_qc(self):
        report = summarize_scoped_samples(
            constant_samples(gpu_watts=100.0, module_watts=150.0)
        )
        self.assertFalse(report["qc_pass"])
        self.assertFalse(report["devices"]["0"]["sampling_gap_qc_pass"])

    def test_decreasing_counter_is_rejected(self):
        rows = constant_samples(gpu_watts=100.0, module_watts=150.0)
        rows[-1]["total_energy_counter_mj"] = -1
        with self.assertRaisesRegex(ValueError, "invalid total_energy_counter_mj"):
            summarize_scoped_samples(rows, maximum_gap_seconds=1.1)


if __name__ == "__main__":
    unittest.main()
