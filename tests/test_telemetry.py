from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.telemetry import integrate_power  # noqa: E402


class TelemetryTests(unittest.TestCase):
    def test_constant_power_with_boundary_interpolation(self):
        samples = [
            {"monotonic_ns": index * 100_000_000, "power_w": 300.0}
            for index in range(12)
        ]
        report = integrate_power(
            samples,
            start_ns=50_000_000,
            end_ns=1_050_000_000,
            maximum_gap_seconds=0.25,
        )
        self.assertAlmostEqual(report["integrated_power_joules"], 300.0)
        self.assertTrue(report["gap_qc_pass"])

    def test_missing_boundary_bracket_is_rejected(self):
        samples = [
            {"monotonic_ns": 100, "power_w": 200.0},
            {"monotonic_ns": 200, "power_w": 200.0},
        ]
        with self.assertRaisesRegex(ValueError, "bracket"):
            integrate_power(samples, start_ns=50, end_ns=150)

    def test_large_gap_is_reported(self):
        samples = [
            {"monotonic_ns": 0, "power_w": 200.0},
            {"monotonic_ns": 1_000_000_000, "power_w": 200.0},
        ]
        report = integrate_power(samples, start_ns=0, end_ns=1_000_000_000)
        self.assertFalse(report["gap_qc_pass"])


if __name__ == "__main__":
    unittest.main()
