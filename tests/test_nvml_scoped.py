from pathlib import Path
import json
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.nvml_scoped import summarize_scoped_samples  # noqa: E402
from active_bytes.nvml_scoped import collect_scoped_samples  # noqa: E402


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
    def test_stop_file_ends_collection_after_two_samples(self):
        class Value:
            def __init__(self, value):
                self.uiVal = value

        class Field:
            def __init__(self, field_id, scope_id, value):
                self.fieldId = field_id
                self.scopeId = scope_id
                self.nvmlReturn = 0
                self.value = Value(value)

        fake_nvml = mock.Mock()
        fake_nvml.nvmlDeviceGetCount.return_value = 1
        fake_nvml.nvmlDeviceGetHandleByIndex.return_value = object()
        fake_nvml.nvmlDeviceGetFieldValues.return_value = [
            Field(185, 0, 100000),
            Field(185, 1, 150000),
            Field(186, 0, 100000),
            Field(186, 1, 150000),
        ]
        fake_nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 1015]
        with TemporaryDirectory() as directory:
            stop = Path(directory) / "stop"
            ready = Path(directory) / "ready.json"
            stop.touch()
            timeline = iter([0.0, 0.0, 0.0, 0.01, 0.01, 0.02])

            with mock.patch("active_bytes.nvml_scoped.importlib.import_module", return_value=fake_nvml), mock.patch(
                "active_bytes.nvml_scoped.time.monotonic", side_effect=timeline
            ), mock.patch("active_bytes.nvml_scoped.time.monotonic_ns", side_effect=range(100, 1000)), mock.patch(
                "active_bytes.nvml_scoped.time.time_ns", side_effect=range(1000, 2000)
            ), mock.patch("active_bytes.nvml_scoped.time.sleep"):
                rows = collect_scoped_samples(
                    duration_seconds=1.0,
                    interval_seconds=0.01,
                    stop_file=stop,
                    ready_file=ready,
                )
            ready_state = json.loads(ready.read_text())["state"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(ready_state, "TELEMETRY_READY")

    def test_module_counter_is_compared_only_with_module_scope(self):
        report = summarize_scoped_samples(
            constant_samples(gpu_watts=100.0, module_watts=150.0),
            maximum_gap_seconds=1.1,
        )
        device = report["devices"]["0"]
        self.assertTrue(report["qc_pass"])
        self.assertAlmostEqual(device["module_counter_relative_error"], 0.0)
        self.assertAlmostEqual(device["cross_scope_difference"], 1 / 3)
        self.assertEqual(device["energy_counter_changed_samples"], 10)
        self.assertAlmostEqual(
            device["energy_counter_median_update_interval_seconds"], 1.0
        )
        self.assertAlmostEqual(device["energy_counter_median_update_joules"], 150.0)

    def test_wrong_module_counter_fails_qc(self):
        rows = constant_samples(gpu_watts=100.0, module_watts=150.0)
        for second, row in enumerate(rows):
            row["total_energy_counter_mj"] = second * 100.0 * 1000
        report = summarize_scoped_samples(rows, maximum_gap_seconds=1.1)
        self.assertFalse(report["qc_pass"])
        self.assertFalse(report["devices"]["0"]["module_counter_qc_pass"])

    def test_lagging_average_is_diagnostic_not_counter_gate(self):
        rows = constant_samples(gpu_watts=100.0, module_watts=150.0)
        for row in rows:
            row["module_average_power_w"] = 120.0
        report = summarize_scoped_samples(rows, maximum_gap_seconds=1.1)
        device = report["devices"]["0"]
        self.assertTrue(report["qc_pass"])
        self.assertAlmostEqual(device["module_counter_relative_error"], 0.0)
        self.assertGreater(device["module_average_counter_relative_error"], 0.1)
        self.assertEqual(
            device["module_counter_comparison_power_field"],
            "module_instant_power_w",
        )

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
