from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.calibration_compare import compare_calibration_doctors  # noqa: E402


def passing_report() -> dict:
    return {
        "qc_pass": True,
        "model": {"id": "model", "revision": "a" * 40},
        "dataset": {
            "id": "dataset",
            "revision": "b" * 40,
            "split": "train",
            "seed": 2027,
            "num_calibration_samples": 8,
            "max_sequence_length": 256,
            "rendered_text_sha256": "c" * 64,
        },
        "recipe": {"weights_quantized": False},
        "runtime": {"packages": {"compressor": "one"}, "cuda": "thirteen"},
        "baseline_parameters_before": {"probe_sha256": "d" * 64},
        "kv_scales": {
            "layers": [
                {"name": "layer.0", "k_scale": 0.25, "v_scale": 0.125},
                {"name": "layer.1", "k_scale": 0.5, "v_scale": 0.0625},
            ]
        },
    }


class CalibrationCompareTests(unittest.TestCase):
    def test_identical_doctors_pass(self):
        first = passing_report()
        report = compare_calibration_doctors(first, deepcopy(first))
        self.assertTrue(report["qc_pass"])
        self.assertEqual(report["compared_scale_count"], 4)
        self.assertEqual(report["maximum_relative_difference"], 0.0)

    def test_scale_drift_fails(self):
        first = passing_report()
        second = deepcopy(first)
        second["kv_scales"]["layers"][0]["k_scale"] = 0.3
        report = compare_calibration_doctors(first, second)
        self.assertFalse(report["qc_pass"])
        self.assertEqual(len(report["out_of_tolerance"]), 1)

    def test_contract_drift_fails(self):
        first = passing_report()
        second = deepcopy(first)
        second["dataset"]["rendered_text_sha256"] = "e" * 64
        report = compare_calibration_doctors(first, second)
        self.assertFalse(report["qc_pass"])
        self.assertEqual(
            report["mismatched_contract_fields"],
            ["dataset.rendered_text_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
