from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.decode_doctor import validate_doctor_observations  # noqa: E402


def valid_observations():
    return [
        {
            "step_id": index,
            "request_id": "doctor",
            "monotonic_start_ns": index * 100 + 1,
            "monotonic_end_ns": index * 100 + 99,
            "cumulative_output_tokens": index + 1,
            "finished": index == 8,
        }
        for index in range(9)
    ]


class DecodeDoctorTests(unittest.TestCase):
    def test_exact_bootstrap_and_decode_growth_passes(self):
        report = validate_doctor_observations(
            valid_observations(), bootstrap_tokens=1, measured_decode_tokens=8
        )
        self.assertTrue(report["qc_pass"])
        self.assertEqual(report["metered_useful_tokens"], 8)

    def test_extra_token_in_one_step_fails(self):
        rows = valid_observations()
        rows[5]["cumulative_output_tokens"] += 1
        report = validate_doctor_observations(
            rows, bootstrap_tokens=1, measured_decode_tokens=8
        )
        self.assertFalse(report["qc_pass"])
        self.assertIn("cumulative output counts", report["qc_reasons"][0])

    def test_early_finish_fails(self):
        rows = valid_observations()
        rows[0]["finished"] = True
        report = validate_doctor_observations(
            rows, bootstrap_tokens=1, measured_decode_tokens=8
        )
        self.assertFalse(report["qc_pass"])

    def test_missing_final_step_fails(self):
        report = validate_doctor_observations(
            valid_observations()[:-1], bootstrap_tokens=1, measured_decode_tokens=8
        )
        self.assertFalse(report["qc_pass"])


if __name__ == "__main__":
    unittest.main()
