from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.batch_doctor import validate_batch_observations  # noqa: E402


def valid_rows(batch=3, measured=2):
    request_ids = [f"batch-doctor-{index}" for index in range(batch)]
    rows = []
    for step in range(measured + 1):
        rows.append(
            {
                "monotonic_start_ns": step * 100 + 1,
                "monotonic_end_ns": step * 100 + 99,
                "cumulative_output_tokens_by_request": {
                    request_id: step + 1 for request_id in request_ids
                },
                "useful_tokens_by_request": {
                    request_id: 1 for request_id in request_ids
                },
                "finished_by_request": {
                    request_id: step == measured for request_id in request_ids
                },
            }
        )
    return rows


class BatchDoctorTests(unittest.TestCase):
    def test_synchronized_batch_passes(self):
        report = validate_batch_observations(
            valid_rows(), batch=3, measured_decode_tokens=2
        )
        self.assertTrue(report["qc_pass"])

    def test_missing_bootstrap_member_fails(self):
        rows = valid_rows()
        del rows[0]["cumulative_output_tokens_by_request"]["batch-doctor-2"]
        report = validate_batch_observations(
            rows, batch=3, measured_decode_tokens=2
        )
        self.assertFalse(report["qc_pass"])

    def test_request_advancing_twice_fails(self):
        rows = valid_rows()
        rows[1]["cumulative_output_tokens_by_request"]["batch-doctor-1"] = 3
        report = validate_batch_observations(
            rows, batch=3, measured_decode_tokens=2
        )
        self.assertFalse(report["qc_pass"])

    def test_early_finish_fails(self):
        rows = valid_rows()
        rows[1]["finished_by_request"]["batch-doctor-0"] = True
        report = validate_batch_observations(
            rows, batch=3, measured_decode_tokens=2
        )
        self.assertFalse(report["qc_pass"])


if __name__ == "__main__":
    unittest.main()
