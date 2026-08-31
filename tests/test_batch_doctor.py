from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.batch_doctor import (  # noqa: E402
    balanced_prompt_lengths,
    validate_batch_observations,
)


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
    def test_balanced_prompts_hit_exact_4k_mean(self):
        prompts = balanced_prompt_lengths(
            target_mean_attended_history_tokens=4096,
            batch=8,
            measured_decode_tokens=1024,
        )
        self.assertEqual(prompts, [3584] * 4 + [3585] * 4)
        self.assertEqual(sum(prompts) / len(prompts) + 511.5, 4096)

    def test_odd_batch_rejects_half_token_mean_prompt(self):
        with self.assertRaisesRegex(ValueError, "cannot realize"):
            balanced_prompt_lengths(
                target_mean_attended_history_tokens=4096,
                batch=7,
                measured_decode_tokens=1024,
            )

    def test_balanced_prompts_hit_exact_16k_mean(self):
        prompts = balanced_prompt_lengths(
            target_mean_attended_history_tokens=16384,
            batch=32,
            measured_decode_tokens=1024,
        )
        self.assertEqual(prompts, [15872] * 16 + [15873] * 16)
        self.assertEqual(sum(prompts) / len(prompts) + 511.5, 16384)

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
