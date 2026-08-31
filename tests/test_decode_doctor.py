from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.decode_doctor import (  # noqa: E402
    _wait_for_external_start_gate,
    validate_doctor_observations,
)


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

    def test_external_gate_writes_ready_then_waits_for_release(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready.json"
            gate = Path(directory) / "go"

            def release():
                time.sleep(0.05)
                gate.touch()

            thread = threading.Thread(target=release)
            thread.start()
            report = _wait_for_external_start_gate(
                ready_file=ready,
                start_gate_file=gate,
                timeout_seconds=1.0,
            )
            thread.join()
            self.assertTrue(ready.exists())
            self.assertIsNotNone(report)
            self.assertGreaterEqual(report["wait_seconds"], 0.04)


if __name__ == "__main__":
    unittest.main()
