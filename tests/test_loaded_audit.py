from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.loaded_audit import align_loaded_audit  # noqa: E402


def telemetry_rows(module_watts=150.0):
    return [
        {
            "gpu_index": 0,
            "monotonic_ns": index * 100_000_000,
            "gpu_average_power_w": 100.0,
            "gpu_instant_power_w": 100.0,
            "module_average_power_w": module_watts,
            "module_instant_power_w": module_watts,
        }
        for index in range(101)
    ]


def doctor_report(counter_joules=1200.0):
    return {
        "boundary": {
            "go_monotonic_ns": 1_000_000_000,
            "decode_done_monotonic_ns": 9_000_000_000,
        },
        "energy": {"module_energy_joules": counter_joules},
        "token_boundary_qc_pass": True,
    }


class LoadedAuditTests(unittest.TestCase):
    def test_constant_power_alignment_passes(self):
        report = align_loaded_audit(
            telemetry_rows(), doctor_report(), maximum_gap_seconds=0.11
        )
        self.assertTrue(report["qc_pass"])
        self.assertAlmostEqual(
            report["integrals"]["gpu_instant_power_w"]["integrated_power_joules"],
            800.0,
        )

    def test_counter_disagreement_fails(self):
        report = align_loaded_audit(
            telemetry_rows(),
            doctor_report(counter_joules=800.0),
            maximum_gap_seconds=0.11,
        )
        self.assertFalse(report["qc_pass"])
        self.assertFalse(report["counter_agreement_qc_pass"])

    def test_failed_token_boundary_fails(self):
        doctor = doctor_report()
        doctor["token_boundary_qc_pass"] = False
        report = align_loaded_audit(
            telemetry_rows(), doctor, maximum_gap_seconds=0.11
        )
        self.assertFalse(report["qc_pass"])


if __name__ == "__main__":
    unittest.main()
