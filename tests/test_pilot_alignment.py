from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.pilot_alignment import align_pilot_repeat  # noqa: E402


def telemetry():
    rows = []
    for second in range(0, 41):
        rows.append(
            {
                "gpu_index": 0,
                "monotonic_ns": second * 1_000_000_000,
                "gpu_instant_power_w": 100.0,
                "module_instant_power_w": 150.0,
            }
        )
    return rows


def repeat():
    episodes = []
    for episode_id in range(5):
        start = episode_id * 8 * 1_000_000_000
        end = start + 6 * 1_000_000_000
        episodes.append(
            {
                "episode_id": episode_id,
                "boundary": {"go_monotonic_ns": start, "done_monotonic_ns": end},
                "decode_seconds": 6.0,
                "metered_useful_tokens": 100,
                "module_counter_joules": 900.0,
            }
        )
    return {
        "qc_pass": True,
        "campaign_lock_sha256": "a" * 64,
        "run": {"run_id": "run"},
        "episodes": episodes,
        "active_bytes": {},
        "model_geometry": {},
        "weights": {},
        "runtime": {},
    }


class PilotAlignmentTests(unittest.TestCase):
    def test_constant_power_repeat_uses_ratio_of_totals(self):
        report = align_pilot_repeat(
            telemetry(), repeat(), host_gpu_index=1, maximum_gap_seconds=1.1
        )
        self.assertTrue(report["qc_pass"], report["qc_reasons"])
        self.assertEqual(report["totals"]["gpu_joules_per_token"], 6.0)
        self.assertEqual(report["totals"]["module_counter_joules_per_token"], 9.0)
        self.assertEqual(report["totals"]["decode_seconds"], 30.0)
        self.assertEqual(report["gpu_index"], 0)
        self.assertEqual(report["host_gpu_index"], 1)

    def test_module_counter_disagreement_fails(self):
        value = repeat()
        value["episodes"][0]["module_counter_joules"] = 100.0
        report = align_pilot_repeat(
            telemetry(), value, maximum_gap_seconds=1.1
        )
        self.assertFalse(report["qc_pass"])


if __name__ == "__main__":
    unittest.main()
