from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.energy import summarize_energy  # noqa: E402


def episode(repeat, episode_id, delta_mj, tokens, seconds=15.0, integrated=None):
    return {
        "schema_version": 1,
        "run_id": f"ab1-pilot-cell-r{repeat:02d}",
        "cell_id": "pilot-cell",
        "repeat": repeat,
        "episode_id": episode_id,
        "boundary": "decode-only",
        "sensor": "DCGM_FI_DEV_TOTAL_ENERGY",
        "counter_start_mj": 10_000,
        "counter_end_mj": 10_000 + delta_mj,
        "counter_read_start_monotonic_ns": 1_000,
        "go_monotonic_ns": 2_000,
        "decode_done_monotonic_ns": 3_000,
        "counter_read_end_monotonic_ns": 4_000,
        "metered_useful_tokens": tokens,
        "decode_seconds": seconds,
        "integrated_power_joules": integrated,
        "qc_pass": True,
    }


class EnergyTests(unittest.TestCase):
    def test_multi_episode_ratio_then_repeat_cv(self):
        records = []
        for repeat, scale in ((1, 0.99), (2, 1.0), (3, 1.01)):
            records.append(episode(repeat, 0, 1000 * scale, 100, integrated=1.0 * scale))
            records.append(episode(repeat, 1, 1000 * scale, 100, integrated=1.0 * scale))
        report = summarize_energy(records, minimum_repeats=3, minimum_decode_seconds=30)
        self.assertTrue(report["qc_pass"], report)
        cell = report["cells"]["pilot-cell"]
        self.assertEqual(cell["valid_repeats"], 3)
        self.assertLess(cell["coefficient_of_variation"], 0.03)

    def test_episode_joules_per_token_are_not_averaged(self):
        records = [
            episode(1, 0, 1000, 10, seconds=15, integrated=1.0),
            episode(1, 1, 9000, 30, seconds=15, integrated=9.0),
        ]
        report = summarize_energy(records, minimum_repeats=1, minimum_decode_seconds=30)
        repeat = report["cells"]["pilot-cell"]["repeats"][0]
        self.assertEqual(repeat["joules_per_useful_token"], 10 / 40)

    def test_counter_decrease_is_rejected(self):
        bad = episode(1, 0, -1, 100, seconds=30)
        report = summarize_energy([bad], minimum_repeats=1, minimum_decode_seconds=30)
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("counter delta" in error for error in report["parsing_errors"]))

    def test_boundary_order_is_rejected(self):
        bad = episode(1, 0, 1000, 100, seconds=30)
        bad["go_monotonic_ns"] = 500
        report = summarize_energy([bad], minimum_repeats=1, minimum_decode_seconds=30)
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("start-read" in error for error in report["parsing_errors"]))


if __name__ == "__main__":
    unittest.main()
