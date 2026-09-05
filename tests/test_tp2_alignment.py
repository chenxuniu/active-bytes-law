from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.tp2_alignment import align_tp2_repeat  # noqa: E402


def telemetry():
    rows = []
    for second in range(0, 41):
        for gpu_index, gpu_power, module_power in (
            (0, 100.0, 150.0),
            (1, 120.0, 170.0),
        ):
            rows.append(
                {
                    "gpu_index": gpu_index,
                    "monotonic_ns": second * 1_000_000_000,
                    "gpu_instant_power_w": gpu_power,
                    "module_instant_power_w": module_power,
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
                "boundary": {
                    "go_monotonic_ns": start,
                    "done_monotonic_ns": end,
                },
                "metered_useful_tokens": 100,
                "module_counter_joules_by_visible_gpu": {
                    "0": 900.0,
                    "1": 1020.0,
                },
            }
        )
    return {
        "qc_pass": True,
        "paper_candidate_measurement": True,
        "campaign_lock_sha256": "a" * 64,
        "run": {"run_id": "tp2-run"},
        "episodes": episodes,
        "active_bytes": {},
        "model_geometry": {},
        "weights": {
            "unique_storage_bytes": 7_619_810_816,
            "accounting_total_unique_storage_bytes": 15_239_621_632,
        },
        "runtime": {
            "tensor_parallel_size": 2,
            "visible_gpu_count": 2,
        },
    }


class TP2AlignmentTests(unittest.TestCase):
    def test_two_devices_are_integrated_and_summed_over_common_boundaries(self):
        report = align_tp2_repeat(
            telemetry(), repeat(), maximum_gap_seconds=1.1
        )
        self.assertTrue(report["qc_pass"], report["qc_reasons"])
        self.assertEqual(report["visible_gpu_indices"], [0, 1])
        self.assertEqual(report["host_gpu_indices"], [0, 1])
        self.assertEqual(report["totals"]["decode_seconds"], 30.0)
        self.assertEqual(report["totals"]["metered_useful_tokens"], 500)
        self.assertEqual(report["totals"]["gpu_joules_per_token"], 13.2)
        self.assertEqual(
            report["totals"]["module_counter_joules_per_token"], 19.2
        )
        self.assertTrue(report["devices"]["0"]["module_counter_qc_pass"])
        self.assertTrue(report["devices"]["1"]["module_counter_qc_pass"])
        self.assertEqual(report["weights"]["unique_storage_bytes"], 15_239_621_632)
        self.assertEqual(
            report["weights"]["driver_worker_unique_storage_bytes"],
            7_619_810_816,
        )

    def test_missing_device_telemetry_is_rejected(self):
        rows = [row for row in telemetry() if row["gpu_index"] == 0]
        with self.assertRaisesRegex(ValueError, "device membership"):
            align_tp2_repeat(rows, repeat(), maximum_gap_seconds=1.1)

    def test_per_device_counter_disagreement_fails_even_if_sum_is_close(self):
        value = repeat()
        value["episodes"][0]["module_counter_joules_by_visible_gpu"] = {
            "0": 600.0,
            "1": 1320.0,
        }
        report = align_tp2_repeat(
            telemetry(), value, maximum_gap_seconds=1.1
        )
        self.assertFalse(report["qc_pass"])
        self.assertFalse(report["devices"]["0"]["module_counter_qc_pass"])
        self.assertFalse(report["devices"]["1"]["module_counter_qc_pass"])
        self.assertAlmostEqual(
            report["totals"]["module_instant_counter_relative_error"], 0.0
        )

    def test_short_paper_repeat_fails_but_qualification_can_be_short(self):
        value = repeat()
        value["episodes"] = value["episodes"][:1]
        paper = align_tp2_repeat(
            telemetry(), value, maximum_gap_seconds=1.1
        )
        self.assertFalse(paper["qc_pass"])
        value["paper_candidate_measurement"] = False
        qualification = align_tp2_repeat(
            telemetry(), value, maximum_gap_seconds=1.1
        )
        self.assertTrue(qualification["qc_pass"], qualification["qc_reasons"])


if __name__ == "__main__":
    unittest.main()
