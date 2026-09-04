from pathlib import Path
from collections import deque
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.pilot_repeat import (  # noqa: E402
    _model_geometry,
    validate_episode_observations,
    validate_scheduler_delta,
)


def valid_observations(request_ids, measured=2):
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


class PilotRepeatTests(unittest.TestCase):
    def test_model_geometry_derives_head_dim_when_config_value_is_none(self):
        class Config:
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            head_dim = None

        class ModelConfig:
            hf_config = Config()

        class Engine:
            model_config = ModelConfig()

        report = _model_geometry(Engine(), "bf16")
        self.assertEqual(report["head_dim"], 128)
        self.assertEqual(report["kv_bytes_per_historical_token"], 131072)

    def test_scheduler_snapshot_accepts_deque_swap_queue(self):
        class Scheduler:
            num_cumulative_preemption = 0
            swapped = deque()

        class Engine:
            scheduler = [Scheduler()]

        from active_bytes.pilot_repeat import _scheduler_snapshot

        report = _scheduler_snapshot(Engine())
        self.assertTrue(report["observable"])
        self.assertEqual(report["schedulers"][0]["swapped_request_count"], 0)

    def test_valid_synchronized_episode_passes(self):
        request_ids = ["r0", "r1"]
        report = validate_episode_observations(
            valid_observations(request_ids),
            request_ids=request_ids,
            measured_decode_tokens=2,
        )
        self.assertTrue(report["qc_pass"])
        self.assertEqual(report["metered_useful_tokens"], 4)

    def test_membership_change_fails(self):
        request_ids = ["r0", "r1"]
        rows = valid_observations(request_ids)
        rows[1]["cumulative_output_tokens_by_request"].pop("r1")
        report = validate_episode_observations(
            rows, request_ids=request_ids, measured_decode_tokens=2
        )
        self.assertFalse(report["qc_pass"])

    def test_scheduler_delta_requires_observable_zero_change(self):
        before = {
            "observable": True,
            "schedulers": [
                {"cumulative_preemptions": 0, "swapped_request_count": 0}
            ],
        }
        after = {
            "observable": True,
            "schedulers": [
                {"cumulative_preemptions": 0, "swapped_request_count": 0}
            ],
        }
        self.assertTrue(validate_scheduler_delta(before, after)["qc_pass"])
        after["schedulers"][0]["cumulative_preemptions"] = 1
        self.assertFalse(validate_scheduler_delta(before, after)["qc_pass"])


if __name__ == "__main__":
    unittest.main()
