from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.validation import validate_static_trace  # noqa: E402


def trace_rows(prompt=4096, batch=8, steps=128):
    request_ids = [f"r{index}" for index in range(batch)]
    rows = []
    for iteration in range(steps):
        rows.append(
            {
                "schema_version": 1,
                "run_id": "ab1-pilot-test-r01",
                "episode_id": 0,
                "iteration_id": iteration,
                "monotonic_start_ns": 1_000_000 + iteration * 10_000,
                "monotonic_end_ns": 1_005_000 + iteration * 10_000,
                "active_request_ids": request_ids,
                "useful_tokens_by_request": {request_id: 1 for request_id in request_ids},
                "metered_useful_output_tokens": batch,
                "runtime_seq_len_raw_by_request": {
                    request_id: prompt + iteration + 1 for request_id in request_ids
                },
                "attended_length_by_request": {
                    request_id: prompt + iteration for request_id in request_ids
                },
                "live_kv_blocks": batch * 10,
                "allocated_kv_blocks": batch * 12,
                "accepted_tokens": batch,
                "rejected_tokens": 0,
                "speculative_draft_tokens": 0,
                "preemptions": 0,
                "swaps": 0,
                "recomputed_tokens": 0,
                "prefix_cache_hits": 0,
                "offloaded_bytes": 0,
                "scheduler_mode": "static-exact-decode",
                "attention_backend": "pinned-backend",
                "graph_mode": "pinned-graph-mode",
                "kv_cache_dtype": "bf16",
                "weight_dtype": "bf16",
            }
        )
    return rows


class TraceValidationTests(unittest.TestCase):
    def test_static_trace_passes_and_has_expected_geometry(self):
        report = validate_static_trace(
            trace_rows(), expected_batch=8, prompt_tokens=4096, measured_decode_tokens=128
        )
        self.assertTrue(report["qc_pass"], report["errors"])
        self.assertEqual(report["metered_useful_tokens"], 1024)
        self.assertEqual(report["effective_batch"], 8)
        self.assertEqual(report["mean_attended_context"], 4159.5)
        self.assertEqual(report["requested_api_output_tokens_per_request"], 129)

    def test_off_by_one_iterations_are_rejected(self):
        report = validate_static_trace(
            trace_rows(steps=127),
            expected_batch=8,
            prompt_tokens=4096,
            measured_decode_tokens=128,
        )
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("128 decode iterations" in error for error in report["errors"]))

    def test_preemption_is_rejected(self):
        rows = trace_rows()
        rows[10]["preemptions"] = 1
        report = validate_static_trace(rows, expected_batch=8, prompt_tokens=4096)
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("preemptions" in error for error in report["errors"]))

    def test_late_entry_is_rejected(self):
        rows = trace_rows()
        changed = deepcopy(rows[2])
        changed["active_request_ids"][-1] = "late"
        changed["useful_tokens_by_request"].pop("r7")
        changed["useful_tokens_by_request"]["late"] = 1
        changed["attended_length_by_request"].pop("r7")
        changed["attended_length_by_request"]["late"] = 4098
        rows[2] = changed
        report = validate_static_trace(rows, expected_batch=8, prompt_tokens=4096)
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("membership changed" in error for error in report["errors"]))

    def test_query_inclusive_length_is_rejected_until_normalized(self):
        rows = trace_rows()
        rows[0]["attended_length_by_request"] = {f"r{i}": 4097 for i in range(8)}
        report = validate_static_trace(rows, expected_batch=8, prompt_tokens=4096)
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("before KV write" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
