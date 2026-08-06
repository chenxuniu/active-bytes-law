from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.accounting import (  # noqa: E402
    active_bytes,
    active_weight_bytes,
    kv_bytes_per_token,
    summarize_trace,
)


class AccountingTests(unittest.TestCase):
    def test_known_kv_constants(self):
        self.assertEqual(kv_bytes_per_token(28, 4, 128, 2), 57_344)
        self.assertEqual(kv_bytes_per_token(28, 4, 128, 1), 28_672)
        self.assertEqual(kv_bytes_per_token(32, 8, 128, 2), 131_072)

    def test_weight_storage_is_deduplicated(self):
        inventory = [
            {"name": "embedding", "physical_storage_id": "s0", "storage_nbytes": 100},
            {"name": "lm_head", "physical_storage_id": "s0", "storage_nbytes": 100},
            {"name": "block", "physical_storage_id": "s1", "storage_nbytes": 300},
        ]
        self.assertEqual(active_weight_bytes(inventory), 400)

    def test_conflicting_storage_size_is_rejected(self):
        with self.assertRaises(ValueError):
            active_weight_bytes(
                [
                    {"physical_storage_id": "s0", "storage_nbytes": 100},
                    {"physical_storage_id": "s0", "storage_nbytes": 101},
                ]
            )

    def test_active_bytes_terms(self):
        result = active_bytes(16_000, 100, 8, 20)
        self.assertEqual(result.weight_bytes_per_token, 2_000)
        self.assertEqual(result.kv_read_bytes_per_token, 2_000)
        self.assertEqual(result.active_bytes_read, 4_000)
        self.assertEqual(result.active_bytes_read_write, 4_100)
        self.assertEqual(result.weight_kv_parity_ratio, 1)

    def test_trace_context_is_useful_token_weighted(self):
        rows = [
            {
                "useful_tokens_by_request": {"short": 1, "long": 1},
                "metered_useful_output_tokens": 2,
                "attended_length_by_request": {"short": 100, "long": 900},
            },
            {
                "useful_tokens_by_request": {"long": 1},
                "metered_useful_output_tokens": 1,
                "attended_length_by_request": {"long": 901},
            },
        ]
        result = summarize_trace(rows)
        self.assertEqual(result["effective_batch"], 1.5)
        self.assertAlmostEqual(result["mean_attended_context"], (100 + 900 + 901) / 3)


if __name__ == "__main__":
    unittest.main()
