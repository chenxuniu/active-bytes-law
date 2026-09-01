from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.kv_calibration import (  # noqa: E402
    KV_OBSERVER,
    calibration_sample_digest,
    kv_scale_report,
    validate_revision,
)


class KVCalibrationTests(unittest.TestCase):
    def test_locked_compressor_uses_registered_minmax_observer_name(self):
        self.assertEqual(KV_OBSERVER, "minmax")

    def test_revision_requires_full_lowercase_commit(self):
        revision = "a" * 40
        self.assertEqual(validate_revision(revision, label="test"), revision)
        for invalid in ("main", "A" * 40, "a" * 39, "g" * 40):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_revision(invalid, label="test")

    def test_sample_digest_is_order_sensitive_and_stable(self):
        self.assertEqual(
            calibration_sample_digest(["a", "b"]),
            calibration_sample_digest(["a", "b"]),
        )
        self.assertNotEqual(
            calibration_sample_digest(["a", "b"]),
            calibration_sample_digest(["b", "a"]),
        )

    def test_scale_report_accepts_complete_nonunity_pairs(self):
        class Attention:
            k_scale = 0.25
            v_scale = 0.5

        class Model:
            def named_parameters(self):
                return []

            def named_buffers(self):
                return []

            def named_modules(self):
                return [("model.layers.0.self_attn", Attention())]

        report = kv_scale_report(Model())
        self.assertEqual(report["complete_layer_count"], 1)
        self.assertTrue(report["finite_positive"])
        self.assertFalse(report["all_unity"])

    def test_scale_report_rejects_zero_and_incomplete_pairs(self):
        class ZeroAttention:
            k_scale = 0.0
            v_scale = 0.5

        class MissingAttention:
            k_scale = 0.5

        class Model:
            def named_parameters(self):
                return []

            def named_buffers(self):
                return []

            def named_modules(self):
                return [("zero", ZeroAttention()), ("missing", MissingAttention())]

        report = kv_scale_report(Model())
        self.assertEqual(report["discovered_parent_count"], 2)
        self.assertEqual(report["complete_layer_count"], 1)
        self.assertFalse(report["finite_positive"])


if __name__ == "__main__":
    unittest.main()
