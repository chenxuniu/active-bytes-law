from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.runtime_audit import (  # noqa: E402
    _known_cache_roots,
    resolve_runtime_contract,
    validate_cache_dtype_contract,
)


class RuntimeAuditTests(unittest.TestCase):
    def test_known_cache_roots_descend_into_cache_engines(self):
        class CacheEngine:
            gpu_cache = ["cache-tensor"]

        class ModelRunner:
            kv_caches = ["runner-cache"]

        class Worker:
            cache_engine = [CacheEngine()]
            gpu_cache = ["worker-cache"]
            model_runner = ModelRunner()

        roots = dict(_known_cache_roots(Worker()))
        self.assertEqual(roots["driver_worker.gpu_cache"], ["worker-cache"])
        self.assertEqual(roots["cache_engine[0].gpu_cache"], ["cache-tensor"])
        self.assertEqual(roots["model_runner.kv_caches"], ["runner-cache"])

    def test_bf16_cache_contract_passes(self):
        report = validate_cache_dtype_contract(
            ["torch.bfloat16"], declared_dtype="bf16"
        )
        self.assertTrue(report["qc_pass"])

    def test_bf16_cache_contract_rejects_fp8(self):
        report = validate_cache_dtype_contract(
            ["torch.float8_e4m3fn"], declared_dtype="bf16"
        )
        self.assertFalse(report["qc_pass"])

    def test_empty_cache_contract_fails(self):
        report = validate_cache_dtype_contract([], declared_dtype="bf16")
        self.assertFalse(report["qc_pass"])

    def test_frozen_runtime_contract_uses_locked_values(self):
        contract = resolve_runtime_contract(
            {"attention_backend": "FLASH_ATTN", "kv_cache_dtype": "bf16"}
        )
        self.assertFalse(contract["compatibility_mode"])
        self.assertEqual(contract["attention_backend"], "FLASH_ATTN")
        self.assertEqual(contract["requested_kv_cache_dtype"], "auto")
        self.assertFalse(contract["calculate_kv_scales"])

    def test_compatibility_runtime_contract_requires_matched_overrides(self):
        parameters = {
            "attention_backend": "FLASH_ATTN",
            "kv_cache_dtype": "fp8_e4m3",
        }
        with self.assertRaises(ValueError):
            resolve_runtime_contract(
                parameters, compatibility_attention_backend="FLASHINFER"
            )
        contract = resolve_runtime_contract(
            parameters,
            compatibility_attention_backend="FLASHINFER",
            compatibility_kv_cache_dtype="fp8_e4m3",
        )
        self.assertTrue(contract["compatibility_mode"])
        self.assertEqual(contract["attention_backend"], "FLASHINFER")
        self.assertEqual(contract["requested_kv_cache_dtype"], "fp8_e4m3")
        self.assertFalse(contract["calculate_kv_scales"])

    def test_scale_calculation_is_compatibility_only(self):
        parameters = {
            "attention_backend": "FLASH_ATTN",
            "kv_cache_dtype": "fp8_e4m3",
        }
        with self.assertRaises(ValueError):
            resolve_runtime_contract(
                parameters, compatibility_calculate_kv_scales=True
            )
        contract = resolve_runtime_contract(
            parameters,
            compatibility_attention_backend="FLASHINFER",
            compatibility_kv_cache_dtype="fp8_e4m3",
            compatibility_calculate_kv_scales=True,
        )
        self.assertTrue(contract["calculate_kv_scales"])


if __name__ == "__main__":
    unittest.main()
