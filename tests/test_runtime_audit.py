from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.runtime_audit import validate_cache_dtype_contract  # noqa: E402


class RuntimeAuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
