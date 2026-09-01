from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.full_calibration import (  # noqa: E402
    contract_sha256,
    validate_full_contract,
)


class FullCalibrationTests(unittest.TestCase):
    def test_contract_digest_is_key_order_independent(self):
        self.assertEqual(
            contract_sha256({"a": 1, "b": 2}),
            contract_sha256({"b": 2, "a": 1}),
        )

    def test_contract_requires_exact_image_packages_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            evidence.write_text('{"qc_pass": true}\n', encoding="utf-8")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            package_contract = {
                "llmcompressor": None,
                "compressed-tensors": None,
                "transformers": None,
                "datasets": None,
                "accelerate": None,
                "torch": None,
            }
            contract = {
                "status": "frozen",
                "runtime": {
                    "calibration_image_id": "image",
                    "packages": package_contract,
                },
                "qualification_evidence": {
                    label: {"sha256": digest}
                    for label in ("doctor_r01", "doctor_r02", "repeat_comparison")
                },
            }
            report = validate_full_contract(
                contract,
                calibration_image_id="image",
                evidence_paths={
                    label: evidence
                    for label in ("doctor_r01", "doctor_r02", "repeat_comparison")
                },
            )
            self.assertTrue(report["qc_pass"])


if __name__ == "__main__":
    unittest.main()
