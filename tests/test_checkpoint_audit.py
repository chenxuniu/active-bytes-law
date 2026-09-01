import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from active_bytes.checkpoint_audit import validate_serialized_checkpoint  # noqa: E402
from active_bytes.full_calibration import contract_sha256  # noqa: E402


class CheckpointAuditTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        contract = {
            "recipe": {
                "kv_cache": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                    "observer": "minmax",
                }
            },
            "expected_model_invariants": {"attention_layers": 2},
        }
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "quantization_status": "frozen",
                "config_groups": {"group": {"weights": None}},
                "kv_cache_scheme": contract["recipe"]["kv_cache"],
            }
        }
        index = {
            "weight_map": {
                "model.layers.0.self_attn.k_scale": "model.safetensors",
                "model.layers.0.self_attn.v_scale": "model.safetensors",
                "model.layers.1.self_attn.k_scale": "model.safetensors",
                "model.layers.1.self_attn.v_scale": "model.safetensors",
            }
        }
        files = {
            "config.json": json.dumps(config),
            "model.safetensors.index.json": json.dumps(index),
            "calibration-contract.json": json.dumps(contract),
            "recipe.yaml": "recipe\n",
        }
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")
        return {
            "contract": {"canonical_sha256": contract_sha256(contract)},
            "checkpoint": {
                "files": [
                    {"path": name, "bytes": (root / name).stat().st_size}
                    for name in files
                ]
            },
        }

    def test_valid_kv_only_serialization_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = validate_serialized_checkpoint(root, self._fixture(root))
            self.assertTrue(report["qc_pass"])
            self.assertEqual(report["k_scale_tensor_count"], 2)
            self.assertEqual(report["v_scale_tensor_count"], 2)

    def test_weight_quantization_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_report = self._fixture(root)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["quantization_config"]["config_groups"]["group"]["weights"] = {
                "num_bits": 8
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            for row in calibration_report["checkpoint"]["files"]:
                if row["path"] == "config.json":
                    row["bytes"] = config_path.stat().st_size
            report = validate_serialized_checkpoint(root, calibration_report)
            self.assertFalse(report["qc_pass"])
            self.assertEqual(report["non_kv_quantization"], ["group.weights"])


if __name__ == "__main__":
    unittest.main()
