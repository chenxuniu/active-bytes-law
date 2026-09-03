from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.model_qualification import evaluate_model_qualification  # noqa: E402


class ModelQualificationTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (
                ROOT
                / "configs"
                / "addenda"
                / "gh200-qwen2p5-14b-qualification-v2.json"
            ).read_text(encoding="utf-8")
        )
        self.lock = json.loads(
            (
                ROOT
                / "results"
                / "manifests"
                / "gh200-qwen2p5-14b-qualification-v2.lock.json"
            ).read_text(encoding="utf-8")
        )
        expected = self.contract["expected"]
        geometry = expected["geometry"]
        self.doctor = {
            "qc_pass": True,
            "non_paper_measurement": True,
            "runtime": {
                "model": expected["model"]["name"],
                "model_revision": expected["model"]["revision"],
            },
            "geometry": {
                "batch": geometry["target_batch"],
                "target_mean_attended_history_tokens": geometry[
                    "target_mean_attended_history_tokens"
                ],
                "mean_attended_history_tokens": geometry[
                    "target_mean_attended_history_tokens"
                ],
                "metered_decode_tokens_per_request": geometry[
                    "metered_decode_tokens_per_request"
                ],
                "decode_seconds": 10.0,
            },
        }
        self.runtime = {
            "qc_pass": True,
            "non_paper_measurement": True,
            "frozen_run_execution": True,
            "campaign_lock_sha256": self.lock["lock_sha256"],
            "runtime": {
                "model": expected["model"]["name"],
                "model_revision": expected["model"]["revision"],
                "attention_backend": "FLASH_ATTN",
                "weight_dtype": "bfloat16",
                "requested_kv_cache_dtype": "auto",
            },
            "cache_contract": {"qc_pass": True},
            "model_geometry": expected["model"]["config"],
            "weights": {"unique_storage_bytes": 30_000_000_000},
            "cache": {
                "tensor_count": 48,
                "gpu_tensor_count": 48,
                "logical_nbytes": 50_000_000_000,
            },
        }

    def evaluate(self, *, doctor=None, runtime=None):
        return evaluate_model_qualification(
            contract=self.contract,
            campaign_lock=self.lock,
            doctor=self.doctor if doctor is None else doctor,
            runtime_audit=self.runtime if runtime is None else runtime,
        )

    def test_complete_qualification_passes(self):
        report = self.evaluate()
        self.assertTrue(report["qc_pass"])
        self.assertTrue(report["qualified_for_campaign_design"])
        self.assertFalse(report["may_enter_paper_outcomes"])
        coordinates = report["observed"][
            "active_byte_coordinates_at_qualification_geometry"
        ]
        self.assertEqual(
            coordinates["kv_read_obligation_bytes_per_useful_token"],
            196608 * 16384,
        )
        self.assertEqual(
            coordinates["weight_read_obligation_bytes_per_useful_token"],
            30_000_000_000 / 16,
        )

    def test_wrong_model_revision_fails_closed(self):
        doctor = deepcopy(self.doctor)
        doctor["runtime"]["model_revision"] = "moving-main"
        report = self.evaluate(doctor=doctor)
        self.assertFalse(report["qc_pass"])
        self.assertIn("batch doctor loaded the wrong model revision", report["qc_reasons"])

    def test_short_decode_fails_campaign_qualification(self):
        doctor = deepcopy(self.doctor)
        doctor["geometry"]["decode_seconds"] = 1.0
        report = self.evaluate(doctor=doctor)
        self.assertFalse(report["qc_pass"])

    def test_wrong_logical_kv_geometry_fails(self):
        runtime = deepcopy(self.runtime)
        runtime["model_geometry"]["logical_kv_bytes_per_attended_token"] = 98304
        report = self.evaluate(runtime=runtime)
        self.assertFalse(report["qc_pass"])
