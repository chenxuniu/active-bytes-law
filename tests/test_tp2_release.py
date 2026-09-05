import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.tp2_release import verify_tp2_release  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TP2ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        freeze = root / "freeze"
        freeze.mkdir()
        gates = {
            "unit_of_analysis": "cell mean over five independent runs",
            "median_absolute_relative_error_maximum": 0.05,
            "maximum_absolute_relative_error_maximum": 0.1,
            "required_cells_passing_ten_percent_error": 6,
            "coefficient_refit_before_gate_decision": False,
            "failed_or_censored_cells_are_not_deleted": True,
        }
        equation = "E_pair = c + alpha * weight + beta * kv + p_time * time"
        addendum = root / "addendum.json"
        addendum.write_text(
            json.dumps(
                {
                    "analysis_policy": {"equation": equation},
                    "holdout_gates": gates,
                }
            ),
            encoding="utf-8",
        )
        addendum_sha = digest(addendum)
        coefficients = {
            "intercept_joules_per_token": 0.02,
            "alpha_weight_joules_per_decimal_gb": 0.2,
            "beta_kv_joules_per_decimal_gb": 0.3,
            "p_time_watts": 200.0,
        }
        coefficient = freeze / "coefficient-artifact.json"
        coefficient.write_text(
            json.dumps(
                {
                    "outcome_contract": {
                        "estimand": "sum_two_gpu_boards_joules_per_useful_token"
                    },
                    "fit": {"coefficients": coefficients},
                }
            ),
            encoding="utf-8",
        )
        residual_range = [-0.03, 0.02]
        envelope = freeze / "discrepancy-envelope.json"
        envelope.write_text(
            json.dumps(
                {
                    "envelope": {
                        "common_residual_range_joules_per_token": residual_range
                    }
                }
            ),
            encoding="utf-8",
        )
        accepted = freeze / "accepted-runs.csv"
        accepted.write_text("run_id\nr1\n", encoding="utf-8")
        summary = freeze / "identification-freeze-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "qc_pass": True,
                    "holdout_release_candidate": True,
                    "campaign_lock_sha256": "i" * 64,
                    "accepted_run_count": 45,
                    "scientific_decisions": {
                        "identification_gate_pass": True,
                        "tp2_functional_form_confirmed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        identification_lock = root / "identification.lock.json"
        identification_lock.write_text(
            json.dumps(
                {
                    "lock_sha256": "i" * 64,
                    "run_count": 45,
                    "run_order": [
                        {
                            "parameters": {
                                "execution_addendum_sha256": addendum_sha
                            }
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        holdout_lock = root / "holdout.lock.json"
        holdout_lock.write_text(
            json.dumps(
                {
                    "lock_sha256": "h" * 64,
                    "run_count": 30,
                    "cell_count": 6,
                    "run_order": [
                        {
                            "run_id": "holdout-r1",
                            "parameters": {
                                "execution_state": "sealed-unreleased",
                                "requires_frozen_identification_release": True,
                                "tensor_parallel_size": 2,
                                "host_gpu_indices": [0, 1],
                                "execution_addendum_sha256": addendum_sha,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        release = root / "release.json"
        release.write_text(
            json.dumps(
                {
                    "release_id": "tp2-test-release",
                    "execution_addendum_sha256": addendum_sha,
                    "identification_campaign": {
                        "campaign_lock_sha256": "i" * 64,
                        "campaign_lock_file_sha256": digest(identification_lock),
                    },
                    "holdout_campaign": {
                        "campaign_lock_sha256": "h" * 64,
                        "campaign_lock_file_sha256": digest(holdout_lock),
                        "run_count": 30,
                        "cell_count": 6,
                        "repetitions_per_cell": 5,
                    },
                    "frozen_artifacts": {
                        name: digest(freeze / name)
                        for name in (
                            "coefficient-artifact.json",
                            "discrepancy-envelope.json",
                            "accepted-runs.csv",
                            "identification-freeze-summary.json",
                        )
                    },
                    "frozen_estimates": {
                        "equation": equation,
                        "coefficients": coefficients,
                        "common_residual_range_joules_per_token": residual_range,
                    },
                    "analysis_equation": equation,
                    "primary_gates": gates,
                    "release_policy": {
                        "released": True,
                        "coefficients_are_immutable": True,
                        "residual_envelope_is_immutable": True,
                        "holdout_may_not_refit": True,
                        "duration_uses_same_interval_observation": True,
                        "both_device_energies_must_be_summed": True,
                        "failed_attempts_are_preserved": True,
                        "nvlink_energy_separately_attributed": False,
                        "tp1_coefficients_are_tp2_predictions": False,
                        "dvfs_claim_authorized": False,
                        "cross_sku_claim_authorized": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return release, freeze, identification_lock, holdout_lock, addendum

    def test_exact_content_addressed_release_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixture(Path(directory))
            report = verify_tp2_release(
                *values,
                expected_release_sha256=digest(values[0]),
            )
            self.assertTrue(report["qc_pass"], report["issues"])

    def test_release_cannot_relax_preregistered_error_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixture(Path(directory))
            release = values[0]
            value = json.loads(release.read_text(encoding="utf-8"))
            value["primary_gates"]["maximum_absolute_relative_error_maximum"] = 0.2
            release.write_text(json.dumps(value), encoding="utf-8")
            report = verify_tp2_release(
                *values,
                expected_release_sha256=digest(release),
            )
            self.assertFalse(report["qc_pass"])
            self.assertTrue(
                any("primary gates" in issue for issue in report["issues"])
            )

    def test_changed_frozen_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixture(Path(directory))
            expected_release_sha = digest(values[0])
            (values[1] / "accepted-runs.csv").write_text(
                "changed\n", encoding="utf-8"
            )
            report = verify_tp2_release(
                *values,
                expected_release_sha256=expected_release_sha,
            )
            self.assertFalse(report["qc_pass"])
            self.assertTrue(
                any("accepted-runs.csv" in issue for issue in report["issues"])
            )


if __name__ == "__main__":
    unittest.main()
