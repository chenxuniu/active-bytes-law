import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.model_replication_release import (  # noqa: E402
    verify_model_replication_release,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelReplicationReleaseTests(unittest.TestCase):
    def fixture(self, root):
        freeze = root / "freeze"
        freeze.mkdir()
        addendum = root / "addendum.json"
        parent_gates = {
            "median_absolute_relative_error_maximum": 0.05,
            "maximum_absolute_relative_error_maximum": 0.1,
            "required_cells_passing_ten_percent_error": 1,
            "coefficient_refit_before_gate_decision": False,
            "failed_or_censored_cells_are_not_deleted": True,
        }
        equation = "E = c + alpha * weight + beta * kv + p_time * time"
        addendum.write_text(
            json.dumps(
                {
                    "holdout_gates": parent_gates,
                    "analysis_policy": {"equation": equation},
                }
            )
        )
        coefficients = {
            "intercept_joules_per_token": 0.01,
            "alpha_weight_joules_per_decimal_gb": 0.1,
            "beta_kv_joules_per_decimal_gb": 0.2,
            "p_time_watts": 100.0,
        }
        coefficient = freeze / "coefficient-artifact.json"
        coefficient.write_text(
            json.dumps(
                {
                    "campaign_lock_sha256": "i" * 64,
                    "fit": {"coefficients": coefficients},
                }
            )
        )
        envelope = freeze / "discrepancy-envelope.json"
        accepted = freeze / "accepted-runs.csv"
        accepted.write_text("run_id\nr1\n")
        envelope.write_text(
            json.dumps(
                {
                    "coefficient_artifact_sha256": digest(coefficient),
                    "envelope": {
                        "common_residual_range_joules_per_token": [-0.02, 0.01]
                    },
                }
            )
        )
        summary = freeze / "identification-freeze-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "qc_pass": True,
                    "holdout_release_candidate": True,
                    "accepted_run_count": 45,
                    "fit_run_count": 30,
                    "calibration_run_count": 15,
                    "campaign_lock_sha256": "i" * 64,
                    "coefficient_artifact": {"sha256": digest(coefficient)},
                    "discrepancy_envelope": {"sha256": digest(envelope)},
                    "accepted_run_table": {"sha256": digest(accepted)},
                    "scientific_decisions": {
                        "identification_gate_pass": True,
                        "traffic_slope_positivity_pass": True,
                        "time_term_finite_nonnegative_pass": True,
                        "cross_model_form_confirmed": False,
                    },
                }
            )
        )
        lock = root / "holdout.lock.json"
        lock_value = {
            "lock_sha256": "h" * 64,
            "run_count": 1,
            "cell_count": 1,
            "run_order": [
                {
                    "run_id": "holdout-r1",
                    "parameters": {
                        "execution_state": "sealed-unreleased",
                        "requires_frozen_identification_release": True,
                        "execution_addendum_sha256": digest(addendum),
                        "target_mean_attended_history_tokens": 6144,
                        "target_batch": 6,
                    },
                }
            ],
        }
        lock.write_text(json.dumps(lock_value))
        release = root / "release.json"
        release_value = {
            "form_replication_addendum_sha256": digest(addendum),
            "identification_campaign": {
                "campaign_lock_sha256": "i" * 64,
                "accepted_run_count": 45,
                "fit_run_count": 30,
                "calibration_run_count": 15,
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
                "common_residual_range_joules_per_token": [-0.02, 0.01],
            },
            "identification_decisions": {
                "identification_gate_pass": True,
                "traffic_slope_familywise_positivity_pass": True,
                "time_term_finite_nonnegative_pass": True,
                "cross_model_form_confirmed": False,
            },
            "holdout_campaign": {
                "campaign_lock_sha256": "h" * 64,
                "campaign_lock_file_sha256": digest(lock),
                "run_count": 1,
                "cell_count": 1,
                "target_batches": [6],
                "target_mean_attended_history_tokens": [6144],
            },
            "primary_gates": parent_gates,
            "release_policy": {
                "released": True,
                "release_requires_exact_artifact_hashes": True,
                "holdout_may_not_refit_coefficients": True,
                "holdout_may_not_recalibrate_the_envelope": True,
                "duration_uses_observed_same_interval_decode_time": True,
                "accepted_attempts_are_never_overwritten": True,
                "cross_model_claim_requires_primary_holdout_gate_pass": True,
                "cross_hardware_claim_authorized": False,
                "dvfs_claim_authorized": False,
            },
        }
        release.write_text(json.dumps(release_value))
        return release, freeze, lock, addendum

    def test_exact_release_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            release, freeze, lock, addendum = self.fixture(Path(directory))
            with patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_FILENAME",
                release.name,
            ), patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_SHA256",
                digest(release),
            ):
                report = verify_model_replication_release(
                    release, freeze, lock, addendum
                )
            self.assertTrue(report["qc_pass"])

    def test_artifact_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            release, freeze, lock, addendum = self.fixture(Path(directory))
            expected_release_sha = digest(release)
            (freeze / "accepted-runs.csv").write_text("changed")
            with patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_FILENAME",
                release.name,
            ), patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_SHA256",
                expected_release_sha,
            ):
                report = verify_model_replication_release(
                    release, freeze, lock, addendum
                )
            self.assertFalse(report["qc_pass"])
            self.assertTrue(any("accepted-runs.csv" in issue for issue in report["issues"]))

    def test_release_cannot_relax_parent_error_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            release, freeze, lock, addendum = self.fixture(Path(directory))
            value = json.loads(release.read_text())
            value["primary_gates"]["maximum_absolute_relative_error_maximum"] = 0.2
            release.write_text(json.dumps(value))
            with patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_FILENAME",
                release.name,
            ), patch(
                "active_bytes.model_replication_release.OFFICIAL_RELEASE_SHA256",
                digest(release),
            ):
                report = verify_model_replication_release(
                    release, freeze, lock, addendum
                )
            self.assertFalse(report["qc_pass"])
            self.assertTrue(
                any("maximum_absolute_relative_error_maximum" in issue for issue in report["issues"])
            )


if __name__ == "__main__":
    unittest.main()
