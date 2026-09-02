import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.evaluation_release import verify_evaluation_release  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationReleaseTests(unittest.TestCase):
    def fixture(self, root: Path):
        freeze = root / "freeze"
        freeze.mkdir()
        coefficient = {
            "campaign_lock_sha256": "identification-lock",
        }
        coefficient_path = freeze / "coefficient-artifact.json"
        coefficient_path.write_text(json.dumps(coefficient))
        coefficient_sha = digest(coefficient_path)
        envelope = {"coefficient_artifact_sha256": coefficient_sha}
        envelope_path = freeze / "discrepancy-envelope.json"
        envelope_path.write_text(json.dumps(envelope))
        (freeze / "accepted-runs.csv").write_text("run_id\nrun-1\n")
        summary = {
            "qc_pass": True,
            "evaluation_release_candidate": True,
            "accepted_run_count": 45,
            "fit_run_count": 30,
            "calibration_run_count": 15,
            "outcome": "gross_gpu_board_joules_per_useful_token",
            "campaign_lock_sha256": "identification-lock",
            "coefficient_artifact": {"sha256": coefficient_sha},
            "discrepancy_envelope": {"sha256": digest(envelope_path)},
            "scientific_decisions": {
                "p2_slope_positivity_pass": True,
                "p3_coefficient_equivalence_pass": False,
            },
        }
        summary_path = freeze / "identification-freeze-summary.json"
        summary_path.write_text(json.dumps(summary))
        lock_path = root / "evaluation-lock.json"
        lock_path.write_text(
            json.dumps({"lock_sha256": "evaluation-lock", "run_count": 30})
        )
        release = {
            "release_id": "release-1",
            "identification_campaign": {
                "campaign_lock_sha256": "identification-lock",
                "accepted_run_count": 45,
                "fit_run_count": 30,
                "calibration_run_count": 15,
                "outcome": "gross_gpu_board_joules_per_useful_token",
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
            "identification_decisions": {
                "two_slope_familywise_positivity_pass": True,
                "single_coefficient_equivalence_pass": False,
            },
            "evaluation_campaign": {
                "campaign_lock_sha256": "evaluation-lock",
                "campaign_lock_file_sha256": digest(lock_path),
                "run_count": 30,
            },
            "release_policy": {
                "released": True,
                "release_requires_exact_artifact_hashes": True,
                "evaluation_may_not_refit_coefficients": True,
                "evaluation_may_not_recalibrate_the_envelope": True,
            },
        }
        release_path = root / "release.json"
        release_path.write_text(json.dumps(release))
        return release_path, freeze, lock_path

    def test_exact_release_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            release, freeze, lock = self.fixture(Path(directory))
            report = verify_evaluation_release(release, freeze, lock)
            self.assertTrue(report["qc_pass"], report["issues"])
            self.assertTrue(report["coefficients_are_immutable"])

    def test_artifact_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            release, freeze, lock = self.fixture(Path(directory))
            with (freeze / "accepted-runs.csv").open("a") as handle:
                handle.write("run-2\n")
            report = verify_evaluation_release(release, freeze, lock)
            self.assertFalse(report["qc_pass"])
            self.assertTrue(
                any("digest mismatch" in reason for reason in report["issues"])
            )


if __name__ == "__main__":
    unittest.main()
