import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.artifacts import validate_manifest  # noqa: E402
from active_bytes.campaign import expand_campaign  # noqa: E402


class ArtifactTests(unittest.TestCase):
    def test_lock_membership_and_artifact_hash(self):
        config = json.loads((ROOT / "configs" / "campaigns" / "pilot.json").read_text())
        lock = expand_campaign(config)
        run = lock["run_order"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "results" / "manifests" / "pilot.lock.json"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(json.dumps(lock))
            artifact_path = root / "results" / "summaries" / "result.json"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text('{"qc_pass": true}\n')
            payload = artifact_path.read_bytes()
            manifest = {
                "run_id": run["run_id"],
                "cell_id": run["cell_id"],
                "split": run["split"],
                "repeat": run["repeat"],
                "campaign_lock_sha256": lock["lock_sha256"],
                "campaign_lock_uri": "results/manifests/pilot.lock.json",
                "artifacts": [
                    {
                        "role": "energy-summary",
                        "artifact_uri": "results/summaries/result.json",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                ],
            }
            report = validate_manifest(manifest, root)
            self.assertTrue(report["qc_pass"], report["errors"])

    def test_absolute_path_is_rejected(self):
        manifest = {
            "run_id": "run",
            "cell_id": "cell",
            "split": "pilot",
            "repeat": 1,
            "campaign_lock_sha256": "0" * 64,
            "campaign_lock_uri": "/tmp/private-lock.json",
            "artifacts": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = validate_manifest(manifest, Path(temporary))
        self.assertFalse(report["qc_pass"])
        self.assertTrue(any("absolute" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
