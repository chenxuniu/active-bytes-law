import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_DRIVER = ROOT / "scripts" / "run_gh200_v1_anchor_batch.sh"


class V1BatchDriverTests(unittest.TestCase):
    def test_run_failure_is_recorded_continued_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            results = root / "results"
            (repo / "results" / "manifests").mkdir(parents=True)
            (repo / "scripts").mkdir()
            lock = {
                "lock_sha256": "test-lock",
                "run_order": [
                    {"order": order, "run_id": f"run-{order}"}
                    for order in range(3)
                ],
            }
            (repo / "results" / "manifests" / "gh200-v1-anchors.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            runner = repo / "scripts" / "run_gh200_v1_anchor_attempt.sh"
            runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    order=$1
                    results_root=${TEL_RESULTS_ROOT:?}
                    run_id="run-${order}"
                    run_root="$results_root/v1-profiler-anchors/$run_id"
                    attempt="$run_root/attempt-$(date +%s%N)"
                    mkdir -p "$attempt"
                    if [[ "$order" == "0" ]] && ! compgen -G "$run_root/batch-failure-*.json" >/dev/null; then
                      echo intentional-test-failure >"$attempt/runner.log"
                      exit 7
                    fi
                    python3 - "$attempt/traffic.json" "$run_id" <<'PY'
                    import json
                    import pathlib
                    import sys
                    pathlib.Path(sys.argv[1]).write_text(json.dumps({
                        "qc_pass": True,
                        "run": {"run_id": sys.argv[2]},
                        "campaign_lock_sha256": "test-lock",
                    }))
                    PY
                    """
                ),
                encoding="utf-8",
            )
            runner.chmod(0o755)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            environment = dict(os.environ)
            environment.update(
                TEL_REPO_ROOT=str(repo),
                TEL_RESULTS_ROOT=str(results),
            )

            first = subprocess.run(
                [str(BATCH_DRIVER), "0", "2", "0"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(first.returncode, 2, first.stdout)
            self.assertIn("status=recorded-failure-continuing", first.stdout)
            self.assertTrue(
                list((results / "v1-profiler-anchors" / "run-0").glob("batch-failure-*.json"))
            )
            for run_id in ("run-1", "run-2"):
                self.assertTrue(
                    list((results / "v1-profiler-anchors" / run_id).glob("attempt-*/traffic.json"))
                )
            first_summaries = list(results.glob("v1-batch-runs/**/batch.summary.json"))
            self.assertEqual(len(first_summaries), 1)
            first_summary = json.loads(first_summaries[0].read_text())
            self.assertEqual(first_summary["status"], "complete-with-recorded-failures")
            self.assertEqual(first_summary["failed_orders"], [0])

            second = subprocess.run(
                [str(BATCH_DRIVER), "0", "2", "0"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("run_id=run-1 status=already-accepted", second.stdout)
            self.assertIn("batch_status=complete", second.stdout)
            self.assertTrue(
                list((results / "v1-profiler-anchors" / "run-0").glob("attempt-*/traffic.json"))
            )


if __name__ == "__main__":
    unittest.main()
