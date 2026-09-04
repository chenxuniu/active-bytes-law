from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_bytes.campaign import expand_campaign, write_lock  # noqa: E402


EXPECTED = {
    "pilot.json": (4, 12),
    "qwen-core.json": (30, 150),
    "weight-treatment.json": (5, 25),
    "window-placebo.json": (9, 45),
    "llama-holdout.json": (12, 60),
    "dynamic.json": (9, 45),
    "ncu-anchors.json": (16, 48),
    "gh200-memory-placebo.json": (1, 3),
    "gh200-primary-bf16.json": (9, 45),
    "gh200-primary-bf16-evaluation.json": (6, 30),
    "gh200-v2-duration-holdout.json": (9, 45),
    "gh200-v1-anchors.json": (12, 60),
    "gh200-qwen2p5-14b-qualification.json": (1, 1),
    "gh200-qwen2p5-14b-qualification-v2.json": (1, 1),
    "gh200-qwen2p5-14b-identification.json": (9, 45),
    "gh200-qwen2p5-14b-holdout.json": (6, 30),
    "gh200-mistral7b-qualification-v1.json": (1, 1),
    "gh200-mistral7b-identification.json": (9, 45),
    "gh200-mistral7b-holdout.json": (6, 30),
}


class CampaignTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "configs" / "campaigns" / name).read_text())

    def test_all_campaign_counts_and_run_ids(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                lock = expand_campaign(self.load(name))
                self.assertEqual((lock["cell_count"], lock["run_count"]), expected)
                run_ids = [run["run_id"] for run in lock["run_order"]]
                self.assertEqual(len(run_ids), len(set(run_ids)))

    def test_expansion_is_byte_deterministic(self):
        config = self.load("qwen-core.json")
        one = json.dumps(expand_campaign(config), sort_keys=True, separators=(",", ":"))
        two = json.dumps(expand_campaign(config), sort_keys=True, separators=(",", ":"))
        self.assertEqual(one, two)

    def test_gh200_primary_keeps_evaluation_in_a_separate_lock(self):
        identification = expand_campaign(self.load("gh200-primary-bf16.json"))
        evaluation = expand_campaign(
            self.load("gh200-primary-bf16-evaluation.json")
        )
        self.assertNotIn(
            "evaluation", {run["split"] for run in identification["run_order"]}
        )
        self.assertEqual(
            {run["split"] for run in evaluation["run_order"]}, {"evaluation"}
        )

    def test_gh200_v1_uses_amended_application_range_replay(self):
        lock = expand_campaign(self.load("gh200-v1-anchors.json"))
        modes = {
            run["parameters"]["profiler_replay_mode"]
            for run in lock["run_order"]
        }
        self.assertEqual(modes, {"app-range"})

    def test_qwen14b_qualification_locks_match_sources_exactly(self):
        pairs = (
            (
                "gh200-qwen2p5-14b-qualification.json",
                "gh200-qwen2p5-14b-qualification.lock.json",
            ),
            (
                "gh200-qwen2p5-14b-qualification-v2.json",
                "gh200-qwen2p5-14b-qualification-v2.lock.json",
            ),
        )
        for source_name, lock_name in pairs:
            with self.subTest(source=source_name):
                source = self.load(source_name)
                observed = json.loads(
                    (ROOT / "results" / "manifests" / lock_name).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(observed, expand_campaign(source))

    def test_qwen14b_replication_locks_match_sources_and_holdout_is_sealed(self):
        pairs = (
            (
                "gh200-qwen2p5-14b-identification.json",
                "gh200-qwen2p5-14b-identification.lock.json",
            ),
            (
                "gh200-qwen2p5-14b-holdout.json",
                "gh200-qwen2p5-14b-holdout.lock.json",
            ),
        )
        expanded = {}
        for source_name, lock_name in pairs:
            with self.subTest(source=source_name):
                source = self.load(source_name)
                expected = expand_campaign(source)
                observed = json.loads(
                    (ROOT / "results" / "manifests" / lock_name).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(observed, expected)
                expanded[source_name] = expected

        identification_coordinates = {
            (
                run["parameters"]["target_mean_attended_history_tokens"],
                run["parameters"]["target_batch"],
            )
            for run in expanded[
                "gh200-qwen2p5-14b-identification.json"
            ]["run_order"]
        }
        holdout = expanded["gh200-qwen2p5-14b-holdout.json"]
        holdout_coordinates = {
            (
                run["parameters"]["target_mean_attended_history_tokens"],
                run["parameters"]["target_batch"],
            )
            for run in holdout["run_order"]
        }
        self.assertTrue(identification_coordinates.isdisjoint(holdout_coordinates))
        self.assertEqual(
            {run["parameters"]["execution_state"] for run in holdout["run_order"]},
            {"sealed-unreleased"},
        )
        self.assertEqual(
            {
                run["parameters"]["requires_frozen_identification_release"]
                for run in holdout["run_order"]
            },
            {True},
        )

    def test_mistral7b_qualification_lock_matches_source_exactly(self):
        source = self.load("gh200-mistral7b-qualification-v1.json")
        observed = json.loads(
            (
                ROOT
                / "results"
                / "manifests"
                / "gh200-mistral7b-qualification-v1.lock.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(observed, expand_campaign(source))

    def test_mistral7b_replication_locks_are_exact_and_holdout_is_sealed(self):
        expanded = {}
        for source_name, lock_name in (
            (
                "gh200-mistral7b-identification.json",
                "gh200-mistral7b-identification.lock.json",
            ),
            (
                "gh200-mistral7b-holdout.json",
                "gh200-mistral7b-holdout.lock.json",
            ),
        ):
            with self.subTest(source=source_name):
                expected = expand_campaign(self.load(source_name))
                observed = json.loads(
                    (ROOT / "results" / "manifests" / lock_name).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(observed, expected)
                expanded[source_name] = expected

        identification_coordinates = {
            (
                run["parameters"]["target_mean_attended_history_tokens"],
                run["parameters"]["target_batch"],
            )
            for run in expanded["gh200-mistral7b-identification.json"]["run_order"]
        }
        holdout = expanded["gh200-mistral7b-holdout.json"]
        holdout_coordinates = {
            (
                run["parameters"]["target_mean_attended_history_tokens"],
                run["parameters"]["target_batch"],
            )
            for run in holdout["run_order"]
        }
        self.assertTrue(identification_coordinates.isdisjoint(holdout_coordinates))
        self.assertEqual(
            {run["parameters"]["execution_state"] for run in holdout["run_order"]},
            {"sealed-unreleased"},
        )
        self.assertEqual(
            {
                run["parameters"]["requires_frozen_identification_release"]
                for run in holdout["run_order"]
            },
            {True},
        )

    def test_pilot_declared_order_is_rotated_across_repeats(self):
        lock = expand_campaign(self.load("pilot.json"))
        cells = [run["cell_id"] for run in lock["run_order"]]
        self.assertEqual(
            cells[:4],
            [
                "p-a-l4096-b8-bf16",
                "p-b-l4096-b8-fp8",
                "p-c-l16384-b32-bf16",
                "p-d-l16384-b32-fp8",
            ],
        )
        self.assertEqual(
            cells[4:8],
            [
                "p-b-l4096-b8-fp8",
                "p-c-l16384-b32-bf16",
                "p-d-l16384-b32-fp8",
                "p-a-l4096-b8-bf16",
            ],
        )

    def test_condition_cannot_leak_across_splits(self):
        config = self.load("pilot.json")
        copied = deepcopy(config["blocks"][0]["cells"][0])
        copied["cell_id"] = "leaked-condition"
        config["blocks"].append(
            {"block_id": "leak", "split": "evaluation", "cells": [copied]}
        )
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            expand_campaign(config)

    def test_frozen_lock_refuses_overwrite(self):
        lock = expand_campaign(self.load("pilot.json"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.lock.json"
            write_lock(path, lock)
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_lock(path, lock)
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
