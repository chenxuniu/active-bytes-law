import json
import hashlib
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_all_json_files_parse(self):
        for directory in (
            ROOT / "configs",
            ROOT / "schemas",
            ROOT / "environment",
            ROOT / "calibration" / "contracts",
        ):
            for path in directory.rglob("*.json"):
                with self.subTest(path=path):
                    json.loads(path.read_text())

    def test_preflight_shell_syntax(self):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / "collect_preflight.sh")], check=True
        )

    def test_calibration_requirements_are_fully_pinned(self):
        requirements = (
            ROOT / "calibration" / "requirements-gh200.txt"
        ).read_text(encoding="utf-8").splitlines()
        active = [line for line in requirements if line and not line.startswith("#")]
        self.assertTrue(active)
        self.assertTrue(all(line.count("==") == 1 for line in active))
        names = [line.split("==", 1)[0] for line in active]
        self.assertEqual(len(names), len(set(names)))

    def test_full_calibration_contract_matches_sidecar(self):
        contract = (
            ROOT
            / "calibration"
            / "contracts"
            / "qwen2p5-7b-fp8-kv-v1.json"
        )
        sidecar = contract.with_suffix(contract.suffix + ".sha256")
        expected, filename = sidecar.read_text(encoding="utf-8").split()
        self.assertEqual(filename, contract.name)
        self.assertEqual(hashlib.sha256(contract.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
