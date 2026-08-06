import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_all_json_files_parse(self):
        for directory in (ROOT / "configs", ROOT / "schemas", ROOT / "environment"):
            for path in directory.rglob("*.json"):
                with self.subTest(path=path):
                    json.loads(path.read_text())

    def test_preflight_shell_syntax(self):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / "collect_preflight.sh")], check=True
        )


if __name__ == "__main__":
    unittest.main()
