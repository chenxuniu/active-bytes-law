from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_publication_safety import scan_text  # noqa: E402


class SafetyTests(unittest.TestCase):
    def test_sensitive_patterns_are_detected(self):
        private_address = "10." + "23.45.67"
        email = "researcher" + "@" + "example.invalid"
        mac = "AA:" + "BB:CC:DD:EE:FF"
        gpu_uuid = "GPU-" + "12345678-1234-1234-1234-123456789abc"
        text = "\n".join((private_address, email, mac, gpu_uuid))
        labels = {label for label, _ in scan_text(text)}
        self.assertIn("non-public IPv4 address", labels)
        self.assertIn("email address", labels)
        self.assertIn("MAC address", labels)
        self.assertIn("GPU UUID", labels)

    def test_sanitized_profile_is_clean(self):
        text = '{"gpu": "NVIDIA H100 80GB HBM3", "os": "Ubuntu 22.04"}'
        self.assertEqual(scan_text(text), [])

    def test_allowlisted_public_package_version_is_clean(self):
        self.assertEqual(scan_text("llmcompressor==0.6.0.1"), [])


if __name__ == "__main__":
    unittest.main()
