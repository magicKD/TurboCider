"""Native offline discovery and help; no service or model needed."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CLI = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")) / "turbocider"


@unittest.skipUnless(CLI.is_file(), "build the native CLI before this test")
class CLIDiscoveryTests(unittest.TestCase):
    def test_help_and_capabilities_need_no_service_or_registry(self):
        with tempfile.TemporaryDirectory(prefix="tc-cli-discovery-") as temporary:
            root = Path(temporary)
            env = dict(os.environ, TURBOCIDER_MODEL_LIBRARY=str(root / "missing-library"),
                       TURBOCIDER_LIBRARY_SETTINGS=str(root / "missing-settings.json"))

            def run(*args):
                return subprocess.run([str(CLI.resolve()), *args], cwd=root, env=env,
                                      capture_output=True, text=True, timeout=10)

            for flag in ("--help", "-h", "help"):
                with self.subTest(flag=flag):
                    result = run(flag)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(result.stderr)
                    for entry in ("capabilities", "installations", "plan", "submit", "cancel"):
                        self.assertIn(entry, result.stdout)
            result = run("capabilities")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(result.stderr)
            contract = json.loads(result.stdout)
            self.assertEqual(contract["protocol"], "turbocider.local")
            self.assertEqual(contract["native_request_validation"], "plan")
            self.assertEqual(contract["workflow"]["submission_retry"], "never_automatic")
            actions = {item["name"]: item for item in contract["actions"]}
            self.assertEqual(actions["submit"]["input_schema"]["required"],
                             ["action", "request", "model_path"])
            self.assertTrue(actions["submit"]["mutates"])
            self.assertFalse(actions["models"]["mutates"])
            for args in ((), ("unknown",), ("--help", "unexpected"), ("capabilities", "unexpected")):
                with self.subTest(invalid=args):
                    result = run(*args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(result.stdout)
                    self.assertIn("--help", result.stderr)
            self.assertEqual(list(root.iterdir()), [], "Read-only discovery created local state")


if __name__ == "__main__":
    unittest.main(verbosity=2)
