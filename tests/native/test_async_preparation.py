"""Host-only ownership/cancellation test; no MLX, Core ML or model load."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AsyncPreparationTests(unittest.TestCase):
    def test_request_ownership_and_errors(self):
        with tempfile.TemporaryDirectory(prefix="tc-async-preparation-") as directory:
            binary = Path(directory) / "test"
            command = [os.environ.get("CXX", "clang++"), "-std=c++20", "-O2", "-pthread",
                       "-Wall", "-Wextra", "-Werror", str(ROOT / "tests/native/async_preparation_test.cpp"),
                       "-o", str(binary)]
            build = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("PASS async preparation", run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
