"""CPU-only bounded memory recovery policy; no model, MLX or GPU work."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class StreamingMemoryRecoveryTests(unittest.TestCase):
    def test_recovery_observation_stability_and_deadline_contracts(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="tc-stream-memory-recovery-") as raw:
            binary = Path(raw) / "recovery"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/native/qwen21_streaming_memory_recovery_test.cpp"),
                "-o", str(binary)], check=True, cwd=ROOT, timeout=30)
            result = subprocess.run([str(binary)], capture_output=True, text=True, cwd=ROOT, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS streaming memory recovery:", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
