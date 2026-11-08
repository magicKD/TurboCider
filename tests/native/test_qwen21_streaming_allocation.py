"""CPU-only required streaming allocation policy; no model, MLX or GPU work."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class StreamingAllocationTests(unittest.TestCase):
    def test_system_managed_required_allocations_keep_optional_cache_guarded(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="tc-stream-allocation-") as raw:
            binary = Path(raw) / "allocation"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/native/qwen21_streaming_allocation_test.cpp"),
                str(ROOT / "native/backends/ane_memory.cpp"), "-o", str(binary)],
                check=True, cwd=ROOT, timeout=30)
            result = subprocess.run([str(binary)], capture_output=True, text=True, cwd=ROOT, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS streaming allocation:", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
