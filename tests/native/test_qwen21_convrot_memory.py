"""CPU-only source-aware Qwen admission, linked to the real residency budget gate."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ConvrotMemoryTests(unittest.TestCase):
    def test_actual_sources_geometry_and_checked_admission(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-convrot-memory-") as directory:
            directory = Path(directory)
            block = directory / "block_residency.o"
            subprocess.run(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-c",
                str(ROOT / "native/runtime/block_residency.c"), "-o", str(block)], check=True, cwd=ROOT)
            binary = directory / "probe"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native/core"), str(ROOT / "tests/native/qwen21_convrot_memory_test.cpp"),
                str(ROOT / "native/runtime/residency.cpp"), str(ROOT / "native/core/common.cpp"),
                str(block), "-o", str(binary)], check=True, cwd=ROOT)
            completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True, cwd=ROOT)
            self.assertIn("PASS ConvRot source-aware memory", completed.stdout)
            print(completed.stdout.strip())


if __name__ == "__main__":
    unittest.main()
