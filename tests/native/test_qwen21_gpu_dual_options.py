"""CPU-only model/recipe admission and source-bound sticky GPU qualification."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class GpuDualOptionsTests(unittest.TestCase):
    def test_strict_request_scope_generation_credit_and_sticky_failure(self):
        with tempfile.TemporaryDirectory(prefix="tc-gpu-dual-options-") as directory:
            binary=Path(directory)/"test"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/qwen21_gpu_dual_options_test.cpp"),"-o",str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=="__main__":unittest.main()
