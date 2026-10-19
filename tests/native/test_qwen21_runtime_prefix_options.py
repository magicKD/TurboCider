"""CPU-only request and actual-runtime admission negative controls."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RuntimePrefixOptionsTests(unittest.TestCase):
    def test_base_only_fixed_healthy_recipe(self):
        with tempfile.TemporaryDirectory(prefix="tc-runtime-prefix-options-") as directory:
            binary = Path(directory) / "options"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/core"),
                            str(ROOT / "tests/native/qwen21_runtime_prefix_options_test.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
