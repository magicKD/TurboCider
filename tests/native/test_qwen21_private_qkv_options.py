"""CPU-only explicit joint runtime QKV/FFN admission controls."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PrivateQkvOptionsTests(unittest.TestCase):
    def test_joint_base_fixed_healthy_recipe(self):
        with tempfile.TemporaryDirectory(prefix="tc-private-qkv-options-") as directory:
            binary = Path(directory) / "options"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/core"),
                            str(ROOT / "tests/native/qwen21_private_qkv_options_test.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
