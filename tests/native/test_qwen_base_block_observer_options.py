"""CPU-only full-base40 diagnostic source admission; no model inference."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BaseBlockObserverOptionsTests(unittest.TestCase):
    def test_full_base_geometry_and_negative_controls(self):
        with tempfile.TemporaryDirectory(prefix="tc-base-observer-") as directory:
            binary = Path(directory) / "test"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/native/qwen21_base_block_observer_options_test.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
