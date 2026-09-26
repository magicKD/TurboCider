import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Qwen21SequenceTests(unittest.TestCase):
    def test_reference_geometry(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-sequence-") as directory:
            binary = Path(directory) / "sequence-test"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/native/qwen21_sequence_test.cpp"),
                            str(ROOT / "native/models/qwen21/sequence.cpp"),
                            str(ROOT / "native/core/common.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
