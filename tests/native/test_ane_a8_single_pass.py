"""CPU-only A8 candidate math/policy; never creates a Metal device or loads models."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class A8SinglePassHostTests(unittest.TestCase):
    def test_independent_two_pass_bits_and_shared_host_policy(self):
        with tempfile.TemporaryDirectory(prefix="tc-a8-single-pass-") as directory:
            binary = Path(directory) / "math"
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "tests/native/ane_a8_single_pass_test.cpp", "native/core/gguf_decode.cpp", "-o", str(binary)],
                           cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
