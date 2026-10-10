from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class GgufKAffineTests(unittest.TestCase):
    def test_mixed_q4km_group_conversion_and_guards(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-k-affine-") as temporary:
            binary = Path(temporary) / "probe"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/core"),
                str(ROOT / "tests/native/gguf_k_affine_test.cpp"),
                str(ROOT / "native/core/gguf_affine.cpp"),
                str(ROOT / "native/core/gguf_decode.cpp"), "-o", str(binary),
            ], check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS 108 Q4_K/Q5_K/Q6_K typed affine cases", result.stdout)


if __name__ == "__main__":
    unittest.main()
