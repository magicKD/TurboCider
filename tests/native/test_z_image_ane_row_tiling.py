"""Compare tiled FFN rows with an independent full-sequence projection."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))


class ANERowTilingTests(unittest.TestCase):
    def test_full_sequence_and_reused_backing(self):
        mlx = Path(os.environ.get("MLX_ROOT", str(Path(sysconfig.get_paths()["purelib"]) / "mlx")))
        with tempfile.TemporaryDirectory(prefix="tc-z-ane-rows-") as directory:
            binary = Path(directory) / "row-test"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx / "include"),
                str(ROOT / "tests/native/z_image_ane_row_tiling_test.cpp"),
                "-L", str(NATIVE), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
                "-Wl,-rpath," + str(NATIVE), "-Wl,-rpath," + str(mlx / "lib"),
                "-o", str(binary),
            ], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS: complete long-caption FFN coverage", result.stdout)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
