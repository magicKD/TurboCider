"""Actual Metal MPP tile/tail/physical-offset regression; no model fixture."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
class DenseGpuProjectionTests(unittest.TestCase):
    def test_tiles_tails_physical_offsets_and_rejections(self):
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        library = (ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-dense-projection-") as directory:
            probe = Path(directory) / "probe"
            result = subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-mmacosx-version-min=26.2", "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx / "include"), str(ROOT / "tests/native/dense_gpu_projection_test.cpp"),
                "-L", str(library), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
                "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(probe)
            ], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(probe)], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS dense GPU projection cases=144 rejected=16", result.stdout)
            print(result.stdout)


if __name__ == "__main__":
    unittest.main()
