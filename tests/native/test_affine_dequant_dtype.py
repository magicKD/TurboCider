"""Opt-in tiny MLX fixture; never runs during timed full-model experiments."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AffineDequantDtypeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit small MLX/Metal fixture opt-in required")
    def test_actual_q4_q8_copied_source_and_dtype(self):
        library = (ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")).resolve()
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        with tempfile.TemporaryDirectory(prefix="tc-affine-dequant-dtype-") as raw:
            binary = Path(raw) / "probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT / "native/core"),"-isystem",str(mlx / "include"),
                str(ROOT / "tests/native/affine_dequant_dtype_test.cpp"),"-L",str(library),"-lturbocider",
                "-L",str(mlx / "lib"),"-lmlx","-Wl,-rpath,"+str(library),"-Wl,-rpath,"+str(mlx / "lib"),"-o",str(binary)],
                cwd=ROOT,check=True,timeout=90)
            result = subprocess.run([str(binary)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS affine dequant dtypes: two actual Q4/Q8 cases",result.stdout)
            print(result.stdout,end="")


if __name__ == "__main__":
    unittest.main()
