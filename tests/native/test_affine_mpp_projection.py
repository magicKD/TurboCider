import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual MLX/Metal opt-in required")
class AffineMppProjectionTests(unittest.TestCase):
    def test_original_typed_coefficients_and_physical_range(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-affine-mpp-") as folder:
            probe=Path(folder)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/affine_mpp_projection_test.cpp"),"-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS MPP register-decode affine F32 cases=200",result.stdout);print(result.stdout)


if __name__=="__main__":unittest.main()
