import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual Metal opt-in required")
class DenseCooperativeTests(unittest.TestCase):
    def test_f32_and_original_dtype_oracles_with_physical_sources(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-dense-coop-") as folder:
            probe=Path(folder)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-isystem",str(mlx/"include"),str(ROOT/"tests/native/dense_cooperative_test.cpp"),"-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS cooperative dense cases=800",result.stdout)
            self.assertIn("PASS cooperative dense rejected_contracts=20",result.stdout)
            print(result.stdout)


if __name__=="__main__":unittest.main()
