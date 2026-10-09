"""Actual Metal/IOSurface candidate; no ANE execution or real model fixture."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit Metal opt-in required")
class LoraUploadTests(unittest.TestCase):
    def test_direct_original_b_projection_to_correction_surface(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        library=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-lora-upload-") as folder:
            binary=Path(folder)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-fobjc-arc","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/lora_upload_test.mm"),"-L",str(library),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-framework","Foundation","-framework","Metal","-framework","IOSurface",
                "-Wl,-rpath,"+str(library),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(binary)],cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS rank-to-surface numeric_cases=144 rejected=8",result.stdout)
            self.assertIn("PASS rank-to-surface guarded_nonfinite_overflow=4 recovered_finite_producer=1",result.stdout)
            print(result.stdout)


if __name__=="__main__":unittest.main()
