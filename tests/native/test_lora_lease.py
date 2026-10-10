import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual MLX/Metal opt-in required")
class LoraLeaseTests(unittest.TestCase):
    def test_verified_held_fd_bind_math_mutation_and_cancellation(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        library=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-lora-lease-") as folder:
            root=Path(folder);binary=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/lora_lease_test.cpp"),"-L",str(library),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(library),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(binary),str(root/"fixtures")],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS leased LoRA stacked_bindings=4 rejected=6",result.stdout)
            print(result.stdout)


if __name__=="__main__":unittest.main()
