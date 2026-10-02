"""Real Metal FP16 decode/GEMM oracle; no mock pass or whole-model qualification."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit real Metal opt-in required")
class AffineF16Tests(unittest.TestCase):
    def test_decoder_and_projection(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        native=ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/quantized-execution")
        with tempfile.TemporaryDirectory(prefix="tc-z-affine-f16-") as raw:
            binary=Path(raw)/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/z_affine_fp16_test.cpp"),str(ROOT/"native/core/gguf_decode.cpp"),
                "-L",str(native),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS GPU affine FP16",result.stdout)
            print(result.stdout,end="")

if __name__=="__main__":unittest.main()
