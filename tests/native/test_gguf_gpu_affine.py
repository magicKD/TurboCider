"""Actual GPU raw-block packing; fields/claims compared with independent CPU oracle."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual Metal opt-in required")
class GpuAffineTests(unittest.TestCase):
    def test_raw_fields_and_storage_lifetime(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        native=ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/quantized-execution")
        with tempfile.TemporaryDirectory(prefix="tc-gguf-gpu-affine-") as raw:
            binary=Path(raw)/"probe"
            flags=["-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror"]
            if os.environ.get("TC_STREAMING_SANITIZER"):
                flags += ["-fsanitize="+os.environ["TC_STREAMING_SANITIZER"],"-fno-omit-frame-pointer"]
            subprocess.run(["xcrun","clang++",*flags,
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/gguf_gpu_affine_test.cpp"),str(ROOT/"native/core/gguf_affine.cpp"),
                str(ROOT/"native/core/gguf_decode.cpp"),"-L",str(native),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(native),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS raw GPU affine",result.stdout);print(result.stdout,end="")

if __name__=="__main__":unittest.main()
