"""CPU-only unavailable-FFN S1 ownership and actual metrics exporter."""
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin","requires Apple MLX CPU and Foundation")
class AneSmoothquantFfnTests(unittest.TestCase):
    def test_cpu_unavailable_ffn_profile_ownership_and_exporter(self):
        mlx_root = Path(os.environ.get("MLX_ROOT",Path(sysconfig.get_paths()["purelib"]) / "mlx"))
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR",ROOT / "build/native")).resolve()
        self.assertTrue((mlx_root / "include/mlx/mlx.h").is_file(),"run with repository MLX dependencies")
        self.assertTrue((native / "libturbocider.dylib").is_file(),"build current native library first")
        with tempfile.TemporaryDirectory(prefix="tc-s1-ffn-cpu-") as temporary:
            directory = Path(temporary).resolve()
            binary = directory / "ffn-cpu"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-fobjc-arc",
                            "-mmacosx-version-min=" + platform.mac_ver()[0],
                            str(ROOT / "tests/native/ane_smoothquant_ffn_test.mm"),
                            "-I",str(ROOT / "native/core"),"-isystem",str(mlx_root / "include"),
                            "-L" + str(native),"-lturbocider","-L" + str(mlx_root / "lib"),"-lmlx",
                            "-Wl,-rpath," + str(native),"-Wl,-rpath," + str(mlx_root / "lib"),
                            "-framework","Foundation","-o",str(binary)],
                           cwd=ROOT,check=True,capture_output=True,text=True,timeout=60)
            environment = {k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
            environment["TURBOCIDER_TEST_MLX_CPU"] = "1"
            result = subprocess.run([str(binary),str(directory)],cwd=ROOT,env=environment,
                                    check=False,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stdout + result.stderr)
            self.assertIn("PASS CPU S1 FFN ownership/fallback/exporter",result.stdout)
            self.assertFalse((directory / "never-created-model-or-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
