"""Synthetic Metal regression of compiled Qwen FFN ownership; no checkpoint."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


@unittest.skipUnless(sys.platform == "darwin", "requires Metal")
class RuntimeActivationTests(unittest.TestCase):
    def test_compiled_ffn_values_and_weight_release(self):
        library = NATIVE / "libturbocider.dylib"
        self.assertTrue(library.is_file(), "build the native library first")
        mlx = Path(os.environ.get("MLX_ROOT", str(Path(sysconfig.get_paths()["purelib"]) / "mlx")))
        self.assertTrue((mlx / "lib/libmlx.dylib").is_file())
        sdk = os.environ.get("SDKROOT", "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk")
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("TURBOCIDER_", "DYLD_"))}
        with tempfile.TemporaryDirectory(prefix="tc-qwen-activation-", dir="/private/tmp") as temporary:
            root = Path(temporary)
            binary = root / "test"
            compiler = "/Library/Developer/CommandLineTools/usr/bin/clang++"
            command = [compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-isysroot", sdk, "-mmacosx-version-min=26.2",
                       "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
                       str(ROOT / "tests/native/qwen21_runtime_activation_test.cpp"),
                       "-L" + str(NATIVE), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl",
                       "-Wl,-rpath," + str(NATIVE), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)]
            compiled = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr + compiled.stdout)
            result = subprocess.run([str(binary), str(root)], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS Qwen runtime activation ownership", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
