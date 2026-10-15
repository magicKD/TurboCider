"""Live-buffer regression for fresh, real compiled Qwen21 Transformers."""

import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX GPU")
class TransformerLifecycleTests(unittest.TestCase):
    def test_compiled_transformers_release_materialized_weights(self):
        mlx_root = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        self.assertTrue((native / "libturbocider.dylib").is_file(),
                        "build the native library before the lifecycle test")
        with tempfile.TemporaryDirectory(prefix="turbocider-qwen21-lifecycle-") as directory:
            binary = Path(directory) / "probe"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra",
                "-mmacosx-version-min=" + platform.mac_ver()[0],
                str(ROOT / "tests/native/qwen21_lifecycle_test.cpp"),
                "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx_root / "include"),
                "-L" + str(native), "-lturbocider",
                "-L" + str(mlx_root / "lib"), "-lmlx",
                "-Wl,-rpath," + str(native),
                "-Wl,-rpath," + str(mlx_root / "lib"),
                "-o", str(binary),
            ], check=True, cwd=ROOT)
            # Deliberately exercise the ordinary compiled GPU graph even if
            # another diagnostic has been launched from the same shell.
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("TURBOCIDER_QWEN21_") and
                           key not in {"MLX_DISABLE_COMPILE",
                                       "TURBOCIDER_DISABLE_FUSED_RMSNORM"}}
            completed = subprocess.run([str(binary)], capture_output=True, text=True,
                                       check=False, cwd=ROOT, env=environment,
                                       timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            measured = json.loads(completed.stdout)
            self.assertEqual(measured["cycles"], 24)
            self.assertEqual(measured["variants"], 8)
            self.assertTrue(measured["deterministic_prefill_decode"])
            self.assertEqual(measured["snapshot_cycles"], 24)
            self.assertTrue(measured["snapshot_cross_transformer_parity"])
            self.assertTrue(measured["dbcache_policy_and_invalidation"])
            self.assertTrue(measured["dbcache_reference_geometry_and_lora"])
            self.assertGreater(measured["weight_bytes_per_base_cycle"], 65536)
            self.assertLessEqual(measured["maximum_active_bytes"],
                                 measured["baseline_active_bytes"] +
                                 measured["idle_tolerance_bytes"])


if __name__ == "__main__":
    unittest.main()
