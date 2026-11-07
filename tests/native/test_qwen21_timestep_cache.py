"""Default-off policy, exact M2 BF16 conditions and detached source lifetimes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TimestepCachePolicyTests(unittest.TestCase):
    def test_strict_flag_and_source_route(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="tc-timestep-policy-") as raw:
            binary = Path(raw) / "policy"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/native/qwen21_timestep_cache_options_test.cpp"),
                str(ROOT / "native/core/common.cpp"), "-o", str(binary)], check=True, cwd=ROOT)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS timestep cache strict request policy", result.stdout)


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit actual Metal opt-in required")
class TimestepCacheMetalTests(unittest.TestCase):
    def test_full_m2_bits_source_owners_and_reset_lifecycle(self):
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        manifest = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
        for source in manifest["inputs"]["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(), source["sha256"],
                "C++ source/header ABI requires matching native tree: " + source["path"])
        mlx = Path(os.environ.get("MLX_ROOT", Path(sysconfig.get_paths()["purelib"]) / "mlx"))
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        sdk = os.environ.get("SDKROOT") or subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        with tempfile.TemporaryDirectory(prefix="tc-timestep-metal-") as raw:
            binary = Path(raw) / "probe"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                "-isysroot", sdk, "-mmacosx-version-min=26.2", '-DTC_TEST_BOUND_NATIVE_ID="' + manifest["runtime_build_id"] + '"',
                "-I", str(ROOT / "native/core"), "-I", str(ROOT / "bindings/c/include"), "-isystem", str(mlx / "include"),
                str(ROOT / "tests/native/qwen21_timestep_cache_test.cpp"), "-L" + str(native), "-lturbocider",
                "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl", "-framework", "Foundation", "-framework", "Metal",
                "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True, cwd=ROOT)
            env = {key: value for key, value in os.environ.items() if not key.startswith("TURBOCIDER_QWEN21_") and key != "MLX_DISABLE_COMPILE"}
            env["MLX_ENABLE_TF32"] = "0"
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=180, env=env, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            measured = json.loads(result.stdout)
            self.assertTrue(measured["synthetic_source_only"])
            self.assertEqual(measured["full_m2_sigma_cases"], 8)
            self.assertEqual(measured["logical_condition_bytes"], 655360)
            self.assertEqual(measured["actual_condition_capacity_bytes"], 655360)
            self.assertTrue(measured["original_source_owners_expired"])
            self.assertTrue(measured["owner_sigma_dtype_binding_retirement_reentry_guards_passed"])
            self.assertEqual(measured["tiny_exact_forward_cases"], 10)
            self.assertEqual(len(measured["output_sha256"]), 16)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
