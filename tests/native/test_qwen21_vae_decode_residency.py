"""Actual Original VAE metadata validation and lazy-owner filtering, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "native MLX library requires macOS")
class VaeDecodeResidencyTests(unittest.TestCase):
    def test_request_metadata_payload_and_owners(self):
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        manifest = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
        for source in manifest["inputs"]["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(), source["sha256"],
                             "native source/header binding differs: " + source["path"])
        mlx = Path(os.environ.get("MLX_ROOT", Path(sysconfig.get_paths()["purelib"]) / "mlx"))
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        sdk = os.environ.get("SDKROOT") or subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        with tempfile.TemporaryDirectory(prefix="tc-qwen-vae-residency-") as raw:
            binary = Path(raw) / "probe"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                "-isysroot", sdk, "-mmacosx-version-min=26.2", '-DTC_TEST_BOUND_NATIVE_ID="' + manifest["runtime_build_id"] + '"',
                "-I", str(ROOT / "native/core"), "-I", str(ROOT / "bindings/c/include"), "-isystem", str(mlx / "include"),
                str(ROOT / "tests/native/qwen21_vae_decode_residency_test.cpp"), "-L" + str(native), "-lturbocider",
                "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl", "-framework", "Foundation", "-framework", "Metal",
                "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True, cwd=ROOT)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            measured = json.loads(result.stdout)
            self.assertEqual(measured["runtime_build_id"], manifest["runtime_build_id"])
            self.assertEqual(measured["status"], "passed")
            self.assertTrue(measured["synthetic_source_only"])
            self.assertTrue(measured["metadata_preflight_no_payload_reads"])
            self.assertEqual(measured["invalid_unused_metadata_cases"], 3)
            self.assertTrue(measured["legacy_and_owner_cases_passed"])
            self.assertTrue(measured["canonical_saved_metadata_passed"])
            self.assertTrue(measured["cancellation_passed"])
            self.assertFalse(measured["gpu_used"])
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
