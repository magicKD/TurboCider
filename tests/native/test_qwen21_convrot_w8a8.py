"""Original-code ConvRot W8A8: independent CPU integer oracle and real GPU dispatch."""
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


@unittest.skipUnless(sys.platform == "darwin", "requires Apple Metal")
class OriginalConvrotW8A8Tests(unittest.TestCase):
    def test_cpu_oracle_precision_and_shared_weights(self):
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        manifest = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
        for source in manifest["inputs"]["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(), source["sha256"],
                             "C++ source/header ABI requires the matching native tree: " + source["path"])
        mlx = Path(os.environ.get("MLX_ROOT", Path(sysconfig.get_paths()["purelib"]) / "mlx"))
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        sdk = os.environ.get("SDKROOT") or subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        with tempfile.TemporaryDirectory(prefix="tc-original-convrot-w8a8-") as directory:
            binary = Path(directory) / "probe"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-Wno-unused-parameter", "-isysroot", sdk, "-mmacosx-version-min=26.2",
                '-DTC_TEST_BOUND_NATIVE_ID="' + manifest["runtime_build_id"] + '"',
                "-I", str(ROOT / "native/core"), "-I", str(ROOT / "bindings/c/include"), "-isystem", str(mlx / "include"),
                str(ROOT / "tests/native/convrot_w8a8_test.cpp"),
                "-L" + str(native), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl",
                "-framework", "Foundation", "-framework", "Metal",
                "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"),
                "-o", str(binary)], cwd=ROOT, check=True)
            result = subprocess.run([str(binary), str(Path(directory) / "synthetic-lora.safetensors")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("qwen_convrot_lora=PASS", result.stdout)
            if "SKIP M5 TensorOps unavailable" in result.stdout:
                self.skipTest("M5 TensorOps unavailable; raw/packed ConvRot LoRA regression passed")
            self.assertIn("PASS ConvRot original-code W8A8", result.stdout)
            self.assertIn("shared_weights=PASS", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
