"""Numerical parity for the real MLX runtime-LoRA projection slice path."""

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


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class ProjectSliceTests(unittest.TestCase):
    def test_fused_gate_up_runtime_lora_slices(self):
        mlx_root = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        self.assertTrue((native / "libturbocider.dylib").is_file(),
                        "build the native library before running the numerical test")
        with tempfile.TemporaryDirectory(prefix="turbocider-project-slice-") as directory:
            binary = Path(directory) / "probe"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra",
                "-mmacosx-version-min=" + platform.mac_ver()[0],
                str(ROOT / "tools/native/mlx_project_slice_probe.cpp"),
                "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx_root / "include"),
                "-L" + str(native), "-lturbocider",
                "-L" + str(mlx_root / "lib"), "-lmlx",
                "-Wl,-rpath," + str(native),
                "-Wl,-rpath," + str(mlx_root / "lib"),
                "-o", str(binary),
            ], check=True, cwd=ROOT)
            arguments = [str(binary)]
            if os.environ.get("TURBOCIDER_TEST_MLX_CPU") == "1":
                arguments.append("--cpu")
            report = subprocess.run(arguments, capture_output=True, text=True,
                                    check=True, cwd=ROOT)
            measured = json.loads(report.stdout)
            self.assertLess(measured["lora_rows_relative_l2"], 1e-5)
            self.assertLess(measured["lora_columns_relative_l2"], 1e-5)
            self.assertLess(measured["fp16_lora_rows_relative_l2"], 1e-5)
            self.assertLess(measured["gate_delta_relative_l2"], 1e-5)
            self.assertLess(measured["up_delta_relative_l2"], 1e-5)
            self.assertEqual(measured["workspace_rows_relative_l2"], 0)
            self.assertTrue(measured["workspace_identity_checks"])
            self.assertTrue(measured["supplied_base_identity_checks"])
            self.assertTrue(measured["disjoint_channel_correction_checks"])


if __name__ == "__main__":
    unittest.main()
