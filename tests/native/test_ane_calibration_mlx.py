"""CPU-only MLX bridge validation with fake sources and tiny bound arrays."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("s1_bridge_sidecar", ROOT / "tools/coreml/create_smoothquant_s1_sidecar.py")
S1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S1)


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX CPU")
class AneCalibrationMlxTests(unittest.TestCase):
    def test_cpu_bridge_request_scope_and_bounded_outputs(self):
        mlx_root = Path(os.environ.get("MLX_ROOT", Path(sysconfig.get_paths()["purelib"]) / "mlx"))
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        self.assertTrue((mlx_root / "include/mlx/mlx.h").is_file(), "run with the repository MLX dependency environment")
        self.assertTrue((native / "libturbocider.dylib").is_file(), "build the current native library first")
        with tempfile.TemporaryDirectory(prefix="tc-calibration-mlx-cpu-") as temporary:
            directory = Path(temporary).resolve()
            binary = directory / "bridge-cpu"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-mmacosx-version-min=" + platform.mac_ver()[0],
                            str(ROOT / "tests/native/ane_calibration_mlx_test.cpp"),
                            "-I", str(ROOT / "native/core"), "-isystem", str(mlx_root / "include"),
                            "-L" + str(native), "-lturbocider", "-L" + str(mlx_root / "lib"), "-lmlx",
                            "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx_root / "lib"),
                            "-o", str(binary)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
            isolated_keys = {"TURBOCIDER_ANE_BACKEND", "TURBOCIDER_PRIVATE_ANE_DATA_PATH", "TURBOCIDER_PRIVATE_ANE_CHANNELS",
                             "TURBOCIDER_PRIVATE_ANE_S1_PROFILE", "TURBOCIDER_RUNTIME_ANE_CHUNKS", "TURBOCIDER_RUNTIME_ANE_PROFILE"}
            environment = {k: v for k, v in os.environ.items()
                           if not k.startswith("TURBOCIDER_ANE_CALIBRATION") and k not in isolated_keys}
            environment["TURBOCIDER_TEST_MLX_CPU"] = "1"
            result = subprocess.run([str(binary), str(directory)], cwd=ROOT, env=environment,
                                    check=False, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS CPU MLX calibration", result.stdout)
            captures = list((directory / "captures").glob("*/capture.json"))
            self.assertEqual(len(captures), 2)
            by_model = {json.loads(path.read_text())["binding"]["model_id"]: path for path in captures}
            for model, path in by_model.items():
                capture = json.loads(path.read_text())
                self.assertTrue(capture["complete"])
                self.assertFalse(capture["performance_sample"])
                self.assertFalse(capture["quantization_qualified"])
                self.assertEqual(capture["selection"]["steps"], [0, 3, 5])
                self.assertEqual(capture["selection"]["layers"], list(range(32)))
                self.assertEqual(capture["selection"]["rows_per_point"], 8)
                self.assertEqual(len(capture["points"]), 96)
                self.assertEqual(capture["used"], {"input_bytes": 96 * 8 * 8 * 2, "statistics_bytes": 128 * 8 * 4})
                self.assertTrue(all(point["rows_observed"] == point["rows"] for point in capture["points"]))
                stats = json.loads((path.parent / "weight_stats.json").read_text())
                self.assertEqual(stats["binding"], capture["binding"])
                self.assertEqual(stats["statistics_scope"], "base_gate_up_all_rows")
                self.assertEqual(len(stats["layers"]), 32)
                self.assertTrue(all(layer["gate_up_channel_max"][3] == 4 for layer in stats["layers"]))
                self.assertTrue(all(point["channel_max"][3] == 16 for point in capture["points"]))
                if model == "qwen-image-2.1":
                    expected_regions = ["text-0", "reference-0", "target"]
                    self.assertEqual(capture["binding"]["reference_count"], 1)
                    self.assertEqual(len(capture["binding"]["loras"]), 1)
                    points = [point for point in capture["points"] if point["step"] == 0]
                else:
                    expected_regions = ["target", "text"]
                    self.assertEqual(capture["binding"]["reference_count"], 0)
                    self.assertEqual(capture["binding"]["loras"], [])
                    points = capture["points"]
                for point in points:
                    self.assertEqual([region["name"] for region in point["regions"]], expected_regions)
                sidecar = S1.build_sidecar(path, path.parent / "weight_stats.json")
                self.assertEqual(sidecar["binding"], capture["binding"])
                self.assertEqual(sidecar["scope"], "full-model-s1")
                self.assertTrue(sidecar["cpu_replay"]["passed"])
                self.assertFalse(sidecar["runtime_applied"])
                self.assertFalse(sidecar["quantization_qualified"])
                self.assertLess(sum(p.stat().st_size for p in path.parent.iterdir()), 160 * 1024)
                if model == "qwen-image-2.1":
                    output = directory / "qwen-s1.json"
                    S1.write_sidecar(output,sidecar)
                    loaded = subprocess.run([str(binary),str(directory),str(output)],cwd=ROOT,env=environment,
                                            check=False,capture_output=True,text=True,timeout=20)
                    self.assertEqual(loaded.returncode,0,loaded.stdout + loaded.stderr)
                    self.assertIn("PASS CPU S1 binding/scales",loaded.stdout)


if __name__ == "__main__":
    unittest.main()
