"""Explicit real-model check: evict a full-size DiT cache during Turbo8."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DUMPS = os.environ.get("TURBOCIDER_QWEN21_CACHE_TRAJECTORY_DUMPS")


@unittest.skipUnless(sys.platform == "darwin" and DUMPS and os.environ.get("TURBOCIDER_TEST_GPU") == "1",
                     "requires explicit matching 512px Turbo8 dumps and actual Metal opt-in")
class CacheTrajectoryTests(unittest.TestCase):
    def test_eviction_after_third_step_matches_complete_uncached_trajectory(self):
        library = ROOT / os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", "build/native")
        model = Path(os.environ.get("TURBOCIDER_QWEN21_CACHE_TRAJECTORY_MODEL",
            ROOT / "models/Qwen-Image-2.1-Turbo-INT8/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"))
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen-cache-trajectory-") as temporary:
            binary = Path(temporary) / "trajectory"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-mmacosx-version-min=26.2", "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
                "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen21_convrot_cache_trajectory_test.cpp"),
                "-L", str(library), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx",
                "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith("TURBOCIDER_")}
            environment.update(MLX_ENABLE_TF32="0", TURBOCIDER_SOURCE_REGISTRATION_CACHE=str(Path(temporary) / "registration"))
            result = subprocess.run([str(binary), str(model), str(Path(DUMPS).resolve())],
                text=True, capture_output=True, env=environment, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1536 W8A8 projections; 253 drained refills", result.stdout)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
