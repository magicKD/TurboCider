"""Actual-library Qwen3-VL ConvRot embedding and compiled W8A8 route checks."""
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


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class ConvrotEncoderRuntimeTests(unittest.TestCase):
    def test_embedding_inverse_source_geometry_and_compiled_routes(self):
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
        manifest = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
        for source in manifest["inputs"]["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(), source["sha256"],
                             "C++ source/header ABI requires the matching native tree: " + source["path"])
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen-convrot-encoder-") as directory:
            binary = Path(directory) / "probe"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-Wno-missing-field-initializers", "-mmacosx-version-min=26.2",
                '-DTC_TEST_BOUND_NATIVE_ID="' + manifest["runtime_build_id"] + '"',
                "-I", str(ROOT / "native/core"), "-I", str(ROOT / "bindings/c/include"),
                "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen21_convrot_encoder_test.cpp"),
                "-L" + str(native), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx",
                "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)
            ], check=True, cwd=ROOT)
            subprocess.run([str(binary)], check=True, cwd=ROOT, timeout=180)


if __name__ == "__main__":
    unittest.main()
