import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit actual Metal opt-in required")
class EncoderPrefillTests(unittest.TestCase):
    def test_typed_kernels_source_fusion_and_processor_lease(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-prefill-") as raw:
            folder = Path(raw)
            (folder / "tokenizer.json").write_text(json.dumps(dict(
                model=dict(type="BPE", vocab={"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "abc": 5}, merges=[["a", "b"], ["ab", "c"]]),
                pre_tokenizer=dict(pretokenizers=[dict(pattern=dict(Regex="[a-z]+"))]), added_tokens=[])))
            mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
            library = ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")
            binary = folder / "probe"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-mmacosx-version-min=26.2",
                "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
                str(ROOT / "tests/native/qwen21_encoder_prefill_test.cpp"), "-L", str(library), "-lturbocider",
                "-L", str(mlx / "lib"), "-lmlx", "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary), str(folder)], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS encoder GPU prefill", result.stdout)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
