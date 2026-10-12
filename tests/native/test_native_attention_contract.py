"""CPU head-layout/ABI contracts and host syntax only; no Metal device creation."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "native host syntax requires macOS clang")
class NativeAttentionContractTests(unittest.TestCase):
    def test_shared_layout_validation_and_host_wrapper_without_device(self):
        with tempfile.TemporaryDirectory(prefix="tc-native-sdpa-cpu-") as directory:
            binary = Path(directory) / "contract"
            common = ["xcrun", "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror"]
            syntax = subprocess.run([*common, "-fobjc-arc", "-fsyntax-only",
                                     "native/backends/metal_graph/native_attention.mm"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            build = subprocess.run([*common, "-O2", "tests/native/native_attention_contract_test.cpp",
                                    "-o", str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=10)
            print(result.stdout, end="")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("native_attention_cpu=PASS", result.stdout)
            self.assertIn("device_created=0 model_loads=0 score_buffer_bytes=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
