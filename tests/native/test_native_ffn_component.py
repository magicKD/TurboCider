"""Native dense FFN host contract and opt-in, bounded Metal component checks."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("tools/native/native_ffn_component.mm",
           "native/backends/metal_graph/native_ffn.mm",
           "native/backends/metal_graph/backend.mm")
IDENTITY_FILES = (*SOURCES, "native/backends/metal_graph/backend.hpp",
                  "native/backends/metal_graph/native_ffn.hpp",
                  "native/backends/metal_graph/kernels/dense_ffn.hpp")


@unittest.skipUnless(sys.platform == "darwin", "Apple Metal host compilation requires macOS")
class NativeFfnComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="tc-native-ffn-component-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "native-ffn-component"
        command = ["xcrun", "clang++", "-std=c++20", "-O2", "-ffp-contract=off", "-fobjc-arc",
                   "-mmacosx-version-min=15.0", "-Wall", "-Wextra", "-Werror",
                   "-Wno-deprecated-declarations", *SOURCES,
                   "-framework", "Foundation", "-framework", "Metal", "-o", str(cls.binary)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.identities = "\n".join(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() + "  " + f
                                   for f in IDENTITY_FILES)

    def run_probe(self, mode):
        result = subprocess.run([str(self.binary), mode], cwd=ROOT, capture_output=True, text=True, timeout=90)
        record = self.identities + "\nmode=" + mode + "\n" + result.stdout + result.stderr
        print(record, end="")
        log = os.environ.get("TURBOCIDER_NATIVE_FFN_LOG")
        if log:
            with Path(log).open("a") as output:
                output.write(record)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_cpu_contract_without_device(self):
        self.assertIn("cpu_contract=PASS", self.run_probe("--cpu-only"))

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_NATIVE_FFN_COMPILE") == "1",
                         "explicit opt-in required for Metal runtime compilation")
    def test_real_shape_compile_without_commands(self):
        self.assertIn("compile_only=PASS command_buffers=0 dispatches=0 models_loaded=0",
                      self.run_probe("--compile-only"))

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_NATIVE_FFN_GPU") == "1",
                         "explicit opt-in required for six tiny GPU dispatches")
    def test_tiny_bf16_matrix_oracle(self):
        self.assertIn("small_tensors=PASS cases=3 dispatches=6 models_loaded=0",
                      self.run_probe("--small-tensors"))


if __name__ == "__main__":
    unittest.main()
