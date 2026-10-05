"""Opt-in actual Metal library/pipeline compilation, with no GPU command submission."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ENABLED = sys.platform == "darwin" and os.environ.get("TURBOCIDER_TEST_A8_METAL_COMPILE") == "1"


@unittest.skipUnless(ENABLED, "explicitly set TURBOCIDER_TEST_A8_METAL_COMPILE=1 for compile-only device access")
class A8MetalCompileTests(unittest.TestCase):
    def test_actual_candidate_library_and_two_function_constants_without_dispatch(self):
        records = ["Metal compile-only check; no GPU commands, ANE calls, models or performance loops."]
        for file in ("tests/native/ane_a8_metal_compile_test.mm", "native/backends/private/ane_transfer_kernels.hpp",
                     "native/backends/private/ane_w8_kernels.hpp"):
            records.append(hashlib.sha256((ROOT / file).read_bytes()).hexdigest() + "  " + file)
        result = None
        try:
            with tempfile.TemporaryDirectory(prefix="tc-a8-metal-compile-") as directory:
                binary = Path(directory) / "compile-only"
                command = ["xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-mmacosx-version-min=15.0",
                           "-Wall", "-Wextra", "-Werror", "-Wno-deprecated-declarations",
                           "tests/native/ane_a8_metal_compile_test.mm", "-framework", "Foundation", "-framework", "Metal",
                           "-o", str(binary)]
                build = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
                records.extend(("host_build_exit=" + str(build.returncode), build.stdout, build.stderr))
                self.assertEqual(build.returncode, 0, build.stderr)
                result = subprocess.run([str(binary), "--compile-only"], cwd=ROOT, capture_output=True, text=True, timeout=90)
                records.extend(("runtime_compile_exit=" + str(result.returncode), result.stdout, result.stderr))
        finally:
            records.append("temporary source/binary directory removed; persistent test source remains in repository")
            contents = "\n".join(records) + "\n"
            print(contents, end="")
            log = os.environ.get("TURBOCIDER_A8_METAL_COMPILE_LOG")
            if log:
                with Path(log).open("a") as output:
                    output.write(contents)
        self.assertIsNotNone(result)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compile_only=PASS command_buffers_created=0 dispatches=0 models_loaded=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
