"""Compare source-independent model math to pinned pre-refactor Git code."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
class StateTests(unittest.TestCase):
    def test_pinned_reference_resident_and_submitted(self):
        old = subprocess.check_output(["git", "show", "3030a77:native/components/text/qwen3.cpp"], cwd=ROOT, text=True)
        pieces = []
        for begin, end in [("static int qwen3_integer_setting", "std::filesystem::path qwen3_checkpoint_path"),
                           ("static bool qwen3_fused_attention_enabled", "static float qwen3_validation_threshold"),
                           ("static float qwen3_validation_threshold", "Qwen3Conditioning Qwen3Conditioning::flux_klein")]:
            pieces.append(old[old.index(begin):old.index(end)])
        function = old[old.index("Tensor qwen3_conditioning("):old.rindex("} // namespace tc::components")]
        pieces.append(function.replace("Tensor qwen3_conditioning(", "Tensor qwen3_conditioning_reference(", 1))
        library = ROOT / os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/quantized-execution")
        import sysconfig
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        with tempfile.TemporaryDirectory(prefix="tc-qwen3-state-") as raw:
            directory = Path(raw)
            baseline = directory / "reference.cpp"
            baseline.write_text('#include "qwen3.hpp"\n#include "coreml.hpp"\n#include <algorithm>\n#include <cmath>\n#include <cerrno>\n#include <cstdlib>\n#include <cstdio>\nnamespace tc::components {\n' + "\n".join(pieces) + "\n}\n")
            binary = directory / "probe"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-mmacosx-version-min=15.0", "-I", str(ROOT / "native/core"),
                "-I", str(ROOT / "native/components/text"), "-I", str(ROOT / "native/backends"),
                "-isystem", str(mlx / "include"), str(ROOT / "tests/native/qwen3_state_test.cpp"), str(baseline),
                "-L", str(library), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx", "-ljaccl",
                "-Wl,-rpath," + str(library), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)],
                check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS Qwen3 state", result.stdout)
            print(result.stdout, end="")
