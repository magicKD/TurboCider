"""Production ConvRot helper parity; GPU execution is explicitly opt-in."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ConvRotFfnContractTests(unittest.TestCase):
    def test_research_opt_in_is_explicit(self):
        source = (ROOT / "native/models/z_image/z_image.cpp").read_text()
        self.assertIn('std::getenv("TURBOCIDER_Z_CONVROT_SHARED_GATE_UP")', source)
        self.assertIn('shared && std::strcmp(shared, "1") == 0', source)
        self.assertIn('return z_image::feed_forward(x, w, prefix, shared &&', source)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_native_shared_rotation_exact(self):
        subprocess.run(["bash", "tools/native/build_convrot_ffn_probe.sh"], cwd=ROOT, check=True)
        result = subprocess.run([str(ROOT / "build/native/convrot-ffn-probe")],
                                cwd=ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout)
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["exact"])
        self.assertEqual(receipt["cases"], 96)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_simd_quad_rotation_against_shared_and_independent_basis(self):
        with tempfile.TemporaryDirectory(prefix="tc-convrot-rotation-") as directory:
            root = Path(directory)
            subprocess.run(["bash", "tools/native/build_convrot_rotation_test.sh"], cwd=ROOT, check=True,
                           env={**os.environ, "TURBOCIDER_NATIVE_OUT": str(root)}, capture_output=True, text=True)
            result = subprocess.run([str(root / "convrot-rotation-test")], cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("72 typed/strided cases bit-exact", result.stdout)
            self.assertIn("256 independent basis rows", result.stdout)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_integrated_quad_large_packed_ffn_exact(self):
        with tempfile.TemporaryDirectory(prefix="tc-convrot-ffn-") as directory:
            root = Path(directory)
            subprocess.run(["bash", "tools/native/build_convrot_ffn_probe.sh"], cwd=ROOT, check=True,
                           env={**os.environ, "TURBOCIDER_NATIVE_OUT": str(root)}, capture_output=True, text=True)
            result = subprocess.run([str(root / "convrot-ffn-probe"), "rotation-integration"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count("PASS integrated ConvRot quad:"), 2)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_convrot_base_channel_projection_preserves_source_basis(self):
        with tempfile.TemporaryDirectory(prefix="tc-convrot-ranges-") as directory:
            root = Path(directory)
            subprocess.run(["bash", "tools/native/build_convrot_ffn_probe.sh"], cwd=ROOT, check=True,
                           env={**os.environ, "TURBOCIDER_NATIVE_OUT": str(root)}, capture_output=True, text=True)
            result = subprocess.run([str(root / "convrot-ffn-probe"), "projection-ranges"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS 24 ConvRot GPU base/channel projection cases", result.stdout)


if __name__ == "__main__":
    unittest.main()
