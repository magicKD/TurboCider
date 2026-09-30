"""Production ConvRot helper parity; GPU execution is explicitly opt-in."""
import json
import os
from pathlib import Path
import subprocess
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


if __name__ == "__main__":
    unittest.main()
