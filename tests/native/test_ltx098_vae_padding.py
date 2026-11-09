"""Exact GPU padding and conservative fallback checks without model weights.

Set TURBOCIDER_LTX098_VAE_PAD_PROBE to a probe built from the current tree.
"""
import json
import os
from pathlib import Path
import subprocess
import unittest

PROBE = Path(os.environ.get("TURBOCIDER_LTX098_VAE_PAD_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "current native LTX VAE pad probe required")
class VAEPaddingTests(unittest.TestCase):
    def test_byte_exact_layouts_and_fallbacks(self):
        cases = ((1, 2, 2, 128, "contiguous", True), (2, 3, 5, 128, "bits", True), (2, 3, 5, 128, "transpose", True),
                 (2, 3, 5, 128, "slice", True), (2, 3, 5, 128, "flip", None),
                 (2, 3, 5, 7, "contiguous", False))
        for f, h, w, c, layout, fused in cases:
            with self.subTest(shape=(f, h, w, c), layout=layout):
                result = subprocess.run([str(PROBE), str(f), str(h), str(w), str(c), "2", layout],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["exact"])
                self.assertEqual(report["fused"], not report["negative_stride"] if fused is None else fused)


if __name__ == "__main__":
    unittest.main()
