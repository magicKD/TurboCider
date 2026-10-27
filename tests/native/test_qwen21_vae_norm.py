"""Real GPU FP32 norm/activation and NHWC-origin contracts; no model files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "requires Apple Metal/MLX")
class VaeNormTests(unittest.TestCase):
    def probe(self, builder, executable):
        with tempfile.TemporaryDirectory(prefix="qwen21-vae-norm-") as directory:
            env = dict(os.environ, TURBOCIDER_NATIVE_OUT=directory,
                       TURBOCIDER_NATIVE_LIBRARY_DIR=str(Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()))
            subprocess.run(["bash", str(ROOT / "tools/native" / builder)],
                           env=env, cwd=ROOT, check=True, capture_output=True, text=True)
            result = subprocess.run([str(Path(directory) / executable)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_fp32_fused_norm_and_real_strided_origins(self):
        report = self.probe("build_qwen21_vae_norm_probe.sh", "qwen21-vae-norm-probe")
        self.assertEqual(report["cases"], 214)
        self.assertEqual(report["rejected"], 5)
        self.assertLess(report["worst_relative_l2"], 2e-6)
        self.assertGreater(report["lazy_metadata_changes"], 0)

    def test_complete_fp32_retry_and_cancellation(self):
        report = self.probe("build_qwen21_vae_fallback_probe.sh", "qwen21-vae-fallback-probe")
        self.assertEqual(report["full_fp32_fallbacks"], 2)
        self.assertEqual(report["rgba_channels"], 4)
        for key in ("cancel_before_decode", "cancel_before_retry", "invalid_input_rejected", "original_source_restored"):
            self.assertTrue(report[key])


if __name__ == "__main__":
    unittest.main()
