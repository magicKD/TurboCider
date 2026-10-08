"""Bounded early-decode consumer: real Metal tests are explicitly opt-in."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AffineDenseWindowTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal opt-in required")
    def test_typed_decode_and_reader_lifetime(self):
        with tempfile.TemporaryDirectory(prefix="tc-affine-dense-window-") as temporary:
            subprocess.run(["bash", "tools/native/build_gpu_weight_consumer_probe.sh"], cwd=ROOT, check=True,
                           env={**os.environ, "TURBOCIDER_NATIVE_OUT": temporary}, capture_output=True, text=True)
            result = subprocess.run([str(Path(temporary) / "gpu-weight-consumer-probe")], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS 12 typed affine dense window cases", result.stdout)
            for reuses in ("0","17"):
                result = subprocess.run([str(Path(temporary) / "gpu-weight-consumer-probe"),
                    "gguf","never-read.gguf","unused", "33","9",reuses], cwd=ROOT,
                    capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("actual-reuses must be 1..16",result.stderr)
                self.assertNotIn("cannot open GGUF",result.stderr)
            for mode in ("gguf-mpp","convrot-mpp"):
                for rows,extra in (("31",[]),("33",["1"])):
                    result=subprocess.run([str(Path(temporary)/"gpu-weight-consumer-probe"),
                        mode,"never-read.gguf","unused",rows,"9",*extra],cwd=ROOT,
                        capture_output=True,text=True,timeout=30)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn("MPP screen requires M>=32 and no actual-reuses option",result.stderr)
                    self.assertNotIn("cannot open GGUF",result.stderr)
                    self.assertNotIn("cannot open ConvRot",result.stderr)


if __name__ == "__main__":
    unittest.main()
