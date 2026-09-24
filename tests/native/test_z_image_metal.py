"""Native GPU parity for specialized Z-Image kernels; no weights/downloads."""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1",
                     "explicit Metal GPU test opt-in required")
class ZImageMetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["bash", "tools/native/build_z_image_metal_probe.sh"],
                       cwd=ROOT, check=True, capture_output=True, text=True)

    def test_fused_kernels(self):
        # Short context, unaligned synthetic tail, 512 image+text, 1024 image+text.
        for rows in (1, 33, 1056, 4128):
            for mode in ("inline", "precompute"):
                with self.subTest(rows=rows, mode=mode):
                    result = subprocess.run([str(ROOT / "build/native/z-image-metal-probe"),
                                             str(rows), "2", mode], cwd=ROOT, check=True,
                                            capture_output=True, text=True, timeout=120)
                    cases = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual(len(cases), 10)
                    self.assertTrue(all(case["passed"] for case in cases), cases)
                    self.assertEqual({x["dtype"] for x in cases[0]["cases"]},
                                     {"bf16", "fp16", "fp32"})
                    for case in cases[1:]:
                        if case.get("norm_mod"):
                            self.assertEqual(case["precompute_gate"],
                                             mode == "precompute" and case["dtype"] != "fp32")
                    self.assertEqual(sum(bool(case.get("gate_norm")) for case in cases), 3)

    def test_virtual_gate_norm_exact(self):
        for rows in (1, 33, 1024, 1056, 4096, 4128):
            for threads in (128, 256, 512):
                with self.subTest(rows=rows, threads=threads):
                    result = subprocess.run(
                        [str(ROOT / "build/native/z-image-gate-norm-probe"),
                         str(rows), str(threads), "2"], cwd=ROOT, check=True,
                        capture_output=True, text=True, timeout=120)
                    cases = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual({x["dtype"] for x in cases}, {"bf16", "fp16", "fp32"})
                    self.assertTrue(all(x["max_abs"] == 0 for x in cases))

    def test_virtual_norm_matches_vector_exactly(self):
        for threads, rows in ((128, 1), (128, 33), (128, 1056), (128, 4128),
                              (256, 1), (256, 33), (256, 1056), (256, 4096), (256, 4128),
                              (512, 1), (512, 33), (512, 1056), (512, 4128)):
            for mode in ("inline", "precompute"):
                with self.subTest(threads=threads, rows=rows, mode=mode):
                    result = subprocess.run(
                        [str(ROOT / "build/native/z-image-metal-probe"),
                         str(rows), "2", mode, "vector", str(threads)],
                        cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
                    cases = [json.loads(line) for line in result.stdout.splitlines()]
                    norms = [case for case in cases if case.get("norm_mod")]
                    self.assertEqual(len(norms), 6)
                    self.assertTrue(all(case["virtual_threads"] == threads and
                                        case["passed"] and case["max_abs"] == 0
                                        for case in norms), norms)

    def test_vector_norm_matches_scalar_fusion_exactly(self):
        for rows in (1, 33, 1056, 4128):
            for mode in ("inline", "precompute"):
                with self.subTest(rows=rows, mode=mode):
                    result = subprocess.run(
                        [str(ROOT / "build/native/z-image-metal-probe"),
                         str(rows), "2", mode, "vector"], cwd=ROOT,
                        check=True, capture_output=True, text=True, timeout=120)
                    cases = [json.loads(line) for line in result.stdout.splitlines()]
                    norms = [case for case in cases if case.get("norm_mod")]
                    self.assertEqual(len(norms), 6)
                    self.assertTrue(all(case["vector_norm"] for case in norms))
                    self.assertTrue(all(case["passed"] and case["max_abs"] == 0
                                        for case in norms), norms)


if __name__ == "__main__":
    unittest.main()
