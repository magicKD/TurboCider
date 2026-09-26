import copy
import sys
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from benchmark_ltx_attention_branch import VARIANTS, validate


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(schema="ltx-attention-branch-probe-v2", rows=65, runs=2,
                         heads=32, dim=128, synthetic=True, warmups=2,
                         variants=[])
        for name in VARIANTS:
            self.data["variants"].append(dict(name=name, total_seconds=4., samples=[4.,4.],
                qkv_seconds=1., gate_seconds=1., core_seconds=1., post_seconds=1.,
                relative_l2_vs_fused=0., max_abs_vs_fused=0.,
                relative_l2_vs_unfused_post=0., max_abs_vs_unfused_post=0.,
                changed_bf16_elements_vs_unfused_post=0))

    def test_complete_data(self):
        validate(self.data, 65, 2)

    def test_invalid_shape_or_missing_mode(self):
        with self.assertRaises(ValueError):
            validate(self.data, 66, 2)
        self.data["variants"].pop()
        with self.assertRaises(ValueError):
            validate(self.data, 65, 2)

    def test_nonfinite_and_incomplete_measurements(self):
        for key, value in (("total_seconds", 0), ("core_seconds", float("nan")),
                           ("relative_l2_vs_fused", float("inf")), ("samples", [1.])):
            data = copy.deepcopy(self.data)
            data["variants"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data, 65, 2)

    def test_invalid_fusion_comparison(self):
        for key, value in (("changed_bf16_elements_vs_unfused_post", -1),
                           ("changed_bf16_elements_vs_unfused_post", True),
                           ("relative_l2_vs_unfused_post", float("nan"))):
            data = copy.deepcopy(self.data)
            data["variants"][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data, 65, 2)

    @unittest.skipUnless((ROOT / "build/native/ltx-attention-branch-probe").exists(),
                         "build the native branch probe first")
    def test_native_rejects_invalid_arguments_before_gpu_creation(self):
        binary = str(ROOT / "build/native/ltx-attention-branch-probe")
        for rows, runs in (("0", "1"), ("-1", "1"), ("16385", "1"),
                           ("65junk", "1"), ("65", "0"), ("65", "101")):
            with self.subTest(rows=rows, runs=runs):
                result = subprocess.run([binary, "missing-shader", rows, runs],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)
