import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from benchmark_ltx_selective_summaries import selective_summary_shader


class SelectiveSummaryTests(unittest.TestCase):
    def test_only_summary_kernel_changes(self):
        source = (ROOT / "native/models/ltx_runtime/ltx_shaders.metal").read_text()
        result = selective_summary_shader(source)
        start = "kernel void ltx_sol_reduce_summaries_bf16("
        end = "kernel void ltx_sol_thresholds_diag_bf16("
        self.assertEqual(source.split(start)[0], result.split(start)[0])
        self.assertEqual(source.split(end)[1], result.split(end)[1])
        self.assertIn("if (args.mode == 1u) return;", result)
        self.assertIn("if (args.mode != 2u) query_sum", result)
        self.assertIn("if (args.mode != 4u) value_sum", result)
        with self.assertRaises(ValueError):
            selective_summary_shader(result)

    def test_stale_source_rejected(self):
        with self.assertRaises(ValueError):
            selective_summary_shader("invalid shader")
