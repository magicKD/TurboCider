import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from benchmark_ltx_query_cache import query_cache_shader


class QueryCacheTests(unittest.TestCase):
    def test_transform_changes_only_the_target_kernel(self):
        source = (ROOT / "native/models/ltx_runtime/ltx_shaders.metal").read_text()
        result = query_cache_shader(source)
        boundary = "kernel void ltx_sol_attention_tiled_bf16"
        self.assertEqual(source.split(boundary)[0], result.split(boundary)[0])
        self.assertEqual(result.count("query_tile = cached_query[dimension_tile];"), 2)
        self.assertEqual(result.count("cached_query[dimension_tile].template load"), 1)
        with self.assertRaises(ValueError):
            query_cache_shader(result)

    def test_stale_source_rejected(self):
        with self.assertRaises(ValueError):
            query_cache_shader("invalid shader")
