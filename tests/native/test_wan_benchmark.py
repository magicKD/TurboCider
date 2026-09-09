"""CPU-only checks for benchmark parsing; no performance assertions."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'wan_benchmark', ROOT / 'tools/validation/wan/benchmark_native.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class WanBenchmarkTests(unittest.TestCase):
    def test_run_and_cache_sequence(self):
        text = '\n'.join(f'; seconds=2.5; run={i}; prompt_cache_hit={int(i > 0)}; '
                         'compiled=1; mlx_peak_bytes=123' for i in range(3))
        self.assertEqual(len(BENCH.parse_rows(text, True, 3)), 3)
        for broken in [text.replace('run=1', 'run=0'),
                       text.replace('prompt_cache_hit=1', 'prompt_cache_hit=0'),
                       text.replace('seconds=2.5', 'seconds=0')]:
            with self.assertRaises(ValueError):
                BENCH.parse_rows(broken, True, 3)

    def test_missing_or_wrong_mode_rejected(self):
        with self.assertRaises(ValueError):
            BENCH.parse_rows('', False, 3)
        with self.assertRaises(ValueError):
            BENCH.parse_rows('; seconds=2; run=0; prompt_cache_hit=0; '
                             'compiled=1; mlx_peak_bytes=123', False, 1)


if __name__ == '__main__':
    unittest.main()
