"""Validate encoder geometry and predeclared measurement protocol."""
import importlib.util
import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qwen_sweep', ROOT / 'tools/native/benchmark_qwen3_encoder_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def manifest(tmp_path, hidden, width, blocks):
    (tmp_path / 'fixture.mlmodelc').mkdir(exist_ok=True)
    value = {'schema_version': 2, 'shape': {
        'K': hidden, 'N': hidden, 'mlp_width': width, 'buckets': [64],
        'ane_mlp_start': 0, 'ane_mlp_end': width // 2,
    }, 'artifacts': {str(i): {'64': 'fixture.mlmodelc'} for i in range(blocks)}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value))
    return path


class GeometryTests(unittest.TestCase):
    def test_predeclared_settling_preserves_all_samples(self):
        original = {"warm_seconds": [100.0, 90.0, 2.0, 4.0], "first_encoder_seconds": 120.0}
        measured = sweep.apply_settling(original, 2, 2)
        self.assertEqual(measured["settling_seconds"], [100.0, 90.0])
        self.assertEqual(measured["warm_seconds"], [2.0, 4.0])
        self.assertEqual(measured["post_first_seconds"], original["warm_seconds"])
        self.assertEqual(measured["first_encoder_seconds"], 120.0)
        self.assertEqual(measured["warm_median_seconds"], 3.0)
        self.assertEqual(sweep.apply_settling(original, 0, 4)["warm_seconds"], original["warm_seconds"])
        for settling, runs in [(-1, 5), (4, 0), (2, 3)]:
            with self.subTest(settling=settling, runs=runs), self.assertRaises(ValueError):
                sweep.apply_settling(original, settling, runs)
        for invalid in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sweep.apply_settling({"warm_seconds": [invalid, 1.0]}, 1, 1)

    def test_predeclared_backend_order_rotates(self):
        self.assertEqual(sweep.backend_order(0, False, 'gpu'), ('gpu', 'hybrid'))
        self.assertEqual(sweep.backend_order(0, False, 'hybrid'), ('hybrid', 'gpu'))
        self.assertEqual(sweep.backend_order(1, False, 'hybrid'), ('gpu', 'hybrid'))
        self.assertEqual(sweep.backend_order(0, True, 'hybrid'), ('hybrid', 'mlx_reference', 'gpu'))
        self.assertEqual(sweep.backend_order(1, True, 'hybrid'), ('mlx_reference', 'gpu', 'hybrid'))
        with self.assertRaises(ValueError): sweep.backend_order(0, False, 'invalid')

    def test_settling_is_executed_by_every_backend(self):
        import argparse
        args = argparse.Namespace(probe=Path('/probe'), weights=Path('/weights'),
                                  runs=10, settling_runs=5, mode='z_image',
                                  encoder_ane_manifest=Path('/manifest'),
                                  mlx_python=Path('/python'), mlx_script=Path('/script'))
        for hybrid in [False, True]:
            self.assertEqual(sweep.command_for(args, 64, Path('/out'), hybrid)[3], '15')
        self.assertEqual(sweep.mlx_command_for(args, 64, Path('/out'))[4], '15')
        summary = dict(ane_calls_session_total=560, ane_first_runtime_calls_session_total=35,
                       ane_subsequent_runtime_calls_session_total=525,
                       coreml_block_count=35, runtime_failed=False, runtime_failures_session_total=0,
                       prefill_fixed_shape=True, prefill_actual_tokens=64)
        self.assertEqual(sweep.hybrid_execution_status(summary, 'z_image', 64, 15, [64]), (True, True))
        self.assertFalse(sweep.hybrid_execution_status(summary, 'z_image', 64, 10, [64])[0])

    def test_fixed_execution_and_flexible_qualification_are_separate(self):
        summary = dict(ane_calls_session_total=162,
                       ane_first_runtime_calls_session_total=27,
                       ane_subsequent_runtime_calls_session_total=135,
                       coreml_block_count=27, runtime_failed=False,
                       runtime_failures_session_total=0,
                       prefill_fixed_shape=True, prefill_actual_tokens=64,
                       qualified_flexible_backing=False)
        self.assertEqual(sweep.hybrid_execution_status(summary, 'flux_klein', 64, 5, [64]), (True, True))
        self.assertEqual(sweep.hybrid_execution_status(summary, 'flux_klein', 64, 5, [64, 128]), (True, False))
        summary['qualified_flexible_backing'] = True
        self.assertEqual(sweep.hybrid_execution_status(summary, 'flux_klein', 64, 5, [64, 128]), (True, True))
        summary['ane_calls_session_total'] = 161
        self.assertFalse(sweep.hybrid_execution_status(summary, 'flux_klein', 64, 5, [64])[0])
        summary['ane_calls_session_total'] = 162
        summary['runtime_failures_session_total'] = 1
        self.assertFalse(sweep.hybrid_execution_status(summary, 'flux_klein', 64, 5, [64])[0])

    def test_supported_and_rejected_geometry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for hidden, width in [(2560, 9728), (4096, 12288)]:
                with self.subTest(hidden=hidden, width=width):
                    value = sweep.load_compiled_manifest(manifest(root, hidden, width, 27), 'flux_klein')
                    self.assertEqual(value['shape']['K'], hidden)
            for hidden, width in [(2560, 12288), (4096, 9728)]:
                with self.subTest(hidden=hidden, width=width), self.assertRaisesRegex(ValueError, 'geometry'):
                    sweep.load_compiled_manifest(manifest(root, hidden, width, 27), 'flux_klein')
            with self.assertRaisesRegex(ValueError, 'geometry'):
                sweep.load_compiled_manifest(manifest(root, 4096, 12288, 35), 'z_image')
            with self.assertRaisesRegex(ValueError, 'missing'):
                sweep.load_compiled_manifest(manifest(root, 2560, 9728, 27), 'z_image')
            self.assertTrue(sweep.load_compiled_manifest(manifest(root, 2560, 9728, 35), 'z_image'))
            with self.assertRaisesRegex(ValueError, 'missing'):
                sweep.load_compiled_manifest(manifest(root, 2560, 9728, 1), 'flux_klein')


if __name__ == '__main__':
    unittest.main()
