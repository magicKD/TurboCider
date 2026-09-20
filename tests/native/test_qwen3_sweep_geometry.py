"""The qualification tool must cover both Klein encoders without mixing geometry."""
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
