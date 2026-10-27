"""Host-only fail-closed contract checks; no model/hardware import side effects."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('runtime_image_collect', ROOT / 'tools/profiling/runtime_image_collect.py')
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


def fixture():
    return dict(model='z-image-turbo', source_files=['original.safetensors'], environment={}, requests=[
        dict(model='z-image-turbo', operation='image.generate', execution='gpu', residency='resident',
             width=768, height=512, steps=8, seed=73)])


class RuntimeImageCollectTests(unittest.TestCase):
    def test_synthetic_fixture_is_explicit_nonzero_and_reproducible(self):
        sys.path.insert(0, str(ROOT / 'tools/profiling'))
        from runtime_image_specs import synthetic_adapter
        import json
        import struct
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.safetensors', Path(d) / 'b.safetensors'
            synthetic_adapter(a)
            synthetic_adapter(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            raw = a.read_bytes()
            n = struct.unpack('<Q', raw[:8])[0]
            h = json.loads(raw[8:8+n])
            self.assertIn('not-trained', h['__metadata__']['fixture'])
            self.assertEqual(len(h) - 1, 6)
            self.assertTrue(any(raw[8+n:]))
            for key, value in h.items():
                if key != '__metadata__':
                    self.assertEqual(value['dtype'], 'BF16')
                    self.assertIn(2, value['shape'])
            with self.assertRaises(FileExistsError): synthetic_adapter(a)

    def test_no_contaminated_diagnostic_shape_or_synthetic_speed_ratio(self):
        sys.path.insert(0, str(ROOT / 'tools/profiling'))
        from runtime_image_audit import pair_speed
        a = fixture()
        a['requests'] *= 3
        s = dict(status='complete', mode='stock', spec=a, report=dict(runs=[dict(request_wall_including_export=t) for t in (9., 8., 8.)]))
        faster = copy.deepcopy(s)
        faster['report']['runs'][1:]=[dict(request_wall_including_export=4.)]*2
        self.assertEqual(pair_speed(s, faster)['ratio'], 2.)
        for kind in ('failed', 'timeline', 'shape', 'synthetic'):
            bad = copy.deepcopy(faster)
            if kind == 'failed': bad['status'] = 'failed'
            if kind == 'timeline': bad['mode'] = 'timeline'
            if kind == 'shape': bad['spec']['requests'][1] = dict(bad['spec']['requests'][1], width=1024)
            if kind == 'synthetic': bad['spec']['expected_lora_projections'] = 3
            self.assertIsNone(pair_speed(s, bad))

    def test_no_reference_or_shortened_z_qualification(self):
        screen.validate_spec(fixture())
        for edit in (dict(operation='image.edit'), dict(inputs=[{'kind': 'image'}]), dict(steps=4)):
            bad = fixture()
            bad['requests'][0].update(edit)
            with self.assertRaises(ValueError):
                screen.validate_spec(bad)

    def test_explicit_fixed_program_and_one_route(self):
        a = fixture()
        a['requests'][0].update(execution='gpu_ane', hybrid_mlp_mode='runtime', allow_approximation=True, ane_manifest='fixed.json')
        with self.assertRaises(ValueError):
            screen.validate_spec(a)
        a['environment']['TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM'] = '1'
        self.assertTrue(screen.validate_spec(a))
        a['requests'].append(fixture()['requests'][0])
        with self.assertRaises(ValueError):
            screen.validate_spec(a)

    def test_no_external_loader_environment(self):
        a = fixture()
        a['environment']['DYLD_INSERT_LIBRARIES'] = 'unrelated'
        with self.assertRaises(ValueError):
            screen.validate_spec(a)

    def test_actual_steps_progress_and_lora_required(self):
        a = fixture()
        m = dict(a['requests'][0], actual_denoise_steps=8, lora_applied_projections=0)
        boundaries = [0] + [v for i in range(1, 8) for v in (i, i)] + [8]
        r = dict(runs=[dict(metrics=m, request_wall_including_export=1.)], progress=[
            dict(request=0, event=dict(phase='denoise', completed=i, total=8)) for i in boundaries])
        screen.validate_run(r, a)
        for kind in ('steps', 'progress', 'lora', 'gpu'):
            bad = copy.deepcopy(r)
            if kind == 'steps': bad['runs'][0]['metrics']['actual_denoise_steps'] = 7
            if kind == 'progress': bad['progress'].pop()
            if kind == 'lora': bad['runs'][0]['metrics']['lora_applied_projections'] = 3
            if kind == 'gpu': bad['runs'][0]['metrics']['hybrid'] = {'enabled': True}
            with self.assertRaises(ValueError): screen.validate_run(bad, a)


if __name__ == '__main__':
    unittest.main()
