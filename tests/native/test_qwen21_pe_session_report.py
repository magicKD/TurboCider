import copy
import unittest

from tools.validation.qwen21_pe_session_report import validate


class PESessionReportTests(unittest.TestCase):
    def fixture(self):
        request = dict(model='qwen-image-2.1', operation='image.edit', prompt='Make it matte',
                       output='edit.png', width=512, height=512, steps=40, seed=42, execution='gpu',
                       prompt_enhance=True, prompt_enhance_edit_experimental=True,
                       prompt_enhancer_path='pe', inputs=[dict(kind='image', role='reference', path='ref.png')])
        report = {key: request[key] for key in ('model', 'operation', 'output', 'width', 'height', 'steps', 'seed')}
        report.update(runtime_backend='mlx_cpp_metal', actual_denoise_steps=40, reference_tokens=4096,
                      warmup=False, plan=dict(prompt_enhance=True, prompt_enhance_edit_experimental=True,
                      prompt_enhancer_path='pe', execution='gpu', stages=[dict(id='prompt_enhance'),
                      dict(id='text_encode', dependencies=['prompt_enhance']), dict(id='denoise'), dict(id='export')]),
                      prompt_enhancement=dict(backend='native_qwen35_pe_i2i_experimental', complete=True,
                      experimental_edit=True, visual_precision='float32', quality_accepted=False,
                      applied_ratio=False, original_prompt='Make it matte', positive_prompt='Make the teapot matte.',
                      generated_tokens=123, chunked_prefill=True, wh_ratio='', ratio_follow='<image1>', seconds=100),
                      timings_seconds=dict(request_wall=200, text_encode=5, image_encode=5, denoise=70, vae_decode=10))
        return request, report

    def test_valid_does_not_claim_quality(self):
        result = validate(*self.fixture())
        self.assertTrue(result['passed'])
        self.assertFalse(result['quality_accepted'])
        self.assertFalse(result['standalone_rewrite_matched'])

    def test_wrong_pe_provenance(self):
        request, report = self.fixture()
        for key, value in dict(backend='native_qwen35_pe_t2i', complete=False, experimental_edit=False,
                               visual_precision='bfloat16', quality_accepted=True, applied_ratio=True,
                               original_prompt='different', positive_prompt=' ', generated_tokens=True,
                               chunked_prefill=1, ratio_follow='<image2>', seconds=float('nan')).items():
            with self.subTest(key=key):
                changed = copy.deepcopy(report)
                changed['prompt_enhancement'][key] = value
                with self.assertRaises(ValueError):
                    validate(request, changed)

    def test_final_request_and_timing_mismatches(self):
        request, report = self.fixture()
        for key, value in dict(width=1024, actual_denoise_steps=39, reference_tokens=0, warmup=True,
                               output='wrong.png', runtime_backend='python').items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate(request, {**report, key: value})
        report['timings_seconds']['request_wall'] = 100
        with self.assertRaisesRegex(ValueError, 'wall excludes'):
            validate(request, report)

    def test_opt_in_and_stage_dependency_required(self):
        request, report = self.fixture()
        with self.assertRaises(ValueError):
            validate({**request, 'prompt_enhance_edit_experimental':False}, report)
        report['plan']['stages'][1]['dependencies'] = []
        with self.assertRaisesRegex(ValueError, 'depend'):
            validate(request, report)

    def test_standalone_comparison_is_explicit(self):
        request, report = self.fixture()
        standalone = dict(complete=True, stopped_eos=True, parse_ok=True, fp32_visual=True,
                          quality_accepted=False, **{key:report['prompt_enhancement'][key] for key in
                          ('positive_prompt', 'generated_tokens', 'wh_ratio', 'ratio_follow')})
        self.assertTrue(validate(request, report, standalone)['standalone_rewrite_matched'])
        standalone['positive_prompt'] = 'Different rewrite'
        with self.assertRaisesRegex(ValueError, 'positive_prompt differs'):
            validate(request, report, standalone)

    def test_multiple_references_and_ratio_index(self):
        request, report = self.fixture()
        request['inputs'].append(dict(kind='image', role='reference', path='mask.png'))
        report['reference_tokens'] = 8192
        report['prompt_enhancement']['ratio_follow'] = '<image2>'
        result = validate(request, report)
        self.assertEqual(result['reference_count'], 2)
        self.assertFalse(result['quality_accepted'])
        for ratio in ('<image0>', '<image3>', '<image01>', 'image2', '<image2> extra'):
            with self.subTest(ratio=ratio):
                report['prompt_enhancement']['ratio_follow'] = ratio
                with self.assertRaisesRegex(ValueError, 'ratio_follow'):
                    validate(request, report)

    def test_reference_limit_and_roles(self):
        request, report = self.fixture()
        request['inputs'] = [dict(kind='image', role='reference', path=f'{i}.png') for i in range(10)]
        report['prompt_enhancement']['ratio_follow'] = '<image10>'
        self.assertEqual(validate(request, report)['reference_count'], 10)
        for inputs in ([], request['inputs'] + [request['inputs'][0]]):
            with self.subTest(count=len(inputs)):
                with self.assertRaisesRegex(ValueError, 'reference count'):
                    validate({**request, 'inputs': inputs}, report)
        for key, value in (('role', 'mask'), ('kind', 'video'), ('path', '')):
            with self.subTest(key=key):
                changed = copy.deepcopy(request)
                changed['inputs'][1][key] = value
                with self.assertRaisesRegex(ValueError, 'reference inputs'):
                    validate(changed, report)


if __name__ == '__main__':
    unittest.main()
