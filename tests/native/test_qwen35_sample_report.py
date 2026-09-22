import copy
from pathlib import Path
import unittest

from tools.validation.qwen35_multimodal_sample_report import load_reference, validate, validate_partial


class SampleReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / 'references/Qwen-Image-2.1/prompt_rewrite/pe_core.py'
        if not source.is_file():
            raise unittest.SkipTest('official PE parser checkout unavailable')
        cls.reference = load_reference(source)

    def fixture(self):
        return dict(prompt_tokens=9, expanded_tokens=12, generated_tokens=3, max_new_tokens=16,
                    next_position=10, rope_delta=-2, cached_tokens=14, stopped_eos=True,
                    complete=True, parse_ok=True, quality_accepted=False, chunked_prefill=True,
                    prefill_seconds=1., decode_seconds=2., thinking='Preserve the shape.',
                    raw='Preserve the shape.</think>\n{"rewritten_prompt":"Make it red.","ratio_follow":"<image1>"}',
                    positive_prompt='Make it red.', wh_ratio='', ratio_follow='<image1>')

    def test_complete_official_parse(self):
        self.assertTrue(validate(self.fixture(), self.reference, True)['complete'])

    def test_truncated_thinking_is_not_complete(self):
        report = self.fixture()
        report.update(generated_tokens=1, max_new_tokens=1, cached_tokens=12, stopped_eos=False,
                      complete=False, parse_ok=False, raw='The', thinking='The',
                      positive_prompt='', ratio_follow='')
        self.assertFalse(validate(report, self.reference)['complete'])
        with self.assertRaisesRegex(ValueError, 'not complete'):
            validate(report, self.reference, True)

    def test_sampled_eos_evidence(self):
        report = self.fixture()
        report.update(generated_ids=[10, 11, 248044], eos_token_id=248044)
        self.assertTrue(validate(report, self.reference, True)['token_eos_checked'])
        for ids in ([10, 11, 12], [248044, 11, 248044], [10, 248044], [10, True, 248044]):
            with self.subTest(ids=ids):
                report['generated_ids'] = ids
                with self.assertRaises(ValueError):
                    validate(report, self.reference)

    def test_partial_report(self):
        partial = dict(status='running', complete=False, quality_accepted=False,
                       generated_tokens=2, generated_ids=[10, 11], max_new_tokens=8,
                       expanded_tokens=12, cached_tokens=14, raw='test', raw_utf8_base64='dGVzdA==')
        self.assertTrue(validate_partial(partial)['passed'])
        partial['cached_tokens'] = 15
        with self.assertRaises(ValueError):
            validate_partial(partial)

    def test_official_decode_verification(self):
        report = self.fixture()
        report.update(generated_ids=[10, 11, 248044], eos_token_id=248044)
        class Decoder:
            def __init__(self, text):
                self.text = text
            def decode(self, ids, skip_special_tokens):
                if ids != [10, 11, 248044] or skip_special_tokens:
                    raise AssertionError('decoder must retain sampled special tokens')
                return self.text
        result = validate(report, self.reference, True, Decoder(report['raw']))
        self.assertTrue(result['tokenizer_decode_checked'])
        with self.assertRaisesRegex(ValueError, 'decode differs'):
            validate(report, self.reference, True, Decoder('incorrect raw text'))
        del report['generated_ids']
        with self.assertRaisesRegex(ValueError, 'requires sampled IDs'):
            validate(report, self.reference, True, Decoder(report['raw']))

    def test_partial_invalid_utf8(self):
        partial = dict(status='running', complete=False, quality_accepted=False,
                       generated_tokens=1, generated_ids=[10], max_new_tokens=8,
                       expanded_tokens=12, cached_tokens=13, raw=None, raw_utf8_base64='/w==')
        self.assertTrue(validate_partial(partial)['passed'])
        partial['raw'] = 'replacement'
        with self.assertRaises(ValueError):
            validate_partial(partial)

    def test_rejects_inconsistent_reports(self):
        for key, value in dict(cached_tokens=15, rope_delta=0, next_position=13,
                               quality_accepted=True, generated_tokens=17, stopped_eos=False,
                               complete=False, positive_prompt='Different answer',
                               ratio_follow='<image2>', thinking='Different thinking',
                               prefill_seconds=float('nan'), max_new_tokens=24001,
                               prompt_tokens=True, parse_ok=1).items():
            with self.subTest(field=key):
                report = copy.deepcopy(self.fixture())
                report[key] = value
                with self.assertRaises(ValueError):
                    validate(report, self.reference)


if __name__ == '__main__':
    unittest.main()
