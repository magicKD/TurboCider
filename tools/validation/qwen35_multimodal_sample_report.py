"""Check native PE-I2I report invariants and official answer parsing.

This is an offline validator, not inference or image-quality acceptance.
"""
import argparse
import base64
import importlib.util
import json
import math
from pathlib import Path
import sys


def load_reference(path):
    spec = importlib.util.spec_from_file_location('qwen21_pe_reference_core', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Native deliberately implements strict JSON, not optional json_repair.
    module.json_repair = None
    return module


def validate(report, reference, require_complete=False, tokenizer=None):
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    for key in ('prompt_tokens', 'expanded_tokens', 'generated_tokens',
                'max_new_tokens', 'next_position', 'rope_delta', 'cached_tokens'):
        check(type(report.get(key)) is int, f'{key} must be an integer')
    for key in ('complete', 'parse_ok', 'stopped_eos', 'quality_accepted', 'chunked_prefill'):
        check(type(report.get(key)) is bool, f'{key} must be boolean')
    check(0 < report['prompt_tokens'] <= report['expanded_tokens'], 'invalid prompt counts')
    check(0 < report['max_new_tokens'] <= 24000, 'invalid edit generation budget')
    check(0 < report['generated_tokens'] <= report['max_new_tokens'], 'invalid generated count')
    check(report['expanded_tokens'] + report['max_new_tokens'] <= 32768, 'context budget exceeded')
    check(0 < report['next_position'] <= report['expanded_tokens'], 'invalid mRoPE origin')
    check(report['rope_delta'] == report['next_position'] - report['expanded_tokens'], 'mRoPE delta mismatch')
    check(report['cached_tokens'] == report['expanded_tokens'] + report['generated_tokens'] - 1,
          'cache count mismatch')
    check(report['stopped_eos'] or report['generated_tokens'] == report['max_new_tokens'],
          'generation stopped without EOS before budget')
    check(report['complete'] == (report['stopped_eos'] and report['parse_ok']), 'completion flag mismatch')
    token_eos_checked = 'generated_ids' in report
    if token_eos_checked:
        ids = report['generated_ids']
        check(type(ids) is list and len(ids) == report['generated_tokens'], 'sampled token count mismatch')
        check(all(type(token) is int and 0 <= token < 248320 for token in ids), 'invalid sampled token IDs')
        check(type(report.get('eos_token_id')) is int and report['eos_token_id'] == 248044, 'wrong PE EOS ID')
        check(248044 not in ids[:-1], 'sampled tokens continue after EOS')
        check(report['stopped_eos'] == (ids[-1] == 248044), 'EOS flag differs from sampled tokens')
    check(not report['quality_accepted'], 'experimental probe cannot accept visual/edit quality')
    for key in ('prefill_seconds', 'decode_seconds'):
        check(type(report.get(key)) in (int, float) and math.isfinite(report[key]) and report[key] >= 0,
              f'invalid {key}')
    for key in ('raw', 'thinking', 'positive_prompt', 'wh_ratio', 'ratio_follow'):
        check(isinstance(report.get(key), str), f'{key} must be text')
    if tokenizer is not None:
        check(token_eos_checked, 'tokenizer verification requires sampled IDs')
        decoded = tokenizer.decode(report['generated_ids'], skip_special_tokens=False)
        check(decoded == report['raw'], 'official tokenizer decode differs from raw text')
    # The chat template already opens thinking before generation begins.
    thinking, answer = reference.split_thinking('<think>\n' + report['raw'])
    parsed = reference.parse_answer(answer, reference.get_profile('edit'))
    check(thinking == report['thinking'], 'official thinking split differs')
    for key in ('positive_prompt', 'wh_ratio', 'ratio_follow', 'parse_ok'):
        check(parsed[key] == report[key], f'official {key} differs')
    if require_complete:
        check(report['complete'], 'native rewrite is not complete')
    return dict(passed=True, complete=report['complete'], generated_tokens=report['generated_tokens'],
                token_eos_checked=token_eos_checked,
                tokenizer_decode_checked=tokenizer is not None,
                quality_accepted=False, scope='report consistency, available sampled IDs and official strict parser; not visual quality')


def validate_partial(report):
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    check(report.get('status') == 'running', 'partial report must be running')
    check(report.get('complete') is False and report.get('quality_accepted') is False,
          'partial report cannot claim completion or quality')
    for key in ('generated_tokens', 'max_new_tokens', 'expanded_tokens', 'cached_tokens'):
        check(type(report.get(key)) is int, f'partial {key} must be an integer')
    check(report['expanded_tokens'] > 0, 'empty partial prompt')
    ids = report.get('generated_ids')
    check(type(ids) is list and len(ids) == report.get('generated_tokens'), 'partial IDs/count mismatch')
    check(all(type(token) is int and 0 <= token < 248320 for token in ids), 'invalid partial token ID')
    check(type(report.get('max_new_tokens')) is int and 0 < report['max_new_tokens'] <= 24000,
          'invalid partial budget')
    check(0 < report['generated_tokens'] < report['max_new_tokens'], 'partial must be below budget')
    check(report['cached_tokens'] == report['expanded_tokens'] + report['generated_tokens'],
          'partial cache count mismatch')
    check(report['expanded_tokens'] + report['max_new_tokens'] <= 32768, 'partial context budget exceeded')
    check(248044 not in ids, 'partial output contains EOS but remains running')
    check(isinstance(report.get('raw_utf8_base64'), str) and report['raw_utf8_base64'],
          'partial raw bytes missing')
    raw = base64.b64decode(report['raw_utf8_base64'], validate=True)
    try:
        expected_raw = raw.decode('utf-8')
    except UnicodeDecodeError:
        expected_raw = None
    check(report.get('raw') == expected_raw, 'partial UTF-8/base64 output mismatch')
    return dict(passed=True, status='running', generated_tokens=report['generated_tokens'],
                token_eos_checked=True, scope='partial sampler evidence only; not a completed rewrite')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference-core', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path,
                        help='Optional official tokenizer.json for exact sampled-ID/raw decode validation (CPU only)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--require-complete', action='store_true')
    mode.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.partial and args.tokenizer:
        parser.error('--tokenizer is currently for terminal reports only')
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report = json.loads(args.report.read_text())
    result = validate_partial(report) if args.partial else validate(
        report, load_reference(args.reference_core), args.require_complete, tokenizer)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
