"""CPU-only integrated experimental PE-I2I report checks, not image quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def validate(request, report, standalone=None):
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    check(request.get('model') == report.get('model') == 'qwen-image-2.1', 'wrong model')
    check(request.get('operation') == report.get('operation') == 'image.edit', 'not an edit')
    check(request.get('prompt_enhance') is True and request.get('prompt_enhance_edit_experimental') is True,
          'missing explicit experimental PE request')
    check(isinstance(request.get('prompt_enhancer_path'), str) and bool(request['prompt_enhancer_path']),
          'missing PE installation')
    check(request.get('execution') == 'gpu' and report.get('runtime_backend') == 'mlx_cpp_metal',
          'expected native GPU edit')
    refs = request.get('inputs')
    check(isinstance(refs, list) and 1 <= len(refs) <= 10, 'invalid reference count')
    check(all(isinstance(ref, dict) and ref.get('kind') == 'image' and ref.get('role') == 'reference'
              and isinstance(ref.get('path'), str) and ref['path'] for ref in refs), 'invalid reference inputs')
    for key in ('width', 'height', 'steps', 'seed'):
        check(type(request.get(key)) is int and type(report.get(key)) is int and
              request[key] == report[key], f'{key} differs from request')
    check(1 <= request['steps'] <= 50 and request['seed'] >= 0, 'invalid sampling settings')
    check(all(64 <= request[key] <= 4096 and request[key] % 32 == 0 for key in ('width', 'height'))
          and request['width'] * request['height'] <= 8388608, 'invalid canvas')
    check(type(report.get('actual_denoise_steps')) is int and report['actual_denoise_steps'] == request['steps'],
          'denoise did not finish requested steps')
    check(report.get('warmup') is False and not report.get('prepared', False), 'not a final generation report')
    check(report.get('output') == request.get('output') and isinstance(report.get('output'), str)
          and bool(report['output']), 'output differs from request')
    check(type(report.get('reference_tokens')) is int and report['reference_tokens'] > 0,
          'missing reference conditioning')
    plan = report.get('plan', {})
    check(plan.get('prompt_enhance') is True and plan.get('prompt_enhance_edit_experimental') is True,
          'experimental PE route lost in plan')
    check(plan.get('prompt_enhancer_path') == request['prompt_enhancer_path'], 'PE installation differs')
    check(plan.get('execution') == 'gpu', 'unexpected runtime execution')
    stages = plan.get('stages', [])
    check(isinstance(stages, list) and all(isinstance(s, dict) for s in stages), 'invalid stage graph')
    ids = [s.get('id') for s in stages]
    check(len(ids) == len(set(ids)) and all(key in ids for key in ('prompt_enhance', 'text_encode', 'denoise', 'export')),
          'missing or repeated PE generation stages')
    text_stage = stages[ids.index('text_encode')]
    check('prompt_enhance' in text_stage.get('dependencies', []), 'text encoder does not depend on PE')
    pe = report.get('prompt_enhancement', {})
    check(pe.get('backend') == 'native_qwen35_pe_i2i_experimental' and pe.get('complete') is True,
          'missing completed native PE-I2I')
    check(pe.get('experimental_edit') is True and pe.get('visual_precision') == 'float32',
          'PE precision/provenance is not experimental FP32')
    check(pe.get('quality_accepted') is False, 'experimental route cannot accept quality')
    check(pe.get('applied_ratio') is False, 'request canvas must remain explicit')
    check(pe.get('original_prompt') == request.get('prompt') and isinstance(request.get('prompt'), str)
          and bool(request['prompt'].strip()), 'original prompt differs')
    check(isinstance(pe.get('positive_prompt'), str) and bool(pe['positive_prompt'].strip()), 'empty rewritten prompt')
    check(type(pe.get('generated_tokens')) is int and 0 < pe['generated_tokens'] <= 24000, 'invalid PE token count')
    check(type(pe.get('chunked_prefill')) is bool, 'invalid prefill mode')
    check(isinstance(pe.get('wh_ratio'), str) and isinstance(pe.get('ratio_follow'), str), 'missing ratio fields')
    if pe['ratio_follow']:
        match = re.fullmatch(r'<image([1-9][0-9]*)>', pe['ratio_follow'])
        check(match is not None and int(match.group(1)) <= len(refs), 'ratio_follow refers to a missing image')
    times = report.get('timings_seconds', {})
    for key, value in [('prompt_enhancement', pe.get('seconds'))] + [
            (key, times.get(key)) for key in ('request_wall', 'text_encode', 'image_encode', 'denoise', 'vae_decode')]:
        check(type(value) in (float, int) and math.isfinite(value) and value >= 0, f'invalid {key} timing')
    check(pe['seconds'] > 0 and times['denoise'] > 0, 'missing inference timing')
    accounted = pe['seconds'] + sum(times[key] for key in ('text_encode', 'image_encode', 'denoise', 'vae_decode'))
    check(accounted <= times['request_wall'] + 0.001, 'request wall excludes recorded stages')
    if standalone is not None:
        check(all(standalone.get(key) is True for key in ('complete', 'stopped_eos', 'parse_ok', 'fp32_visual')),
              'standalone rewrite is not complete FP32 evidence')
        check(standalone.get('quality_accepted') is False, 'standalone report claims quality')
        for key in ('positive_prompt', 'generated_tokens', 'wh_ratio', 'ratio_follow'):
            check(pe[key] == standalone.get(key), f'Session/standalone {key} differs')
    return dict(passed=True, reference_count=len(refs), generated_tokens=pe['generated_tokens'],
                standalone_rewrite_matched=standalone is not None, quality_accepted=False,
                scope='integrated report contract only; not sampled-ID proof, input-tensor identity, image quality or ANE placement')


def main():
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--standalone-rewrite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request, report = (json.loads(path.read_text()) for path in (args.request, args.report))
    standalone = json.loads(args.standalone_rewrite.read_text()) if args.standalone_rewrite else None
    result = validate(request, report, standalone)
    # Native request paths are relative to the invocation directory, not to
    # the JSON's directory. Invoke this validator from the same working dir.
    path = Path(report['output'])
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGBA' or image.size != (request['width'], request['height']):
            raise ValueError('PNG mode/dimensions do not match native RGBA output')
        image.verify()
    result['image'] = dict(path=str(path), mode='RGBA', width=request['width'], height=request['height'],
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
