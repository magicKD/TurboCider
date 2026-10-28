#!/usr/bin/env python3
"""Source-bound native image windows; stock timing and selector diagnostics separate.

A spec contains model/model_path, explicit environment, source_files and requests.
It cannot qualify arbitrary models or pixel quality. Full hashes/array comparisons
belong after all inference windows, never inside the measured request loop.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/validation'))
sys.path.insert(0, str(ROOT / 'tools/profiling/qwen_base'))
from collect import Observer, sha256_file, write_json, validate_fixed_program
from analyze import observer_contract
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled


def validate_spec(spec):
    model = spec['model']
    if model not in ('qwen-image-2.1', 'z-image-turbo'):
        raise ValueError('only the inspected native image families are admitted')
    requests = spec['requests']
    if not 1 <= len(requests) <= 8 or not spec.get('source_files'):
        raise ValueError('bounded explicit requests and original source files required')
    env = spec['environment']
    if any(not k.startswith('TURBOCIDER_') or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('only explicit string native experiment controls allowed')
    routes = {q['execution'] for q in requests}
    if len(routes) != 1 or not routes <= {'gpu', 'gpu_ane'}:
        raise ValueError('one route per process; compare independently initialized windows')
    for q in requests:
        if (q['model'] != model or q['operation'] not in ('image.generate', 'image.edit') or
                q['residency'] != 'resident' or q.get('dump_tensors') or
                type(q['steps']) is not int or not 1 <= q['steps'] <= 40 or
                any(type(q[k]) is not int or not 128 <= q[k] <= 2048 or q[k] % 16 for k in ('width', 'height'))):
            raise ValueError('explicit complete resident image request required')
        if model == 'z-image-turbo' and (q['operation'] != 'image.generate' or q.get('inputs') or q['steps'] != 8):
            raise ValueError('Z Turbo is T2I only and retains all eight steps')
        if routes == {'gpu_ane'} and (q.get('hybrid_mlp_mode') != 'runtime' or
                not q.get('allow_approximation') or not q.get('ane_manifest') or
                env.get('TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM') != '1'):
            raise ValueError('explicit approximation and fixed runtime program required')
    return routes == {'gpu_ane'}


def snapshot(paths):
    result = {}
    for name in paths:
        p = Path(name)
        s = p.stat()
        with p.open('rb') as f:
            prefix = f.read(65536)
        t = p.stat()
        fields = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if fields(s) != fields(t):
            raise ValueError('source generation changed during bounded snapshot')
        result[str(p.absolute())] = dict(canonical=str(p.resolve(strict=True)), generation=fields(s),
                                       prefix_sha256=hashlib.sha256(prefix).hexdigest())
    return result


def validate_run(report, spec, partial=False):
    runtime = validate_spec(spec)
    runs = report['runs']
    if not runs or len(runs) > len(spec['requests']) or (not partial and len(runs) != len(spec['requests'])):
        raise ValueError('every requested image must complete')
    previous_blocks = previous_calls = 0
    for i, row in enumerate(runs):
        m, q = row['metrics'], spec['requests'][i]
        keys = ('model', 'operation', 'width', 'height', 'steps', 'seed')
        if any(m.get(k) != q[k] for k in keys) or m.get('actual_denoise_steps') != q['steps']:
            raise ValueError('actual model/shape/sampler differs')
        if m.get('lora_applied_projections', 0) != spec.get('expected_lora_projections', 0):
            raise ValueError('actual LoRA binding differs')
        events = [x['event'] for x in report['progress'] if x['request'] == i and x['event'].get('phase') == 'denoise']
        # Z emits both the previous step's completion and next step's start;
        # Qwen emits one boundary. Preserve and check the actual family ABI.
        boundaries = ([0] + [v for j in range(1, q['steps']) for v in (j, j)] + [q['steps']]
                      if spec['model'] == 'z-image-turbo' else list(range(q['steps'] + 1)))
        if ([x['completed'] for x in events] != boundaries or
                any(x['total'] != q['steps'] for x in events)):
            raise ValueError('complete actual denoise progress required')
        wall = row['request_wall_including_export']
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError('finite complete request wall required')
        if not runtime:
            if m.get('hybrid') or m.get('encoder_hybrid'):
                raise ValueError('GPU control unexpectedly executed ANE')
            continue
        h = m.get('hybrid') or {}
        w = h.get('runtime_weight') or {}
        descriptor = json.loads(Path(q['ane_manifest']).read_text())
        blocks, calls = w.get('hybrid_blocks_session_total'), h.get('runtime_calls_session_total')
        if (h.get('bucket') != descriptor['rows'] or h.get('hidden') != descriptor['hidden'] or
                h.get('mlp_width') != descriptor['width'] or h.get('runtime_failed') is not False or
                w.get('executor_backend') != 'private_ane' or w.get('data_path') != 'w8a8_hadamard' or
                w.get('partition_axis') != 'intermediate_channels' or
                w.get('ane_channels') != int(spec['environment']['TURBOCIDER_PRIVATE_ANE_CHANNELS']) or
                any(w.get(k) != 0 for k in ('fallback_blocks_session_total', 'overflow_retries_session_total')) or
                w.get('headroom_scale') != 1 or type(blocks) is not int or blocks - previous_blocks != 32 * q['steps'] or
                type(calls) is not int or calls <= previous_calls):
            raise ValueError('healthy actual fixed-program channel FFN required; no fallback/retry credit')
        validate_fixed_program([m])
        previous_blocks, previous_calls = blocks, calls


def child(args, spec):
    observer = Observer(args.observer) if args.mode == 'timeline' else None
    lib = C.CDLL(str(args.library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_runtime_build_identity.restype = C.c_void_p
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    cb_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, cb_type, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def take(p):
        if not p.value:
            return ''
        try:
            return C.string_at(p).decode()
        finally:
            lib.tc_string_free(p)

    progress, errors = [], []
    current = -1
    @cb_type
    def callback(raw, _):
        try:
            progress.append(dict(request=current, event=json.loads(raw), at_ns=time.monotonic_ns()))
        except Exception as e:
            errors.append(str(e))

    report = dict(status='running', loaded_runtime_id=take(C.c_void_p(lib.tc_runtime_build_identity())),
                  runs=[], progress=progress)
    engine, error = C.c_void_p(), C.c_void_p()
    try:
        if observer:
            observer.lib.tc_qwen_profile_scope(-1)
        if lib.tc_engine_create_model(spec['model'].encode(), spec['model_path'].encode(), C.byref(engine), C.byref(error)):
            raise RuntimeError(take(error))
        for i, original in enumerate(spec['requests']):
            current = i
            q = dict(original, output=str(args.output / f'{i}.png'))
            write_json(args.output / f'request-{i}.json', q)
            if observer:
                observer.lib.tc_qwen_profile_scope(i)
            begin = observer.now() if observer else None
            out, error = C.c_void_p(), C.c_void_p()
            start = time.perf_counter()
            code = lib.tc_engine_generate(engine, json.dumps(q).encode(), callback, None, C.byref(out), C.byref(error))
            wall = time.perf_counter() - start
            end = observer.now() if observer else None
            value, failure = take(out), take(error)
            if code or errors:
                raise RuntimeError(failure or errors[0])
            if observer and observer.lib.tc_qwen_profile_wait(2000):
                raise RuntimeError('GPU completion observer did not drain')
            report['runs'].append(dict(index=i, metrics=json.loads(value), begin_ns=begin, end_ns=end,
                                       request_wall_including_export=wall))
            write_json(args.output / 'report.json', report)
            validate_run(report, spec, partial=True)
            print(f'complete request {i}: {wall:.6f}s', flush=True)
        report['status'] = 'complete'
    except Exception as e:
        report.update(status='failed', error=str(e))
        raise
    finally:
        if observer:
            observer.lib.tc_qwen_profile_scope(-3)
        if engine.value:
            lib.tc_engine_free(engine)
        if observer:
            write_json(args.output / 'observer.json', observer.take('tc_qwen_profile_finish'))
        write_json(args.output / 'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('library', 'spec', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--mode', choices=('stock', 'timeline'), default='stock')
    p.add_argument('--observer', type=Path, default=ROOT / 'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--timeout-seconds', type=float, default=1200)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    for key in ('library', 'spec', 'observer'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    a.output = a.output.absolute()
    spec = json.loads(a.spec.read_text())
    runtime = validate_spec(spec)
    if a.child:
        child(a, spec)
        return
    if not math.isfinite(a.timeout_seconds) or not 1 <= a.timeout_seconds <= 3600:
        p.error('owned deadline must be finite and bounded')
    if shutil.disk_usage(ROOT).free < (2 << 30) + (256 << 20):
        p.error('2GiB disk reserve plus bounded outputs required')
    manifest = json.loads((a.library.parent / 'runtime-build/runtime-build-manifest.json').read_text())
    def check_native():
        if any(sha256_file(ROOT / x['path']) != x['sha256'] for x in manifest['inputs']['sources']):
            raise ValueError('working native source differs from loaded library')
    check_native()
    sources = [__file__, str(ROOT / 'tools/profiling/qwen_base/collect.py'), str(ROOT / 'tools/profiling/qwen_base/analyze.py')]
    files = list(dict.fromkeys(spec['source_files'] + [str(a.library), str(a.spec), str(a.observer)] + sources +
                              [q['ane_manifest'] for q in spec['requests'] if q.get('ane_manifest')]))
    before = snapshot(files)
    a.output.mkdir(parents=True, exist_ok=False)
    trial = a.output / 'trial'
    trial.mkdir()
    summary = dict(status='running', mode=a.mode, runtime=runtime, qualification_passed=False,
                   spec=spec, source_snapshot=before, runtime_build_id=manifest['runtime_build_id'],
                   scope='full requests; bounded prefix/generation binding, not full checkpoint hashes or semantic quality')
    write_json(a.output / 'summary.json', summary)
    write_json(a.output / 'collector-source.json', {s: Path(s).read_text() for s in sources})
    load = LoadObservation(a.output / 'load.jsonl')
    env = {k: v for k, v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k != 'MLX_DISABLE_COMPILE'}
    env.update(spec['environment'])
    command = [sys.executable, '-B', __file__, '--child', '--library', str(a.library), '--spec', str(a.spec),
               '--mode', a.mode, '--observer', str(a.observer), '--output', str(trial)]
    try:
        with (a.output / 'stdout.log').open('x') as stdout, (a.output / 'stderr.log').open('x') as stderr:
            summary['memory'] = run_sampled(command, repo=ROOT, output=a.output, stem='trial', env=env,
                stdout=stdout, stderr=stderr, timeout=a.timeout_seconds, interval_ms=100, max_gap_ms=500, observer=load)
        report = json.loads((trial / 'report.json').read_text())
        validate_run(report, spec)
        if report['status'] != 'complete' or report['loaded_runtime_id'] != manifest['runtime_build_id']:
            raise ValueError('actual loaded native build or completion differs')
        if a.mode == 'timeline':
            summary['observer'] = observer_contract(json.loads((trial / 'observer.json').read_text()), report['runs'], runtime)
            if len(report['runs']) > 1 and runtime and not all(summary['observer'][k] for k in ('hot_no_compile_observed', 'hot_no_model_load_observed')):
                raise ValueError('actual hot compile/load observed')
        summary['png_sha256'] = {x.name: sha256_file(x) for x in sorted(trial.glob('*.png'))}
        if len(summary['png_sha256']) != len(spec['requests']):
            raise ValueError('every complete PNG required')
        check_native()
        if snapshot(files) != before:
            raise ValueError('source generation changed across window')
        summary['load'] = load.verify()
        summary['status'] = 'complete'
    except Exception as e:
        summary.update(status='failed', error=str(e))
    write_json(a.output / 'summary.json', summary)
    if summary['status'] != 'complete':
        raise RuntimeError(summary['error'])
    print(json.dumps(dict(status=summary['status'], output=str(a.output))))


if __name__ == '__main__':
    main()
