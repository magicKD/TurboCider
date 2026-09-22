#!/usr/bin/env python3
"""Compare a frozen legacy Z-Image library with the generic manual executor.

One engine/library per subprocess; serial ABBA sessions, first image excluded
from warm medians. Requires a native build exporting the candidate constructor.
The exact layout is explicit: a legacy memory budget is not a public tier or
an exact-request budget. No catalog is installed or changed.
"""
import argparse
import copy
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from benchmark_z_image_streaming import process_memory, vm_counters


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def worker(job):
    out = Path(job['output'])
    out.mkdir()
    lib = C.CDLL(job['library'])
    lib.tc_string_free.argtypes = [C.c_void_p]
    ctor = getattr(lib, 'tc_engine_create_model_candidate' if job['generic'] else 'tc_engine_create_model')
    ctor.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
                                     C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def take(pointer):
        if not pointer.value:
            return None
        value = C.string_at(pointer).decode()
        lib.tc_string_free(pointer)
        return value

    def create():
        engine, error = C.c_void_p(), C.c_void_p()
        status = ctor(b'z-image-turbo', job['model'].encode(), C.byref(engine), C.byref(error))
        message = take(error)
        if status:
            raise RuntimeError(message)
        return engine

    report = dict(job, runs=[], library_sha256=hashlib.sha256(Path(job['library']).read_bytes()).hexdigest(),
                  environment={k: v for k, v in os.environ.items() if k.startswith(('TURBOCIDER_', 'MLX_', 'TC_'))})
    engine = None
    try:
        if job['lifecycle'] == 'persistent':
            engine = create()
        for index in range(job['runs']):
            request = copy.deepcopy(job['request'])
            output = out / f'{index}.png'
            if job['generic']:
                request['outputs'][0]['path'] = str(output)
            else:
                request['output'] = str(output)
            samples, errors, stopped = [], [], threading.Event()
            def sample():
                while not stopped.is_set():
                    try:
                        samples.append(process_memory())
                    except Exception as error:
                        errors.append(str(error))
                    stopped.wait(.1)
            before = vm_counters()
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            full_start = time.perf_counter()
            try:
                if job['lifecycle'] == 'per_request':
                    engine = create()
                value, error = C.c_void_p(), C.c_void_p()
                start = time.perf_counter()
                status = lib.tc_engine_generate(engine, json.dumps(request, ensure_ascii=False).encode(),
                                               None, None, C.byref(value), C.byref(error))
                wall = time.perf_counter() - start
                raw, message = take(value), take(error)
                if job['lifecycle'] == 'per_request':
                    lib.tc_engine_free(engine)
                    engine = None
                full_wall = time.perf_counter() - full_start
            finally:
                stopped.set()
                sampler.join()
            after = vm_counters()
            row = {'index': index, 'status': status, 'error': message,
                   'wall_seconds': wall, 'create_generate_free_seconds': full_wall,
                   'metrics': json.loads(raw) if raw else None,
                   'sha256': hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
                   'process_samples': samples, 'sampling_errors': errors,
                   'peak_phys_footprint_bytes': max((x['phys_footprint_bytes'] for x in samples), default=None),
                   'system_vm_delta_bytes': {k: after[k] - before[k] for k in before}}
            report['runs'].append(row)
            write_json(out / 'report.json', report)
            if status:
                raise RuntimeError(message)
    finally:
        if engine:
            lib.tc_engine_free(engine)


def generic_request(legacy, args):
    if legacy.get('loras'):
        raise ValueError('LoRA is outside this comparison')
    execution = {'policy': legacy.get('execution', 'gpu'), 'streaming': {
        'enabled': True, 'schema_version': 1, 'selection': 'manual', 'retention': 'request',
        'stages': {'denoiser': {'residency': 'streamed', 'block_group_size': 1,
            'slot_count': args.slots, 'resident_prefix_blocks': args.prefix,
            'prefetch_distance': args.distance, 'io_workers': args.workers}}}}
    for key in ('ane_manifest', 'allow_approximation'):
        if key in legacy:
            execution[key] = legacy[key]
    return {'schema_version': 2, 'model': 'z-image-turbo', 'operation': 'image.generate',
            'inputs': [{'kind': 'text', 'role': 'prompt', 'text': legacy['prompt']}],
            'outputs': [{'kind': 'image', 'path': 'replaced-by-worker.png',
                         'width': legacy['width'], 'height': legacy['height']}],
            'sampling': {'seed': legacy['seed'], 'steps': legacy['steps']},
            'execution': execution,
            'parameters': {'dynamic_text': legacy.get('dynamic_text', True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-job', type=Path)
    parser.add_argument('--baseline-library', type=Path)
    parser.add_argument('--framework-library', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--request', type=Path, help='Existing legacy streamed request JSON')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prefix', type=int, default=10)
    parser.add_argument('--slots', type=int, default=3)
    parser.add_argument('--distance', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=2, help='Number of ABBA cycles')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--lifecycle', choices=['persistent', 'per_request'], default='persistent')
    args = parser.parse_args()
    if args.worker_job:
        worker(json.loads(args.worker_job.read_text()))
        return
    if not all((args.baseline_library, args.framework_library, args.model, args.request, args.output)):
        parser.error('library, model, request and output paths are required')
    if args.cycles < 1 or args.runs < 2:
        parser.error('use at least one ABBA cycle and two images per session')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    legacy = json.loads(args.request.read_text())
    if legacy.get('residency') != 'streamed':
        raise ValueError('expected an explicit legacy streamed request')
    generic = generic_request(legacy, args)
    order = ['legacy', 'generic', 'generic', 'legacy'] * args.cycles
    manifest = {'order': order, 'lifecycle': args.lifecycle, 'warmup_per_session': 1,
                'scope': 'same weights/workload/layout; manual candidate, not public tier qualification',
                'memory_scope': 'sampled process footprint + native MLX allocator; VM deltas are system-wide',
                'results': []}
    for index, arm in enumerate(order):
        job = {'generic': arm == 'generic', 'model': str(args.model.resolve()),
               'library': str((args.framework_library if arm == 'generic' else args.baseline_library).resolve()),
               'request': generic if arm == 'generic' else legacy,
               'runs': args.runs, 'lifecycle': args.lifecycle,
               'output': str(out / f'{index:02d}-{arm}')}
        job_path = out / f'{index:02d}-{arm}.json'
        write_json(job_path, job)
        print(f'START {index + 1}/{len(order)} {arm}', flush=True)
        with (out / f'{index:02d}-{arm}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker-job', str(job_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result_path = Path(job['output']) / 'report.json'
        report = json.loads(result_path.read_text())
        manifest['results'].append({'arm': arm, 'report': str(result_path)})
        write_json(out / 'sessions.json', manifest)
        print('DONE', arm, [round(x['wall_seconds'], 3) for x in report['runs']], flush=True)
    arms = {arm: [] for arm in ('legacy', 'generic')}
    hashes = set()
    for entry in manifest['results']:
        report = json.loads(Path(entry['report']).read_text())
        hashes.update(x['sha256'] for x in report['runs'])
        arms[entry['arm']].extend(report['runs'][1:])
    summary = {'all_png_bytes_identical': len(hashes) == 1, 'sha256': sorted(hashes), 'arms': {}}
    for arm, rows in arms.items():
        summary['arms'][arm] = {'warm_samples': len(rows),
            'wall_median_seconds': statistics.median(x['wall_seconds'] for x in rows),
            'denoise_median_seconds': statistics.median(x['metrics']['timings_seconds']['denoise'] for x in rows),
            'create_generate_free_median_seconds': statistics.median(x['create_generate_free_seconds'] for x in rows)}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['all_png_bytes_identical']:
        raise RuntimeError('output mismatch: inspect saved images and reports')


if __name__ == '__main__':
    main()
