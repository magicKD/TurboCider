#!/usr/bin/env python3
"""Compare fixed K2 Z-Image D/Q tuples with a predeclared mirrored order.

Exploratory candidate-only measurement, not public preset qualification. Every
request uses a fresh engine in one process; F_NOCACHE and numeric kernels are
unchanged. All images must match an independently supplied baseline digest.
"""
import argparse
import ctypes as C
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_streaming_campaign import load_native_library, create_native_engine, consume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error('--rounds must be positive even and at least 2')
    if len(args.expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_sha256):
        parser.error('--expected-sha256 must be a lowercase SHA-256 digest')
    request = json.loads(args.request.read_text())
    stage = request['execution']['streaming']['stages']['denoiser']
    assert request['model'] == 'z-image-turbo'
    assert request['execution']['policy'] == 'gpu'
    assert stage['slot_count'] == 2 and stage['block_group_size'] == 1
    assert request['execution']['streaming']['selection'] == 'manual'
    assert len(request['outputs']) == 1 and request['outputs'][0]['kind'] == 'image'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence = [(0, 1), (1, 2), (1, 1), (0, 2)]
    schedule = [pair for index in range(args.rounds)
                for pair in (sequence if index % 2 == 0 else sequence[::-1])]
    config = dict(library=str(args.library.resolve()), model_path=str(args.model.resolve()),
                  model_id='z-image-turbo', constructor='candidate')
    plan = dict(schema_version=1, scope=__doc__, request=request, config=config,
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                warmup=[0, 1], measured_schedule=schedule,
                expected_image_sha256=args.expected_sha256,
                timing_scope='engine creation through safe engine destruction, including export; artifact hash and audit serialization excluded')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    lib = load_native_library(config)
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    rows = []
    for index, (distance, workers) in enumerate([(0, 1), *schedule]):
        folder = output / ('warmup' if index == 0 else f'run-{index:02d}')
        folder.mkdir()
        current = deepcopy(request)
        current['outputs'][0]['path'] = str(folder / 'image.png')
        current_stage = current['execution']['streaming']['stages']['denoiser']
        current_stage.update(prefetch_distance=distance, io_workers=workers)
        if lib._tc_streaming_audit_reset:
            lib._tc_streaming_audit_reset()
        engine = None
        with (folder / 'events.jsonl').open('w') as events:
            @callback_type
            def event(raw, unused):
                events.write(raw.decode() + '\n')
            started = time.perf_counter()
            try:
                engine = create_native_engine(lib, config)
                result, error = C.c_void_p(), C.c_void_p()
                status = lib.tc_engine_generate(engine, json.dumps(current).encode(), event, None,
                                               C.byref(result), C.byref(error))
                value, failure = consume(lib, result), consume(lib, error)
            finally:
                if engine is not None:
                    lib.tc_engine_free(engine)
            wall = time.perf_counter() - started
        row = dict(index=index, warmup=index == 0, distance=distance, workers=workers,
                   wall_seconds=wall, status=status, error=failure,
                   result=json.loads(value) if value else None)
        (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        assert status == 0, row
        actual = row['result']['block_streaming']['actual_layout']
        assert actual['prefetch_distance'] == distance and actual['io_workers'] == workers
        assert actual['slot_count'] == 2 and actual['drained'] is True
        assert row['result']['block_streaming']['request_slot_fills'] == actual['group_count'] * actual['pass_count']
        row['image_sha256'] = hashlib.sha256((folder / 'image.png').read_bytes()).hexdigest()
        assert row['image_sha256'] == args.expected_sha256, row
        if lib._tc_streaming_audit_snapshot:
            value, error = C.c_void_p(), C.c_void_p()
            code = lib._tc_streaming_audit_snapshot(C.byref(value), C.byref(error))
            payload, failure = consume(lib, value), consume(lib, error)
            assert code == 0, failure
            row['audit'] = json.loads(payload)
        (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        if index:
            rows.append(row)
        print('PASS', index, f'D{distance}/Q{workers}', f'{wall:.6f}s', flush=True)
    summary = dict(schema_version=1, scope='Exploratory small-sample tuple comparison; no release qualification',
                   library_sha256=plan['library_sha256'], image_sha256=args.expected_sha256,
                   measured_requests=len(rows), variants=[])
    for distance, workers in sequence:
        selected = [row for row in rows if (row['distance'], row['workers']) == (distance, workers)]
        walls = [row['wall_seconds'] for row in selected]
        summary['variants'].append(dict(distance=distance, workers=workers, count=len(selected),
            wall_seconds=walls, wall_median=statistics.median(walls),
            denoise_seconds=[row['result']['timings_seconds']['denoise'] for row in selected],
            wait_seconds=[row['result']['block_streaming']['request_wait_seconds'] for row in selected]))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
