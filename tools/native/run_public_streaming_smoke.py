#!/usr/bin/env python3
"""Real public-semantics image smoke using an explicitly test-only catalog.

Requires a test-hook library. Template calibration/review fields in this catalog
are NOT release evidence. Run under collect_streaming_memory.py for full lifetime
sampling. No production catalog is installed or modified.
"""
import argparse
import ctypes as C
import hashlib
import json
import time
from pathlib import Path
from run_streaming_campaign import load_native_library, create_native_engine, consume


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.request.read_text())
    if (request.get('operation') != 'image.generate' or len(request.get('outputs', [])) != 1
            or request.get('execution', {}).get('streaming', {}).get('schema_version') != 2):
        raise ValueError('requires a schema-2 public image request')
    request['outputs'][0]['path'] = str(output / 'image.png')
    config = dict(library=str(args.library.resolve()), model_path=str(args.model.resolve()),
                  model_id=request['model'], constructor='public', execution_container='cli_worker',
                  verify_streaming_sources=True, test_streaming_catalog=str(args.catalog.resolve()))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    write(output / 'plan.json', dict(scope=__doc__, config=config, request=request,
          cancel_at_step=args.cancel_at_step, library_sha256=digest(args.library),
          catalog_sha256=digest(args.catalog), script_sha256=digest(Path(__file__))))
    library = load_native_library(config)
    library.tc_engine_resolve_streaming_json.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_cancel.argtypes = [C.c_void_p]
    engine = None
    started = time.monotonic()
    observation = dict(status='failed', cleanup_returned=False, cancellation_sent=False)
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    try:
        engine = create_native_engine(library, config)
        write(output / 'source-verification.json', library._tc_source_verification)
        result, error = C.c_void_p(), C.c_void_p()
        status = library.tc_engine_resolve_streaming_json(engine, json.dumps(request).encode(), C.byref(result), C.byref(error))
        raw, failure = consume(library, result), consume(library, error)
        write(output / 'resolve-result.json', dict(status=status, error=failure, result=json.loads(raw) if raw else None))
        if status:
            raise RuntimeError(failure)
        resolution = json.loads(raw)
        if resolution['status'] != 'resolved' or resolution['selection']['execution_container'] != 'cli_worker':
            raise ValueError('unexpected resolution')
        request['execution']['streaming'] = resolution['exact_selector']
        write(output / 'exact-request.json', request)
        callback_errors = []
        with (output / 'events.jsonl').open('w') as events:
            @callback_type
            def event(raw, unused):
                try:
                    value = json.loads(raw)
                    events.write(json.dumps(value) + '\n'); events.flush()
                    if (args.cancel_at_step is not None and value['phase'] == 'denoise'
                            and value['completed'] >= args.cancel_at_step and not observation['cancellation_sent']):
                        observation['cancellation_sent'] = True
                        observation['cancellation_wall_seconds'] = time.monotonic() - started
                        library.tc_engine_cancel(engine)
                except Exception as exc:
                    callback_errors.append(str(exc)); library.tc_engine_cancel(engine)
            result, error = C.c_void_p(), C.c_void_p()
            status = library.tc_engine_generate(engine, json.dumps(request).encode(), event, None, C.byref(result), C.byref(error))
            raw, failure = consume(library, result), consume(library, error)
        value = json.loads(raw) if raw else None
        write(output / 'result.json', dict(status=status, error=failure, result=value, callback_errors=callback_errors))
        if callback_errors:
            raise RuntimeError(str(callback_errors))
        if args.cancel_at_step is not None:
            if not observation['cancellation_sent'] or status == 0 or raw or 'cancel' not in failure.lower():
                raise RuntimeError('expected cancellation did not reject generation')
            if (output / 'image.png').exists():
                raise RuntimeError('cancelled generation left an image')
            observation['status'] = 'cancelled_as_expected'
        else:
            if status:
                raise RuntimeError(failure)
            summary = value['public_streaming']
            if (summary['actual_plan_verified'] is not True or
                    summary['resolution_digest'] != resolution['resolution_digest'] or
                    summary['record_digest'] != resolution['selection']['record_digest'] or
                    summary['actual_layout_digest'] != resolution['selection']['layout_digest'] or
                    summary['execution_container'] != 'cli_worker' or not summary['receipt_digest']):
                raise ValueError('actual execution receipt differs from resolution')
            from PIL import Image
            with Image.open(output / 'image.png') as image:
                image.load()
                if image.format != 'PNG' or image.size != (request['outputs'][0]['width'], request['outputs'][0]['height']):
                    raise ValueError('output image shape/format mismatch')
            observation.update(status='succeeded', image_sha256=digest(output / 'image.png'))
    except Exception as exc:
        observation['error'] = str(exc)
        raise
    finally:
        if engine is not None:
            library.tc_engine_free(engine)
            observation['cleanup_returned'] = True
        observation['wall_seconds'] = time.monotonic() - started
        if 'cancellation_wall_seconds' in observation:
            observation['cancellation_to_cleanup_seconds'] = observation['wall_seconds'] - observation['cancellation_wall_seconds']
        write(output / 'observation.json', observation)
    print(observation['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'model', 'request', 'catalog', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--cancel-at-step', type=int)
    args = parser.parse_args()
    if args.cancel_at_step is not None and args.cancel_at_step < 0:
        parser.error('--cancel-at-step must be nonnegative')
    run(args)
