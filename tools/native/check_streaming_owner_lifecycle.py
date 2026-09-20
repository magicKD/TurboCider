#!/usr/bin/env python3
"""Opt-in, real-weight Flux/Z-Image streaming lifecycle check in a disposable process.

Quarantine mode requires a native test-hook build; success/cancel modes also
work with ordinary builds. An injected false drain result models unknown
completion; it does not hang the GPU. The unsafe case intentionally retains its
engine until this process exits. Run each mode in a separate process.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

from run_streaming_campaign import load_native_library, create_native_engine, consume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-id', choices=['flux2-klein-4b','z-image-turbo'], default='flux2-klein-4b')
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['success', 'cancel-retry', 'quarantine'], required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(args.request).read_text())
    request['outputs'][0]['path'] = str((output / 'image.png').resolve())
    config = dict(library=args.library, constructor='candidate', model_path=args.model,
                  model_id=args.model_id)
    lib = load_native_library(config)
    lib.tc_engine_cancel.argtypes = [C.c_void_p]
    lib.tc_engine_load.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_unload.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_prepare.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    fault = getattr(lib, 'tc_engine_test_streaming_drain_failure', None)
    retained = getattr(lib, 'tc_engine_test_streaming_retained_engines', None)
    if args.mode == 'quarantine' and (fault is None or retained is None):
        raise RuntimeError('Quarantine mode requires a native test-hook build')
    if fault is not None:
        fault.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_void_p)]
    if retained is not None:
        retained.restype = C.c_uint64
    engine = create_native_engine(lib, config)
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    events = []
    cancel = args.mode != 'success'
    block_phase = 'transformer_block' if args.model_id.startswith('flux') else 'z_image_denoise_block'

    @callback_type
    def event(raw, unused):
        value = json.loads(raw)
        events.append(value)
        if cancel and value['phase'] == block_phase:
            lib.tc_engine_cancel(engine)

    def call(function, *parameters):
        result, error = C.c_void_p(), C.c_void_p()
        start = time.perf_counter()
        status = function(*parameters, C.byref(result), C.byref(error))
        return dict(status=status, result=consume(lib, result), error=consume(lib, error),
                    wall_seconds=time.perf_counter() - start)

    report = dict(model=args.model_id, mode=args.mode, test_hooks_present=fault is not None, library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest())
    if args.mode == 'quarantine':
        error = C.c_void_p()
        status = lib.tc_engine_test_streaming_drain_failure(engine, 1, C.byref(error))
        failure = consume(lib, error)
        assert status == 0, failure
    payload = json.dumps(request).encode()
    first = call(lib.tc_engine_generate, engine, payload, event, None)
    report['first'] = first
    assert any(item['phase'] == block_phase for item in events), first
    if args.mode == 'quarantine':
        assert first['status'] == 1 and 'streaming_process_quarantined:' in first['error'], first
        assert 'cancel' in first['error'].lower() and 'GPU drain incomplete' in first['error'], first
        count = len(events)
        retries = {
            'generate': call(lib.tc_engine_generate, engine, payload, event, None),
            'load': call(lib.tc_engine_load, engine, event, None),
            'unload': call(lib.tc_engine_unload, engine),
            'prepare': call(lib.tc_engine_prepare, engine, payload, 1, event, None),
        }
        for name, value in retries.items():
            assert value['status'] == 1 and 'streaming_process_quarantined:' in value['error'], (name, value)
        assert count == len(events), 'Rejected work emitted callbacks'
        before = lib.tc_engine_test_streaming_retained_engines()
        lib.tc_engine_free(engine)
        engine = None
        assert lib.tc_engine_test_streaming_retained_engines() == before + 1
        # Creation is metadata-only; replacing the engine must not recover GPU use.
        replacement = create_native_engine(lib, config)
        value = call(lib.tc_engine_load, replacement, event, None)
        assert value['status'] == 1 and 'streaming_process_quarantined:' in value['error'], value
        lib.tc_engine_free(replacement)
        assert lib.tc_engine_test_streaming_retained_engines() == before + 2
        assert count == len(events)
        report.update(retries=retries, replacement_load=value, retained_engines=before + 2)
    else:
        if args.mode == 'cancel-retry':
            assert first['status'] == 2, first
            assert not (output / 'image.png').exists(), 'Cancelled request published an image'
            cancel = False
            report['retry'] = call(lib.tc_engine_generate, engine, payload, event, None)
            assert report['retry']['status'] == 0, report['retry']
        else:
            assert first['status'] == 0, first
        digest = hashlib.sha256((output / 'image.png').read_bytes()).hexdigest()
        report['image_sha256'] = digest
        if args.expected_sha256:
            assert digest == args.expected_sha256, digest
        lib.tc_engine_free(engine)
        engine = None
        if retained is not None:
            assert retained() == 0
    report['status'] = 'PASS'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
