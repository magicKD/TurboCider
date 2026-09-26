#!/usr/bin/env python3
"""Fixed-container metadata discovery on the production (empty) catalog."""
import ctypes
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
library = ctypes.CDLL(str(Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR', root / 'build/native')) / 'libturbocider.dylib'))
library.tc_string_free.argtypes = [ctypes.c_void_p]
for name in ['tc_streaming_options_json', 'tc_worker_streaming_options_json']:
    function = getattr(library, name)
    function.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int

def call(name, request):
    output, error = ctypes.c_void_p(), ctypes.c_void_p()
    status = getattr(library, name)(json.dumps(request).encode(), ctypes.byref(output), ctypes.byref(error))
    try:
        return status, ctypes.string_at(output).decode() if output else None, ctypes.string_at(error).decode() if error else None
    finally:
        library.tc_string_free(output)
        library.tc_string_free(error)

for model in ['z-image-turbo', 'flux2-klein-4b']:
    request = json.loads((root / 'tests/fixtures/streaming/worker-request-v1.json').read_text())
    request['model'] = model
    request['sampling']['steps'] = 9 if model == 'z-image-turbo' else 4
    embedded = call('tc_streaming_options_json', request)
    worker = call('tc_worker_streaming_options_json', request)
    assert embedded[0] == worker[0] == 0, (embedded, worker)
    left, right = json.loads(embedded[1]), json.loads(worker[1])
    assert left['execution_container'] == 'embedded_app' and right['execution_container'] == 'cli_worker'
    assert right['schema_version'] == 2 and right['query_status'] == 'catalog_empty'
    assert len(right['targets']) == 5 and all(t['status'] == 'catalog_empty' for t in right['targets'])
    left['execution_container'] = 'cli_worker'
    assert left == right, 'metadata behavior changed beyond the explicit fixed container'
    for scope in ['execution', 'selector']:
        invalid = json.loads(json.dumps(request))
        destination = invalid['execution'] if scope == 'execution' else invalid['execution']['streaming']
        destination['execution_container'] = 'embedded_app'
        result = call('tc_worker_streaming_options_json', invalid)
        assert result[0] == 1 and result[1] is None and result[2], result
    print(model, 'fixed container, legacy metadata compatibility, JSON container override rejection PASS')
