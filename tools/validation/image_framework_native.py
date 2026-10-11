"""Shared C API request-wall observer; no model work occurs on import."""
import ctypes as C
import json
import time

from flux9_bf16_framework_screen import write_json


def generate_requests(library, model_id, model_path, requests, output, route):
    lib = C.CDLL(str(library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(pointer):
        value = C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:
            lib.tc_string_free(pointer)
        return value

    engine, error = C.c_void_p(), C.c_void_p()
    start = time.perf_counter()
    if lib.tc_engine_create_model(model_id.encode(), str(model_path).encode(), C.byref(engine), C.byref(error)):
        raise RuntimeError(consume(error))
    report = dict(engine=route, constructor_seconds=time.perf_counter() - start, runs=[])
    try:
        for index, request in enumerate(requests):
            write_json(output / f"request-{index}.json", request)
            result, error = C.c_void_p(), C.c_void_p()
            start = time.perf_counter()
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), None, None, C.byref(result), C.byref(error))
            wall = time.perf_counter() - start
            text, failure = consume(result), consume(error)
            if status:
                raise RuntimeError(failure)
            report["runs"].append(dict(index=index, request_wall_including_export=wall, metrics=json.loads(text)))
            write_json(output / "report.json", report)
    finally:
        lib.tc_engine_free(engine)
    return report
