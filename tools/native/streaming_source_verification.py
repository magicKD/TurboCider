"""Explicit native verification shared by worker tools; JSON never grants trust."""
import ctypes as c
import json
import time


def verify_native_sources(library, engine):
    try:
        verify = library.tc_engine_verify_streaming_sources_json
    except AttributeError as exc:
        raise RuntimeError("native library lacks explicit source verification") from exc
    verify.argtypes = [c.c_void_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    verify.restype = c.c_int
    result, error = c.c_void_p(), c.c_void_p()
    started = time.perf_counter()
    try:
        status = verify(engine, c.byref(result), c.byref(error))
        elapsed = time.perf_counter() - started
        failure = c.string_at(error).decode() if error.value else None
        if status:
            raise RuntimeError(f"source verification failed ({status}): {failure}")
        if not result.value:
            raise RuntimeError("source verification returned no proof report")
        report = json.loads(c.string_at(result).decode())
        if (report.get("status") != "verified" or
                report.get("proof_scope") != "native_process_generation" or
                not report.get("files")):
            raise RuntimeError("source verification returned an invalid proof report")
        return {"scope": "engine_setup", "wall_seconds": elapsed, "report": report}
    finally:
        if result.value:
            library.tc_string_free(result)
        if error.value:
            library.tc_string_free(error)
