"""Native bindings shared by the candidate streaming gate tests."""

import ctypes as C
import os
from pathlib import Path


def load_candidate_library(root: Path) -> C.CDLL:
    native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", root / "build/native")).resolve()
    library = C.CDLL(str(native / "libturbocider.dylib"))
    library.tc_engine_create_model.argtypes = [
        C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    library.tc_engine_create_model_candidate.argtypes = library.tc_engine_create_model.argtypes
    library.tc_engine_generate.argtypes = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    library.tc_engine_prepare.argtypes = [
        C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    library.tc_engine_resolve_streaming_json.argtypes = [
        C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    library.tc_engine_free.argtypes = [C.c_void_p]
    library.tc_string_free.argtypes = [C.c_void_p]
    return library
