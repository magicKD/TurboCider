"""Native bindings shared by the candidate streaming gate tests."""

import ctypes as C
import json
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


class CandidateStreamingGate:
    def __init__(self, root: Path, model: str):
        self.library = load_candidate_library(root)
        self.model_id = model.encode()

    def _consume(self, pointer: C.c_void_p) -> str:
        if not pointer.value:
            return ""
        value = C.string_at(pointer).decode()
        self.library.tc_string_free(pointer)
        return value

    def create(self, root: Path, *, candidate: bool = False) -> C.c_void_p:
        constructor = (self.library.tc_engine_create_model_candidate if candidate
                       else self.library.tc_engine_create_model)
        engine, error = C.c_void_p(), C.c_void_p()
        status = constructor(
            self.model_id, str(root).encode(), C.byref(engine), C.byref(error),
        )
        failure = self._consume(error)
        assert status == 0 and engine.value, failure
        return engine

    def free(self, engine: C.c_void_p) -> None:
        self.library.tc_engine_free(engine)

    def _call(self, function, *arguments, result_output=True) -> tuple[int, str, str]:
        result, error = C.c_void_p(), C.c_void_p()
        result_pointer = C.byref(result) if result_output else None
        status = function(*arguments, result_pointer, C.byref(error))
        return status, self._consume(result), self._consume(error)

    def generate(self, engine: C.c_void_p, request: dict) -> tuple[int, str, str]:
        return self._call(
            self.library.tc_engine_generate, engine, json.dumps(request).encode(),
            None, None,
        )

    def prepare(self, engine: C.c_void_p, request: dict) -> tuple[int, str, str]:
        return self._call(
            self.library.tc_engine_prepare, engine, json.dumps(request).encode(),
            0, None, None,
        )

    def resolve(self, engine: C.c_void_p, request: dict) -> tuple[int, str, str]:
        return self.resolve_raw(engine, json.dumps(request).encode())

    def resolve_raw(self, engine, request_json, result_output=True) -> tuple[int, str, str]:
        return self._call(
            self.library.tc_engine_resolve_streaming_json, engine, request_json,
            result_output=result_output,
        )


def image_generation_request(model: str, prompt: str, output_path: str) -> dict:
    return {
        "schema_version": 2,
        "model": model,
        "operation": "image.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": prompt}],
        "outputs": [{
            "kind": "image", "path": output_path,
            "width": 64, "height": 64,
        }],
        "sampling": {"seed": 42, "steps": 1},
        "execution": {
            "policy": "gpu",
            "streaming": {
                "schema_version": 1,
                "enabled": True,
                "selection": "manual",
                "retention": "request",
                "stages": {"denoiser": {
                    "residency": "streamed",
                    "block_group_size": 1,
                    "slot_count": 2,
                    "resident_prefix_blocks": 0,
                    "prefetch_distance": 0,
                    "io_workers": 1,
                }},
            },
        },
    }


def memory_tier_request(value: dict) -> dict:
    value["execution"]["streaming"] = {
        "schema_version": 2,
        "enabled": True,
        "selection": "memory_tier",
        "retention": "request",
        "target_request_memory_bytes": 12 << 30,
    }
    return value
