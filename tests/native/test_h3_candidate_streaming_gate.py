#!/usr/bin/env python3
"""Host-only proof that H3 exact execution is private-candidate scoped."""

from __future__ import annotations

import ctypes as C
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LIB = C.CDLL(str(ROOT / "build/native/libturbocider.dylib"))
LIB.tc_engine_create_model.argtypes = [
    C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
]
LIB.tc_engine_create_model_candidate.argtypes = LIB.tc_engine_create_model.argtypes
LIB.tc_engine_generate.argtypes = [
    C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
]
LIB.tc_engine_free.argtypes = [C.c_void_p]
LIB.tc_string_free.argtypes = [C.c_void_p]


def consume(pointer: C.c_void_p) -> str:
    if not pointer.value:
        return ""
    value = C.string_at(pointer).decode()
    LIB.tc_string_free(pointer)
    return value


def create(symbol, root: Path) -> C.c_void_p:
    engine, error = C.c_void_p(), C.c_void_p()
    status = symbol(
        b"minimax-h3-turbo", str(root).encode(),
        C.byref(engine), C.byref(error),
    )
    failure = consume(error)
    assert status == 0 and engine.value, failure
    return engine


def generate(engine: C.c_void_p, request: dict) -> tuple[int, str]:
    result, error = C.c_void_p(), C.c_void_p()
    status = LIB.tc_engine_generate(
        engine, json.dumps(request).encode(), None, None,
        C.byref(result), C.byref(error),
    )
    consume(result)
    return status, consume(error)


def request(slot_count: int = 2) -> dict:
    return {
        "schema_version": 2,
        "model": "minimax-h3-turbo",
        "operation": "video.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "gate"}],
        "outputs": [{
            "kind": "video", "path": "/tmp/not-generated-h3.mp4",
            "width": 64, "height": 64, "frames": 22, "fps": 24,
            "audio": False,
        }],
        "sampling": {"seed": 42, "steps": 4},
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
                    "slot_count": slot_count,
                    "resident_prefix_blocks": 0,
                    "prefetch_distance": 1 if slot_count > 1 else 0,
                    "io_workers": 1,
                }},
            },
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tc-h3-candidate-gate-") as raw:
        root = Path(raw)

        public = create(LIB.tc_engine_create_model, root)
        try:
            status, error = generate(public, request())
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
        finally:
            LIB.tc_engine_free(public)

        candidate = create(LIB.tc_engine_create_model_candidate, root)
        try:
            status, error = generate(candidate, request(slot_count=1))
            assert status != 0
            assert "streaming_route_unsupported: H3 exact candidate" in error, error

            status, error = generate(candidate, request())
            assert status != 0
            assert "streaming_layout_not_certified" not in error, error
            assert "streaming_route_unsupported: H3 exact candidate" not in error, error
        finally:
            LIB.tc_engine_free(candidate)

    api = (ROOT / "native/api/c_api.mm").read_text()
    session = (ROOT / "native/platform/apple/h3_session.mm").read_text()
    assert 'std::strcmp(id, "minimax-h3-turbo")' in api
    assert "create_h3_candidate(path)" in api
    assert "allow_experimental_streaming_" in session
    assert "private candidate constructor" in session
    assert "exact_streaming_finished" in session

    print("PASS H3 public gate remains fail-closed; private candidate validates the frozen exact tuple")


if __name__ == "__main__":
    main()
