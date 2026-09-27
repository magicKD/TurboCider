#!/usr/bin/env python3
"""Host-only proof that Z-Image generic streaming is candidate scoped."""

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
LIB.tc_engine_prepare.argtypes = [
    C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
]
LIB.tc_engine_resolve_streaming_json.argtypes = [
    C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
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
        b"z-image-turbo", str(root).encode(),
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


def prepare(engine: C.c_void_p, request: dict) -> tuple[int, str]:
    result, error = C.c_void_p(), C.c_void_p()
    status = LIB.tc_engine_prepare(
        engine, json.dumps(request).encode(), 0, None, None,
        C.byref(result), C.byref(error),
    )
    consume(result)
    return status, consume(error)


def resolve(engine: C.c_void_p, request: dict) -> tuple[int, str]:
    result, error = C.c_void_p(), C.c_void_p()
    status = LIB.tc_engine_resolve_streaming_json(
        engine, json.dumps(request).encode(),
        C.byref(result), C.byref(error),
    )
    value, failure = consume(result), consume(error)
    if status == 0:
        assert value and not failure, (value, failure)
    else:
        assert not value and failure, (value, failure)
    return status, failure


def resolve_raw(engine, request_json, result_output=True) -> tuple[int, str, str]:
    result, error = C.c_void_p(), C.c_void_p()
    result_pointer = C.byref(result) if result_output else None
    status = LIB.tc_engine_resolve_streaming_json(
        engine, request_json, result_pointer, C.byref(error),
    )
    return status, consume(result), consume(error)


def request() -> dict:
    return {
        "schema_version": 2,
        "model": "z-image-turbo",
        "operation": "image.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "gate"}],
        "outputs": [{
            "kind": "image", "path": "/tmp/not-generated-z-image.png",
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


def selector_request() -> dict:
    value = request()
    value["execution"]["streaming"] = {
        "schema_version": 2,
        "enabled": True,
        "selection": "memory_tier",
        "retention": "request",
        "target_request_memory_bytes": 12 << 30,
    }
    return value


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tc-z-image-candidate-gate-") as raw:
        root = Path(raw)
        for relative in (
            "split_files/text_encoders/qwen_3_4b.safetensors",
            "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "split_files/vae/ae.safetensors",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        tokenizer = root / "tokenizer/tokenizer.json"
        tokenizer.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.write_text(json.dumps({
            "model": {"type": "BPE", "vocab": {"g": 0}, "merges": []},
            "pre_tokenizer": {
                "pretokenizers": [{"pattern": {"Regex": "."}}]
            },
            "added_tokens": [],
        }))

        public = create(LIB.tc_engine_create_model, root)
        try:
            status, error = generate(public, request())
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
            status, error = generate(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
            status, error = resolve(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error

            # Request-only validation and engine identity precede the empty
            # production catalog. These fixtures must not reach model probing,
            # tokenizer construction, GPU state, or session execution.
            conflict = selector_request()
            conflict["execution"]["memory_budget_bytes"] = 0
            status, error = resolve(public, conflict)
            assert status != 0
            assert "streaming_config_conflict" in error, error
            assert "catalog_has_no_public_records" not in error, error

            hybrid = selector_request()
            hybrid["execution"].update({
                "policy": "gpu_ane",
                "allow_approximation": True,
                "ane_manifest": "/tmp/not-opened-z-image-ane.json",
            })
            status, error = resolve(public, hybrid)
            assert status != 0
            assert "streaming_route_unsupported" in error, error
            assert "catalog_has_no_public_records" not in error, error

            compiled = selector_request()
            compiled["parameters"] = {"compile_gpu": True}
            status, error = resolve(public, compiled)
            assert status != 0
            assert "streaming_route_unsupported" in error, error
            assert "compiled GPU graphs" in error, error

            wrong_model = selector_request()
            wrong_model["model"] = "flux2-klein-9b"
            status, error = resolve(public, wrong_model)
            assert status != 0
            assert "streaming_engine_model_mismatch" in error, error
            assert "catalog_has_no_public_records" not in error, error

            malformed = selector_request()
            malformed["execution"]["streaming"][
                "target_request_memory_bytes"] = 14 << 30
            status, error = resolve(public, malformed)
            assert status != 0
            assert "not a published target" in error, error

            # C ABI failure ownership: no result on failure, exactly one error.
            status, value, failure = resolve_raw(
                public, json.dumps(selector_request()).encode())
            assert status != 0 and not value
            assert "catalog_has_no_public_records" in failure, failure

            # Null inputs are rejected without dereferencing output storage.
            status, value, failure = resolve_raw(
                None, json.dumps(selector_request()).encode())
            assert status != 0 and not value
            assert "missing streaming resolve input/output" in failure, failure
            status, value, failure = resolve_raw(public, None)
            assert status != 0 and not value
            assert "missing streaming resolve input/output" in failure, failure
            status, _, failure = resolve_raw(
                public, json.dumps(selector_request()).encode(),
                result_output=False)
            assert status != 0
            assert "missing streaming resolve input/output" in failure, failure
        finally:
            LIB.tc_engine_free(public)

        candidate = create(LIB.tc_engine_create_model_candidate, root)
        try:
            status, error = generate(candidate, request())
            assert status != 0
            assert "streaming_layout_not_certified" not in error, error
            status, error = prepare(candidate, selector_request())
            assert status != 0
            assert "streaming_prepare_unsupported" in error, error
            status, error = resolve(candidate, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
        finally:
            LIB.tc_engine_free(candidate)

    api = (ROOT / "native/api/c_api.mm").read_text()
    implementation = (ROOT / "native/models/z_image/z_image.cpp").read_text()
    assert 'std::strcmp(id, "z-image-turbo")' in api
    assert "z_image_exact_streaming_requested" in implementation
    assert "ZImageExactStream" in implementation
    assert "else if (legacy_streamed)" in implementation

    print("PASS Z-Image public gate remains fail-closed; exact resolve stops at empty catalog; private candidate reaches the generic route")


if __name__ == "__main__":
    main()
