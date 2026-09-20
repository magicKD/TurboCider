#!/usr/bin/env python3
"""Host-only proof that FLUX 9B generic streaming is candidate scoped."""

from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
MODEL = os.environ.get("TURBOCIDER_TEST_FLUX_MODEL", "flux2-klein-9b")
assert MODEL in ("flux2-klein-9b", "flux2-klein-4b")
LIB = C.CDLL(str(NATIVE / "libturbocider.dylib"))
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
        MODEL.encode(), str(root).encode(),
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
    consume(result)
    return status, consume(error)


def request() -> dict:
    return {
        "schema_version": 2,
        "model": MODEL,
        "operation": "image.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "g"}],
        "outputs": [{
            "kind": "image", "path": "/tmp/not-generated-flux-9b.png",
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


def fixture(root: Path) -> None:
    tokenizer = root / "tokenizer/tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"g": 0}, "merges": []},
        "pre_tokenizer": {"pretokenizers": [{"pattern": {"Regex": "."}}]},
        "added_tokens": [],
    }))
    values = {
        "transformer/config.json": {
            "num_attention_heads": 24 if MODEL.endswith("4b") else 32, "attention_head_dim": 128,
            "num_layers": 5 if MODEL.endswith("4b") else 8, "num_single_layers": 20 if MODEL.endswith("4b") else 24,
            "in_channels": 128, "guidance_embeds": False,
            "joint_attention_dim": 7680 if MODEL.endswith("4b") else 12288,
        },
        "text_encoder/config.json": {
            "hidden_size": 2560 if MODEL.endswith("4b") else 4096, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8,
        },
        "vae/config.json": {"latent_channels": 32},
    }
    for relative, value in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tc-flux-candidate-gate-") as raw:
        root = Path(raw)
        fixture(root)

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
    pipeline = (ROOT / "native/models/flux2/pipeline.cpp").read_text()
    transformer = (ROOT / "native/models/flux2/flux_transformer.cpp").read_text()
    assert 'std::strcmp(id, "flux2-klein-9b")' in api
    assert "flux_exact_streaming_requested" in pipeline
    assert "FluxExactStream" in pipeline
    assert "MultiPoolPolicy::retain_all" in transformer

    print("PASS " + MODEL + " public gate remains fail-closed; exact resolve stops at empty catalog; private candidate reaches the generic route")


if __name__ == "__main__":
    main()
