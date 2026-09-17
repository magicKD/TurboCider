#!/usr/bin/env python3
"""Host-only proof that FLUX 9B generic streaming is candidate scoped."""

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
        b"flux2-klein-9b", str(root).encode(),
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


def request() -> dict:
    return {
        "schema_version": 2,
        "model": "flux2-klein-9b",
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
            "num_attention_heads": 32, "attention_head_dim": 128,
            "num_layers": 8, "num_single_layers": 24,
            "in_channels": 128, "guidance_embeds": False,
            "joint_attention_dim": 12288,
        },
        "text_encoder/config.json": {
            "hidden_size": 4096, "num_hidden_layers": 36,
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
        finally:
            LIB.tc_engine_free(public)

        candidate = create(LIB.tc_engine_create_model_candidate, root)
        try:
            status, error = generate(candidate, request())
            assert status != 0
            assert "streaming_layout_not_certified" not in error, error
        finally:
            LIB.tc_engine_free(candidate)

    api = (ROOT / "native/api/c_api.mm").read_text()
    pipeline = (ROOT / "native/models/flux2/pipeline.cpp").read_text()
    transformer = (ROOT / "native/models/flux2/flux_transformer.cpp").read_text()
    assert 'std::strcmp(id, "flux2-klein-9b")' in api
    assert "flux_exact_streaming_requested" in pipeline
    assert "FluxExactStream" in pipeline
    assert "MultiPoolPolicy::retain_all" in transformer

    print("PASS FLUX 9B public gate remains fail-closed; private candidate reaches the generic route")


if __name__ == "__main__":
    main()
