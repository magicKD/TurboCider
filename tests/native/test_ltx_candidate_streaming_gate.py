#!/usr/bin/env python3
"""Host-only proof that exact execution is candidate-scoped, not public."""

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


def consume(pointer):
    if not pointer.value:
        return ""
    value = C.string_at(pointer).decode()
    LIB.tc_string_free(pointer)
    return value


def create(symbol, root):
    engine, error = C.c_void_p(), C.c_void_p()
    status = symbol(
        b"ltx-2.5-distilled", str(root).encode(),
        C.byref(engine), C.byref(error),
    )
    failure = consume(error)
    assert status == 0 and engine.value, failure
    return engine


def generate(engine, request):
    result, error = C.c_void_p(), C.c_void_p()
    status = LIB.tc_engine_generate(
        engine, json.dumps(request).encode(), None, None,
        C.byref(result), C.byref(error),
    )
    value, failure = consume(result), consume(error)
    return status, value, failure


def main():
    request = {
        "schema_version": 2,
        "model": "ltx-2.5-distilled",
        "operation": "video.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "gate"}],
        "outputs": [{
            "kind": "video", "path": "/tmp/not-generated.mp4",
            "width": 64, "height": 64, "frames": 9, "fps": 24,
            "audio": False,
        }],
        "sampling": {"seed": 42, "steps": 11},
        "execution": {
            "policy": "gpu",
            "ltx_backend": "c_metal",
            "streaming": {
                "schema_version": 1,
                "enabled": True,
                "selection": "manual",
                "retention": "request",
                "stages": {"denoiser": {
                    "residency": "streamed",
                    "block_group_size": 1,
                    "slot_count": 1,
                    "resident_prefix_blocks": 1,
                    "prefetch_distance": 0,
                    "io_workers": 1,
                }},
            },
        },
    }
    with tempfile.TemporaryDirectory(prefix="tc-ltx-candidate-gate-") as raw:
        root = Path(raw)
        for relative in (
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

        public = create(LIB.tc_engine_create_model, root)
        try:
            status, _, error = generate(public, request)
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
        finally:
            LIB.tc_engine_free(public)

        candidate = create(LIB.tc_engine_create_model_candidate, root)
        LIB.tc_engine_free(candidate)

        # Candidate execution itself is covered by the opt-in real-Metal
        # harness. Keep this default contract host-only while proving the
        # authorization bit can only be set by the private constructor.
        source = (ROOT / "native/api/c_api.mm").read_text()
        assert "bool allow_experimental_streaming = false;" in source
        assert "(*engine)->allow_experimental_streaming = true;" in source
        assert "!request.streaming.active() ||\n                            e->allow_experimental_streaming" in source
        for symbol in (
            "tc_engine_test_ltx_exact_destroy_failures",
            "tc_engine_test_ltx_exact_cancel_first_fill",
            "tc_engine_test_ltx_process_quarantine_count",
            "tc_engine_test_ltx_retry_process_quarantine",
        ):
            try:
                getattr(LIB, symbol)
            except AttributeError:
                continue
            raise AssertionError(f"release library exports private test hook: {symbol}")

    print("PASS LTX public gate remains fail-closed; private candidate authority; release has no lifecycle test hooks")


if __name__ == "__main__":
    main()
