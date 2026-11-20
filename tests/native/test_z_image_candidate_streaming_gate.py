#!/usr/bin/env python3
"""Host-only proof that Z-Image generic streaming is candidate scoped."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from streaming_gate_api import (
    CandidateStreamingGate, image_generation_request, memory_tier_request,
)


ROOT = Path(__file__).resolve().parents[2]
GATE = CandidateStreamingGate(ROOT, "z-image-turbo")



def resolve(engine, request: dict) -> tuple[int, str]:
    status, value, failure = GATE.resolve(engine, request)
    if status == 0:
        assert value and not failure, (value, failure)
    else:
        assert not value and failure, (value, failure)
    return status, failure


def request() -> dict:
    return image_generation_request("z-image-turbo", "gate", "/tmp/not-generated-z-image.png")


def selector_request() -> dict:
    return memory_tier_request(request())


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

        public = GATE.create(root)
        try:
            status, _, error = GATE.generate(public, request())
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
            status, _, error = GATE.generate(public, selector_request())
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
            status, value, failure = GATE.resolve_raw(
                public, json.dumps(selector_request()).encode())
            assert status != 0 and not value
            assert "catalog_has_no_public_records" in failure, failure

            # Null inputs are rejected without dereferencing output storage.
            status, value, failure = GATE.resolve_raw(
                None, json.dumps(selector_request()).encode())
            assert status != 0 and not value
            assert "missing streaming resolve input/output" in failure, failure
            status, value, failure = GATE.resolve_raw(public, None)
            assert status != 0 and not value
            assert "missing streaming resolve input/output" in failure, failure
            status, _, failure = GATE.resolve_raw(
                public, json.dumps(selector_request()).encode(),
                result_output=False)
            assert status != 0
            assert "missing streaming resolve input/output" in failure, failure
        finally:
            GATE.free(public)

        candidate = GATE.create(root, candidate=True)
        try:
            status, _, error = GATE.generate(candidate, request())
            assert status != 0
            assert "streaming_layout_not_certified" not in error, error
            status, _, error = GATE.prepare(candidate, selector_request())
            assert status != 0
            assert "streaming_prepare_unsupported" in error, error
            status, error = resolve(candidate, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
        finally:
            GATE.free(candidate)

    api = (ROOT / "native/api/c_api.mm").read_text()
    implementation = (ROOT / "native/models/z_image/z_image.cpp").read_text()
    assert 'std::strcmp(id, "z-image-turbo")' in api
    assert "z_image_exact_streaming_requested" in implementation
    assert "ZImageExactStream" in implementation
    assert "else if (legacy_streamed)" in implementation

    print("PASS Z-Image public gate remains fail-closed; exact resolve stops at empty catalog; private candidate reaches the generic route")


if __name__ == "__main__":
    main()
