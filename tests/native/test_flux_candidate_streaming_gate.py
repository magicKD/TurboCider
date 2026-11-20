#!/usr/bin/env python3
"""Host-only proof that FLUX 9B generic streaming is candidate scoped."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from streaming_gate_api import CandidateStreamingGate


ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("TURBOCIDER_TEST_FLUX_MODEL", "flux2-klein-9b")
assert MODEL in ("flux2-klein-9b", "flux2-klein-4b")
GATE = CandidateStreamingGate(ROOT, MODEL)



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

        public = GATE.create(root)
        try:
            status, _, error = GATE.generate(public, request())
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
            status, _, error = GATE.generate(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
            status, _, error = GATE.resolve(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
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
            status, _, error = GATE.resolve(candidate, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
        finally:
            GATE.free(candidate)

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
