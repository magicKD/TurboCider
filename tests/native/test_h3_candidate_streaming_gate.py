#!/usr/bin/env python3
"""Host-only proof that H3 exact execution is private-candidate scoped."""

from __future__ import annotations

from pathlib import Path
import tempfile

from streaming_gate_api import CandidateStreamingGate


ROOT = Path(__file__).resolve().parents[2]
GATE = CandidateStreamingGate(ROOT, "minimax-h3-turbo")



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
    with tempfile.TemporaryDirectory(prefix="tc-h3-candidate-gate-") as raw:
        root = Path(raw)

        public = GATE.create(root)
        try:
            status, _, error = GATE.generate(public, request())
            assert status != 0
            assert "streaming_layout_not_certified" in error, error
            status, _, error = GATE.generate(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
            status, _, error = GATE.prepare(public, selector_request())
            assert status != 0
            assert "streaming_prepare_unsupported" in error, error
            status, _, error = GATE.resolve(public, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error
        finally:
            GATE.free(public)

        candidate = GATE.create(root, candidate=True)
        try:
            status, _, error = GATE.generate(candidate, request(slot_count=1))
            assert status != 0
            assert "streaming_route_unsupported: H3 exact candidate" in error, error
            status, _, error = GATE.prepare(candidate, selector_request())
            assert status != 0
            assert "streaming_prepare_unsupported" in error, error
            status, _, error = GATE.resolve(candidate, selector_request())
            assert status != 0
            assert "catalog_has_no_public_records" in error, error

            status, _, error = GATE.generate(candidate, request())
            assert status != 0
            assert "streaming_layout_not_certified" not in error, error
            assert "streaming_route_unsupported: H3 exact candidate" not in error, error
        finally:
            GATE.free(candidate)

    api = (ROOT / "native/api/c_api.mm").read_text()
    session = (ROOT / "native/platform/apple/h3_session.mm").read_text()
    assert 'std::strcmp(id, "minimax-h3-turbo")' in api
    assert "create_h3_candidate(path)" in api
    assert "allow_experimental_streaming_" in session
    assert "private candidate constructor" in session
    assert "exact_streaming_finished" in session

    print("PASS H3 public gate remains fail-closed; exact resolve stops at empty catalog; private candidate validates the frozen exact tuple")


if __name__ == "__main__":
    main()
