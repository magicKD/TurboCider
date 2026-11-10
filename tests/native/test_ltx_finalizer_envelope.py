#!/usr/bin/env python3
"""Exercise the fd-only LTX public exec-finalizer envelope boundary."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINALIZER = Path(os.environ.get(
    "TURBOCIDER_LTX_FINALIZER_BINARY", str(ROOT / "build/native/ltx-video-finalizer")
))


def invoke(envelope: dict | None, telemetry: dict | str | None = None) -> subprocess.CompletedProcess[str]:
    if not FINALIZER.is_file():
        raise RuntimeError(f"LTX finalizer is missing: {FINALIZER}")
    environment = os.environ.copy()
    environment.pop("TURBOCIDER_LTX_PUBLIC_ENVELOPE_FD", None)
    environment.pop("TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD", None)
    environment.pop("TURBOCIDER_LTX_STAGE_TELEMETRY", None)
    environment.pop("TURBOCIDER_LTX_REQUEST_START_NS", None)
    if telemetry is not None:
        environment["TURBOCIDER_LTX_STAGE_TELEMETRY"] = telemetry if isinstance(telemetry, str) else json.dumps(telemetry)
    descriptor = -1
    path: Path | None = None
    try:
        if envelope is not None:
            handle = tempfile.NamedTemporaryFile(
                prefix="tc-ltx-public-envelope-", suffix=".json", delete=False
            )
            path = Path(handle.name)
            with handle:
                handle.write(json.dumps(envelope, separators=(",", ":")).encode())
            descriptor = os.open(path, os.O_RDONLY)
            os.set_inheritable(descriptor, True)
            environment["TURBOCIDER_LTX_PUBLIC_ENVELOPE_FD"] = str(descriptor)
        return subprocess.run(
            [
                str(FINALIZER), "/missing/video-vae.safetensors",
                "/missing/video-latent.bf16", "1", "1", "1",
                "/tmp/tc-ltx-finalizer-envelope.mp4",
                "video.generate", "gpu",
            ],
            env=environment,
            pass_fds=(() if descriptor < 0 else (descriptor,)),
            text=True,
            capture_output=True,
            timeout=30,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if path is not None:
            path.unlink(missing_ok=True)


def valid_envelope() -> dict:
    return {
        "format": "turbocider-ltx-public-finalizer-envelope-v1",
        "schema_version": 1,
        "public_streaming": {"actual_plan_verified": True},
        "block_residency": {},
        "block_streaming": {},
        "streaming_stages": [{"stage_index": 0}, {"stage_index": 1}],
        "streaming_boundaries": [{"boundary_index": 0}],
        "streaming_receipt": {"schema_version": 3},
    }


def main() -> None:
    private = invoke(None)
    assert private.returncode == 1
    assert "cannot open latent input" in private.stderr
    assert "public streaming finalizer envelope" not in private.stderr

    accepted = invoke(valid_envelope())
    assert accepted.returncode == 1
    assert "cannot open latent input" in accepted.stderr
    assert "public streaming finalizer envelope" not in accepted.stderr

    incomplete = valid_envelope()
    incomplete["public_streaming"] = {"actual_plan_verified": False}
    rejected = invoke(incomplete)
    assert rejected.returncode == 1
    assert "incomplete LTX public streaming finalizer envelope" in rejected.stderr

    wrong_format = valid_envelope()
    wrong_format["format"] = "untrusted"
    rejected = invoke(wrong_format)
    assert rejected.returncode == 1
    assert "incomplete LTX public streaming finalizer envelope" in rejected.stderr

    for bad_schema in (None, "1", True, 1.5, [], {}):
        malformed = valid_envelope()
        malformed["schema_version"] = bad_schema
        rejected = invoke(malformed)
        assert rejected.returncode == 1, rejected.stderr
        assert "incomplete LTX public streaming finalizer envelope" in rejected.stderr
    for bad_verified in (None, "true", 1, [], {}):
        malformed = valid_envelope()
        malformed["public_streaming"]["actual_plan_verified"] = bad_verified
        rejected = invoke(malformed)
        assert rejected.returncode == 1, rejected.stderr
        assert "incomplete LTX public streaming finalizer envelope" in rejected.stderr
    for field in valid_envelope():
        malformed = valid_envelope()
        del malformed[field]
        rejected = invoke(malformed)
        assert rejected.returncode == 1, (field, rejected.stderr)
        assert "incomplete LTX public streaming finalizer envelope" in rejected.stderr
    telemetry = {
        "schema_version": 1, "steps": 11, "seed": 42,
        "plan": {"model": "ltx-2.5-distilled"},
        "timings_seconds": {key: 1.0 for key in (
            "text_encoder_load", "text_encode", "connector", "pre_model_load",
            "model_load", "conditioning_and_rng", "stage1", "latent_upsample", "stage2", "denoise")},
        "denoise_step_seconds": {"stage1": [0.1] * 8, "stage2": [0.2] * 3},
    }
    accepted = invoke(None, telemetry)
    assert "cannot open latent input" in accepted.stderr, accepted.stderr
    for seed in (0, (1 << 64) - 1):
        boundary = dict(telemetry, seed=seed)
        accepted = invoke(None, boundary)
        assert "cannot open latent input" in accepted.stderr, accepted.stderr
    for key in ("steps", "seed"):
        for bad in (None, True, -1, 1.5, "42", [], {}):
            broken = dict(telemetry, **{key: bad})
            rejected = invoke(None, broken)
            assert "invalid LTX request identity" in rejected.stderr, rejected.stderr
        broken = dict(telemetry)
        del broken[key]
        assert "invalid LTX request identity" in invoke(None, broken).stderr
    for bad in (-1, True, "1", None):
        broken = json.loads(json.dumps(telemetry))
        broken["timings_seconds"]["text_encode"] = bad
        rejected = invoke(None, broken)
        assert rejected.returncode == 1 and "invalid LTX stage duration" in rejected.stderr, rejected.stderr
    for stage, values in (("stage1", [0.1]*7), ("stage2", []), ("stage2", [0.1,True,0.3])):
        broken = json.loads(json.dumps(telemetry))
        broken["denoise_step_seconds"][stage] = values
        rejected = invoke(None, broken)
        assert rejected.returncode == 1 and "invalid LTX step" in rejected.stderr, rejected.stderr
    print("PASS LTX exec finalizer envelope shape/type rejection (no model/GPU work)")


if __name__ == "__main__":
    main()
