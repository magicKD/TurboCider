#!/usr/bin/env python3
"""Verify the LTX 2.5 audio bundle against ModelScope and write its manifest.

The runtime deliberately refuses an unverified audio safetensors file.  This
tool is the only supported way to create the sidecar: it obtains the official
ModelScope file record, checks the installed bytes and safetensors structure,
then writes a small provenance document next to the artifact.  The model
weights themselves are never copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any


ENDPOINT = "https://modelscope.cn"
MODEL_ID = "Lightricks/LTX-2.5"
ARTIFACT_NAME = "ltx-2.5-audio-vae-bf16.safetensors"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model_record() -> tuple[dict[str, Any], str]:
    url = (f"{ENDPOINT}/api/v1/models/{MODEL_ID}/repo/files"
           "?Revision=master&Recursive=true")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    document = json.loads(raw)
    files = document.get("Data", {}).get("Files", [])
    for record in files:
        if record.get("Path") == f"vae/{ARTIFACT_NAME}":
            return {
                "name": record["Path"],
                "size": record["Size"],
                "sha256": record["Sha256"],
                "revision": record["Revision"],
            }, hashlib.sha256(raw).hexdigest()
    fail(f"ModelScope did not list vae/{ARTIFACT_NAME}")


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            fail(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= 64 * 1024 * 1024:
            fail(f"unsafe safetensors header length: {header_bytes}")
        raw = stream.read(header_bytes)
    if len(raw) != header_bytes:
        fail(f"truncated safetensors header: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        fail("safetensors header is not an object")
    return value


def validate_structure(path: Path) -> dict[str, Any]:
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    required_prefixes = {
        "audio_vae.",
        "vocoder.vocoder.",
        "vocoder.bwe_generator.",
        "vocoder.mel_stft.",
    }
    missing = [prefix for prefix in required_prefixes
               if not any(key.startswith(prefix) for key in keys)]
    if missing:
        fail(f"audio bundle is missing tensor groups: {', '.join(sorted(missing))}")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        fail("safetensors metadata is not an object")
    config_text = metadata.get("config")
    if not isinstance(config_text, str):
        fail("audio bundle does not contain the LTX config metadata")
    config = json.loads(config_text)
    preprocessing = config["audio_vae"]["preprocessing"]
    if preprocessing["audio"]["sampling_rate"] != 16000:
        fail("audio VAE does not declare 16 kHz input")
    if not preprocessing["audio"]["stereo"]:
        fail("audio VAE does not declare stereo input")
    bwe = config["vocoder"]["bwe"]
    if bwe["output_sampling_rate"] != 48000:
        fail("BWE vocoder does not declare 48 kHz output")
    return {
        "tensor_count": len(keys),
        "model_version": metadata.get("model_version", ""),
        "input_sampling_rate": 16000,
        "output_sampling_rate": 48000,
        "channels": 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.endpoint != ENDPOINT or args.model_id != MODEL_ID:
        fail("only the pinned Lightricks/LTX-2.5 ModelScope source is accepted")
    root = args.model_root.expanduser().resolve()
    artifact = (args.artifact or (root / "vae" / ARTIFACT_NAME)).expanduser().absolute()
    if not artifact.is_file():
        fail(f"audio artifact is missing: {artifact}")
    record, response_sha = fetch_model_record()
    expected_bytes = int(record["size"])
    expected_sha = str(record["sha256"]).lower()
    actual_bytes = artifact.stat().st_size
    actual_sha = sha256(artifact)
    if actual_bytes != expected_bytes:
        fail(f"size mismatch: local={actual_bytes} ModelScope={expected_bytes}")
    if actual_sha != expected_sha:
        fail(f"SHA-256 mismatch: local={actual_sha} ModelScope={expected_sha}")
    structure = validate_structure(artifact)
    manifest = {
        "schema": "turbocider-ltx-audio-assets-modelscope-v1",
        "repository": MODEL_ID,
        "source": {
            "provider": "modelscope",
            "endpoint": ENDPOINT,
            "model_id": MODEL_ID,
            "path": f"vae/{ARTIFACT_NAME}",
            "revision": record.get("revision") or record.get("Revision", ""),
            "api_sha256": response_sha,
        },
        "artifact": {
            "filename": artifact.name,
            "bytes": actual_bytes,
            "sha256": actual_sha,
        },
        "components": {
            "audio_vae_decoder": "audio_vae",
            "base_vocoder": "vocoder.vocoder",
            "bwe_vocoder": "vocoder.bwe_generator",
            "mel_stft": "vocoder.mel_stft",
        },
        "sample_rates": [16000, 48000],
        "channels": 2,
        "safetensors": structure,
    }
    destination = artifact.parent / "ltx-audio.manifest.json"
    if not args.dry_run:
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"[ready] {destination}")
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
