#!/usr/bin/env python3
"""Download the published FastH3 Preview v1 checkpoint from ModelScope.

The ModelScope SDK's normal snapshot path keeps a second content-addressed
cache.  This utility intentionally downloads selected files directly into the
final component tree, one resumable ``.part`` file at a time.  That matters for
the 66.25 GB (61.70 GiB) transformer on a nearly full Apple Silicon SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))
from prepare_fasth3_mlx import download_file, ensure_transformer_index  # noqa: E402


DEFAULT_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
# Backward-compatible import for local tooling/tests; callers should prefer
# the explicit --model-id argument when downloading the VSA checkpoint.
MODEL_ID = DEFAULT_MODEL_ID
MODEL_ENDPOINT = "https://modelscope.cn"
TRANSFORMER_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
RUNTIME_PREFIXES = (
    "text_encoder/",
    "tokenizer/",
    "processor/",
    "vae/",
    "audio_vae/",
    "scheduler/",
    "audio_scheduler/",
)
RUNTIME_ROOT_FILES = {
    "checkpoint_content.json",
    "checkpoint_metadata.json",
    "configuration.json",
    "fastvideo_inference.json",
    "modular_model_index.json",
    "provenance.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(path if path.exists() else path.parent).free


def wanted(path: str, component: str) -> bool:
    if component == "transformer":
        return path.startswith("transformer/")
    if component == "text_encoder":
        return path.startswith("text_encoder/")
    if component == "vae":
        return path.startswith("vae/")
    if component == "audio_vae":
        return path.startswith("audio_vae/")
    if component == "runtime":
        return path in RUNTIME_ROOT_FILES or path.startswith(RUNTIME_PREFIXES)
    return path in RUNTIME_ROOT_FILES or path.startswith(("transformer/",) + RUNTIME_PREFIXES)


def main() -> None:
    try:
        from modelscope.hub.api import HubApi
    except ModuleNotFoundError as error:
        raise SystemExit(
            "ModelScope SDK is required for downloads; install it in the active Python "
            "environment before running this command"
        ) from error
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="final model directory, e.g. models/FastH3-ModelScope")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="ModelScope repository (Dense-DataFree or VSA-DataFree)")
    parser.add_argument("--component", choices=("transformer", "text_encoder", "vae", "audio_vae",
                                                 "runtime", "all"),
                        default="transformer")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--verify-content", action="store_true",
                        help="verify shard SHA-256 against checkpoint_content.json")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    api = HubApi(endpoint=MODEL_ENDPOINT)
    model_id = args.model_id.strip()
    if "/" not in model_id or model_id.startswith("/"):
        raise ValueError(f"invalid ModelScope model id: {model_id!r}")
    files = api.get_model_files(model_id, revision=args.revision)
    records = {entry["Path"]: int(entry.get("Size", 0)) for entry in files}
    selected = sorted(path for path in records if wanted(path, args.component))
    if args.verify_content and "checkpoint_content.json" in records and "checkpoint_content.json" not in selected:
        selected.append("checkpoint_content.json")
        selected.sort()
    if not selected:
        raise RuntimeError(f"no ModelScope files selected for component {args.component}")
    bytes_needed = sum(records[path] for path in selected if not (root / path).is_file())
    free = available_bytes(root.parent) / 2**30
    if free < bytes_needed / 2**30 + args.min_free_gib:
        raise RuntimeError(
            f"{args.component} needs {bytes_needed / 2**30:.2f} GiB and only {free:.2f} GiB is free; "
            f"free space or select a smaller component"
        )
    base = f"{MODEL_ENDPOINT}/models/{model_id}/resolve/{args.revision}"
    for path in selected:
        target = root / path
        if target.is_file() and records[path] and target.stat().st_size == records[path]:
            print(f"[skip] {path}", flush=True)
            continue
        if target.is_file():
            actual = target.stat().st_size
            print(f"[replace] {path}: local size {actual} != ModelScope size {records[path]}", flush=True)
            target.unlink()
        download_file(f"{base}/{path}?download=true", target)
        if records[path] and target.stat().st_size != records[path]:
            raise RuntimeError(f"ModelScope file size mismatch for {path}: {target.stat().st_size} != {records[path]}")

    if args.component == "transformer":
        # The current ModelScope tree omits the diffusers index.  Synthesize a
        # deterministic local one from shard headers once all shards are done;
        # FastVideo can glob the shards, while TurboCider uses this index for
        # validation and provenance.
        ensure_transformer_index(root / "transformer")

    manifest_path = root / "modelscope_download.json"
    previous = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            previous = {}
    previous_files = previous.get("files", {}) if isinstance(previous.get("files"), dict) else {}
    previous_files.update({path: {"bytes": records[path]} for path in selected})
    previous_components = previous.get("components", [])
    if not isinstance(previous_components, list):
        previous_components = [previous.get("component")] if previous.get("component") else []
    components = sorted({str(item) for item in previous_components if item} | {args.component})
    manifest = {
        "schema": "turbocider-modelscope-fasth3-v1",
        "model_id": model_id,
        "revision": args.revision,
        "component": args.component,
        "components": components,
        "endpoint": MODEL_ENDPOINT,
        "files": dict(sorted(previous_files.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.verify_content and (root / "checkpoint_content.json").is_file():
        content = json.loads((root / "checkpoint_content.json").read_text())
        for record in content.get("files", []):
            path = root / record["path"]
            if path.is_file() and path.stat().st_size == int(record["bytes"]):
                actual = sha256_file(path)
                if actual != record["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch for {record['path']}: {actual} != {record['sha256']}")
    print(f"[ready] {root} component={args.component} files={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
