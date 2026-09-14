#!/usr/bin/env python3
"""Assemble a vpipe VDN workspace from existing ModelScope assets.

This helper never downloads or copies model weights.  It validates the
ModelScope provenance manifests, creates relative symlinks for the ordinary
FL2VA base and runtime components, and writes a pipeline spec whose VDN and
Turbo adapter references are absolute paths.  The reference vpipe checkout is
not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_REPOSITORY = "MiniMax/MiniMax-H3"
VDN_REPOSITORY = "OpenVDN/vdn-minimax-h3"
FASTH3_MODEL = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
STAGE = "stage-dmd-step-250"


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"missing {label}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        fail(f"refusing to overwrite existing path: {target}")
    target.symlink_to(os.path.relpath(source, target.parent))


def validate(base_root: Path, runtime_root: Path, vdn_root: Path) -> dict[str, Any]:
    base = base_root / "FL2VA" / "transformer"
    require_file(base / "config.json", "ModelScope FL2VA config")
    require_file(base / "model.safetensors.index.json", "ModelScope FL2VA index")
    base_provenance = base_root / "modelscope_download.json"
    require_file(base_provenance, "ModelScope FL2VA provenance")
    base_meta = read_json(base_provenance)
    if base_meta.get("repository") != BASE_REPOSITORY:
        fail(f"ordinary base is not {BASE_REPOSITORY}: {base_provenance}")
    if base_meta.get("partition") != "FL2VA":
        fail(f"ordinary base is not the FL2VA partition: {base_provenance}")

    runtime_provenance = runtime_root / "modelscope_download.json"
    # The downloader writes this manifest only after a complete component.
    # A partial text-encoder download must not be mistaken for a runnable
    # vpipe tree.
    require_file(runtime_provenance, "FastH3 ModelScope provenance")
    runtime_meta = read_json(runtime_provenance)
    if runtime_meta.get("model_id") != FASTH3_MODEL or runtime_meta.get("endpoint") != "https://modelscope.cn":
        fail(f"runtime provenance is not the approved ModelScope FastH3 model: {runtime_provenance}")
    for path, label in [
        (runtime_root / "text_encoder" / "config.json", "FastH3 text encoder config"),
        (runtime_root / "text_encoder" / "model.safetensors.index.json", "FastH3 text encoder index"),
        (runtime_root / "processor" / "tokenizer.json", "FastH3 processor tokenizer"),
        (runtime_root / "audio_vae" / "config.json", "FastH3 audio VAE config"),
    ]:
        require_file(path, label)

    stage = vdn_root / STAGE
    vdn_provenance = vdn_root / "modelscope_download.json"
    require_file(vdn_provenance, "ModelScope VDN provenance")
    vdn_meta = read_json(vdn_provenance)
    if vdn_meta.get("repository") != VDN_REPOSITORY:
        fail(f"VDN provenance is not {VDN_REPOSITORY}: {vdn_provenance}")
    for path, label in [
        (stage / "linear_branch" / "config.json", "VDN linear branch config"),
        (stage / "linear_branch" / "model.safetensors", "VDN linear branch weights"),
        (stage / "adapters" / "turbo" / "adapter_model.safetensors", "VDN Turbo adapter"),
    ]:
        require_file(path, label)

    # vpipe's H3 decoder needs the original MiniMax source/config envelope,
    # not merely a diffusers VAE with a compatible-looking class name.  The
    # FastH3 preview VAE is deliberately not used here: it is sharded under
    # ``vae/`` and lacks ``video_vae/source`` plus the audio metadata envelope.
    video_vae = base / "../video_vae"
    audio_vae = base / "../audio_vae"
    for path, label in [
        (video_vae / "config.json", "MiniMax FL2VA video VAE wrapper config"),
        (video_vae / "source" / "config.json", "MiniMax FL2VA video VAE source config"),
        (video_vae / "source" / "model.safetensors", "MiniMax FL2VA video VAE weights"),
        (audio_vae / "config.json", "MiniMax FL2VA audio VAE config"),
        (audio_vae / "metadata.json", "MiniMax FL2VA audio VAE metadata"),
        (audio_vae / "model.safetensors", "MiniMax FL2VA audio VAE weights"),
    ]:
        require_file(path, label)
    return {
        "base": str(base),
        "base_config_sha256": sha256(base / "config.json"),
        "base_index_sha256": sha256(base / "model.safetensors.index.json"),
        "runtime_provenance": str(runtime_provenance),
        "vdn_stage": str(stage),
        "vdn_branch_sha256": sha256(stage / "linear_branch" / "model.safetensors"),
        "vdn_adapter_sha256": sha256(stage / "adapters" / "turbo" / "adapter_model.safetensors"),
        "video_vae": str(video_vae),
        "audio_vae": str(audio_vae),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=ROOT / "models" / "MiniMax-H3-ModelScope")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "models" / "FastH3-ModelScope")
    parser.add_argument("--vdn-root", type=Path, default=ROOT / "models" / "VDN-H3-ModelScope")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    base_root = args.base_root.expanduser().resolve()
    runtime_root = args.runtime_root.expanduser().resolve()
    vdn_root = args.vdn_root.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    if workspace.exists() and any(workspace.iterdir()):
        fail(f"workspace must be new or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)

    evidence = validate(base_root, runtime_root, vdn_root)
    model_root = workspace / "models" / "MiniMax-H3-FL2VA-ModelScope"
    fl2va = model_root / "FL2VA"
    link(base_root / "FL2VA" / "transformer", fl2va / "transformer")
    link(runtime_root / "text_encoder", fl2va / "text_encoder")
    link(runtime_root / "processor", fl2va / "processor")
    link(runtime_root / "processor", fl2va / "tokenizer")
    # Keep the original MiniMax VAE layout.  FastH3's diffusers VAE remains
    # available in its own ModelScope tree for TurboCider's MLX path, but it
    # is not a vpipe H3 decoder input.
    link(base_root / "FL2VA" / "video_vae", fl2va / "video_vae")
    link(base_root / "FL2VA" / "audio_vae", fl2va / "audio_vae")

    pipeline = {
        "id": "turbocider-vdn-modelscope",
        "stages": [
            {"id": "cfg", "type": "minimax-h3-model-config", "config": {
                "video_shift": 12, "audio_shift": 3, "condition_timestep": 1,
                "audio_seconds": 0, "linear_branch": str(vdn_root / STAGE),
                "lora": str(vdn_root / STAGE / "adapters" / "turbo" / "adapter_model.safetensors"),
                "lora_scale": 1,
            }},
            {"id": "model", "type": "model-select", "config": {"hf_dir": str(model_root)}},
            {"id": "prompt", "type": "text-prompt", "config": {
                "text": "cinematic red fox running through a snowy forest",
            }},
            {"id": "condition", "type": "diffusion-conditioner", "iports":[
                {"src":"prompt","oport":0},{"src":"","oport":0},{"src":"model","oport":0}],
                "config": {"unload_when_idle":"always"}},
            {"id": "generate", "type": "generate-video", "iports":[
                {"src":"condition","oport":0},{"src":"","oport":0},{"src":"model","oport":0},
                {"src":"","oport":0},{"src":"","oport":0},{"src":"","oport":0},
                {"src":"","oport":0},{"src":"","oport":0},{"src":"","oport":0},{"src":"cfg","oport":0}],
                "config": {"height":544,"width":960,"frames":124,"fps":24,"steps":6,"seed":6,
                           "i8_gemm":True,"unload_when_idle":"always"}},
            {"id":"audio","type":"audio-vae-decode","iports":[{"src":"generate","oport":1},{"src":"model","oport":0}],"config":{}},
            {"id":"video","type":"vae-decode","iports":[{"src":"generate","oport":0},{"src":"model","oport":0}],"config":{}},
            {"id":"rgb","type":"rgb-to-video","iports":[{"src":"video","oport":0}],"config":{"fps":24}},
            {"id":"save","type":"save-video","iports":[{"src":"rgb","oport":0},{"src":"audio","oport":0}],"config":{
                "output_url":"turbocider-vdn-vpipe-workspace.mp4","enable_video":True,"enable_audio":True}},
        ],
        "subpipelines": [],
    }
    (workspace / "turbocider-vdn.vpipeline").write_text(json.dumps(pipeline, indent=2) + "\n")
    manifest = {"schema":"turbocider-vpipe-vdn-modelscope-workspace-v1",
               "modelscope_only":True,"workspace":str(workspace),"model_root":str(model_root),
               "pipeline":str(workspace / "turbocider-vdn.vpipeline"),"evidence":evidence}
    (workspace / "workspace.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
