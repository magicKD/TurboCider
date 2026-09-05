#!/usr/bin/env python3
"""Export a deterministic ComfyUI LTX-2.5 spatial-upscaler fixture.

The input is stored as raw BF16 words in BCFHW layout. The reference output
is stored as contiguous little-endian FP32 in the same layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_upsampler_class(comfy_root: Path):
    module_path = comfy_root / "comfy/ldm/lightricks/latent_upsampler.py"
    spec = importlib.util.spec_from_file_location(
        "comfy_ltx_latent_upsampler", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LatentUpsampler


def checkpoint_config(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw = metadata.get("config") or metadata.get("embedded_config")
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed.get("config", parsed)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=Path("/Users/james/ComfyUI-Installs/ComfyUI/ComfyUI"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    latent_upsampler = load_upsampler_class(args.comfy_root)
    config = checkpoint_config(args.checkpoint)
    config.update(
        {
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "spatial_scale": 2.0,
            "rational_resampler": False,
        }
    )
    operations = SimpleNamespace(
        Conv2d=torch.nn.Conv2d,
        Conv3d=torch.nn.Conv3d,
        GroupNorm=torch.nn.GroupNorm,
    )
    model = latent_upsampler.from_config(config, operations=operations)
    state = load_file(str(args.checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model = model.to(device=device, dtype=dtype).eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    shape = (1, 128, args.frames, args.height, args.width)
    latent_cpu = torch.randn(shape, generator=generator, dtype=torch.float32)
    latent = latent_cpu.to(device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(latent)
            synchronize(device)
        start = time.perf_counter()
        output = model(latent)
        synchronize(device)
        elapsed = time.perf_counter() - start

    input_bf16 = (
        latent.detach().cpu().contiguous().view(torch.uint16).numpy()
    )
    output_f32 = output.detach().float().cpu().contiguous().numpy()
    input_path = args.out_dir / "input_bf16.raw"
    output_path = args.out_dir / "output_f32.raw"
    input_bf16.tofile(input_path)
    output_f32.tofile(output_path)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "shape": list(shape),
        "output_shape": list(output_f32.shape),
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "elapsed_seconds": elapsed,
        "input": input_path.name,
        "output": output_path.name,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
