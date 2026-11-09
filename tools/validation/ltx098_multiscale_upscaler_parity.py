#!/usr/bin/env python3
"""Compare the native multiscale upsampler boundary with ComfyUI's model.

Uses a native --trace run's step_6 and upsampled_latent tensors, the same
official 0.9.8 weights, BF16 and VAE channel statistics. This component
comparison does not establish end-to-end perceptual equivalence.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--native", type=Path, help="optional candidate tensor; default is trace/upsampled_latent.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output report must not exist")
    sys.path.insert(0, str(args.comfy.resolve(strict=True)))
    import comfy.ops
    from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU required for the ComfyUI comparison")
    device = torch.device("mps")
    up = args.model / "ltxv-spatial-upscaler-0.9.8.safetensors"
    checkpoint = args.model / "ltxv-2b-0.9.8-distilled.safetensors"
    with safe_open(up, framework="pt", device="cpu") as source:
        config = json.loads(source.metadata()["config"])
    model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init)
    model.load_state_dict(load_file(str(up)))
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        mean = source.get_tensor("vae.per_channel_statistics.mean-of-means").to(device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
        std = source.get_tensor("vae.per_channel_statistics.std-of-means").to(device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
    state = load_file(str(args.trace / "step_6.safetensors"))["value"].permute(0, 4, 1, 2, 3).to(device=device, dtype=torch.bfloat16)
    native = load_file(str(args.native or (args.trace / "upsampled_latent.safetensors")))["value"].float()
    with torch.inference_mode():
        reference = (model(state * std + mean) - mean) / std
        reference = reference.permute(0, 2, 3, 4, 1).float().cpu()
    delta = native - reference
    report = {
        "schema": "ltx098-upscaler-parity-v1", "scope": "BF16 ComfyUI MPS vs native MLX GPU; matching native stage-1 latent and VAE statistics",
        "shape": list(native.shape), "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)).item(),
        "max_absolute_error": delta.abs().max().item(), "mean_absolute_error": delta.abs().mean().item(),
        "all_finite": bool(torch.isfinite(native).all() and torch.isfinite(reference).all()),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
