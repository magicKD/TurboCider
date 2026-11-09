#!/usr/bin/env python3
"""Compare low-resolution Euler states/predictions, excluding the locked frame."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output report must not exist")
    sigmas = [1., .9937, .9875, .9812, .975, .9094, .725]
    rows = []
    for index, sigma in enumerate(sigmas):
        comfy = load_file(str(args.comfy / f"step_{index}.safetensors"))
        x = comfy["x"].permute(0, 2, 3, 4, 1).float()[:, 1:]
        denoised = comfy["denoised"].permute(0, 2, 3, 4, 1).float()[:, 1:]
        state_name = "initial_noise" if index == 0 else f"step_{index-1}"
        native_x = load_file(str(args.native / f"{state_name}.safetensors"))["value"].float()[:, 1:]
        native_v = load_file(str(args.native / f"prediction_{index}.safetensors"))["value"].float()[:, 1:]
        velocity = (x - denoised) / sigma
        def compare(reference, actual):
            delta = actual - reference
            return {"relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
                    "mean_absolute": delta.abs().mean().item()}
        rows.append({"step": index, "sigma": sigma, "state": compare(x, native_x),
                     "velocity": compare(velocity, native_v)})
    result = {"schema": "ltx098-multiscale-steps-v1", "scope": "First-pass motion latents only; Comfy Euler callback versus native; first conditioning frame excluded", "steps": rows}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
