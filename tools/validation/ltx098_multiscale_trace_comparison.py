#!/usr/bin/env python3
"""Compare diagnostic ComfyUI and native 7+AdaIN+3 latent boundaries."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    mapping = [("first_condition", "first_frame"), ("first_pass", "step_6"),
               ("upsampled", "upsampled_latent"), ("adain", "adain_latent"), ("final", "step_9")]
    report = {"schema": "ltx098-multiscale-trace-v1", "scope": "native MLX versus ComfyUI MPS; matching seed, strength=1 and official AdaIN; independent backend arithmetic",
              "boundaries": {}}
    for comfy_name, native_name in mapping:
        expected = load_file(str(args.comfy / f"{comfy_name}_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1).float()
        actual = load_file(str(args.native / f"{native_name}.safetensors"))["value"].float()
        if comfy_name == "first_condition":
            expected = expected[:, :1]
        if expected.shape != actual.shape:
            raise ValueError(f"{comfy_name}: shape mismatch {expected.shape} != {actual.shape}")
        delta = actual - expected
        report["boundaries"][comfy_name] = {"shape": list(actual.shape),
            "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
            "mean_absolute": delta.abs().mean().item(), "max_absolute": delta.abs().max().item(),
            "cosine": F.cosine_similarity(expected.flatten(), actual.flatten(), dim=0).item()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
