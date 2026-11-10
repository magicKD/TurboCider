#!/usr/bin/env python3
"""Compare native BF16 PrunaVAED against the official patched Diffusers decoder.

Run in an isolated Python 3.11 environment with torch, diffusers==0.38.0,
accelerate, numpy and safetensors. No Python dependency is used at runtime.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from diffusers import AutoencoderKLLTX2Video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=[2, 2, 3], metavar=("F", "H", "W"))
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    shim = args.model_root / "patch_diffusers.py"
    if hashlib.sha256(shim.read_bytes()).hexdigest() != "dfac5ca02891f3d7061f54b1410db65d1dfb91e558bee93ca571ee1ed066dbf3":
        raise ValueError("Official compatibility shim differs from the inspected revision")
    spec = importlib.util.spec_from_file_location("pruna_official_patch", shim)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_pruna_ltx2_decoder()
    dtype = getattr(torch, args.dtype)
    vae = AutoencoderKLLTX2Video.from_pretrained(
        args.model_root / "vae", torch_dtype=dtype, local_files_only=True).eval()
    del vae.encoder
    frames, height, width = args.shape
    # BF16 inputs shared by both backends; non-square geometry detects swapped
    # spatial patch axes. The oracle keeps upstream normalization and reshapes.
    rng = np.random.default_rng(42)
    value = rng.normal(size=(1, frames, height, width, 128)).astype(np.float32)
    bits = value.view(np.uint32)
    bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    latent_path = args.output / "input.bf16"
    bf16.tofile(latent_path)
    value = (bf16.astype(np.uint32) << 16).view(np.float32)
    latent = torch.from_numpy(value).permute(0, 4, 1, 2, 3).to(dtype)
    with torch.inference_mode():
        latent = latent * vae.latents_std.reshape(1, -1, 1, 1, 1) + vae.latents_mean.reshape(1, -1, 1, 1, 1)
        reference = vae.decode(latent, return_dict=False)[0].float().numpy()
    reference.tofile(args.output / "reference.f32")
    env = os.environ.copy()
    env["TURBOCIDER_LTX_VAE_TILED"] = "0"
    env.pop("TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD", None)
    native_path = args.output / "native.bf16"
    subprocess.run([str(args.binary.resolve()), str(args.model_root / "vae/diffusion_pytorch_model.safetensors"),
                    str(latent_path), *map(str, args.shape), str(native_path),
                    str(args.output / "timing.json")], env=env, check=True)
    native = (np.fromfile(native_path, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(reference.shape)
    error = native - reference
    rmse = float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
    finite = bool(np.isfinite(native).all() and np.isfinite(reference).all())
    report = dict(shape=list(reference.shape), oracle_dtype=args.dtype, finite=finite,
                  rmse=rmse, mae=float(np.mean(np.abs(error))), max_abs=float(np.max(np.abs(error))),
                  psnr_range_2=float(20 * np.log10(2 / max(rmse, 1e-12))),
                  # Accumulation/rounding differs between MLX BF16 and PyTorch.
                  passed=finite and rmse < 0.025 and float(np.max(np.abs(error))) < 0.2)
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
