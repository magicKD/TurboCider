#!/usr/bin/env python3
"""Run FastVideo's MLX Wan2.2 DiT with an isolated I2V conditioning wrapper.

FastVideo's current MLX example is text-to-video only. This experiment imports
its unchanged DiT blocks directly, supplies the captured native text/first
frame, and implements first-frame conditioning around the three DMD steps.
It is not the official FastVideo end-to-end image-to-video CLI.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

from wan22_schedule import warped_sigmas


def rotary(frames, height, width, head_dim):
    """Full-dimension interleaved Wan rotary values for the patch grid."""
    dimensions = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
    positions = np.indices((frames, height, width)).reshape(3, -1)
    cosine, sine = [], []
    for coordinate, dim in zip(positions, dimensions):
        angles = coordinate[:, None].astype(np.float32) * (1.0 / np.power(
            np.float32(10000), np.arange(0, dim, 2, dtype=np.float32) / dim))[None, :]
        cosine.append(np.repeat(np.cos(angles), 2, axis=1))
        sine.append(np.repeat(np.sin(angles), 2, axis=1))
    return np.concatenate(cosine, 1), np.concatenate(sine, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fastvideo", "checkpoint", "config", "conditioning", "job", "comfy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    repo, checkpoint, config, conditioning, job, comfy = (
        path.resolve(strict=True) for path in
        (args.fastvideo, args.checkpoint, args.config, args.conditioning, args.job, args.comfy))
    # The package's top-level initializer imports CUDA-serving dependencies.
    # Loading the untouched MLX submodules directly avoids that unrelated path.
    package = types.ModuleType("fastvideo")
    package.__path__ = [str(repo / "fastvideo")]
    sys.modules["fastvideo"] = package
    mlx_package = types.ModuleType("fastvideo.mlx_runtime")
    mlx_package.__path__ = [str(repo / "fastvideo/mlx_runtime")]
    sys.modules["fastvideo.mlx_runtime"] = mlx_package
    import mlx.core as mx
    import torch
    from safetensors.torch import save_file
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.sampling import add_noise, pred_noise_to_pred_video

    sys.path.insert(0, str(comfy))
    from comfy.latent_formats import Wan22

    request = json.loads(job.read_text())[0]["request"]
    inputs = mx.load(str(conditioning))
    text, first = inputs["text"], inputs["first_latent"]
    frames, width, height = request["frames"], request["width"], request["height"]
    if (request["model"] != "wan2.2-ti2v-5b-fastwan" or
            first.shape != (1, 48, 1, height // 16, width // 16) or
            text.shape != (1, 512, 4096) or frames % 4 != 1):
        raise ValueError("captured conditioning does not match the saved Wan2.2 request")
    model_config = json.loads(config.read_text())
    if (model_config["num_layers"], model_config["num_attention_heads"], model_config["in_channels"]) != (30, 24, 48):
        raise ValueError("unexpected FastVideo checkpoint configuration")
    args.output.mkdir(parents=True)
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    started = time.perf_counter()
    model = mlx_wan22_dit_from_diffusers_safetensors(checkpoint, config, dtype=args.dtype, compile=False)
    load_seconds = time.perf_counter() - started
    text = text.astype(dtype)
    shape = (1, 48, (frames - 1) // 4 + 1, height // 16, width // 16)
    generator = torch.Generator(device="cpu").manual_seed(request["seed"])
    noise = lambda: mx.array(torch.randn(shape, generator=generator, dtype=torch.float32).numpy())
    latents = noise()
    grid = (shape[2], shape[3] // 2, shape[4] // 2)
    cos, sin = (mx.array(value) for value in rotary(*grid, 128))
    sigmas = warped_sigmas()
    first_tokens = grid[1] * grid[2]
    total_tokens = grid[0] * first_tokens
    elapsed = []
    for step, sigma in enumerate(sigmas):
        start = time.perf_counter()
        model_input = mx.concatenate([first, latents[:, :, 1:]], axis=2)
        times = mx.concatenate([mx.zeros((1, first_tokens)),
                                mx.full((1, total_tokens - first_tokens), sigma * 1000.0)], axis=1)
        hidden = model._patch_embed(model_input.astype(dtype))
        temb, modulation, encoded_text = model._condition(times, text)
        mx.eval(hidden, temb, modulation, encoded_text)
        for index, block in enumerate(model.blocks):
            hidden = block(hidden, encoded_text, modulation, cos, sin)
            mx.eval(hidden)
            if index % 5 == 0:
                print(f"step={step + 1}/3 block={index + 1}/30 mlx_active={mx.get_active_memory()}", flush=True)
        prediction = model._output(hidden, temb, batch=1, frames=shape[2], height=shape[3], width=shape[4])
        clean = pred_noise_to_pred_video(prediction.astype(mx.float32), model_input.astype(mx.float32), sigma)
        clean = mx.concatenate([first, clean[:, :, 1:]], axis=2)
        latents = add_noise(clean, noise(), sigmas[step + 1]) if step < 2 else clean
        mx.eval(latents)
        elapsed.append(time.perf_counter() - start)
    normalized = np.array(latents.astype(mx.float32))
    raw = Wan22().process_out(torch.from_numpy(normalized)).to(torch.bfloat16).contiguous()
    save_file({"latents": raw}, str(args.output / "latent.safetensors"))
    report = {"schema": "fastvideo-wan22-i2v-wrapper-v1", "dtype": args.dtype,
              "scope": "FastVideo MLX DiT, custom I2V/DMD wrapper and native captured conditioning; no decoder",
              "fastvideo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "shape": list(shape), "seed": request["seed"], "sigmas": sigmas,
              "load_seconds": load_seconds, "denoise_seconds": elapsed,
              "mlx_peak_bytes": mx.get_peak_memory()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
