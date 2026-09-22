"""Matched 40-step mflux oracle for the native generation diagnostic.

Consumes the native run's verified prompt conditioning, independently creates
noise/schedule, runs mflux DiT+VAE, and saves its RGB result for visual review.
No part of this script is called by production/native generation.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

def timing_summary(native_times, times, steps):
    if steps < 1 or len(native_times) != steps or len(times) != steps:
        raise ValueError("step timing count does not match schedule")
    if not all(math.isfinite(t) and t > 0 for t in native_times + times):
        raise ValueError("invalid step timing evidence")
    return {
        "native_step_seconds": native_times, "mflux_step_seconds": times,
        "native_first_step_seconds": native_times[0], "mflux_first_step_seconds": times[0],
        "native_denoise_seconds": sum(native_times), "mflux_denoise_seconds": sum(times),
        "denoise_speedup_mflux_over_native": sum(times) / sum(native_times),
        "native_median_step_seconds": statistics.median(native_times[1:]) if steps > 1 else None,
        "mflux_median_step_seconds": statistics.median(times[1:]) if steps > 1 else None,
    }


def main():
    import mlx.core as mx
    from PIL import Image
    import numpy as np
    from qwen21_vae_parity import canonical

    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Write machine-readable timing and parity evidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-config", type=Path, help="Align mflux VAE normalization with the official config used by current native builds")
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.config.config import Config
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
    from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
    from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping

    native = mx.load(str(args.native))
    height, width = native["pixels"].shape[1:3]
    steps = native["sigmas"].shape[0] - 1
    if steps < 1:
        raise ValueError("at least one denoise step is required")
    ModelConfig.precision = mx.bfloat16
    config = Config(ModelConfig.qwen_image_21(), num_inference_steps=steps,
                    height=height, width=width, guidance=1.0)
    sigmas = config.scheduler.sigmas
    schedule_error = mx.max(mx.abs(sigmas - native["sigmas"])).item()
    if not math.isfinite(schedule_error):
        raise ValueError("mflux produced a non-finite schedule; this step count is not comparable")
    if schedule_error >= 1e-6:
        raise ValueError(f"native/mflux schedules differ by {schedule_error}")
    latents = mx.random.normal(native["initial"].shape, key=mx.random.key(args.seed)).astype(mx.bfloat16)
    assert mx.array_equal(latents, native["initial"]).item(), "seed/noise differs"
    model = Qwen21Transformer()
    weights = mx.load(str(args.model / "diffusion_models/qwen_image_2.1_bf16.safetensors"))
    params = []
    for key, tensor in weights.items():
        if key.endswith(".img_mlp.gate_up.weight"):
            gate, up = mx.split(tensor, 2, axis=0)
            params.extend([(key.replace("gate_up", "gate_layer"), gate), (key.replace("gate_up", "proj"), up)])
        else:
            params.append((key.replace("modulation.1", "modulation.layers.1"), tensor))
    model.load_weights(params, strict=False)
    mx.eval(model.parameters())
    del weights, params
    times = []
    for step in range(steps):
        start = time.perf_counter()
        noise = model(step, config, latents, native["text"])
        latents = config.scheduler.step(noise, step, latents)
        mx.eval(latents)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(json.dumps({"mflux_step": step + 1, "seconds": elapsed}), flush=True)
    delta = latents.astype(mx.float32) - native["latents"].astype(mx.float32)
    latent_rrmse = mx.sqrt(mx.mean(delta * delta) / mx.mean(latents.astype(mx.float32)**2)).item()
    del model, noise
    gc.collect()
    mx.clear_cache()
    vae = Qwen21VAE()
    if args.vae_config:
        vae_config = json.loads(args.vae_config.read_text())
        object.__setattr__(vae, "LATENTS_MEAN", vae_config["latents_mean"])
        object.__setattr__(vae, "LATENTS_STD", vae_config["latents_std"])
    weights = {canonical(k): (mx.squeeze(v, 2) if v.ndim == 5 and v.shape[2] == 1 else v)
               for k, v in mx.load(str(args.model / "vae/qwen_image_2.1_vae_bf16.safetensors")).items() if "time_conv" not in k}
    params = []
    for target in Qwen21WeightMapping.get_vae_mapping():
        value = weights[target.from_pattern[0]]
        params.append((target.to_pattern, target.transform(value) if target.transform else value))
    vae.load_weights(params, strict=True)
    spatial = latents.reshape(1, height // 16, width // 16, 64).transpose(0, 3, 1, 2)
    mean, std = mx.array(vae.LATENTS_MEAN).reshape(1, 64, 1, 1), mx.array(vae.LATENTS_STD).reshape(1, 64, 1, 1)
    pixels = vae.decoder(vae.post_quant_conv(spatial * std + mean)).transpose(0, 2, 3, 1)
    rgb = mx.clip(pixels[..., :3] * 0.5 + 0.5, 0, 1)
    native_rgb = mx.clip(native["pixels"][..., :3] * 0.5 + 0.5, 0, 1)
    rmse = mx.sqrt(mx.mean((rgb - native_rgb)**2)).item()
    x, y = rgb.flatten() - mx.mean(rgb), native_rgb.flatten() - mx.mean(native_rgb)
    correlation = (mx.sum(x * y) / mx.sqrt(mx.sum(x * x) * mx.sum(y * y))).item()
    Image.fromarray(np.array(mx.round(rgb[0] * 255).astype(mx.uint8))).save(args.output)
    native_times = native["step_seconds"].tolist()
    passed = all(math.isfinite(v) for v in (latent_rrmse, rmse, correlation)) and correlation > 0.98 and rmse < 0.06
    report = {"schedule_max_abs": schedule_error, "seed_equal": True,
              "passed": passed, "width": width, "height": height, "steps": steps, "seed": args.seed,
              "scope": "matched DiT steps and decoded RGB; shared native text conditioning; excludes text encoding and end-to-end loading time",
              "native_artifact": str(args.native), "mflux_source": str(args.mflux),
              "vae_constants": str(args.vae_config) if args.vae_config else "unmodified-mflux",
              "latent_relative_rmse": latent_rrmse, "rgb_rmse": rmse, "rgb_correlation": correlation,
              **timing_summary(native_times, times, steps)}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert passed, "matched generation diagnostic image gate failed"


if __name__ == "__main__":
    main()
