#!/usr/bin/env python3
"""Compare the real LTX 0.9.8 native VAE encoder with stock ComfyUI.

The two GPU implementations run sequentially. This is an encoder component
check, not end-to-end video validation or a speed comparison.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(1, 9), default=1)
    parser.add_argument("--size", type=int, choices=(64, 128, 512), default=64)
    args = parser.parse_args()
    checkpoint, comfy, probe = (p.resolve(strict=True) for p in (args.checkpoint, args.comfy, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")
    sys.path.insert(0, str(comfy))
    # ComfyUI's optional global CLI parser must not consume this script's args.
    sys.argv = sys.argv[:1]
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from comfy.ldm.lightricks.vae.causal_video_autoencoder import Encoder

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        config = json.loads(source.metadata()["config"])["vae"]
        prefix = "vae.encoder."
        state = {name[len(prefix):]: source.get_tensor(name) for name in source.keys() if name.startswith(prefix)}
        mean = source.get_tensor("vae.per_channel_statistics.mean-of-means").to("mps")
        stddev = source.get_tensor("vae.per_channel_statistics.std-of-means").to("mps")
    with torch.device("meta"):
        encoder = Encoder(dims=config["dims"], in_channels=config["in_channels"],
                          out_channels=config["latent_channels"], blocks=config["encoder_blocks"],
                          patch_size=config["patch_size"], norm_layer=config["norm_layer"],
                          latent_log_var=config["latent_log_var"],
                          spatial_padding_mode=config.get("spatial_padding_mode", "zeros"))
    encoder.load_state_dict(state, strict=True, assign=True)
    del state
    encoder = encoder.to(device="mps", dtype=torch.bfloat16).eval()
    torch.manual_seed(213)
    pixels = (torch.rand((1, args.frames, args.size, args.size, 3), dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    started = time.perf_counter()
    with torch.inference_mode():
        encoded = encoder(pixels.permute(0, 4, 1, 2, 3).to("mps"))[:, :128]
        encoded = (encoded - mean.reshape(1, 128, 1, 1, 1)) / stddev.reshape(1, 128, 1, 1, 1)
        reference = encoded.permute(0, 2, 3, 4, 1).contiguous().cpu().float()
    reference_seconds = time.perf_counter() - started
    del encoder, encoded, mean, stddev
    gc.collect()
    torch.mps.empty_cache()
    output.mkdir(parents=True)
    save_file({"pixels": pixels.contiguous()}, str(output / "input.safetensors"))
    save_file({"latents": reference.contiguous()}, str(output / "reference.safetensors"))
    started = time.perf_counter()
    run = subprocess.run([str(probe), str(checkpoint), str(output / "input.safetensors"),
                          str(output / "native.safetensors")], capture_output=True, text=True, timeout=600)
    native_seconds = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    actual = load_file(str(output / "native.safetensors"))["latents"].float()
    if actual.shape != reference.shape:
        raise ValueError(f"latent shape mismatch: {actual.shape} != {reference.shape}")
    difference = actual - reference
    report = {"schema": "ltx098-vae-encoder-parity-v1", "frames": args.frames, "size": args.size,
              "shape": list(actual.shape), "relative_l2": (difference.norm() / reference.norm()).item(),
              "cosine": F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item(),
              "max_abs": difference.abs().max().item(), "all_finite": bool(torch.isfinite(actual).all()),
              "reference_seconds_excluding_load": reference_seconds,
              "native_process_seconds_including_load": native_seconds,
              "probe_stdout": run.stdout, "probe_stderr": run.stderr}
    report["passed"] = report["all_finite"] and report["relative_l2"] < .05 and report["cosine"] > .998
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
