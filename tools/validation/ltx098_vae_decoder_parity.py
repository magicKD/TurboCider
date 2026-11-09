#!/usr/bin/env python3
"""Compare the real LTX 0.9.8 native VAE decoder with stock ComfyUI.

Fixed explicit decode noise isolates decoder math. This does not establish
full-video quality or a fair end-to-end performance comparison.
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
    parser.add_argument("--latent-frames", type=int, choices=(1, 2), default=2)
    parser.add_argument("--latent-size", type=int, choices=(2, 4, 16), default=2)
    args = parser.parse_args()
    checkpoint, comfy, probe = (p.resolve(strict=True) for p in (args.checkpoint, args.comfy, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")
    sys.path.insert(0, str(comfy))
    sys.argv = sys.argv[:1]
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from comfy.ldm.lightricks.vae.causal_video_autoencoder import Decoder

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        config = json.loads(source.metadata()["config"])["vae"]
        prefix = "vae.decoder."
        state = {name[len(prefix):]: source.get_tensor(name) for name in source.keys() if name.startswith(prefix)}
        mean = source.get_tensor("vae.per_channel_statistics.mean-of-means").to("mps")
        stddev = source.get_tensor("vae.per_channel_statistics.std-of-means").to("mps")
    with torch.device("meta"):
        decoder = Decoder(dims=config["dims"], in_channels=config["latent_channels"],
                          out_channels=config["out_channels"], blocks=config["decoder_blocks"],
                          patch_size=config["patch_size"], norm_layer=config["norm_layer"],
                          causal=config["causal_decoder"], timestep_conditioning=config["timestep_conditioning"],
                          spatial_padding_mode=config.get("spatial_padding_mode", "reflect"))
    decoder.load_state_dict(state, strict=True, assign=True)
    del state
    decoder = decoder.to(device="mps", dtype=torch.bfloat16).eval()
    torch.manual_seed(214)
    shape = (1, args.latent_frames, args.latent_size, args.latent_size, 128)
    latents = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    noise = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    started = time.perf_counter()
    with torch.inference_mode():
        value = noise.to("mps") * .025 + latents.to("mps") * .975
        value = value.permute(0, 4, 1, 2, 3).contiguous()
        value = value * stddev.reshape(1, 128, 1, 1, 1) + mean.reshape(1, 128, 1, 1, 1)
        decoded = decoder(value, timestep=.05)
        reference = decoded.permute(0, 2, 3, 4, 1).contiguous().cpu().float()
    reference_seconds = time.perf_counter() - started
    del decoder, value, decoded, mean, stddev
    gc.collect()
    torch.mps.empty_cache()
    output.mkdir(parents=True)
    save_file({"latents": latents.contiguous(), "noise": noise.contiguous()}, str(output / "input.safetensors"))
    save_file({"pixels": reference.contiguous()}, str(output / "reference.safetensors"))
    started = time.perf_counter()
    run = subprocess.run([str(probe), str(checkpoint), str(output / "input.safetensors"),
                          str(output / "native.safetensors")], capture_output=True, text=True, timeout=600)
    native_seconds = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    actual_bf16 = load_file(str(output / "native.safetensors"))["pixels"]
    if actual_bf16.dtype != torch.bfloat16:
        raise ValueError(f"decoder did not preserve BF16: {actual_bf16.dtype}")
    actual = actual_bf16.float()
    if actual.shape != reference.shape:
        raise ValueError(f"pixel shape mismatch: {actual.shape} != {reference.shape}")
    difference = actual - reference
    report = {"schema": "ltx098-vae-decoder-parity-v1", "latent_frames": args.latent_frames,
              "latent_size": args.latent_size, "shape": list(actual.shape),
              "relative_l2": (difference.norm() / reference.norm()).item(),
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
