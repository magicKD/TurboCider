#!/usr/bin/env python3
"""Decode captured latent trajectories with the same observed Comfy VAE noise.

This isolates denoiser/decoder differences from independent decoder RNG streams;
it does not replace quality measurements of the actual generated MP4 files.
"""
import argparse
import json
from pathlib import Path
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--comfy-diagnostic", type=Path, required=True)
    parser.add_argument("--comfy-trace", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint, probe, diagnostic, comfy_trace, baseline_trace, candidate_trace = (p.resolve(strict=True) for p in
        (args.checkpoint, args.probe, args.comfy_diagnostic, args.comfy_trace, args.baseline_trace, args.candidate_trace))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")

    import numpy as np
    from PIL import Image
    import torch
    from safetensors.torch import load_file, save_file

    def native(directory, name):
        return load_file(str(directory / f"{name}.safetensors"))["value"]

    noise = load_file(str(comfy_trace / "decode_noise.safetensors"))["noise"]
    noise = noise.permute(0, 2, 3, 4, 1).contiguous().to(torch.bfloat16)
    comfy_final = load_file(str(diagnostic / "0-final_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1)
    latents = {"comfy": comfy_final, "baseline": native(baseline_trace, "step_7"),
               "candidate": native(candidate_trace, "step_7")}
    if tuple(noise.shape) != (1, 16, 16, 16, 128) or any(value.shape != noise.shape for value in latents.values()):
        raise ValueError("expected complete 512x512x121 latent/noise geometry")
    paths = sorted(diagnostic.glob("0-frame_*.png"))
    if len(paths) != 121:
        raise ValueError("all 121 Comfy raw PNG frames are required")
    output.mkdir(parents=True)
    report = {"schema": "ltx098-common-noise-decode-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "probe_sha256": sha256(probe), "noise_sha256": sha256(comfy_trace / "decode_noise.safetensors"),
              "scope": "All three latents decoded by native VAE with identical observed Comfy BF16 noise; RGB compared to Comfy PNG; diagnostic only",
              "runs": []}
    try:
        for name, latent in latents.items():
            input_path, output_path = output / f"{name}-input.safetensors", output / f"{name}-pixels.safetensors"
            save_file({"latents": latent.to(torch.bfloat16).contiguous(), "noise": noise}, str(input_path))
            result = subprocess.run([str(probe), str(checkpoint), str(input_path), str(output_path)],
                                    capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"{name} decoder failed: {result.stderr[-4000:]}")
            pixels = load_file(str(output_path))["pixels"]
            if tuple(pixels.shape) != (1, 121, 512, 512, 3):
                raise ValueError(f"{name} decoder returned unexpected pixels")
            frame_mae, motion_errors, previous = [], [], None
            for index, path in enumerate(paths):
                reference = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                candidate = np.clip((pixels[0, index].float().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
                a, b = reference.astype(np.int16), candidate.astype(np.int16)
                frame_mae.append(float(np.abs(a - b).mean()))
                if previous is not None:
                    ref_energy, candidate_energy = (float(np.abs(current - old).mean())
                        for current, old in zip((a, b), previous))
                    motion_errors.append(abs(ref_energy - candidate_energy) / max(ref_energy, candidate_energy, 1.0))
                previous = (a, b)
            row = {"latent_source": name, "frames": len(frame_mae),
                   "mean_mae_255": float(np.mean(frame_mae)), "maximum_mae_255": max(frame_mae),
                   "maximum_motion_relative_error": max(motion_errors)}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
