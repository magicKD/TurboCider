#!/usr/bin/env python3
"""Compare GPU/hybrid latent videos through one VAE and identical noise.

This removes H.264 and decoder RNG differences; the existing RGB thresholds
are applied to 121 uncompressed, quantized RGB frames as a separate diagnostic.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ltx098_comfy_reference import MODEL_SHA256, sha256

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from video_quality_gate import _motion_energy, _rgb_metrics, passes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-probe", type=Path, required=True)
    parser.add_argument("--baseline-latent", type=Path, required=True)
    parser.add_argument("--candidate-latent", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint, decoder, baseline, candidate, noise_path = (path.resolve(strict=True) for path in
        (args.checkpoint, args.decoder_probe, args.baseline_latent, args.candidate_latent, args.noise))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from pinned source")
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    latent = {"baseline": load_file(str(baseline))["value"], "candidate": load_file(str(candidate))["value"]}
    noise = load_file(str(noise_path))["value"]
    if tuple(noise.shape) != (1, 16, 16, 16, 128) or any(value.shape != noise.shape for value in latent.values()):
        raise ValueError("expected complete 512x512x121 latent/noise geometry")
    output.mkdir(parents=True)
    report = {"schema": "ltx098-hybrid-raw-quality-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "decoder_probe_sha256": sha256(decoder), "noise_sha256": sha256(noise_path),
              "baseline_latent_sha256": sha256(baseline), "candidate_latent_sha256": sha256(candidate),
              "scope": "Same native VAE, same BF16 decoder noise, uncompressed uint8 RGB; separate diagnostic from encoded MP4 gate"}
    try:
        pixels = {}
        for name, value in latent.items():
            input_path, output_path = output / f"{name}-input.safetensors", output / f"{name}-pixels.safetensors"
            save_file({"latents": value.to(torch.bfloat16).contiguous(), "noise": noise.to(torch.bfloat16).contiguous()}, str(input_path))
            run = subprocess.run([str(decoder), str(checkpoint), str(input_path), str(output_path)],
                                 capture_output=True, text=True, timeout=600)
            if run.returncode:
                raise RuntimeError(f"{name} decoder failed: {run.stderr[-4000:]}")
            pixels[name] = load_file(str(output_path))["pixels"]
            if tuple(pixels[name].shape) != (1, 121, 512, 512, 3):
                raise ValueError(f"{name} decoder output geometry differs")
        frame_metrics, motion_errors, previous = [], [], None
        for index in range(121):
            frames = [np.clip((pixels[name][0, index].float().numpy() + 1) * 127.5, 0, 255).astype(np.uint8).tobytes()
                      for name in ("baseline", "candidate")]
            frame_metrics.append(_rgb_metrics(frames[0], frames[1]))
            if previous is not None:
                left = _motion_energy(previous[0], frames[0])
                right = _motion_energy(previous[1], frames[1])
                motion_errors.append(abs(left - right) / max(left, right, 1.0))
            previous = frames
        mean = lambda key: sum(row[key] for row in frame_metrics) / len(frame_metrics)
        metrics = {"shape_equal": True, "frame_count_equal": True, "fps_equal": True, "finite": True,
                   "frame_count_compared": len(frame_metrics), "mean_correlation": mean("correlation"),
                   "minimum_correlation": min(row["correlation"] for row in frame_metrics),
                   "mean_cosine": mean("cosine"), "mean_mae_255": mean("mae_255"),
                   "maximum_motion_relative_error": max(motion_errors), "frame_metrics": frame_metrics}
        delta = latent["baseline"].float() - latent["candidate"].float()
        report.update(status="complete", metrics=metrics, aligned_rgb_gate_passed=passes(metrics),
                      latent_relative_l2=(delta.norm() / latent["baseline"].float().norm().clamp_min(1e-30)).item())
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"aligned_rgb_gate_passed": report["aligned_rgb_gate_passed"],
                      "latent_relative_l2": report["latent_relative_l2"],
                      **{key: report["metrics"][key] for key in ("mean_correlation", "minimum_correlation", "mean_cosine", "mean_mae_255", "maximum_motion_relative_error")}}, indent=2))


if __name__ == "__main__":
    main()
