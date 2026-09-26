"""CPU-only matched native generation report; numerical metrics do not prove semantics."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = (load_file(str(p)) for p in (args.baseline, args.candidate))
    inputs = {key: torch.equal(baseline[key], candidate[key]) for key in ("text", "initial", "sigmas")}
    if not all(inputs.values()):
        raise ValueError(f"Unmatched benchmark inputs: {inputs}")
    a, b = baseline["latents"].float(), candidate["latents"].float()
    latents_rrms = float(((a - b).square().mean() / a.square().mean()).sqrt())
    pixels_a, pixels_b = ((data["pixels"].float() * .5 + .5).clamp(0, 1) for data in (baseline, candidate))
    rgb_a, rgb_b = pixels_a[..., :3], pixels_b[..., :3]
    x, y = rgb_a.flatten() - rgb_a.mean(), rgb_b.flatten() - rgb_b.mean()
    a_times, b_times = (data["step_seconds"].tolist() for data in (baseline, candidate))
    report = {"scope": "matched numerical/step-timing comparison; visual and semantic inspection required",
              "inputs_equal": inputs, "latent_relative_rmse": latents_rrms,
              "rgb_rmse": float((rgb_a - rgb_b).square().mean().sqrt()),
              "rgb_correlation": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
              "alpha_rmse": float((pixels_a[..., 3] - pixels_b[..., 3]).square().mean().sqrt()),
              "baseline_median_step_seconds": statistics.median(a_times[1:]),
              "candidate_median_step_seconds": statistics.median(b_times[1:]),
              "median_step_speedup": statistics.median(a_times[1:]) / statistics.median(b_times[1:]),
              "denoise_speedup_including_prefill": sum(a_times) / sum(b_times),
              "baseline_first_step_seconds": a_times[0], "candidate_first_step_seconds": b_times[0],
              "candidate_finite": bool(torch.isfinite(candidate["pixels"]).all())}
    print(json.dumps(report, indent=2), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
