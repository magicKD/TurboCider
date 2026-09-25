"""CPU-only matched native generation report; numerical metrics do not prove semantics."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def matched_inputs(baseline, candidate):
    extra = lambda data: {key for key in data if key == "image_slots" or key.startswith("reference")}
    if extra(baseline) != extra(candidate):
        raise ValueError("Unmatched reference tensor keys")
    keys = ("text", "initial", "sigmas", *sorted(extra(baseline)))
    inputs = {key: torch.equal(baseline[key], candidate[key]) for key in keys}
    if not all(inputs.values()):
        raise ValueError(f"Unmatched benchmark inputs: {inputs}")
    return inputs


def coreml_prediction_summary(data, times, *, w8a8):
    if "coreml_step_seconds" not in data and "coreml_step_calls" not in data:
        return {}
    if "coreml_step_seconds" not in data or "coreml_step_calls" not in data:
        raise ValueError("Core ML coverage requires both per-step prediction timing and call counts")
    recorded = data["coreml_step_seconds"].tolist()
    if len(recorded) != len(times) or any(value < 0 or value > elapsed for value, elapsed in zip(recorded, times)):
        raise ValueError("Invalid per-step Core ML prediction timing")
    if recorded[0] != 0 or len(recorded) < 2 or any(value <= 0 for value in recorded[1:]):
        raise ValueError("Expected no prefill Core ML calls and one prediction window per decode step")
    calls = data["coreml_step_calls"].tolist()
    if (len(calls) != len(times) or calls[0] != 0 or
            any(value != calls[1] or value not in (29, 30, 31, 32) for value in calls[1:])):
        raise ValueError("Qwen21 W8A8 coverage must be at least 29/32 layers on each decode step")
    summary = {"coreml_prediction_seconds_total": sum(recorded),
            "coreml_prediction_median_per_decode_step_seconds": statistics.median(recorded[1:]),
            "coreml_prediction_median_per_ffn_call_ms": statistics.median(recorded[1:]) * 1000 / calls[1],
            "coreml_calls_per_decode_step": calls[1],
            "hybrid_ffn_layer_coverage": calls[1] / 32,
            "coreml_timing_scope": "prediction API wall time; overlaps GPU work, do not subtract from step wall"}
    if w8a8:
        summary["w8a8_ffn_layer_coverage"] = calls[1] / 32
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = (load_file(str(p)) for p in (args.baseline, args.candidate))
    with safe_open(str(args.candidate), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata() or {}
    if "coreml_step_seconds" in candidate and metadata.get("experimental_w8a8") not in ("true", "false"):
        raise ValueError("Core ML candidate requires explicit experimental_w8a8 precision provenance")
    inputs = matched_inputs(baseline, candidate)
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
              "candidate_finite": bool(torch.isfinite(candidate["pixels"]).all()),
              **coreml_prediction_summary(candidate, b_times,
                                          w8a8=metadata.get("experimental_w8a8") == "true")}
    print(json.dumps(report, indent=2), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
