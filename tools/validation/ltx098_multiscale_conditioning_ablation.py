#!/usr/bin/env python3
"""Replay the first multiscale step with native versus Comfy first frames."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--valid-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must not exist")
    snapshot = load_file(str(args.comfy / "first_steps/step_0.safetensors"))
    state = snapshot["x"].permute(0, 2, 3, 4, 1).contiguous()
    denoised = snapshot["denoised"].permute(0, 2, 3, 4, 1).contiguous()
    native_noise = load_file(str(args.native / "initial_noise.safetensors"))["value"]
    if not torch.equal(state, native_noise):
        raise ValueError("the initial noise tensors differ")
    comfy_first = load_file(str(args.comfy / "first_condition_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1)[:, :1]
    native_first = load_file(str(args.native / "first_frame.safetensors"))["value"]
    text = load_file(str(args.native / "text.safetensors"))["value"].to(torch.bfloat16)
    if not 0 < args.valid_tokens <= text.shape[1]:
        raise ValueError("invalid valid-token count")
    mask = torch.full((1, 1, 1, text.shape[1]), -10000., dtype=torch.bfloat16)
    mask[..., :args.valid_tokens] = 0
    expected = (state - denoised)[:, 1:].float()
    args.output.mkdir(parents=True)
    report = {"schema": "ltx098-multiscale-conditioning-v1",
              "scope": "First stage sigma=1; Comfy motion-frame velocity versus native transformer with native text; same initial noise; swap first-frame latent only",
              "runs": []}
    for name, first in (("native", native_first), ("comfy", comfy_first)):
        value = state.clone()
        value[:, :1] = first.float()
        input_path = args.output / f"{name}-input.safetensors"
        output_path = args.output / f"{name}-prediction.safetensors"
        save_file({"latents": value.to(torch.bfloat16).contiguous(), "text": text.contiguous(), "text_mask": mask}, str(input_path))
        run = subprocess.run([str(args.probe.resolve(strict=True)), str(args.checkpoint.resolve(strict=True)), str(input_path),
                              str(output_path), "28", "1", "1.0"], capture_output=True, text=True, timeout=600)
        if run.returncode:
            raise RuntimeError(run.stderr)
        prediction = load_file(str(output_path))["output"].float()
        delta = prediction[:, 1:] - expected
        row = {"first_frame": name, "relative_l2": (delta.norm() / expected.norm()).item(),
               "mean_absolute": delta.abs().mean().item()}
        if name == "native":
            captured = load_file(str(args.native / "prediction_0.safetensors"))["value"]
            row["matches_native_trace"] = bool(torch.equal(prediction, captured.float()))
        report["runs"].append(row)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
