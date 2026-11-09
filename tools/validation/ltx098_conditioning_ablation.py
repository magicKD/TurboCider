#!/usr/bin/env python3
"""Replay the first LTX Euler step with native/Comfy conditioning combinations.

The stock Comfy trace is the reference; its velocity on motion frames follows
from x - denoised at sigma=1. This is a diagnostic, not an end-to-end gate.
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
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-changed-replay", action="store_true",
                        help="compare an intentionally changed transformer against the captured native baseline")
    args = parser.parse_args()
    checkpoint, probe, diagnostic, comfy_trace, native_trace = (p.resolve(strict=True) for p in
        (args.checkpoint, args.probe, args.comfy_diagnostic, args.comfy_trace, args.native_trace))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")

    import torch
    from safetensors.torch import load_file, save_file

    def native(name):
        return load_file(str(native_trace / f"{name}.safetensors"))["value"]

    def stats(reference, candidate):
        reference, candidate = reference.float().contiguous(), candidate.float().contiguous()
        if reference.shape != candidate.shape:
            raise ValueError("tensor shapes differ")
        difference = reference - candidate
        return {"relative_l2": (difference.norm() / reference.norm().clamp_min(1e-30)).item(),
                "mae": difference.abs().mean().item(), "max_abs": difference.abs().max().item(),
                "exact": bool(torch.equal(reference, candidate))}

    snapshot = load_file(str(comfy_trace / "step_0.safetensors"))
    if snapshot["sigma"].item() != 1.0 or snapshot["sigma_hat"].item() != 1.0:
        raise ValueError("requires the sigma=1 first Euler step")
    state = snapshot["x"].permute(0, 2, 3, 4, 1).contiguous()
    denoised = snapshot["denoised"].permute(0, 2, 3, 4, 1).contiguous()
    if not torch.equal(state, native("initial_noise")):
        raise ValueError("native and Comfy initial noise differ")
    comfy_first = load_file(str(diagnostic / "0-condition_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1)[:, :1]
    text_data = load_file(str(diagnostic / "0-text_00001_.safetensors"))
    if tuple(state.shape) != (1, 16, 16, 16, 128) or tuple(text_data["conditioning"].shape) != (1, 128, 4096):
        raise ValueError("requires the 512x512x121 reference capture")
    valid = text_data["attention_mask"]
    if not torch.equal(valid, (torch.arange(valid.shape[1])[None, :] < valid.sum()).long()):
        raise ValueError("non-contiguous attention mask unsupported")
    mask = torch.where(valid[:, None, None, :].bool(), 0., -10000.).to(torch.bfloat16).contiguous()
    text = {"comfy": text_data["conditioning"].to(torch.bfloat16), "native": native("text")}
    first = {"comfy": comfy_first, "native": native("first_frame")}
    expected_velocity = (state - denoised)[:, 1:]
    output.mkdir(parents=True)
    report = {"schema": "ltx098-conditioning-ablation-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "probe_sha256": sha256(probe), "valid_text_tokens": int(valid.sum()),
              "scope": "first Euler step; actual 512x512x121 latent; strength 1; Comfy and native condition combinations",
              "text_valid": stats(text["comfy"][:, :valid.sum()], text["native"][:, :valid.sum()]),
              "first_frame": stats(first["comfy"], first["native"]), "runs": []}
    try:
        for text_source, frame_source in (("native", "native"), ("comfy", "native"),
                                          ("native", "comfy"), ("comfy", "comfy")):
            label = f"text-{text_source}-frame-{frame_source}"
            x = state.clone()
            x[:, :1] = first[frame_source].float()
            input_path, output_path = output / f"{label}-input.safetensors", output / f"{label}-prediction.safetensors"
            save_file({"latents": x.to(torch.bfloat16).contiguous(), "text": text[text_source].contiguous(),
                       "text_mask": mask}, str(input_path))
            command = [str(probe), str(checkpoint), str(input_path), str(output_path), "28", "1", "1.0"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"{label}: {result.stderr[-4000:]}")
            prediction = load_file(str(output_path))["output"].float()
            row = {"text": text_source, "first_frame": frame_source,
                   "velocity_motion_frames": stats(expected_velocity, prediction[:, 1:]),
                   "denoised_motion_frames": stats(denoised[:, 1:], state[:, 1:] - prediction[:, 1:])}
            if label == "text-native-frame-native":
                row["native_trace_prediction"] = stats(native("prediction_0"), prediction)
                if not row["native_trace_prediction"]["exact"] and not args.allow_changed_replay:
                    raise ValueError("native/native replay differs from captured native prediction")
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
