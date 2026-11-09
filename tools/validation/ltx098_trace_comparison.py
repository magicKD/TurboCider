#!/usr/bin/env python3
"""Compare real ComfyUI and native LTX 0.9.8 conditioning/latent traces.

PNG-versus-raw RGB diagnostics remove MP4 encoder differences, but the two
decoders still use different random noise streams. No quality gate is inferred.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True, help="Comfy diagnostic output/diagnostic directory")
    parser.add_argument("--native", type=Path, required=True, help="native trace-0 directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comfy, native = args.comfy.resolve(strict=True), args.native.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    def native_tensor(name):
        return load_file(str(native / f"{name}.safetensors"))["value"]

    def metrics(a, b):
        a, b = a.float().contiguous(), b.float().contiguous()
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {a.shape} != {b.shape}")
        delta = a - b
        return {"shape": list(a.shape), "relative_l2": (delta.norm() / a.norm().clamp_min(1e-30)).item(),
                "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                "max_abs": delta.abs().max().item(), "exact": bool(torch.equal(a, b))}

    cond = load_file(str(comfy / "0-condition_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1)
    final = load_file(str(comfy / "0-final_00001_.latent"))["latent_tensor"].permute(0, 2, 3, 4, 1)
    text = load_file(str(comfy / "0-text_00001_.safetensors"))
    valid = int(text["attention_mask"].sum().item())
    noise = torch.randn((1, 128, 16, 16, 16), generator=torch.Generator().manual_seed(42)).permute(0, 2, 3, 4, 1)
    report = {"schema": "ltx098-trace-comparison-v1", "valid_text_tokens": valid,
              "text_valid": metrics(text["conditioning"][:, :valid], native_tensor("text")[:, :valid]),
              "first_frame_latent": metrics(cond[:, :1], native_tensor("first_frame")),
              "initial_noise": metrics(noise, native_tensor("initial_noise")),
              "final_latent": metrics(final, native_tensor("step_7")),
              "final_first_frame": metrics(final[:, :1], native_tensor("step_7")[:, :1])}
    pixels = native_tensor("pixels").float().numpy()[0]
    paths = sorted(comfy.glob("0-frame_*.png"))
    if pixels.shape != (121, 512, 512, 3) or len(paths) != 121:
        raise ValueError("diagnostic requires all 121 original-resolution frames")
    previous = None
    frames, motion = [], []
    for index, path in enumerate(paths):
        reference = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        candidate = np.clip((pixels[index] + 1) * 127.5, 0, 255).astype(np.uint8)
        a, b = reference.astype(np.float32), candidate.astype(np.float32)
        delta = a - b
        frames.append({"frame": index, "mae_255": float(np.abs(delta).mean()),
                       "rmse_255": float(np.sqrt(np.mean(delta * delta)))})
        if previous is not None:
            left = float(np.abs(reference.astype(np.int16) - previous[0]).mean())
            right = float(np.abs(candidate.astype(np.int16) - previous[1]).mean())
            motion.append({"from_frame": index - 1, "reference_energy": left, "candidate_energy": right,
                           "relative_difference": abs(left - right) / max(left, right, 1.0)})
        previous = (reference.astype(np.int16), candidate.astype(np.int16))
    report["raw_rgb"] = {"frames": len(frames), "mean_mae_255": float(np.mean([row["mae_255"] for row in frames])),
                         "maximum_mae_255": max(row["mae_255"] for row in frames),
                         "maximum_motion_relative_error": max(row["relative_difference"] for row in motion),
                         "largest_motion_differences": sorted(motion, key=lambda row: row["relative_difference"], reverse=True)[:10],
                         "frame_metrics": frames, "motion_metrics": motion}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rgb"}, indent=2))
    print(json.dumps({key: value for key, value in report["raw_rgb"].items() if key not in ("frame_metrics", "motion_metrics")}, indent=2))


if __name__ == "__main__":
    main()
