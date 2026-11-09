#!/usr/bin/env python3
"""Compare a diagnostic stock-Comfy Euler trace with native LTX 0.9.8.

The two processes run separately, so the metrics include accumulated differences
in text/image conditioning and inference; they are not isolated kernel errors.
The supported trace is exactly the fixed eight-step strength=1 workflow.
"""
import argparse
import json
from pathlib import Path

from ltx098_comfy_reference import SIGMAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comfy, native = args.comfy.resolve(strict=True), args.native.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")

    import torch
    from safetensors.torch import load_file

    def native_tensor(name):
        return load_file(str(native / f"{name}.safetensors"))["value"].float()

    def stats(reference, value):
        reference, value = reference.float().contiguous(), value.float().contiguous()
        if reference.shape != value.shape:
            raise ValueError(f"tensor geometry differs: {reference.shape} vs {value.shape}")
        delta = reference - value
        return {"relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
                "mae": delta.abs().mean().item(), "max_abs": delta.abs().max().item(),
                "exact": bool(torch.equal(reference, value))}

    sigmas = [float(value.strip()) for value in SIGMAS.split(",")]
    if len(sigmas) != 9:
        raise ValueError("eight-step sigma schedule required")
    first = native_tensor("first_frame")
    report = {"schema": "ltx098-euler-trace-comparison-v1",
              "scope": "Separate complete 512x512x121 strength=1 runs; accumulated conditioning and inference differences",
              "prediction_compared": (native / "prediction_0.safetensors").is_file(), "steps": []}
    for i in range(8):
        saved = load_file(str(comfy / f"step_{i}.safetensors"))
        if abs(saved["sigma"].item() - sigmas[i]) > 1e-6 or abs(saved["sigma_hat"].item() - sigmas[i]) > 1e-6:
            raise ValueError(f"trace sigma mismatch at step {i}")
        x = saved["x"].permute(0, 2, 3, 4, 1).contiguous().float()
        denoised = saved["denoised"].permute(0, 2, 3, 4, 1).contiguous().float()
        reference_next = x + (x - denoised) / saved["sigma"] * (sigmas[i + 1] - sigmas[i])
        native_before = native_tensor("initial_noise" if i == 0 else f"step_{i - 1}")
        native_after = native_tensor(f"step_{i}")
        row = {"step": i, "sigma": sigmas[i],
               "input": stats(x, native_before), "input_first_frame": stats(x[:, :1], native_before[:, :1]),
               "input_motion_frames": stats(x[:, 1:], native_before[:, 1:]),
               "output": stats(reference_next, native_after),
               "output_motion_frames": stats(reference_next[:, 1:], native_after[:, 1:])}
        if report["prediction_compared"]:
            model_input = native_tensor(f"model_input_{i}")
            prediction = native_tensor(f"prediction_{i}")
            native_denoised = model_input - prediction * sigmas[i]
            native_denoised[:, :1] = first
            row["denoised"] = stats(denoised, native_denoised)
            row["denoised_first_frame"] = stats(denoised[:, :1], native_denoised[:, :1])
            row["denoised_motion_frames"] = stats(denoised[:, 1:], native_denoised[:, 1:])
        report["steps"].append(row)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in report["steps"]:
        print(json.dumps({"step": row["step"], "input_relative_l2": row["input"]["relative_l2"],
                          "output_relative_l2": row["output"]["relative_l2"],
                          "denoised_relative_l2": row.get("denoised", {}).get("relative_l2")}))


if __name__ == "__main__":
    main()
