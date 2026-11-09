#!/usr/bin/env python3
"""Compare current and FP32-intermediate BF16 SiLU against PyTorch MPS."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe, output = args.probe.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU unavailable")
    generator = torch.Generator().manual_seed(941)
    x = (torch.randn((131072,), generator=generator) * 3).to(torch.bfloat16)
    reference = F.silu(x.to("mps")).cpu().float()
    output.mkdir(parents=True)
    save_file({"x": x}, str(output / "input.safetensors"))
    run = subprocess.run([str(probe), str(output / "input.safetensors"), str(output / "output.safetensors")],
                         capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise RuntimeError(run.stderr)
    report = {"schema": "ltx098-silu-screen-v1", "scope": "synthetic BF16 values; native MLX vs PyTorch MPS"}
    for name, output_tensor in load_file(str(output / "output.safetensors")).items():
        actual = output_tensor.float()
        delta = actual - reference
        report[name] = {"relative_l2": (delta.norm() / reference.norm()).item(), "max_abs": delta.abs().max().item(),
                        "exact_fraction": (actual == reference).float().mean().item()}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
