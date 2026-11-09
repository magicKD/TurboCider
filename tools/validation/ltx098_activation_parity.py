#!/usr/bin/env python3
"""Compare legacy and FP32-intermediate LTX GELU against PyTorch MPS BF16."""
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
    generator = torch.Generator().manual_seed(452)
    values = (torch.randn((131072,), generator=generator) * 3).to(torch.bfloat16)
    reference = F.gelu(values.to("mps"), approximate="tanh").cpu()
    output.mkdir(parents=True)
    save_file({"input": values}, str(output / "input.safetensors"))
    run = subprocess.run([str(probe), str(output / "input.safetensors"), str(output / "output.safetensors")],
                         capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise RuntimeError(run.stderr)
    results = load_file(str(output / "output.safetensors"))
    report = {"schema": "ltx098-activation-parity-v1", "reference": "pytorch_mps_gelu_tanh_bf16", "count": values.numel()}
    for name in ("old", "candidate"):
        delta = results[name].float() - reference.float()
        report[name] = {"relative_l2": (delta.norm() / reference.float().norm()).item(),
                        "max_abs": delta.abs().max().item(),
                        "exact_fraction": (results[name] == reference).float().mean().item()}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
