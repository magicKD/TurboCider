#!/usr/bin/env python3
"""Compare LTX RMSNorm rounding variants against PyTorch MPS BF16."""
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
    x = torch.randn((1, 128, 2048), generator=generator).to(torch.bfloat16)
    weight = (1 + torch.randn((2048,), generator=generator) * .25).to(torch.bfloat16)
    output.mkdir(parents=True)
    save_file({"x": x, "weight": weight}, str(output / "input.safetensors"))
    report = {"schema": "ltx098-rms-screen-v1", "scope": "synthetic BF16 activations/weights; native MLX vs PyTorch MPS", "epsilons": {}}
    for epsilon in (1e-6, 1e-5):
        path = output / f"eps-{epsilon}.safetensors"
        result = subprocess.run([str(probe), str(output / "input.safetensors"), str(path), str(epsilon)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
        values = load_file(str(path))
        x_mps, weight_mps = x.to("mps"), weight.to("mps")
        references = {"unweighted": F.rms_norm(x_mps, (2048,), eps=epsilon).cpu().float(),
                      "weighted": F.rms_norm(x_mps, (2048,), weight=weight_mps, eps=epsilon).cpu().float()}
        rows = {}
        for name, value in values.items():
            reference = references[name.split("_")[0]]
            actual = value.float()
            diff = actual - reference
            rows[name] = {"relative_l2": (diff.norm() / reference.norm()).item(),
                          "max_abs": diff.abs().max().item(),
                          "exact_fraction": (actual == reference).float().mean().item()}
        report["epsilons"][str(epsilon)] = rows
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
