#!/usr/bin/env python3
"""Check BF16 multiply-add rounding against PyTorch MPS addcmul."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, help="also compare the native MLX implementation")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU unavailable")
    generator = torch.Generator().manual_seed(1983)
    a, b, c = ((torch.randn(131072, generator=generator) * scale).to("mps", torch.bfloat16)
               for scale in (1., 1., .4))

    def report(reference, candidate):
        reference, candidate = reference.cpu().float(), candidate.cpu().float()
        delta = candidate - reference
        return {"relative_l2": (delta.norm() / reference.norm()).item(), "max_abs": delta.abs().max().item(),
                "exact_fraction": (candidate == reference).float().mean().item()}

    residual = torch.addcmul(a, b, c)
    modulation = torch.addcmul(b, b, c)
    result = {"schema": "ltx098-addcmul-parity-v1", "scope": "PyTorch MPS BF16 operator rounding; synthetic values",
              "residual": {"eager_bf16": report(residual, a + b * c),
                           "fp32_single_round": report(residual, (a.float() + b.float() * c.float()).bfloat16())},
              "modulation": {"eager_bf16": report(modulation, b * (1 + c)),
                             "fp32_single_round": report(modulation, (b.float() + b.float() * c.float()).bfloat16())}}
    if args.probe:
        from safetensors.torch import load_file, save_file
        probe = args.probe.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="ltx098-addcmul-") as temporary:
            directory = Path(temporary)
            result["native"] = {}
            for name, values in (("equal_shape", (a, b, c)),
                                 ("broadcast", (a[:32768].reshape(1, 16, 2048), b[:32768].reshape(1, 16, 2048), c[:2048].reshape(1, 1, 2048)))):
                x, y, z = values
                input_path, output_path = directory / f"{name}-in.safetensors", directory / f"{name}-out.safetensors"
                save_file({key: value.cpu().contiguous() for key, value in zip(("a", "b", "c"), values)}, str(input_path))
                run = subprocess.run([str(probe), str(input_path), str(output_path)], capture_output=True, text=True, timeout=120)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                actual = load_file(str(output_path))["output"]
                result["native"][name] = report(torch.addcmul(x, y, z), actual)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
