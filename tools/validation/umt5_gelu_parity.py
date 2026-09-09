"""Isolate BF16 GELU arithmetic using identical saved UMT5 activations.

Development only. No model or production precision changes are made here.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="reference.safetensors from --trace-first-block")
    parser.add_argument("--probe", type=Path, help="link-tested native umt5-gelu-probe")
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch

    x = mx.load(args.input)["first_gate_input"]
    t = torch.from_numpy(np.asarray(x.astype(mx.float32)).copy()).to("mps", dtype=torch.bfloat16)
    reference = {}
    with torch.inference_mode():
        reference["power"] = torch.pow(t, 3.0)
        reference["scaled_power"] = 0.044715 * reference["power"]
        reference["inner"] = t + reference["scaled_power"]
        reference["tanh_input"] = math.sqrt(2 / math.pi) * reference["inner"]
        reference["tanh"] = torch.tanh(reference["tanh_input"])
        reference["raised"] = 1.0 + reference["tanh"]
        reference["output"] = (0.5 * t) * reference["raised"]
    report = {}
    for mode in ["bf16_constants", "fp32_scalar_then_round"]:
        def mul(value, scalar):
            if mode == "bf16_constants":
                return value * mx.array(scalar, dtype=mx.bfloat16)
            return (value.astype(mx.float32) * mx.array(scalar, dtype=mx.float32)).astype(mx.bfloat16)

        stages = {}
        stages["power"] = mx.power(x, mx.array(3.0, dtype=mx.bfloat16))
        stages["scaled_power"] = mul(stages["power"], 0.044715)
        stages["inner"] = x + stages["scaled_power"]
        stages["tanh_input"] = mul(stages["inner"], math.sqrt(2 / math.pi))
        stages["tanh"] = mx.tanh(stages["tanh_input"])
        stages["raised"] = mx.array(1.0, dtype=mx.bfloat16) + stages["tanh"]
        stages["output"] = mul(x, 0.5) * stages["raised"]
        report[mode] = {}
        for name, value in stages.items():
            got = np.asarray(value.astype(mx.float32)).astype(np.float64)
            expected = reference[name].float().cpu().numpy().astype(np.float64)
            delta = got - expected
            report[mode][name] = {"exact": bool(np.array_equal(got, expected)),
                                 "mismatched": int(np.count_nonzero(delta)),
                                 "max_abs": float(np.max(np.abs(delta))),
                                 "rmse": float(np.sqrt(np.mean(delta * delta)))}
    if args.probe:
        with tempfile.TemporaryDirectory(prefix="tc-umt5-gelu-") as directory:
            output = Path(directory) / "native.safetensors"
            subprocess.run([str(args.probe.resolve(strict=True)), str(Path(args.input).resolve(strict=True)),
                            str(output)], check=True)
            value = mx.load(str(output))["output"]
            got = np.asarray(value.astype(mx.float32))
            expected = reference["output"].float().cpu().numpy()
            report["native_output"] = {"exact": bool(value.dtype == mx.bfloat16 and np.array_equal(got, expected)),
                                       "mismatched": int(np.count_nonzero(got - expected))}
    print(json.dumps(report, indent=2))
    if not all(stage["exact"] for stage in report["fp32_scalar_then_round"].values()):
        raise SystemExit("scalar boundary regression")
    if args.probe and not report["native_output"]["exact"]:
        raise SystemExit("native GELU differs from the same-input reference")


if __name__ == "__main__":
    main()
