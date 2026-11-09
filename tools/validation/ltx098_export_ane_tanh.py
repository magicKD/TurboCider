#!/usr/bin/env python3
"""Export a runtime-weight LTX FFN with explicit tanh-GELU.

This is an isolated Core ML experiment: CPU_AND_NE may still fall back to CPU.
Optional runtime bias inputs are experimental; no weights or production path
are included.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/coreml"))
from export_runtime_ane import export, geometry


def explicit_program(spec):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import numpy as np

    def projection(x, weight):
        columns = []
        for n in range(0, weight.shape[0], spec["tile_n"]):
            partial = None
            for k in range(0, x.shape[1], spec["tile_k"]):
                a = mb.slice_by_index(x=x, begin=[0, k], end=[spec["rows"], min(k + spec["tile_k"], x.shape[1])])
                b = mb.slice_by_index(x=weight, begin=[n, k],
                    end=[min(n + spec["tile_n"], weight.shape[0]), min(k + spec["tile_k"], x.shape[1])])
                product = mb.matmul(x=a, y=b, transpose_y=True)
                partial = product if partial is None else mb.add(x=partial, y=product)
            columns.append(partial)
        return columns[0] if len(columns) == 1 else mb.concat(values=columns, axis=1)

    signatures = [mb.TensorSpec(shape=tuple(shape), dtype=types.fp16) for shape in spec["inputs"].values()]

    def activation(value):
        cube = mb.mul(x=mb.mul(x=value, y=value), y=value)
        inner = mb.mul(x=mb.add(x=value, y=mb.mul(x=cube, y=np.float16(.044715))), y=np.float16(.7978845608))
        return mb.mul(x=mb.mul(x=value, y=np.float16(.5)),
                      y=mb.add(x=mb.tanh(x=inner), y=np.float16(1)))

    if spec["biases"]:
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, wu, wd, bu, bd):
            value = mb.add(x=projection(x, wu), y=bu)
            return mb.identity(x=mb.add(x=projection(activation(value), wd), y=bd), name="y")
    else:
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, wu, wd):
            return mb.identity(x=projection(activation(projection(x, wu)), wd), name="y")

    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=2048)
    parser.add_argument("--tile-n", type=int, default=2048)
    parser.add_argument("--bias-inputs", action="store_true", help="experimental graph with runtime up/down bias inputs; not the production runtime ABI")
    args = parser.parse_args()
    spec = geometry("gelu", args.rows, 2048, 8192, args.tile_k, args.tile_n)
    spec["gelu_lowering"] = "explicit_tanh_fp16"
    if args.bias_inputs:
        spec["backend"] = "ltx098_experimental_bias_fp16"
        spec["biases"] = True
        spec["inputs"].update(bu=[8192], bd=[2048])
    result = export(args.output, spec, program_factory=explicit_program)
    print(f"exported {args.output} ({result['gelu_lowering']})")


if __name__ == "__main__":
    main()
