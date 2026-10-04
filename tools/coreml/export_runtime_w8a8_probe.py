"""Export independent FP16/frozen-QDQ/runtime-QDQ MatMul feasibility controls.

Normalized FP16 interfaces are not memory-compressed INT8 slots. QDQ structure
does not prove ANE placement or physical INT8 arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np


def geometry(rows, hidden, width, arm, tile_k=None, tile_n=None):
    if arm not in ("fp16", "runtime_qdq", "frozen_qdq"):
        raise ValueError("unknown arm")
    for value in (rows, hidden, width):
        if type(value) is not int or not 1 <= value <= 32768:
            raise ValueError("invalid dimension")
    if hidden * 127 * 128 > np.iinfo(np.int32).max:
        raise ValueError("unsafe integer dot bound")
    tile_k = hidden if tile_k is None else tile_k
    tile_n = width if tile_n is None else tile_n
    for value in (tile_k, tile_n):
        if type(value) is not int or not 1 <= value <= 32768:
            raise ValueError("invalid tile")
    nodes = ((hidden + tile_k - 1) // tile_k) * ((width + tile_n - 1) // tile_n)
    if nodes > 4096:
        raise ValueError("probe graph exceeds 4096 MatMul nodes")
    return {"schema": "tc-runtime-w8a8-proof-v1", "arm": arm, "rows": rows,
            "hidden": hidden, "width": width, "tile_k": tile_k, "tile_n": tile_n,
            "layout": "out_in", "io_dtype": "fp16", "normalization_scale": 1 / 128,
            "requested_arithmetic": "int8_candidate" if arm != "fp16" else "fp16",
            "arithmetic_evidence": "unknown", "production_qualified": False}


def make_program(spec, weight_codes=None):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if spec["arm"] == "frozen_qdq":
        if (weight_codes is None or weight_codes.dtype != np.int8
                or weight_codes.shape != (spec["width"], spec["hidden"])):
            raise ValueError("frozen arm requires exact I8 weight fixture")

    def qdq(value, name):
        quantized = mb.quantize(input=value, scale=np.float16(1 / 128),
                                zero_point=np.int8(0), output_dtype="int8", name=name + "_q")
        return mb.dequantize(input=quantized, scale=np.float16(1 / 128),
                             zero_point=np.int8(0), name=name + "_dq")

    def project(x, w):
        if spec["arm"] != "fp16":
            x = qdq(x, "activation")
            if spec["arm"] == "frozen_qdq":
                w = mb.dequantize(input=weight_codes, scale=np.float16(1 / 128),
                                   zero_point=np.int8(0), name="frozen_weight_dq")
            else:
                w = qdq(w, "weight")
        columns = []
        for n in range(0, spec["width"], spec["tile_n"]):
            end_n = min(n + spec["tile_n"], spec["width"])
            partial = None
            for k in range(0, spec["hidden"], spec["tile_k"]):
                end_k = min(k + spec["tile_k"], spec["hidden"])
                a = mb.slice_by_index(x=x, begin=[0, k], end=[spec["rows"], end_k])
                b = mb.slice_by_index(x=w, begin=[n, k], end=[end_n, end_k])
                value = mb.matmul(x=a, y=b, transpose_y=True,
                                  name=f"integer_candidate_n{n}_k{k}")
                partial = value if partial is None else mb.add(x=partial, y=value)
            columns.append(partial)
        value = columns[0] if len(columns) == 1 else mb.concat(values=columns, axis=1)
        return mb.identity(x=value, name="y")

    x = mb.TensorSpec(shape=(spec["rows"], spec["hidden"]), dtype=types.fp16)
    if spec["arm"] == "frozen_qdq":
        @mb.program(input_specs=[x], opset_version=ct.target.macOS15)
        def program(x):
            return project(x, None)
    else:
        weight = mb.TensorSpec(shape=(spec["width"], spec["hidden"]), dtype=types.fp16)

        @mb.program(input_specs=[x, weight], opset_version=ct.target.macOS15)
        def program(x, w):
            return project(x, w)
    return program


def graph_operations(model):
    spec = model.get_spec()
    result = []

    def visit(block):
        for op in block.operations:
            result.append(op.type)
            for nested in op.blocks:
                visit(nested)

    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit(block)
    return result


def export(destination, spec, weight_codes=None):
    import coremltools as ct

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-qdq-proof-", dir=destination.parent) as temporary:
        root = Path(temporary)
        program = make_program(spec, weight_codes)
        before = [op.op_type for op in program.functions["main"].operations]
        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        model.save(str(root / "graph.mlpackage"))
        compiled = Path(ct.models.utils.compile_model(str(root / "graph.mlpackage")))
        try:
            shutil.copytree(compiled, root / "graph.mlmodelc")
        finally:
            # Only the temporary directory returned by this compile invocation.
            shutil.rmtree(compiled)
        metadata = {**spec, "coremltools": ct.__version__, "macos": platform.mac_ver()[0],
                    "program_ops_before_convert": before,
                    "program_ops_after_convert": graph_operations(model), "model_loads_required": 1,
                    "weight_fixture_sha256": (hashlib.sha256(weight_codes.tobytes()).hexdigest()
                                              if weight_codes is not None else None)}
        metadata["files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(root.rglob("*")) if p.is_file()}
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        root.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--arm", choices=["fp16", "runtime_qdq", "frozen_qdq"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-k", type=int)
    parser.add_argument("--tile-n", type=int)
    args = parser.parse_args()
    spec = geometry(args.rows, args.hidden, args.width, args.arm, args.tile_k, args.tile_n)
    weights = (np.random.default_rng(42).integers(-128, 128, size=(args.width, args.hidden), dtype=np.int8)
               if args.arm == "frozen_qdq" else None)
    print(json.dumps(export(args.output, spec, weights), indent=2))


if __name__ == "__main__":
    main()
