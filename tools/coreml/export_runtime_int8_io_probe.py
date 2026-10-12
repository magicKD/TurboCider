"""Public macOS26 INT8 runtime-weight MatMul with explicit compressed IO.

Checkpoint-independent feasibility control, not production/ANE-arithmetic proof.
Use isolated coremltools>=9; the existing project environment is not upgraded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np


def geometry(rows, hidden, width, arm):
    if arm not in ("int8", "fp16"):
        raise ValueError("arm requires int8 or fp16")
    if any(type(v) is not int or v < 1 for v in (rows, hidden, width)) or rows > 4224 or hidden > 4096 or width > 16384:
        raise ValueError("probe dimensions require rows<=4224, hidden<=4096, width<=16384")
    if 2*(hidden*rows+width*hidden+width*rows) > 512 << 20:
        raise ValueError("probe declared FP16 slots exceed 512MiB bound")
    return {"schema": "tc-public-runtime-int8-io-probe-v1", "rows": rows,
            "hidden": hidden, "width": width, "arm": arm,
            "input_dtype": arm, "output_dtype": "fp16", "minimum_macos": "26.0",
            "inputs": {"x": [hidden, rows], "w": [width, hidden]},
            "outputs": {"y": [width, rows]}, "normalization_scale": 1/128,
            "arithmetic_evidence": "unknown", "observed_ane_residency": "unknown",
            "production_qualified": False}


def make_program(spec):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    if not hasattr(ct.target, "macOS26"):
        raise RuntimeError("INT8 public IO requires isolated coremltools>=9 with macOS26 target")
    dtype = types.int8 if spec["arm"] == "int8" else types.fp16
    signatures = [mb.TensorSpec(shape=tuple(shape), dtype=dtype)
                  for shape in spec["inputs"].values()]

    @mb.program(input_specs=signatures, opset_version=ct.target.macOS26)
    def program(x, w):
        if spec["arm"] == "int8":
            x = mb.dequantize(input=x, scale=np.float16(1/128), name="activation_dq")
            w = mb.dequantize(input=w, scale=np.float16(1/128), name="weight_dq")
        return mb.identity(x=mb.matmul(x=w, y=x), name="y")
    return program


def export(destination, spec):
    import coremltools as ct
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("output already exists")
    program = make_program(spec)
    dtype = np.int8 if spec["arm"] == "int8" else np.float16
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".public-int8-io-", dir=destination.parent) as temporary:
        root = Path(temporary)
        model = ct.convert(program, convert_to="mlprogram",
                           minimum_deployment_target=ct.target.macOS26,
                           inputs=[ct.TensorType(name=name, shape=shape, dtype=dtype)
                                   for name, shape in spec["inputs"].items()],
                           outputs=[ct.TensorType(name="y", dtype=np.float16)],
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        features = model.get_spec().description
        enum = ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT8 if spec["arm"] == "int8" else ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
        if (set(value.name for value in features.input) != {"x", "w"} or
                any(value.type.multiArrayType.dataType != enum for value in features.input) or
                len(features.output) != 1 or features.output[0].type.multiArrayType.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16):
            raise ValueError("converted public graph did not retain declared INT8/FP16 interfaces")
        model.save(str(root / "graph.mlpackage"))
        compiled = Path(ct.models.utils.compile_model(str(root / "graph.mlpackage")))
        try:
            shutil.copytree(compiled, root / "graph.mlmodelc")
        finally:
            shutil.rmtree(compiled) # only this compiler-owned temporary artifact
        metadata = {**spec, "coremltools": ct.__version__, "macos": platform.mac_ver()[0],
                    "compiled_model": "graph.mlmodelc", "files": {
                        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(root.rglob("*")) if path.is_file()}}
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
        root.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--arm", choices=("int8", "fp16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, geometry(args.rows,args.hidden,args.width,args.arm))))


if __name__ == "__main__":
    main()
