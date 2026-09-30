"""Offline, checkpoint-independent FP16 runtime-weight Core ML micrographs.

Every projection is a model input in checkpoint [out,in] order. GEMM tiling
is inside one graph/prediction, not host dispatches. No model/LoRA weights or
undocumented compiled-model formats are used. This is a research artifact;
CPU_AND_NE is a scheduling policy, not evidence of ANE residency.
"""

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from pathlib import Path


def geometry(kind, rows, hidden, width, tile_k, tile_n, silu_lowering="exp", lora_inputs=False):
    if kind not in ("matmul", "swiglu", "gelu"):
        raise ValueError("unsupported runtime graph kind")
    if silu_lowering not in ("builtin", "sigmoid", "exp"):
        raise ValueError("unsupported SiLU lowering")
    if type(lora_inputs) is not bool or (lora_inputs and kind != "swiglu"):
        raise ValueError("runtime LoRA activation inputs require SwiGLU")
    for value in (rows, hidden, width, tile_k, tile_n):
        if type(value) is not int or not 1 <= value <= 32768:
            raise ValueError("dimensions and tiles must be integers in [1,32768]")
    inputs = {"x": [rows, hidden]}
    if kind == "matmul":
        inputs["w"] = [width, hidden]
    else:
        if kind == "swiglu":
            inputs["wg"] = [width, hidden]
        inputs["wu"] = [width, hidden]
        inputs["wd"] = [hidden, width]
    outputs = {"y": [rows, width if kind == "matmul" else hidden]}
    if lora_inputs:
        inputs.update(dg=[rows, width], du=[rows, width])
        outputs["h"] = [rows, width]
    result = {"schema_version": 1, "backend": "runtime_weight_fp16",
            "graph_version": 2 if lora_inputs else 1, "kind": kind, "rows": rows,
            "hidden": hidden, "width": width, "tile_k": tile_k,
            "tile_n": tile_n, "layout": "out_in", "inputs": inputs,
            "outputs": outputs,
            "biases": False, "silu_lowering": silu_lowering}
    if lora_inputs:
        result["lora_inputs"] = True
    return result


def make_program(spec):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def projection(x, weight):
        columns = []
        reduction, outputs = x.shape[1], weight.shape[0]
        for begin_n in range(0, outputs, spec["tile_n"]):
            end_n = min(begin_n + spec["tile_n"], outputs)
            partial = None
            for begin_k in range(0, reduction, spec["tile_k"]):
                end_k = min(begin_k + spec["tile_k"], reduction)
                a = mb.slice_by_index(x=x, begin=[0, begin_k], end=[spec["rows"], end_k])
                b = mb.slice_by_index(x=weight, begin=[begin_n, begin_k], end=[end_n, end_k])
                product = mb.matmul(x=a, y=b, transpose_y=True)
                partial = product if partial is None else mb.add(x=partial, y=product)
            columns.append(partial)
        return columns[0] if len(columns) == 1 else mb.concat(values=columns, axis=1)

    def swiglu(x, wg, wu, wd, dg=None, du=None):
        import numpy as np
        gate, up = projection(x, wg), projection(x, wu)
        if dg is not None:
            gate, up = mb.add(x=gate, y=dg), mb.add(x=up, y=du)
        if spec["silu_lowering"] == "sigmoid":
            activated = mb.mul(x=gate, y=mb.sigmoid(x=gate))
        elif spec["silu_lowering"] == "exp":
            activated = mb.real_div(x=gate, y=mb.add(x=np.float16(1),
                y=mb.exp(x=mb.mul(x=gate, y=np.float16(-1)))))
        else:
            activated = mb.silu(x=gate)
        hidden = mb.mul(x=activated, y=up)
        output = mb.identity(x=projection(hidden, wd), name="y")
        return (output, mb.identity(x=hidden, name="h")) if dg is not None else output

    signatures = [mb.TensorSpec(shape=tuple(shape), dtype=types.fp16)
                  for shape in spec["inputs"].values()]
    if spec["kind"] == "matmul":
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, w):
            return mb.identity(x=projection(x, w), name="y")
    elif spec.get("lora_inputs", False):
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, wg, wu, wd, dg, du):
            return swiglu(x, wg, wu, wd, dg, du)
    elif spec["kind"] == "swiglu":
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, wg, wu, wd):
            return swiglu(x, wg, wu, wd)
    else:
        @mb.program(input_specs=signatures, opset_version=ct.target.macOS15)
        def program(x, wu, wd):
            hidden = mb.gelu(x=projection(x, wu), mode="TANH_APPROXIMATION")
            return mb.identity(x=projection(hidden, wd), name="y")
    return program


def export(destination, spec, *, program_factory=make_program, compute_precision=None):
    import coremltools as ct

    destination = Path(destination)
    # Do not overwrite even an empty existing directory or a dangling symlink.
    if destination.exists() or destination.is_symlink():
        raise ValueError("output already exists; choose a new artifact directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-ane-", dir=destination.parent) as scratch:
        root = Path(scratch)
        model = ct.convert(program_factory(spec), convert_to="mlprogram",
                           minimum_deployment_target=ct.target.macOS15,
                           compute_precision=(ct.precision.FLOAT16 if compute_precision is None
                                              else compute_precision),
                           skip_model_load=True)
        package = root / "graph.mlpackage"
        model.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        try:
            shutil.copytree(compiled, root / "graph.mlmodelc")
        finally:
            # compile_model returns its own temporary directory, not user data.
            shutil.rmtree(compiled)
        metadata = {**spec, "coremltools": ct.__version__, "macos": platform.mac_ver()[0],
                    "compiled_model": "graph.mlmodelc", "observed_ane_residency": "unknown"}
        metadata["files"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
        }
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        # Keep the source package for reproduction/compilation on another OS.
        root.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("matmul", "swiglu", "gelu"), default="swiglu")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--tile-k", type=int, default=1024)
    parser.add_argument("--tile-n", type=int, default=1024)
    parser.add_argument("--silu-lowering", choices=("builtin", "sigmoid", "exp"), default="exp",
                        help="equivalent SiLU formulas; builtin/sigmoid failed a large-shape NE self-test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-inputs", action="store_true",
                        help="SwiGLU activation corrections and hidden output; no adapter weights in the graph")
    args = parser.parse_args()
    spec = geometry(args.kind, args.rows, args.hidden, args.width, args.tile_k, args.tile_n,
                    args.silu_lowering, args.lora_inputs)
    print(json.dumps(export(args.output, spec), sort_keys=True))


if __name__ == "__main__":
    main()
