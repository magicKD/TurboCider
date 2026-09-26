#!/usr/bin/env python3
"""Export one Gemma gated-MLP channel split for GPU+ANE experiments.

Gemma4-LTX stores gate/up/down as per-output-channel INT8 ConvRot weights.
The ANE owns a 256-aligned prefix of intermediate channels.  Its Core ML
branch folds the input ConvRot into gate/up, applies GELU-tanh and the gated
product, then folds the intermediate ConvRot into the down prefix.  The GPU
suffix remains in the original INT8+scale representation and is written as
small row/column slices for the native runtime.

This exporter only creates an artifact; the native Gemma encoder does not
enable it until a matching runtime quality/speed sweep has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path


SCHEMA = "ltx-gemma-ane-mlp-v1"
GROUP = 256
HIDDEN = 3840
INTERMEDIATE = 15360
LAYERS = 48
SUPPORTED_ROWS = (64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--rows", default="64,128,256,512,1024")
    parser.add_argument("--ane-intermediate", type=int, required=True)
    parser.add_argument(
        "--min-profitable-rows",
        help=("comma-separated BUCKET:MIN_ACTUAL_ROWS crossovers; omitted "
              "buckets allow exact execution only"),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiled-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", encoded)[0]
        if length > 100_000_000:
            raise ValueError("safetensors header is unreasonably large")
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError("truncated safetensors JSON")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("safetensors header must be an object")
    return 8 + length, decoded


def tensor_info(header: dict[str, object], name: str, dtype: str,
                shape: tuple[int, ...]) -> tuple[int, int]:
    value = header.get(name)
    if not isinstance(value, dict) or value.get("dtype") != dtype:
        raise ValueError(f"missing {dtype} tensor {name}")
    actual = tuple(int(item) for item in value.get("shape", ()))
    if actual != shape:
        raise ValueError(f"{name}: expected {shape}, found {actual}")
    offsets = value.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{name}: invalid data offsets")
    start, stop = (int(item) for item in offsets)
    if start < 0 or stop <= start:
        raise ValueError(f"{name}: invalid data range")
    return start, stop


def mapped_tensor(path: Path, data_offset: int, info: tuple[int, int],
                  dtype, shape):
    import numpy as np

    start, stop = info
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if stop - start != expected:
        raise ValueError(f"tensor byte count mismatch: {stop - start} != {expected}")
    return np.memmap(path, mode="r", dtype=dtype,
                     offset=data_offset + start, shape=shape)


def round_bf16(values):
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def convrot_last_axis(values):
    import numpy as np

    if values.shape[-1] % GROUP:
        raise ValueError("ConvRot axis must be divisible by 256")
    result = np.ascontiguousarray(values, dtype=np.float32)
    flat = result.reshape(-1, GROUP)
    for stride in (1, 4, 16, 64):
        view = flat.reshape(-1, GROUP // (4 * stride), 4, stride)
        a, b = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
        c, d = view[:, :, 2, :].copy(), view[:, :, 3, :].copy()
        view[:, :, 0, :] = a + b + c - d
        view[:, :, 1, :] = a + b - c + d
        view[:, :, 2, :] = a - b + c + d
        view[:, :, 3, :] = -a + b + c + d
    flat *= np.float32(0.0625)
    return result


def absorb_convrot(weight_i8, scale_f32):
    import numpy as np

    dequantized = (np.asarray(weight_i8, dtype=np.float32) *
                   np.asarray(scale_f32, dtype=np.float32).reshape(-1, 1))
    return convrot_last_axis(round_bf16(dequantized)).astype(np.float16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(path: Path, array, files: dict[str, object]) -> None:
    import numpy as np

    value = np.ascontiguousarray(array)
    value.tofile(path)
    files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def build_model(rows: tuple[int, ...], hidden: int, intermediate: int,
                gate_weight, up_weight, down_weight):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Symbol, types

    gate = np.ascontiguousarray(gate_weight[:, :, None, None])
    up = np.ascontiguousarray(up_weight[:, :, None, None])
    down = np.ascontiguousarray(down_weight[:, :, None, None])
    row_dimension = rows[0] if len(rows) == 1 else Symbol("rows")

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, hidden, 1, row_dimension),
                                   dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        gate_value = mb.conv(x=x, weight=mb.const(val=gate, name="gate_weight"),
                             pad_type="valid", strides=[1, 1],
                             dilations=[1, 1], groups=1, name="gate")
        up_value = mb.conv(x=x, weight=mb.const(val=up, name="up_weight"),
                           pad_type="valid", strides=[1, 1],
                           dilations=[1, 1], groups=1, name="up")
        activated = mb.gelu(x=gate_value, mode="TANH_APPROXIMATION",
                            name="gelu")
        product = mb.mul(x=activated, y=up_value, name="gated_product")
        return mb.conv(x=product, weight=mb.const(val=down, name="down_weight"),
                       pad_type="valid", strides=[1, 1],
                       dilations=[1, 1], groups=1, name="y")

    convert_arguments = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.macOS15,
        "compute_precision": ct.precision.FLOAT16,
    }
    if len(rows) > 1:
        convert_arguments.update({
            "source": "milinternal",
            "inputs": [ct.TensorType(
                name="x",
                shape=ct.EnumeratedShapes(
                    [(1, hidden, 1, value) for value in rows],
                    default=(1, hidden, 1, rows[0]),
                ),
                dtype=np.float16,
            )],
        })
    model = ct.convert(program, **convert_arguments)
    row_label = "/".join(str(value) for value in rows)
    model.input_description["x"] = (
        f"row-major FP16 [{row_label},{hidden}] viewed as [1,{hidden},1,rows]"
    )
    model.output_description["y"] = (
        f"row-major FP16 [{row_label},{hidden}] Gemma gated-MLP partial"
    )
    model.user_defined_metadata.update({
        "ltx.schema": SCHEMA,
        "ltx.shape": f"{row_label}x{hidden}x{intermediate}",
        "ltx.activation": "gelu-tanh-gated",
        "ltx.convrot": "absorbed-256",
    })
    return model


def compile_model(package: Path, destination: Path) -> None:
    import coremltools as ct

    loaded = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    temporary = Path(loaded.get_compiled_model_path())
    try:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(temporary, destination)
    finally:
        loaded = None
        if temporary.resolve() != destination.resolve():
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.compiled_only and not args.compile:
        raise SystemExit("--compiled-only requires --compile")
    if args.block < 0 or args.block >= LAYERS:
        raise SystemExit("--block must be in [0, 47]")
    try:
        rows = tuple(int(value) for value in args.rows.split(","))
    except ValueError as error:
        raise SystemExit("--rows must be comma-separated integers") from error
    if not rows or rows != tuple(sorted(set(rows))) or any(
            value not in SUPPORTED_ROWS for value in rows):
        raise SystemExit(f"--rows must be increasing values from {SUPPORTED_ROWS}")
    minimum_profitable_rows = {}
    if args.min_profitable_rows:
        for entry in args.min_profitable_rows.split(","):
            try:
                bucket_text, minimum_text = entry.split(":", 1)
                bucket, minimum = int(bucket_text), int(minimum_text)
            except ValueError as error:
                raise SystemExit(
                    "--min-profitable-rows must contain BUCKET:MIN pairs"
                ) from error
            if (bucket not in rows or str(bucket) in minimum_profitable_rows or
                    minimum < 1 or minimum > bucket):
                raise SystemExit(
                    "each minimum-profitable-row policy must name one selected "
                    "bucket and stay within 1...BUCKET"
                )
            minimum_profitable_rows[str(bucket)] = minimum
    ane = args.ane_intermediate
    if ane <= 0 or ane >= INTERMEDIATE or ane % GROUP:
        raise SystemExit("--ane-intermediate must be a proper 256-aligned prefix")
    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        raise SystemExit("checkpoint must be a regular non-symlink file")
    if args.dry_run:
        print(json.dumps({
            "schema": SCHEMA, "block": args.block, "rows": list(rows),
            "hidden": HIDDEN, "intermediate": INTERMEDIATE,
            "ane_intermediate": ane, "gpu_intermediate": INTERMEDIATE - ane,
        }, sort_keys=True))
        return

    try:
        import numpy as np
        import coremltools as ct
    except ImportError as error:
        raise SystemExit("export requires numpy and coremltools") from error

    data_offset, header = read_header(args.checkpoint)
    prefix = f"model.layers.{args.block}.mlp."
    names = {}
    for stem, shape in (
        ("gate_proj", (INTERMEDIATE, HIDDEN)),
        ("up_proj", (INTERMEDIATE, HIDDEN)),
        ("down_proj", (HIDDEN, INTERMEDIATE)),
    ):
        weight_name = f"{prefix}{stem}.weight"
        scale_name = f"{prefix}{stem}.weight_scale"
        names[stem] = (
            mapped_tensor(args.checkpoint, data_offset,
                          tensor_info(header, weight_name, "I8", shape),
                          np.int8, shape),
            mapped_tensor(args.checkpoint, data_offset,
                          tensor_info(header, scale_name, "F32", (shape[0], 1)),
                          np.dtype("<f4"), (shape[0], 1)),
        )
        metadata = header.get(f"{prefix}{stem}.comfy_quant")
        if not isinstance(metadata, dict) or metadata.get("dtype") != "U8":
            raise ValueError(f"{weight_name} is missing comfy_quant metadata")

    output = args.out_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; pass --force to replace")
    files: dict[str, object] = {}
    gate_weight, gate_scale = names["gate_proj"]
    up_weight, up_scale = names["up_proj"]
    down_weight, down_scale = names["down_proj"]
    save_array(output / "gpu_gate.weight.i8", gate_weight[ane:], files)
    save_array(output / "gpu_gate.scale.f32", gate_scale[ane:], files)
    save_array(output / "gpu_up.weight.i8", up_weight[ane:], files)
    save_array(output / "gpu_up.scale.f32", up_scale[ane:], files)
    save_array(output / "gpu_down.weight.i8", down_weight[:, ane:], files)
    save_array(output / "gpu_down.scale.f32", down_scale, files)

    print("folding Gemma input ConvRot into ANE gate/up", flush=True)
    ane_gate = absorb_convrot(gate_weight[:ane], gate_scale[:ane])
    ane_up = absorb_convrot(up_weight[:ane], up_scale[:ane])
    print("folding Gemma intermediate ConvRot into ANE down", flush=True)
    ane_down = absorb_convrot(down_weight[:, :ane], down_scale)
    base_model = build_model(rows, HIDDEN, ane, ane_gate, ane_up, ane_down)
    artifacts: dict[str, str] = {}
    compiled: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ltx_gemma_ane_export_",
                                      dir=output) as temporary:
        package = output / "gemma_mlp_fp16.mlpackage"
        base_package = Path(temporary) / "base.mlpackage"
        base_model.save(base_package)
        if package.exists():
            shutil.rmtree(package)
        shutil.copytree(base_package, package)
        if not args.compiled_only:
            artifacts["fp16"] = package.name
        if args.compile:
            destination = output / "gemma_mlp_fp16.mlmodelc"
            compile_model(package, destination)
            compiled["fp16"] = destination.name
        if args.compiled_only:
            shutil.rmtree(package)

    stat = args.checkpoint.stat()
    manifest = {
        "schema": SCHEMA,
        "block_index": args.block,
        "shape": {
            "rows": rows[0], "supported_rows": list(rows),
            "hidden": HIDDEN, "intermediate": INTERMEDIATE,
            **({"minimum_profitable_rows": minimum_profitable_rows}
               if minimum_profitable_rows else {}),
        },
        "partition": {
            "ane": {"start": 0, "width": ane},
            "gpu": {"start": ane, "width": INTERMEDIATE - ane},
        },
        "source": {
            "path": str(args.checkpoint.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(args.checkpoint),
            "prefix": prefix,
        },
        "activation": "gelu_tanh_gated",
        "convrot": {"group_size": GROUP, "ane_weights": "absorbed_fp16",
                    "gpu_weights": "checkpoint_int8"},
        "tensor_abi": {
            "logical": [rows[0], HIDDEN],
            "supported_logical": [[value, HIDDEN] for value in rows],
            "coreml": [1, HIDDEN, 1, rows[0]],
            "strides": [rows[0] * HIDDEN, 1, rows[0] * HIDDEN, HIDDEN],
            "input": "caller-owned-row-major-fp16",
            "output": "caller-owned-row-major-fp16-partial",
        },
        "artifacts": artifacts, "compiled_artifacts": compiled,
        "files": files,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
