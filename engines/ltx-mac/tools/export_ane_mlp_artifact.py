#!/usr/bin/env python3
"""Export a real LTX-2.5 video-FFN channel split for GPU+ANE tests.

The checkpoint uses per-output-channel INT8 weights and applies ConvRot to
both Linear inputs.  A split owns complete, 256-aligned intermediate channel
groups.  For the ANE shard, both ConvRot transforms are folded into the
dequantized weights, leaving a fixed-shape Conv/GELU/Conv Core ML graph.  The
GPU suffix remains in the checkpoint's original INT8+scale representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile


SCHEMA = "ltx-ane-mlp-v1"
GROUP = 256
SUPPORTED_ROWS = (1001, 4004)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument(
        "--rows", choices=("1001", "4004", "dual"), required=True,
        help="fixed row count or one model with 1001/4004 enumerated shapes",
    )
    parser.add_argument("--ane-intermediate", type=int, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("fp16", "int8_pc"),
        default=("fp16",),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compiled-only", action="store_true",
        help="discard mlpackage after producing mlmodelc",
    )
    parser.add_argument(
        "--reuse-gpu-files-from", type=Path,
        help="hardlink identical GPU complement files from another artifact",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        length = struct.unpack("<Q", encoded)[0]
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError(f"{path}: truncated safetensors JSON")
    return 8 + length, json.loads(payload)


def tensor_info(
    header: dict[str, object], name: str, dtype: str, shape: tuple[int, ...]
) -> tuple[int, int]:
    value = header.get(name)
    if not isinstance(value, dict) or value.get("dtype") != dtype:
        raise ValueError(f"missing {dtype} tensor {name}")
    actual_shape = tuple(int(item) for item in value.get("shape", ()))
    if actual_shape != shape:
        raise ValueError(f"{name}: expected {shape}, found {actual_shape}")
    offsets = value.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{name}: invalid data offsets")
    start, stop = (int(item) for item in offsets)
    if start < 0 or stop <= start:
        raise ValueError(f"{name}: invalid tensor range")
    return start, stop


def mapped_tensor(path: Path, data_offset: int, info: tuple[int, int],
                  dtype, shape):
    import numpy as np

    start, stop = info
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if stop - start != expected:
        raise ValueError(f"tensor byte count {stop - start} != {expected}")
    return np.memmap(
        path, mode="r", dtype=dtype, offset=data_offset + start, shape=shape
    )


def round_bf16(values):
    """Round float32 exactly as the native MPSGraph dequantization path."""
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def convrot_last_axis(values):
    """Apply H4^k / sqrt(N) on the last axis without a dense matmul."""
    import numpy as np

    if values.shape[-1] % GROUP:
        raise ValueError("ConvRot axis must be divisible by 256")
    result = np.ascontiguousarray(values, dtype=np.float32)
    flat = result.reshape(-1, GROUP)
    for stride in (1, 4, 16, 64):
        view = flat.reshape(-1, GROUP // (4 * stride), 4, stride)
        a = view[:, :, 0, :].copy()
        b = view[:, :, 1, :].copy()
        c = view[:, :, 2, :].copy()
        d = view[:, :, 3, :].copy()
        view[:, :, 0, :] = a + b + c - d
        view[:, :, 1, :] = a + b - c + d
        view[:, :, 2, :] = a - b + c + d
        view[:, :, 3, :] = -a + b + c + d
    flat *= np.float32(0.0625)
    return result


def absorb_convrot(weight_i8, scale_f32):
    import numpy as np

    dequantized = (
        np.asarray(weight_i8, dtype=np.float32) *
        np.asarray(scale_f32, dtype=np.float32).reshape(-1, 1)
    )
    return convrot_last_axis(round_bf16(dequantized)).astype(np.float16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array(path: Path, array, files: dict[str, object]) -> None:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
    }


def reuse_gpu_files(source: Path, destination: Path,
                    block: int, gpu: int) -> dict[str, object]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA or
        manifest.get("block_index") != block or
        manifest.get("partition", {}).get("gpu", {}).get("width") != gpu
    ):
        raise ValueError(f"incompatible reusable artifact {source}")
    files: dict[str, object] = {}
    for name in (
        "gpu_fc1.weight.i8", "gpu_fc1.scale.f32",
        "gpu_fc2.weight.i8", "gpu_fc2.scale.f32",
    ):
        source_file = source / name
        destination_file = destination / name
        destination_file.unlink(missing_ok=True)
        try:
            os.link(source_file, destination_file)
        except OSError:
            shutil.copyfile(source_file, destination_file)
        entry = manifest.get("files", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path}: missing file metadata {name}")
        files[name] = dict(entry)
    return files


def build_model(rows: tuple[int, ...], hidden: int, intermediate: int,
                fc1_weight, fc2_weight):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Symbol, types

    fc1 = np.ascontiguousarray(fc1_weight[:, :, None, None])
    fc2 = np.ascontiguousarray(fc2_weight[:, :, None, None])

    row_dimension = rows[0] if len(rows) == 1 else Symbol("rows")
    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, hidden, 1, row_dimension), dtype=types.fp16
            )
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        projected = mb.conv(
            x=x,
            weight=mb.const(val=fc1, name="fc1_weight"),
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="fc1",
        )
        activated = mb.gelu(
            x=projected, mode="TANH_APPROXIMATION", name="gelu"
        )
        return mb.conv(
            x=activated,
            weight=mb.const(val=fc2, name="fc2_weight"),
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="y",
        )

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
        f"row-major FP16 [{row_label},{hidden}] viewed as "
        f"[1,{hidden},1,rows]"
    )
    model.output_description["y"] = (
        f"row-major FP16 [{row_label},{hidden}] ANE partial"
    )
    model.user_defined_metadata.update({
        "ltx.schema": SCHEMA,
        "ltx.shape": f"{row_label}x{hidden}x{intermediate}",
        "ltx.convrot": "absorbed-256",
        "ltx.activation": "gelu-tanh",
    })
    return model


def quantize_int8(model):
    import coremltools.optimize as cto
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=0,
        )
    )
    return cto.coreml.linear_quantize_weights(model, config)


def compile_model(package: Path, destination: Path) -> None:
    import coremltools as ct

    loaded = ct.models.MLModel(
        str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
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
    if args.block < 0 or args.block >= 48:
        raise SystemExit("block must be in [0, 47]")
    hidden = 4096
    intermediate = 16384
    rows_values = SUPPORTED_ROWS if args.rows == "dual" else (int(args.rows),)
    default_rows = rows_values[0]
    ane = args.ane_intermediate
    if ane <= 0 or ane >= intermediate or ane % GROUP:
        raise SystemExit("ANE intermediate must be a 256-aligned proper shard")
    gpu = intermediate - ane
    if args.dry_run:
        print(json.dumps({
            "schema": SCHEMA,
            "block": args.block,
            "rows": list(rows_values),
            "hidden": hidden,
            "intermediate": intermediate,
            "gpu_intermediate": gpu,
            "ane_intermediate": ane,
            "variants": args.variants,
        }, sort_keys=True))
        return

    try:
        import numpy as np
        import coremltools as ct
    except ImportError as error:
        raise SystemExit(
            "export requires numpy and coremltools; use mac_transformer's "
            ".venv or another Core ML tools environment"
        ) from error

    data_offset, header = read_header(args.checkpoint)
    prefix = (
        "model.diffusion_model.transformer_blocks."
        f"{args.block}.ff"
    )
    fc1_name = f"{prefix}.net.0.proj"
    fc2_name = f"{prefix}.net.2"
    fc1_weight_info = tensor_info(
        header, f"{fc1_name}.weight", "I8", (intermediate, hidden)
    )
    fc1_scale_info = tensor_info(
        header, f"{fc1_name}.weight_scale", "F32", (intermediate, 1)
    )
    fc2_weight_info = tensor_info(
        header, f"{fc2_name}.weight", "I8", (hidden, intermediate)
    )
    fc2_scale_info = tensor_info(
        header, f"{fc2_name}.weight_scale", "F32", (hidden, 1)
    )
    if f"{fc1_name}.bias" in header or f"{fc2_name}.bias" in header:
        raise SystemExit("video FFN unexpectedly contains a bias")

    fc1_weight = mapped_tensor(
        args.checkpoint, data_offset, fc1_weight_info,
        np.int8, (intermediate, hidden),
    )
    fc1_scale = mapped_tensor(
        args.checkpoint, data_offset, fc1_scale_info,
        np.dtype("<f4"), (intermediate, 1),
    )
    fc2_weight = mapped_tensor(
        args.checkpoint, data_offset, fc2_weight_info,
        np.int8, (hidden, intermediate),
    )
    fc2_scale = mapped_tensor(
        args.checkpoint, data_offset, fc2_scale_info,
        np.dtype("<f4"), (hidden, 1),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; pass --force to replace")

    if args.reuse_gpu_files_from:
        files = reuse_gpu_files(
            args.reuse_gpu_files_from, args.out_dir, args.block, gpu
        )
    else:
        files: dict[str, object] = {}
        save_array(
            args.out_dir / "gpu_fc1.weight.i8", fc1_weight[ane:], files
        )
        save_array(
            args.out_dir / "gpu_fc1.scale.f32", fc1_scale[ane:], files
        )
        save_array(
            args.out_dir / "gpu_fc2.weight.i8", fc2_weight[:, ane:], files
        )
        save_array(
            args.out_dir / "gpu_fc2.scale.f32", fc2_scale, files
        )

    print("folding input ConvRot into ANE FC1", flush=True)
    ane_fc1 = absorb_convrot(fc1_weight[:ane], fc1_scale[:ane])
    print("folding hidden ConvRot into ANE FC2", flush=True)
    ane_fc2 = absorb_convrot(fc2_weight[:, :ane], fc2_scale)
    base_model = build_model(rows_values, hidden, ane, ane_fc1, ane_fc2)
    artifacts: dict[str, str] = {}
    compiled: dict[str, str] = {}
    with tempfile.TemporaryDirectory(
        prefix="ltx_ane_export_", dir=args.out_dir
    ) as temporary_directory:
        base_package = Path(temporary_directory) / "base.mlpackage"
        base_model.save(base_package)
        for variant in args.variants:
            package = args.out_dir / f"mlp_{variant}.mlpackage"
            if package.exists():
                shutil.rmtree(package)
            if variant == "fp16":
                shutil.copytree(base_package, package)
            else:
                source = ct.models.MLModel(
                    str(base_package), skip_model_load=True
                )
                quantize_int8(source).save(package)
            if not args.compiled_only:
                artifacts[variant] = package.name
            if args.compile:
                destination = args.out_dir / f"mlp_{variant}.mlmodelc"
                compile_model(package, destination)
                compiled[variant] = destination.name
            if args.compiled_only:
                shutil.rmtree(package)

    source_stat = args.checkpoint.stat()
    manifest = {
        "schema": SCHEMA,
        "block_index": args.block,
        "shape": {
            "rows": default_rows,
            "supported_rows": list(rows_values),
            "hidden": hidden,
            "intermediate": intermediate,
        },
        "partition": {
            "ane": {"start": 0, "width": ane},
            "gpu": {"start": ane, "width": gpu},
        },
        "source": {
            "path": str(args.checkpoint.resolve()),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "prefix": prefix,
        },
        "activation": "gelu_tanh",
        "convrot": {
            "group_size": GROUP,
            "ane_weights": "absorbed_fp16",
            "gpu_weights": "checkpoint_int8",
        },
        "tensor_abi": {
            "logical": [default_rows, hidden],
            "supported_logical": [
                [value, hidden] for value in rows_values
            ],
            "coreml": [1, hidden, 1, default_rows],
            "strides": [default_rows * hidden, 1,
                        default_rows * hidden, hidden],
            "input": "caller-owned-row-major-fp16",
            "output": "caller-owned-row-major-fp16-partial",
        },
        "artifacts": artifacts,
        "compiled_artifacts": compiled,
        "files": files,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
