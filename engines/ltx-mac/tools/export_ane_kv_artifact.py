#!/usr/bin/env python3
"""Export one fixed-shape LTX-2.5 Video-text K/V projection for Core ML/ANE.

The caller selects the text-row specialization.  It
applies the block's prompt affine transform, while this graph folds ConvRot
into the checkpoint's INT8 projection weights and computes K, K RMSNorm and
V.  Outputs use the GPU attention path's head-major layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from export_ane_mlp_artifact import (
    absorb_convrot,
    compile_model,
    mapped_tensor,
    quantize_int8,
    read_header,
    tensor_info,
)


SCHEMA = "ltx-ane-text-kv-v1"
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--text-rows", type=int, default=1024)
    parser.add_argument(
        "--variants", nargs="+", choices=("fp16", "int8_pc"),
        default=("fp16",),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiled-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bf16_tensor(path: Path, data_offset: int, info, shape):
    import numpy as np

    bits = mapped_tensor(path, data_offset, info, np.dtype("<u2"), shape)
    expanded = np.asarray(bits, dtype=np.uint32) << np.uint32(16)
    return expanded.view(np.float32).astype(np.float16)


def load_projection(path: Path, data_offset: int, header, name: str):
    import numpy as np

    weight = mapped_tensor(
        path, data_offset,
        tensor_info(header, f"{name}.weight", "I8", (HIDDEN, HIDDEN)),
        np.int8, (HIDDEN, HIDDEN),
    )
    scale = mapped_tensor(
        path, data_offset,
        tensor_info(
            header, f"{name}.weight_scale", "F32", (HIDDEN, 1)
        ),
        np.dtype("<f4"), (HIDDEN, 1),
    )
    bias = bf16_tensor(
        path, data_offset,
        tensor_info(header, f"{name}.bias", "BF16", (HIDDEN,)),
        (HIDDEN,),
    )
    return absorb_convrot(weight, scale), bias


def load_weights(checkpoint: Path, block: int):
    data_offset, header = read_header(checkpoint)
    prefix = (
        "model.diffusion_model.transformer_blocks."
        f"{block}.attn2"
    )
    key_weight, key_bias = load_projection(
        checkpoint, data_offset, header, f"{prefix}.to_k"
    )
    value_weight, value_bias = load_projection(
        checkpoint, data_offset, header, f"{prefix}.to_v"
    )
    key_norm = bf16_tensor(
        checkpoint, data_offset,
        tensor_info(
            header, f"{prefix}.k_norm.weight", "BF16", (HIDDEN,)
        ),
        (HIDDEN,),
    )
    return {
        "key_weight": key_weight,
        "key_bias": key_bias,
        "value_weight": value_weight,
        "value_bias": value_bias,
        "key_norm": key_norm,
    }, prefix


def build_model(values, text_rows: int):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def conv_weight(name):
        return np.ascontiguousarray(values[name][:, :, None, None])

    def token_major(x, name):
        squeezed = mb.squeeze(x=x, axes=[2], name=f"{name}_squeeze")
        return mb.transpose(
            x=squeezed, perm=[0, 2, 1], name=f"{name}_token_major"
        )

    def head_major(x, name):
        reshaped = mb.reshape(
            x=x, shape=(1, text_rows, HEADS, HEAD_DIM),
            name=f"{name}_reshape",
        )
        return mb.transpose(
            x=reshaped, perm=[0, 2, 1, 3], name=name
        )

    def key_norm(x):
        square = mb.mul(x=x, y=x, name="key_square")
        mean = mb.reduce_mean(
            x=square, axes=[-1], keep_dims=True, name="key_mean"
        )
        inverse = mb.rsqrt(
            x=mb.add(x=mean, y=np.float16(1.0e-6)),
            name="key_rsqrt",
        )
        normalized = mb.mul(x=x, y=inverse, name="key_normalized")
        return mb.mul(
            x=normalized,
            y=np.ascontiguousarray(
                values["key_norm"].reshape(1, 1, HIDDEN)
            ),
            name="key_weighted_norm",
        )

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, HIDDEN, 1, text_rows), dtype=types.fp16
            )
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        key = mb.conv(
            x=x, weight=conv_weight("key_weight"),
            bias=values["key_bias"], name="key_projection",
        )
        value = mb.conv(
            x=x, weight=conv_weight("value_weight"),
            bias=values["value_bias"], name="value_projection",
        )
        key = head_major(
            key_norm(token_major(key, "key")), "key"
        )
        value = head_major(token_major(value, "value"), "value")
        return key, value

    model = ct.convert(
        program, convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    specification = model.get_spec()
    if len(specification.description.input) != 1 or \
            len(specification.description.output) != 2:
        raise ValueError("unexpected Core ML K/V model interface")
    specification.description.input[0].name = "x"
    specification.description.output[0].name = "key"
    specification.description.output[1].name = "value"
    model = ct.models.MLModel(specification, weights_dir=model.weights_dir)
    model.user_defined_metadata.update({
        "ltx.schema": SCHEMA,
        "ltx.block": "video_text_kv",
        "ltx.text_rows": str(text_rows),
        "ltx.convrot": "absorbed-256",
        "ltx.precision": "fp16",
    })
    return model


def main() -> None:
    args = parse_args()
    if args.block < 0 or args.block >= 48:
        raise SystemExit("block must be in [0, 47]")
    if args.text_rows <= 0:
        raise SystemExit("--text-rows must be positive")
    if args.compiled_only and not args.compile:
        raise SystemExit("--compiled-only requires --compile")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} is not empty; pass --force")
    if args.force:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values, prefix = load_weights(args.checkpoint, args.block)
    base_model = build_model(values, args.text_rows)
    artifacts: dict[str, str] = {}
    compiled_artifacts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(
        prefix="ltx_ane_kv_export_", dir=args.out_dir
    ) as temporary_directory:
        base_package = Path(temporary_directory) / "base.mlpackage"
        base_model.save(base_package)
        for variant in args.variants:
            package = args.out_dir / f"text_kv_{variant}.mlpackage"
            if package.exists():
                shutil.rmtree(package)
            if variant == "fp16":
                shutil.copytree(base_package, package)
            else:
                source = __import__("coremltools").models.MLModel(
                    str(base_package), skip_model_load=True
                )
                quantize_int8(source).save(package)
            if not args.compiled_only:
                artifacts[variant] = package.name
            if args.compile:
                compiled = args.out_dir / f"text_kv_{variant}.mlmodelc"
                compile_model(package, compiled)
                compiled_artifacts[variant] = compiled.name
            if args.compiled_only:
                shutil.rmtree(package)

    source_stat = args.checkpoint.stat()
    manifest = {
        "schema": SCHEMA,
        "block_index": args.block,
        "shape": {
            "text_rows": args.text_rows,
            "hidden": HIDDEN,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
        },
        "source": {
            "path": str(args.checkpoint.resolve()),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "prefix": prefix,
        },
        "convrot": {
            "group_size": 256,
            "weights": "absorbed_fp16",
        },
        "tensor_abi": {
            "input_logical": [args.text_rows, HIDDEN],
            "input_coreml": [1, HIDDEN, 1, args.text_rows],
            "output_head_major": [1, HEADS, args.text_rows, HEAD_DIM],
            "dtype": "fp16",
        },
        "artifacts": artifacts,
        "compiled_artifacts": compiled_artifacts,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
