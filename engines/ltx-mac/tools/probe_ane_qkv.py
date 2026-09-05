#!/usr/bin/env python3
"""Measure real LTX-2.5 self-attention QKV + full-inner Norm on ANE.

This is the placement gate for sequence-parallel QKV.  The split is along
tokens, so each device still normalizes a complete 4,096-wide Q/K row and the
result remains mathematically valid without a cross-device reduction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from export_ane_mlp_artifact import (
    absorb_convrot,
    compile_model,
    mapped_tensor,
    quantize_int8,
    read_header,
    tensor_info,
)


HIDDEN = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1001)
    parser.add_argument(
        "--variant", choices=("fp16", "int8_pc"), default="int8_pc"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--keep", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compiled-only", action="store_true",
        help="when exporting, retain only the compiled mlmodelc artifact",
    )
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
        path,
        data_offset,
        tensor_info(header, f"{name}.weight", "I8", (HIDDEN, HIDDEN)),
        np.int8,
        (HIDDEN, HIDDEN),
    )
    scale = mapped_tensor(
        path,
        data_offset,
        tensor_info(
            header, f"{name}.weight_scale", "F32", (HIDDEN, 1)
        ),
        np.dtype("<f4"),
        (HIDDEN, 1),
    )
    bias = bf16_tensor(
        path,
        data_offset,
        tensor_info(header, f"{name}.bias", "BF16", (HIDDEN,)),
        (HIDDEN,),
    )
    return absorb_convrot(weight, scale), bias


def load_weights(checkpoint: Path, block: int):
    data_offset, header = read_header(checkpoint)
    prefix = f"model.diffusion_model.transformer_blocks.{block}.attn1"
    values = {}
    for field, suffix in (("query", "to_q"), ("key", "to_k"),
                          ("value", "to_v")):
        weight, bias = load_projection(
            checkpoint, data_offset, header, f"{prefix}.{suffix}"
        )
        values[f"{field}_weight"] = weight
        values[f"{field}_bias"] = bias
    for field, suffix in (("query_norm", "q_norm.weight"),
                          ("key_norm", "k_norm.weight")):
        values[field] = bf16_tensor(
            checkpoint,
            data_offset,
            tensor_info(
                header, f"{prefix}.{suffix}", "BF16", (HIDDEN,)
            ),
            (HIDDEN,),
        )
    return values, prefix


def build_model(values, rows: int):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def conv_weight(name: str):
        return np.ascontiguousarray(values[name][:, :, None, None])

    def rms_norm(x, weight, name: str):
        square = mb.mul(x=x, y=x, name=f"{name}_square")
        mean = mb.reduce_mean(
            x=square, axes=[1], keep_dims=True, name=f"{name}_mean"
        )
        inverse = mb.rsqrt(
            x=mb.add(x=mean, y=np.float16(1.0e-6)),
            name=f"{name}_inverse",
        )
        normalized = mb.mul(x=x, y=inverse, name=f"{name}_normalized")
        scale = np.ascontiguousarray(weight.reshape(1, HIDDEN, 1, 1))
        return mb.mul(x=normalized, y=scale, name=name)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HIDDEN, 1, rows), dtype=types.fp16)
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        query = mb.conv(
            x=x,
            weight=conv_weight("query_weight"),
            bias=values["query_bias"],
            name="query_projection",
        )
        key = mb.conv(
            x=x,
            weight=conv_weight("key_weight"),
            bias=values["key_bias"],
            name="key_projection",
        )
        value = mb.conv(
            x=x,
            weight=conv_weight("value_weight"),
            bias=values["value_bias"],
            name="value",
        )
        query = rms_norm(query, values["query_norm"], "query")
        key = rms_norm(key, values["key_norm"], "key")
        return query, key, value

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    model.user_defined_metadata.update({
        "ltx.schema": "ltx-ane-qkv-sequence-probe-v1",
        "ltx.rows": str(rows),
        "ltx.hidden": str(HIDDEN),
        "ltx.norm": "full-inner-4096",
        "ltx.convrot": "absorbed-256",
    })
    return model


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    args = parse_args()
    if args.block < 0 or args.block >= 48:
        raise SystemExit("block must be in [0, 47]")
    if args.rows <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("invalid rows/warmup/iterations")
    if args.compiled_only and not args.compile:
        raise SystemExit("--compiled-only requires --compile")

    try:
        import coremltools as ct
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "use gpu_ane/mac_transformer/.venv with Core ML tools"
        ) from error

    temporary = Path(tempfile.mkdtemp(prefix="ltx_ane_qkv_"))
    package = temporary / f"qkv_{args.variant}.mlpackage"
    try:
        load_started = time.perf_counter()
        values, prefix = load_weights(args.checkpoint, args.block)
        weight_load_seconds = time.perf_counter() - load_started

        build_started = time.perf_counter()
        model = build_model(values, args.rows)
        if args.variant == "int8_pc":
            model = quantize_int8(model)
        model.save(package)
        build_seconds = time.perf_counter() - build_started
        package_bytes = tree_bytes(package)

        model_load_started = time.perf_counter()
        runtime = ct.models.MLModel(
            str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        model_load_seconds = time.perf_counter() - model_load_started

        rng = np.random.default_rng(42)
        features = {
            "x": (rng.standard_normal((1, HIDDEN, 1, args.rows)) * 0.1)
            .astype(np.float16)
        }
        first_started = time.perf_counter()
        result = runtime.predict(features)
        first_ms = (time.perf_counter() - first_started) * 1000.0
        for _ in range(args.warmup):
            result = runtime.predict(features)
        timings = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            result = runtime.predict(features)
            timings.append((time.perf_counter() - started) * 1000.0)
        timings.sort()
        outputs = [np.asarray(value) for value in result.values()]
        finite = all(bool(np.isfinite(value).all()) for value in outputs)
        rms = math.sqrt(
            sum(float(np.sum(value.astype(np.float32) ** 2)) for value in outputs)
            / sum(value.size for value in outputs)
        )
        p50 = statistics.median(timings)
        p95 = timings[
            min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)
        ]
        print(
            f"checkpoint={args.checkpoint} attention={prefix} "
            f"rows={args.rows} hidden={HIDDEN} variant={args.variant}"
        )
        print(
            f"weight_load_s={weight_load_seconds:.3f} "
            f"build_s={build_seconds:.3f} load_s={model_load_seconds:.3f} "
            f"package_mib={package_bytes / (1 << 20):.3f}"
        )
        print(
            f"backend=coreml-cpu-and-ne first_ms={first_ms:.3f} "
            f"warmup={args.warmup} iterations={args.iterations} "
            f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
            f"finite={int(finite)} rms={rms:.9g}"
        )
        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(package, args.keep)
            print(f"artifact={args.keep}")
        if args.export_dir:
            if args.export_dir.exists() and any(args.export_dir.iterdir()):
                if not args.force:
                    raise SystemExit(
                        f"{args.export_dir} is not empty; pass --force"
                    )
                shutil.rmtree(args.export_dir)
            args.export_dir.mkdir(parents=True, exist_ok=True)
            package_name = f"qkv_{args.variant}.mlpackage"
            exported_package = args.export_dir / package_name
            artifacts = {}
            if not args.compiled_only:
                shutil.copytree(package, exported_package)
                artifacts[args.variant] = package_name
            compiled_artifacts = {}
            if args.compile:
                compiled_name = f"qkv_{args.variant}.mlmodelc"
                compile_model(
                    package if args.compiled_only else exported_package,
                    args.export_dir / compiled_name,
                )
                compiled_artifacts[args.variant] = compiled_name
            source_stat = args.checkpoint.stat()
            manifest = {
                "schema": "ltx-ane-qkv-sequence-v1",
                "block_index": args.block,
                "shape": {"rows": args.rows, "hidden": HIDDEN},
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
                    "input_logical": [args.rows, HIDDEN],
                    "coreml": [1, HIDDEN, 1, args.rows],
                    "outputs": ["query", "key", "value"],
                    "dtype": "fp16",
                    "qk_norm": "full-inner-4096",
                },
                "artifacts": artifacts,
                "compiled_artifacts": compiled_artifacts,
            }
            manifest_path = args.export_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"manifest={manifest_path}")
        if not finite:
            raise SystemExit(1)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
