#!/usr/bin/env python3
"""Probe a selector-based grouped LTX Video self-QKV Core ML model.

The production QKV sequence split currently owns one Core ML session per
Transformer block.  This probe places a small number of consecutive blocks in
one ML Program and selects exactly one branch for each prediction.  It measures
whether Core ML executes only the selected branch and whether one persistent
session makes alternating block calls cheaper than separate sessions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from export_ane_mlp_artifact import compile_model, quantize_int8
from probe_ane_qkv import HIDDEN, build_model as build_single_model, load_weights


def parse_blocks(value: str) -> list[int]:
    blocks: list[int] = []
    for item in value.split(","):
        block = int(item.strip())
        if block < 0 or block >= 48:
            raise argparse.ArgumentTypeError("blocks must be in [0, 47]")
        if block not in blocks:
            blocks.append(block)
    if len(blocks) < 2 or len(blocks) > 8:
        raise argparse.ArgumentTypeError("select 2 to 8 distinct blocks")
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--blocks", type=parse_blocks, default="0,1")
    parser.add_argument("--rows", type=int, default=1575)
    parser.add_argument(
        "--variant", choices=("fp16", "int8_pc"), default="int8_pc"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--compare-independent", action="store_true")
    parser.add_argument("--compare-multifunction", action="store_true")
    parser.add_argument("--keep", type=Path)
    return parser.parse_args()


def build_model(block_values: list[tuple[int, dict]], rows: int):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def emit_qkv(x, block: int, values: dict):
        prefix = f"block_{block}"

        def conv_weight(name: str):
            return np.ascontiguousarray(values[name][:, :, None, None])

        def rms_norm(tensor, weight, name: str):
            square = mb.mul(x=tensor, y=tensor, name=f"{name}_square")
            mean = mb.reduce_mean(
                x=square, axes=[1], keep_dims=True, name=f"{name}_mean"
            )
            inverse = mb.rsqrt(
                x=mb.add(x=mean, y=np.float16(1.0e-6)),
                name=f"{name}_inverse",
            )
            normalized = mb.mul(
                x=tensor, y=inverse, name=f"{name}_normalized"
            )
            scale = np.ascontiguousarray(weight.reshape(1, HIDDEN, 1, 1))
            return mb.mul(x=normalized, y=scale, name=name)

        query = mb.conv(
            x=x,
            weight=conv_weight("query_weight"),
            bias=values["query_bias"],
            name=f"{prefix}_query_projection",
        )
        key = mb.conv(
            x=x,
            weight=conv_weight("key_weight"),
            bias=values["key_bias"],
            name=f"{prefix}_key_projection",
        )
        value = mb.conv(
            x=x,
            weight=conv_weight("value_weight"),
            bias=values["value_bias"],
            name=f"{prefix}_value",
        )
        query = rms_norm(
            query, values["query_norm"], f"{prefix}_query"
        )
        key = rms_norm(key, values["key_norm"], f"{prefix}_key")
        return query, key, value

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HIDDEN, 1, rows), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(x, selector):
        scalar_selector = mb.squeeze(
            x=selector, axes=[0], name="scalar_selector"
        )

        def select(index: int):
            block, values = block_values[index]
            if index == len(block_values) - 1:
                return emit_qkv(x, block, values)
            predicate = mb.equal(
                x=scalar_selector,
                y=np.int32(index),
                name=f"select_block_{block}",
            )
            return mb.cond(
                pred=predicate,
                _true_fn=lambda: emit_qkv(x, block, values),
                _false_fn=lambda: select(index + 1),
                name=f"block_{block}_branch",
            )

        query, key, value = select(0)
        return (
            mb.identity(x=query, name="query"),
            mb.identity(x=key, name="key"),
            mb.identity(x=value, name="value"),
        )

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    model.user_defined_metadata.update({
        "ltx.schema": "ltx-ane-qkv-selector-group-probe-v1",
        "ltx.rows": str(rows),
        "ltx.hidden": str(HIDDEN),
        "ltx.blocks": ",".join(str(block) for block, _ in block_values),
        "ltx.selector": "zero-based-index-into-ltx.blocks",
    })
    return model


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def timed_predictions(runtime, features: dict, warmup: int,
                      iterations: int) -> tuple[dict, float, float, float]:
    started = time.perf_counter()
    result = runtime.predict(features)
    first_ms = (time.perf_counter() - started) * 1000.0
    for _ in range(warmup):
        result = runtime.predict(features)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = runtime.predict(features)
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
    return result, first_ms, p50, p95


def output_metrics(reference: dict, candidate: dict) -> tuple[float, float]:
    import numpy as np

    reference_flat = np.concatenate([
        np.asarray(reference[name], dtype=np.float32).reshape(-1)
        for name in ("query", "key", "value")
    ])
    candidate_flat = np.concatenate([
        np.asarray(candidate[name], dtype=np.float32).reshape(-1)
        for name in ("query", "key", "value")
    ])
    difference = candidate_flat - reference_flat
    reference_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    rel_l2 = float(np.linalg.norm(difference)) / max(reference_norm, 1.0e-30)
    cosine = float(np.dot(reference_flat, candidate_flat)) / max(
        reference_norm * candidate_norm, 1.0e-30
    )
    return cosine, rel_l2


def output_stats(result: dict) -> tuple[bool, float]:
    import numpy as np

    outputs = [np.asarray(result[name], dtype=np.float32)
               for name in ("query", "key", "value")]
    finite = all(bool(np.isfinite(output).all()) for output in outputs)
    if not finite:
        return False, float("nan")
    squared = sum(float(np.sum(output * output)) for output in outputs)
    elements = sum(output.size for output in outputs)
    return True, math.sqrt(squared / elements)


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("invalid rows/warmup/iterations")
    try:
        import coremltools as ct
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "use gpu_ane/mac_transformer/.venv with Core ML tools"
        ) from error

    temporary = Path(tempfile.mkdtemp(prefix="ltx_ane_qkv_group_"))
    package = temporary / f"qkv_group_{args.variant}.mlpackage"
    compiled = temporary / f"qkv_group_{args.variant}.mlmodelc"
    try:
        load_started = time.perf_counter()
        block_values = []
        for block in args.blocks:
            values, _ = load_weights(args.checkpoint, block)
            block_values.append((block, values))
        weight_load_s = time.perf_counter() - load_started

        build_started = time.perf_counter()
        model = build_model(block_values, args.rows)
        if args.variant == "int8_pc":
            model = quantize_int8(model)
        model.save(package)
        build_s = time.perf_counter() - build_started

        compile_started = time.perf_counter()
        compile_model(package, compiled)
        compile_s = time.perf_counter() - compile_started

        load_started = time.perf_counter()
        runtime = ct.models.MLModel(
            str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        grouped_load_s = time.perf_counter() - load_started

        rng = np.random.default_rng(42)
        features_x = (
            rng.standard_normal((1, HIDDEN, 1, args.rows)) * 0.1
        ).astype(np.float16)
        grouped_results: dict[int, dict] = {}
        grouped_p50: dict[int, float] = {}
        for index, block in enumerate(args.blocks):
            features = {
                "x": features_x,
                "selector": np.asarray([index], dtype=np.int32),
            }
            result, first_ms, p50, p95 = timed_predictions(
                runtime, features, args.warmup, args.iterations
            )
            grouped_results[block] = result
            grouped_p50[block] = p50
            finite, rms = output_stats(result)
            print(
                f"grouped block={block} selector={index} first_ms={first_ms:.3f} "
                f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
                f"finite={int(finite)} rms={rms:.9g}",
                flush=True,
            )

        alternating = []
        for iteration in range(args.iterations):
            started = time.perf_counter()
            for index in range(len(args.blocks)):
                runtime.predict({
                    "x": features_x,
                    "selector": np.asarray([index], dtype=np.int32),
                })
            alternating.append((time.perf_counter() - started) * 1000.0)
        grouped_cycle_p50 = statistics.median(alternating)

        print(
            f"grouped blocks={','.join(map(str, args.blocks))} rows={args.rows} "
            f"variant={args.variant} weight_load_s={weight_load_s:.3f} "
            f"build_s={build_s:.3f} compile_s={compile_s:.3f} "
            f"load_s={grouped_load_s:.3f} "
            f"package_mib={tree_bytes(package) / (1 << 20):.3f} "
            f"compiled_mib={tree_bytes(compiled) / (1 << 20):.3f} "
            f"cycle_p50_ms={grouped_cycle_p50:.3f}",
            flush=True,
        )

        independent_packages = []
        independent_models = []
        independent_results: dict[int, dict] = {}
        if args.compare_independent or args.compare_multifunction:
            independent_build_s = 0.0
            independent_load_s = 0.0
            for block, values in block_values:
                started = time.perf_counter()
                independent_model = build_single_model(values, args.rows)
                if args.variant == "int8_pc":
                    independent_model = quantize_int8(independent_model)
                independent_package = (
                    temporary / f"qkv_block_{block}_{args.variant}.mlpackage"
                )
                independent_model.save(independent_package)
                independent_packages.append(independent_package)
                independent_build_s += time.perf_counter() - started

            if args.compare_independent:
                for independent_package in independent_packages:
                    started = time.perf_counter()
                    independent_models.append(ct.models.MLModel(
                        str(independent_package),
                        compute_units=ct.ComputeUnit.CPU_AND_NE,
                    ))
                    independent_load_s += time.perf_counter() - started

                independent_p50 = {}
                for block, independent in zip(args.blocks, independent_models):
                    result, first_ms, p50, p95 = timed_predictions(
                        independent, {"x": features_x},
                        args.warmup, args.iterations,
                    )
                    independent_results[block] = result
                    independent_p50[block] = p50
                    cosine, rel_l2 = output_metrics(
                        result, grouped_results[block]
                    )
                    finite, rms = output_stats(result)
                    print(
                        f"independent block={block} first_ms={first_ms:.3f} "
                        f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
                        f"finite={int(finite)} rms={rms:.9g} "
                        f"grouped_cosine={cosine:.9f} "
                        f"grouped_rel_l2={rel_l2:.9f}",
                        flush=True,
                    )

                alternating = []
                for _ in range(args.iterations):
                    started = time.perf_counter()
                    for independent in independent_models:
                        independent.predict({"x": features_x})
                    alternating.append((time.perf_counter() - started) * 1000.0)
                independent_cycle_p50 = statistics.median(alternating)
                speedup = independent_cycle_p50 / grouped_cycle_p50
                print(
                    f"comparison independent_build_s={independent_build_s:.3f} "
                    f"independent_load_s={independent_load_s:.3f} "
                    f"independent_cycle_p50_ms={independent_cycle_p50:.3f} "
                    f"grouped_cycle_p50_ms={grouped_cycle_p50:.3f} "
                    f"grouped_cycle_speedup={speedup:.3f}",
                    flush=True,
                )

        if args.compare_multifunction:
            from coremltools.models.utils import (
                MultiFunctionDescriptor,
                save_multifunction,
            )

            multifunction_package = temporary / "qkv_multifunction.mlpackage"
            descriptor = MultiFunctionDescriptor()
            function_names = []
            for block, independent_package in zip(
                    args.blocks, independent_packages):
                function_name = f"block_{block}"
                descriptor.add_function(
                    str(independent_package), "main", function_name
                )
                function_names.append(function_name)
            descriptor.default_function_name = function_names[0]
            started = time.perf_counter()
            save_multifunction(descriptor, str(multifunction_package))
            multifunction_build_s = time.perf_counter() - started

            multifunction_models = []
            multifunction_load_s = 0.0
            for function_name in function_names:
                started = time.perf_counter()
                multifunction_models.append(ct.models.MLModel(
                    str(multifunction_package),
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    function_name=function_name,
                ))
                multifunction_load_s += time.perf_counter() - started

            for block, multifunction in zip(
                    args.blocks, multifunction_models):
                result, first_ms, p50, p95 = timed_predictions(
                    multifunction, {"x": features_x},
                    args.warmup, args.iterations,
                )
                reference = independent_results.get(block)
                if reference is None:
                    single_index = args.blocks.index(block)
                    reference = ct.models.MLModel(
                        str(independent_packages[single_index]),
                        compute_units=ct.ComputeUnit.CPU_AND_NE,
                    ).predict({"x": features_x})
                cosine, rel_l2 = output_metrics(reference, result)
                finite, rms = output_stats(result)
                print(
                    f"multifunction block={block} first_ms={first_ms:.3f} "
                    f"p50_ms={p50:.3f} p95_ms={p95:.3f} "
                    f"finite={int(finite)} rms={rms:.9g} "
                    f"independent_cosine={cosine:.9f} "
                    f"independent_rel_l2={rel_l2:.9f}",
                    flush=True,
                )

            alternating = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                for multifunction in multifunction_models:
                    multifunction.predict({"x": features_x})
                alternating.append((time.perf_counter() - started) * 1000.0)
            multifunction_cycle_p50 = statistics.median(alternating)
            print(
                f"multifunction build_s={multifunction_build_s:.3f} "
                f"load_s={multifunction_load_s:.3f} "
                f"package_mib={tree_bytes(multifunction_package) / (1 << 20):.3f} "
                f"cycle_p50_ms={multifunction_cycle_p50:.3f}",
                flush=True,
            )

        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(temporary, args.keep)
            print(f"artifacts={args.keep}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
