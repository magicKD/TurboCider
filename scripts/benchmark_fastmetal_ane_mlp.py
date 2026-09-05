#!/usr/bin/env python3
"""Benchmark a real FastMetal Wan FFN split across MLX/Metal and Core ML/ANE.

This is an experimental admission gate, not a production backend.  It reads one
block from a packed FastMetal checkpoint, exports a fixed-shape Core ML model
for the leading FFN channels, and overlaps that branch with the remaining MLX
quantized branch.  The report includes the boundary and residual-add cost, so a
fast standalone Core ML result cannot be mistaken for an end-to-end win.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


HIDDEN = 1536
INTERMEDIATE = 8960
GROUP_SIZE = 64
BITS = 8
MODE = "affine"


@dataclass(frozen=True)
class PackedMLP:
    fc1_weight: Any
    fc1_scales: Any
    fc1_biases: Any
    fc1_bias: Any
    fc2_weight: Any
    fc2_scales: Any
    fc2_biases: Any
    fc2_bias: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a real FastMetal INT8 FFN GPU/ANE channel split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8190)
    parser.add_argument("--ane-width", type=int, default=4096)
    parser.add_argument("--variant", choices=("fp16", "int8_pc"), default="int8_pc")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-compile-mlx", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.bridge_dir.is_dir():
        parser.error(f"bridge directory does not exist: {args.bridge_dir}")
    if args.block < 0:
        parser.error("block must be non-negative")
    if args.rows <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("rows/iterations must be positive and warmup non-negative")
    if not 0 < args.ane_width < INTERMEDIATE or args.ane_width % GROUP_SIZE:
        parser.error(f"ane-width must be a {GROUP_SIZE}-aligned proper FFN shard")
    return args


def load_packed_mlp(checkpoint: Path, block: int, mx: Any) -> PackedMLP:
    from safetensors import safe_open

    prefix = f"blocks.{block}.ffn"
    names = {
        "fc1_weight": f"{prefix}.fc_in.weight",
        "fc1_scales": f"{prefix}.fc_in.weight.scales",
        "fc1_biases": f"{prefix}.fc_in.weight.biases",
        "fc1_bias": f"{prefix}.fc_in.bias",
        "fc2_weight": f"{prefix}.fc_out.weight",
        "fc2_scales": f"{prefix}.fc_out.weight.scales",
        "fc2_biases": f"{prefix}.fc_out.weight.biases",
        "fc2_bias": f"{prefix}.fc_out.bias",
    }
    with safe_open(checkpoint, framework="np") as handle:
        missing = sorted(set(names.values()) - set(handle.keys()))
        if missing:
            raise ValueError(f"checkpoint is missing block {block} FFN tensors: {missing}")
        arrays = {field: mx.array(handle.get_tensor(name)) for field, name in names.items()}
    mx.eval(*arrays.values())
    result = PackedMLP(**arrays)
    if tuple(result.fc1_weight.shape) != (INTERMEDIATE, HIDDEN // 4):
        raise ValueError(f"unexpected FC1 packed shape: {result.fc1_weight.shape}")
    if tuple(result.fc2_weight.shape) != (HIDDEN, INTERMEDIATE // 4):
        raise ValueError(f"unexpected FC2 packed shape: {result.fc2_weight.shape}")
    return result


def quantized_linear(
    mx: Any,
    x: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    bias: Any | None,
) -> Any:
    output = mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    ).astype(mx.float16)
    return output if bias is None else output + bias


def make_mlx_branches(mx: Any, nn: Any, weights: PackedMLP, ane_width: int):
    packed_per_group = GROUP_SIZE // (32 // BITS)
    packed_prefix = ane_width // (32 // BITS)
    scale_prefix = ane_width // GROUP_SIZE

    def branch(
        x: Any,
        output_slice: slice,
        packed_slice: slice,
        scale_slice: slice,
        include_output_bias: bool,
    ) -> Any:
        hidden = quantized_linear(
            mx,
            x,
            weights.fc1_weight[output_slice],
            weights.fc1_scales[output_slice],
            weights.fc1_biases[output_slice],
            weights.fc1_bias[output_slice],
        )
        hidden = nn.gelu_approx(hidden)
        return quantized_linear(
            mx,
            hidden,
            weights.fc2_weight[:, packed_slice],
            weights.fc2_scales[:, scale_slice],
            weights.fc2_biases[:, scale_slice],
            weights.fc2_bias if include_output_bias else None,
        )

    def full(x: Any) -> Any:
        return branch(x, slice(None), slice(None), slice(None), True)

    def prefix(x: Any) -> Any:
        return branch(
            x,
            slice(0, ane_width),
            slice(0, packed_prefix),
            slice(0, scale_prefix),
            True,
        )

    def suffix(x: Any) -> Any:
        return branch(
            x,
            slice(ane_width, INTERMEDIATE),
            slice(packed_prefix, INTERMEDIATE // (32 // BITS)),
            slice(scale_prefix, INTERMEDIATE // GROUP_SIZE),
            False,
        )

    if packed_prefix % packed_per_group:
        raise AssertionError("packed split does not end on a quantization group")
    return full, prefix, suffix


def dequantized_prefix(weights: PackedMLP, ane_width: int, mx: Any) -> tuple[np.ndarray, ...]:
    packed_prefix = ane_width // (32 // BITS)
    scale_prefix = ane_width // GROUP_SIZE
    fc1 = mx.dequantize(
        weights.fc1_weight[:ane_width],
        weights.fc1_scales[:ane_width],
        weights.fc1_biases[:ane_width],
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
        dtype=mx.float16,
    )
    fc2 = mx.dequantize(
        weights.fc2_weight[:, :packed_prefix],
        weights.fc2_scales[:, :scale_prefix],
        weights.fc2_biases[:, :scale_prefix],
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
        dtype=mx.float16,
    )
    mx.eval(fc1, fc2)
    return (
        np.ascontiguousarray(np.asarray(fc1), dtype=np.float16),
        np.ascontiguousarray(np.asarray(weights.fc1_bias[:ane_width]), dtype=np.float16),
        np.ascontiguousarray(np.asarray(fc2), dtype=np.float16),
        np.ascontiguousarray(np.asarray(weights.fc2_bias), dtype=np.float16),
    )


def build_coreml_model(
    *,
    rows: int,
    ane_width: int,
    fc1_weight: np.ndarray,
    fc1_bias: np.ndarray,
    fc2_weight: np.ndarray,
    fc2_bias: np.ndarray,
) -> Any:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    fc1 = np.ascontiguousarray(fc1_weight[:, :, None, None])
    fc2 = np.ascontiguousarray(fc2_weight[:, :, None, None])

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, rows), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        projected = mb.conv(
            x=x,
            weight=mb.const(val=fc1, name="fc1_weight"),
            bias=mb.const(val=fc1_bias, name="fc1_bias"),
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="fc1",
        )
        activated = mb.gelu(x=projected, mode="TANH_APPROXIMATION", name="gelu")
        return mb.conv(
            x=activated,
            weight=mb.const(val=fc2, name="fc2_weight"),
            bias=mb.const(val=fc2_bias, name="fc2_bias"),
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="y",
        )

    return ct.convert(
        program,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def quantize_coreml_int8(model: Any) -> Any:
    import coremltools.optimize as cto
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=0,
        )
    )
    return cto.coreml.linear_quantize_weights(model, config)


def compiled_artifact(
    *,
    artifact_dir: Path,
    block: int,
    checkpoint_digest: str,
    rows: int,
    ane_width: int,
    variant: str,
    weights: tuple[np.ndarray, ...],
    force: bool = False,
) -> Path:
    import coremltools as ct

    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"block{block}_mlp_{checkpoint_digest[:16]}_"
        f"m{rows}_a{ane_width}_{variant}"
    )
    package = artifact_dir / f"{stem}.mlpackage"
    compiled = artifact_dir / f"{stem}.mlmodelc"
    if force:
        shutil.rmtree(package, ignore_errors=True)
        shutil.rmtree(compiled, ignore_errors=True)
    if compiled.is_dir():
        return compiled
    if not package.is_dir():
        model = build_coreml_model(
            rows=rows,
            ane_width=ane_width,
            fc1_weight=weights[0],
            fc1_bias=weights[1],
            fc2_weight=weights[2],
            fc2_bias=weights[3],
        )
        if variant == "int8_pc":
            model = quantize_coreml_int8(model)
        model.save(package)
    loaded = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    temporary = Path(loaded.get_compiled_model_path())
    shutil.copytree(temporary, compiled)
    return compiled


def load_bridge(bridge_dir: Path) -> Any:
    bridge_text = str(bridge_dir.resolve())
    if bridge_text not in sys.path:
        sys.path.insert(0, bridge_text)
    return importlib.import_module("_flux2_ane_bridge")


def timed_mlx(mx: Any, operation: Callable[[Any], Any], x: Any, warmup: int, iterations: int):
    for _ in range(warmup):
        mx.eval(operation(x))
    elapsed: list[float] = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = operation(x)
        mx.eval(output)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return output, elapsed


def array_metrics(actual: Any, reference: Any) -> dict[str, Any]:
    actual_np = np.asarray(actual).astype(np.float32)
    reference_np = np.asarray(reference).astype(np.float32)
    delta = actual_np - reference_np
    actual_sq = float(np.sum(np.square(actual_np), dtype=np.float64))
    reference_sq = float(np.sum(np.square(reference_np), dtype=np.float64))
    dot = float(np.sum(actual_np * reference_np, dtype=np.float64))
    denominator = reference_sq**0.5
    cosine_denominator = (actual_sq * reference_sq)**0.5
    return {
        "finite": bool(np.isfinite(actual_np).all()),
        "cosine": float(dot / cosine_denominator)
        if cosine_denominator
        else None,
        "relative_l2": float(np.sum(np.square(delta), dtype=np.float64)**0.5 / denominator)
        if denominator
        else None,
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "reference_rms": float(np.sqrt(np.mean(np.square(reference_np)))),
        "actual_rms": float(np.sqrt(np.mean(np.square(actual_np)))),
    }


def latency_metrics(values: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    import mlx.core as mx
    from mlx import nn

    weights = load_packed_mlp(args.checkpoint, args.block, mx)
    checkpoint_digest = sha256_file(args.checkpoint)
    full, prefix, suffix = make_mlx_branches(mx, nn, weights, args.ane_width)
    if not args.no_compile_mlx:
        full = mx.compile(full)
        prefix = mx.compile(prefix)
        suffix = mx.compile(suffix)

    generator = np.random.default_rng(args.seed)
    input_np = generator.normal(0.0, 0.25, (1, args.rows, HIDDEN)).astype(np.float16)
    x = mx.array(input_np)
    mx.eval(x)

    full_output, full_times = timed_mlx(mx, full, x, args.warmup, args.iterations)
    prefix_output, prefix_times = timed_mlx(mx, prefix, x, args.warmup, args.iterations)
    suffix_output, suffix_times = timed_mlx(mx, suffix, x, args.warmup, args.iterations)
    split_reference = prefix_output + suffix_output
    mx.eval(split_reference)

    dense_prefix = dequantized_prefix(weights, args.ane_width, mx)
    artifact = compiled_artifact(
        artifact_dir=args.artifact_dir,
        block=args.block,
        checkpoint_digest=checkpoint_digest,
        rows=args.rows,
        ane_width=args.ane_width,
        variant=args.variant,
        weights=dense_prefix,
    )
    bridge_module = load_bridge(args.bridge_dir)
    bridge = bridge_module.Flux2ANEModel(
        model_path=str(artifact),
        function_name="main",
        m=args.rows,
        k=HIDDEN,
        n=HIDDEN,
        cache_bindings=True,
        fast_prediction=False,
    )
    ane_output = mx.zeros((1, args.rows, HIDDEN), dtype=mx.float16)
    mx.eval(ane_output)

    for _ in range(args.warmup):
        bridge.predict(x, ane_output)
    ane_times: list[float] = []
    ane_calls: list[dict[str, Any]] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = dict(bridge.predict(x, ane_output))
        ane_times.append((time.perf_counter() - started) * 1000.0)
        ane_calls.append(result)
    mx.eval(ane_output)

    def run_hybrid() -> tuple[Any, dict[str, Any]]:
        gpu_output = suffix(x)
        mx.async_eval(gpu_output)
        call = dict(bridge.predict(x, ane_output))
        joined = gpu_output + ane_output
        mx.eval(joined)
        return joined, call

    for _ in range(args.warmup):
        run_hybrid()
    hybrid_times: list[float] = []
    hybrid_calls: list[dict[str, Any]] = []
    hybrid_output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        hybrid_output, call = run_hybrid()
        hybrid_times.append((time.perf_counter() - started) * 1000.0)
        hybrid_calls.append(call)

    full_median = statistics.median(full_times)
    hybrid_median = statistics.median(hybrid_times)
    report = {
        "schema": "turbocider-fastmetal-ane-mlp-probe-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "block": args.block,
        "shape": {
            "rows": args.rows,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "ane_intermediate": args.ane_width,
            "gpu_intermediate": INTERMEDIATE - args.ane_width,
        },
        "variant": args.variant,
        "mlx_compiled": not args.no_compile_mlx,
        "artifact": str(artifact.resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency": {
            "mlx_full": latency_metrics(full_times),
            "mlx_prefix_reference": latency_metrics(prefix_times),
            "mlx_gpu_suffix": latency_metrics(suffix_times),
            "coreml_ane_prefix": latency_metrics(ane_times),
            "hybrid_total": latency_metrics(hybrid_times),
            "speedup_vs_mlx_full": full_median / hybrid_median,
        },
        "accuracy": {
            "mlx_split_vs_full": array_metrics(split_reference, full_output),
            "coreml_prefix_vs_mlx_prefix": array_metrics(ane_output, prefix_output),
            "hybrid_vs_mlx_split": array_metrics(hybrid_output, split_reference),
            "hybrid_vs_mlx_full": array_metrics(hybrid_output, full_output),
        },
        "coreml": {
            "compute_units": "CPUAndNeuralEngine",
            "load": dict(bridge.load_metrics()),
            "ane_calls": ane_calls,
            "hybrid_calls": hybrid_calls,
            "all_output_backings_used": all(
                call.get("output_backing_used", False) for call in ane_calls + hybrid_calls
            ),
        },
        "scope": (
            "One real FastMetal block FFN with caller-owned MLX buffers. "
            "ANE attribution still requires Instruments; production admission also requires full-DiT E2E parity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
