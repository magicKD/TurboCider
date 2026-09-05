#!/usr/bin/env python3
"""Benchmark MLX LTX-2.5 media components from local Comfy checkpoints.

The checkpoints used by ComfyUI retain PyTorch convolution layouts while the
reference ``ltx-2-mlx`` modules expect MLX channel-last layouts.  This adapter
performs that conversion explicitly, validates every loaded parameter shape,
and benchmarks the same 480p-class tensors used by the native runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from bench_mlx_comfy_block import SafetensorsMMap
from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.model.video_vae.normalization import pixel_norm
from ltx_core_mlx.model.video_vae.sampling import (
    pixel_shuffle_3d,
    unpatchify_spatial,
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="component", required=True)

    upsampler = subparsers.add_parser("upsampler")
    add_common_arguments(upsampler)
    upsampler.add_argument("--frames", type=positive_int, default=13)
    upsampler.add_argument("--height", type=positive_int, default=7)
    upsampler.add_argument("--width", type=positive_int, default=11)
    upsampler.add_argument(
        "--reference-dtype", choices=("bf16", "f32"), default="f32"
    )

    vae = subparsers.add_parser("vae")
    add_common_arguments(vae)
    vae.add_argument("--frames", type=positive_int, default=13)
    vae.add_argument("--height", type=positive_int, default=14)
    vae.add_argument("--width", type=positive_int, default=22)
    vae.add_argument(
        "--reference-dtype", choices=("bf16", "f32"), default="bf16"
    )
    vae.add_argument(
        "--input-layout", choices=("bcfhw", "bfhwc"), default="bcfhw"
    )
    vae.add_argument("--materialize-stages", action="store_true")
    vae.add_argument("--dump-stages", type=Path)
    return parser.parse_args()


def load_bf16_raw(path: Path, shape: tuple[int, ...]) -> mx.array:
    count = math.prod(shape)
    raw = np.memmap(path, dtype="<u2", mode="r", shape=(count,))
    bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return mx.array(bits.view(np.float32).reshape(shape)).astype(mx.bfloat16)


def deterministic_bf16(shape: tuple[int, ...]) -> mx.array:
    count = math.prod(shape)
    indices = np.arange(count, dtype=np.int64)
    values = ((indices % 257) - 128).astype(np.float32) / np.float32(64.0)
    return mx.array(values.reshape(shape)).astype(mx.bfloat16)


def convert_convolution_layout(
    value: mx.array, target_shape: tuple[int, ...], source_key: str
) -> mx.array:
    if tuple(value.shape) == target_shape:
        return value
    if value.ndim == 5:
        converted = value.transpose(0, 2, 3, 4, 1)
    elif value.ndim == 4:
        converted = value.transpose(0, 2, 3, 1)
    else:
        converted = value
    if tuple(converted.shape) != target_shape:
        raise ValueError(
            f"shape mismatch for {source_key}: source={tuple(value.shape)} "
            f"converted={tuple(converted.shape)} expected={target_shape}"
        )
    return converted


def load_module_weights(
    module: object,
    reader: SafetensorsMMap,
    source_key: Callable[[str], str],
) -> tuple[int, int]:
    weights: list[tuple[str, mx.array]] = []
    source_names: set[str] = set()
    source_bytes = 0
    for target, parameter in tree_flatten(module.parameters()):
        source = source_key(target)
        entry = reader.entry(source)
        begin, end = (int(value) for value in entry["data_offsets"])
        value = convert_convolution_layout(
            reader.mlx_array(source), tuple(parameter.shape), source
        )
        weights.append((target, value))
        source_names.add(source)
        source_bytes += end - begin
    module.load_weights(weights, strict=True)
    mx.eval(*[value for _, value in weights])
    return len(source_names), source_bytes


def benchmark(
    run: Callable[[], mx.array], warmup: int, iterations: int
) -> tuple[mx.array, list[float]]:
    for _ in range(warmup):
        output = run()
        mx.eval(output)
        del output
    gc.collect()

    timings: list[float] = []
    final_output: mx.array | None = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = run()
        mx.eval(output)
        timings.append(time.perf_counter() - started)
        if final_output is not None:
            del final_output
        final_output = output
    assert final_output is not None
    timings.sort()
    return final_output, timings


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    bits = np.asarray(words, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def compare_raw(
    actual: mx.array, reference_path: Path, reference_dtype: str
) -> tuple[float, float, float, int]:
    actual_f32 = np.asarray(actual.astype(mx.float32)).reshape(-1)
    if reference_dtype == "bf16":
        reference = np.memmap(reference_path, dtype="<u2", mode="r")
    else:
        reference = np.memmap(reference_path, dtype="<f4", mode="r")
    if reference.size != actual_f32.size:
        raise ValueError(
            f"reference elements={reference.size}, output elements={actual_f32.size}"
        )

    delta_l2 = 0.0
    reference_l2 = 0.0
    actual_l2 = 0.0
    dot = 0.0
    max_abs = 0.0
    nonfinite = 0
    chunk = 1 << 20
    for begin in range(0, reference.size, chunk):
        end = min(reference.size, begin + chunk)
        left = actual_f32[begin:end].astype(np.float64)
        if reference_dtype == "bf16":
            right = bf16_words_to_float32(reference[begin:end]).astype(np.float64)
        else:
            right = np.asarray(reference[begin:end], dtype=np.float64)
        finite = np.isfinite(left) & np.isfinite(right)
        nonfinite += int(finite.size - np.count_nonzero(finite))
        if not np.all(finite):
            left = left[finite]
            right = right[finite]
        delta = left - right
        delta_l2 += float(np.dot(delta, delta))
        reference_l2 += float(np.dot(right, right))
        actual_l2 += float(np.dot(left, left))
        dot += float(np.dot(left, right))
        if delta.size:
            max_abs = max(max_abs, float(np.max(np.abs(delta))))

    tiny = np.finfo(np.float64).tiny
    rel_l2 = math.sqrt(delta_l2) / max(math.sqrt(reference_l2), tiny)
    cosine = dot / max(math.sqrt(actual_l2 * reference_l2), tiny)
    return rel_l2, cosine, max_abs, nonfinite


def print_timings(timings: list[float]) -> None:
    p50 = timings[(len(timings) - 1) // 2]
    p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
    print(
        f"warm_iterations={len(timings)} p50_seconds={p50:.6f} "
        f"p95_seconds={p95:.6f} mean_seconds={sum(timings) / len(timings):.6f}"
    )


def dump_f32(directory: Path, name: str, value: mx.array) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    materialized = mx.contiguous(value.astype(mx.float32))
    mx.eval(materialized)
    np.asarray(materialized).tofile(directory / f"{name}.f32")
    print(f"dump_stage={name} shape={tuple(value.shape)}")


def dump_vae_stages(
    model: VideoDecoder, latent: mx.array, directory: Path
) -> None:
    output_dtype = latent.dtype
    parameters = tree_flatten(model.parameters())
    weights_dtype = parameters[0][1].dtype if parameters else output_dtype
    if latent.dtype != weights_dtype:
        latent = latent.astype(weights_dtype)

    value = latent.transpose(0, 2, 3, 4, 1)
    dump_f32(directory, "00_input", value)
    value = model.denormalize_latent(value)
    dump_f32(directory, "01_denorm", value)
    value = model.conv_in(value)
    dump_f32(directory, "02_conv_in", value)

    upsample_index = 0
    stage_names = (
        "03_res0",
        "04_up1",
        "05_res2",
        "06_up3",
        "07_res4",
        "08_up5",
        "09_res6",
        "10_up7",
        "11_res8",
    )
    for index, block in enumerate(model.up_blocks):
        value = block(value)
        if index % 2 == 1:
            spatial_factor, temporal_factor = model._upsample_config[
                upsample_index
            ]
            value = pixel_shuffle_3d(
                value,
                spatial_factor=spatial_factor,
                temporal_factor=temporal_factor,
            )
            if temporal_factor > 1:
                value = value[:, 1:, :, :, :]
            upsample_index += 1
        dump_f32(directory, stage_names[index], value)

    value = nn.silu(pixel_norm(value))
    dump_f32(directory, "12_pre_out", value)
    value = model.conv_out(value)
    dump_f32(directory, "13_conv_out", value)
    value = unpatchify_spatial(value, patch_size=4)
    value = value.transpose(0, 4, 1, 2, 3).astype(output_dtype)
    dump_f32(directory, "14_output", value)


def run_upsampler(args: argparse.Namespace) -> None:
    reader = SafetensorsMMap(args.checkpoint)
    raw_config = reader.metadata.get("config")
    config = json.loads(raw_config) if isinstance(raw_config, str) else {}
    config = config.get("config", config)

    started = time.perf_counter()
    model = LatentUpsampler.from_config(config)
    tensor_count, weight_bytes = load_module_weights(model, reader, lambda name: name)
    setup_seconds = time.perf_counter() - started

    shape = (1, int(config.get("in_channels", 128)), args.frames, args.height, args.width)
    latent = load_bf16_raw(args.input, shape) if args.input else deterministic_bf16(shape)
    output, timings = benchmark(lambda: model(latent), args.warmup, args.iterations)

    print(f"component=upsampler checkpoint={args.checkpoint}")
    print(
        f"weights={tensor_count} weight_gib={weight_bytes / 1024**3:.6f} "
        f"setup_seconds={setup_seconds:.6f}"
    )
    print(f"input_shape={shape} output_shape={tuple(output.shape)} dtype={output.dtype}")
    print_timings(timings)
    if args.reference:
        rel_l2, cosine, max_abs, nonfinite = compare_raw(
            output, args.reference, args.reference_dtype
        )
        print(
            f"reference={args.reference} reference_dtype={args.reference_dtype} "
            f"rel_l2={rel_l2:.9g} cosine={cosine:.9g} "
            f"max_abs={max_abs:.9g} nonfinite={nonfinite}"
        )


def vae_source_key(target: str) -> str:
    if target == "per_channel_statistics.mean":
        return "per_channel_statistics.mean-of-means"
    if target == "per_channel_statistics.std":
        return "per_channel_statistics.std-of-means"
    return "decoder." + target


def run_vae(args: argparse.Namespace) -> None:
    reader = SafetensorsMMap(args.checkpoint)
    raw_config = reader.metadata.get("config")
    config = json.loads(raw_config) if isinstance(raw_config, str) else {}
    vae_config = config.get("vae", config)

    started = time.perf_counter()
    model = VideoDecoder(
        causal=bool(vae_config.get("causal_decoder", False)),
        spatial_padding_mode=str(vae_config.get("spatial_padding_mode", "zeros")),
    )
    tensor_count, weight_bytes = load_module_weights(model, reader, vae_source_key)
    setup_seconds = time.perf_counter() - started

    shape = (1, 128, args.frames, args.height, args.width)
    if args.input and args.input_layout == "bfhwc":
        token_shape = (1, args.frames, args.height, args.width, 128)
        latent = load_bf16_raw(args.input, token_shape).transpose(0, 4, 1, 2, 3)
    else:
        latent = load_bf16_raw(args.input, shape) if args.input else deterministic_bf16(shape)
    output, timings = benchmark(
        lambda: model.decode(latent, _materialize_stages=args.materialize_stages),
        args.warmup,
        args.iterations,
    )
    if args.dump_stages:
        dump_vae_stages(model, latent, args.dump_stages)

    print(f"component=video_vae checkpoint={args.checkpoint}")
    print(
        f"weights={tensor_count} weight_gib={weight_bytes / 1024**3:.6f} "
        f"setup_seconds={setup_seconds:.6f} materialize_stages={args.materialize_stages}"
    )
    print(
        f"input_shape={shape} input_layout={args.input_layout} "
        f"output_shape={tuple(output.shape)} dtype={output.dtype}"
    )
    print_timings(timings)
    if args.reference:
        rel_l2, cosine, max_abs, nonfinite = compare_raw(
            output, args.reference, args.reference_dtype
        )
        print(
            f"reference={args.reference} reference_dtype={args.reference_dtype} "
            f"rel_l2={rel_l2:.9g} cosine={cosine:.9g} "
            f"max_abs={max_abs:.9g} nonfinite={nonfinite}"
        )


def main() -> None:
    args = parse_args()
    if args.component == "upsampler":
        run_upsampler(args)
    else:
        run_vae(args)


if __name__ == "__main__":
    main()
