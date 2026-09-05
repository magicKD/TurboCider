#!/usr/bin/env python3
"""Benchmark all 48 ltx-2-mlx blocks from a local Comfy ConvRot checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from bench_mlx_comfy_block import (
    HADAMARD_256,
    SafetensorsMMap,
    load_block_weights,
    quantized_layers,
    random_bf16,
    replace_quantized_linears,
    video_positions,
)
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--text-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def target_to_source_top(name: str) -> str:
    return name.replace(".linear1.", ".linear_1.").replace(".linear2.", ".linear_2.")


def load_top_weights(model: LTXModel, reader: SafetensorsMMap) -> None:
    prefix = "model.diffusion_model."
    weights: list[tuple[str, mx.array]] = []
    for name, parameter in tree_flatten(model.parameters()):
        if name.startswith("transformer_blocks."):
            continue
        value = reader.mlx_array(prefix + target_to_source_top(name))
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"shape mismatch for {name}: loaded {value.shape}, expected {parameter.shape}"
            )
        weights.append((name, value))
    model.load_weights(weights, strict=False)
    mx.eval(*[value for _, value in weights])


def load_model(reader: SafetensorsMMap) -> tuple[LTXModel, float]:
    raw_config = reader.metadata.get("config")
    if not isinstance(raw_config, str):
        raise ValueError("checkpoint has no embedded transformer config")
    config = LTXModelConfig.from_checkpoint_config(json.loads(raw_config))
    if config.num_layers != 48:
        raise ValueError(f"expected 48 blocks, got {config.num_layers}")

    started = time.perf_counter()
    model = LTXModel(config)
    load_top_weights(model, reader)
    for block_index, block in enumerate(model.transformer_blocks):
        source_prefix = f"model.diffusion_model.transformer_blocks.{block_index}."
        layers = quantized_layers(reader, source_prefix)
        if len(layers) != 28:
            raise ValueError(f"block {block_index} has {len(layers)} quantized linears")
        replace_quantized_linears(block, layers)
        load_block_weights(block, reader, source_prefix, layers)
        mx.eval(block.parameters())
        gc.collect()
        if (block_index + 1) % 8 == 0:
            print(f"loaded_blocks={block_index + 1}/48", flush=True)
    mx.eval(model.parameters(), HADAMARD_256)
    return model, time.perf_counter() - started


def benchmark_stage(
    model: LTXModel,
    rows: int,
    text_rows: int,
    warmup: int,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + rows)
    video = random_bf16(rng, (1, rows, 128), 0.25)
    audio = random_bf16(rng, (1, 101, 128), 0.25)
    video_text = random_bf16(rng, (1, text_rows, 4096), 0.1)
    audio_text = random_bf16(rng, (1, text_rows, 2048), 0.1)
    active_video_positions = video_positions(rows)
    active_audio_positions = mx.array(
        (np.arange(101, dtype=np.float32) % np.float32(20.0))[:, None]
    )[None]
    timestep = mx.array([0.5], dtype=mx.bfloat16)

    def run() -> tuple[mx.array, mx.array]:
        output = model(
            video,
            audio,
            timestep,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
            video_positions=active_video_positions,
            audio_positions=active_audio_positions,
        )
        mx.eval(*output)
        return output

    for _ in range(warmup):
        output = run()
        del output
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = run()
        timings.append(time.perf_counter() - started)
        del output
    timings.sort()
    p50 = timings[(len(timings) - 1) // 2]
    p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]
    return p50, p95


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.text_rows <= 0:
        raise ValueError("warmup/text rows/iterations are invalid")
    reader = SafetensorsMMap(args.checkpoint)
    model, setup_seconds = load_model(reader)
    print(f"checkpoint={args.checkpoint}")
    print(f"setup_seconds={setup_seconds:.6f} resident_blocks=48")

    stages: list[tuple[str, int]] = []
    if args.stage in {"stage1", "both"}:
        stages.append(("stage1", 1001))
    if args.stage in {"stage2", "both"}:
        stages.append(("stage2", 4004))
    for name, rows in stages:
        p50, p95 = benchmark_stage(
            model,
            rows,
            args.text_rows,
            args.warmup,
            args.iterations,
            args.seed,
        )
        print(
            f"stage={name} video_rows={rows} audio_rows=101 text_rows={args.text_rows} "
            f"warmup={args.warmup} iterations={args.iterations} "
            f"p50_seconds={p50:.6f} p95_seconds={p95:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
