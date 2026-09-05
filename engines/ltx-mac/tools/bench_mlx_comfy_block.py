#!/usr/bin/env python3
"""Run one ltx-2-mlx block directly from a Comfy INT8 ConvRot checkpoint.

This is a local reference adapter, not a new model format.  Comfy's signed
tensorwise INT8 rows are represented exactly with MLX affine q8 by shifting
each byte to unsigned, repeating the row scale for every MLX group, and using
``bias=-128*scale``.  A custom QuantizedLinear applies the checkpoint's H256
ConvRot transform before MLX quantized_matmul.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map_with_path

from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock


GROUP_SIZE = 64
BITS = 8
CONVROT_GROUP_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--video-rows", type=int, default=None)
    parser.add_argument("--audio-rows", type=int, default=101)
    parser.add_argument("--text-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SafetensorsMMap:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors file: {path}")
            header_size = struct.unpack("<Q", raw)[0]
            header = json.loads(stream.read(header_size))
        self.data_start = 8 + header_size
        metadata = header.get("__metadata__", {})
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.header = {key: value for key, value in header.items() if key != "__metadata__"}

    def entry(self, key: str) -> dict[str, Any]:
        try:
            entry = self.header[key]
        except KeyError as exc:
            raise KeyError(f"checkpoint has no tensor {key}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"invalid safetensors entry for {key}")
        return entry

    def array(self, key: str) -> np.ndarray:
        entry = self.entry(key)
        begin, end = (int(value) for value in entry["data_offsets"])
        shape = tuple(int(value) for value in entry["shape"])
        dtype = entry["dtype"]
        dtype_map = {
            "I8": np.dtype("i1"),
            "U8": np.dtype("u1"),
            "F16": np.dtype("<f2"),
            "F32": np.dtype("<f4"),
            "BF16": np.dtype("<u2"),
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype} for {key}")
        result = np.memmap(
            self.path,
            dtype=dtype_map[dtype],
            mode="r",
            offset=self.data_start + begin,
            shape=shape,
            order="C",
        )
        if result.nbytes != end - begin:
            raise ValueError(f"tensor byte count mismatch for {key}")
        return result

    def mlx_array(self, key: str) -> mx.array:
        entry = self.entry(key)
        value = self.array(key)
        if entry["dtype"] == "BF16":
            bits = np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
            return mx.array(bits.view(np.float32)).astype(mx.bfloat16)
        return mx.array(np.asarray(value))


def custom_hadamard_256() -> mx.array:
    h4 = np.asarray(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=np.float32,
    )
    table = h4
    while table.shape[0] < CONVROT_GROUP_SIZE:
        table = np.kron(table, h4).astype(np.float32, copy=False)
    table *= np.float32(1.0 / math.sqrt(CONVROT_GROUP_SIZE))
    return mx.array(table).astype(mx.bfloat16)


HADAMARD_256 = custom_hadamard_256()


class ConvRotQuantizedLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool):
        super().__init__()
        if input_dims % CONVROT_GROUP_SIZE:
            raise ValueError(f"ConvRot input dim {input_dims} is not divisible by 256")
        self.group_size = GROUP_SIZE
        self.bits = BITS
        self.weight = mx.zeros((output_dims, input_dims // 4), dtype=mx.uint32)
        groups = input_dims // GROUP_SIZE
        self.scales = mx.zeros((output_dims, groups), dtype=mx.float32)
        self.biases = mx.zeros((output_dims, groups), dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((output_dims,), dtype=mx.bfloat16)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        rotated = x.reshape(-1, CONVROT_GROUP_SIZE) @ HADAMARD_256.astype(x.dtype)
        rotated = rotated.reshape(shape)
        output = mx.quantized_matmul(
            rotated,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            output = output + self.bias
        return output


def source_to_target(name: str) -> str:
    return (
        name.replace(".to_out.0.", ".to_out.")
        .replace("audio_ff.net.0.proj.", "audio_ff.proj_in.")
        .replace("audio_ff.net.2.", "audio_ff.proj_out.")
        .replace("ff.net.0.proj.", "ff.proj_in.")
        .replace("ff.net.2.", "ff.proj_out.")
    )


def target_to_source(name: str) -> str:
    return (
        name.replace(".to_out.", ".to_out.0.")
        .replace("audio_ff.proj_in.", "audio_ff.net.0.proj.")
        .replace("audio_ff.proj_out.", "audio_ff.net.2.")
        .replace("ff.proj_in.", "ff.net.0.proj.")
        .replace("ff.proj_out.", "ff.net.2.")
    )


def quantized_layers(reader: SafetensorsMMap, prefix: str) -> set[str]:
    layers: set[str] = set()
    for key in reader.header:
        if key.startswith(prefix) and key.endswith(".comfy_quant"):
            relative = key[len(prefix) : -len(".comfy_quant")]
            layers.add(source_to_target(relative + ".x")[:-2])
    return layers


def replace_quantized_linears(block: nn.Module, layers: set[str]) -> None:
    def replace(path: str, module: nn.Module) -> nn.Module:
        if path not in layers:
            return module
        if not isinstance(module, nn.Linear):
            raise TypeError(f"quantized checkpoint path {path} is not an MLX Linear")
        output_dims, input_dims = (int(value) for value in module.weight.shape)
        return ConvRotQuantizedLinear(input_dims, output_dims, "bias" in module)

    leaves = tree_map_with_path(replace, block.leaf_modules(), is_leaf=nn.Module.is_module)
    block.update_modules(leaves)


def pack_q8(weight: np.ndarray) -> np.ndarray:
    if weight.dtype != np.int8 or weight.ndim != 2 or weight.shape[1] % 4:
        raise ValueError(f"invalid Comfy INT8 weight {weight.dtype} {weight.shape}")
    unsigned = np.bitwise_xor(np.asarray(weight).view(np.uint8), np.uint8(0x80))
    return np.ascontiguousarray(unsigned).view("<u4").reshape(weight.shape[0], weight.shape[1] // 4)


def load_block_weights(
    block: nn.Module,
    reader: SafetensorsMMap,
    source_prefix: str,
    layers: set[str],
) -> None:
    weights: list[tuple[str, mx.array]] = []
    for target, parameter in tree_flatten(block.parameters()):
        parts = target.rsplit(".", 1)
        layer = parts[0] if len(parts) == 2 else ""
        leaf = parts[-1]
        source_layer = target_to_source(layer + ".x")[:-2]
        source_base = source_prefix + source_layer
        if layer in layers and leaf in {"weight", "scales", "biases"}:
            raw_weight = reader.array(source_base + ".weight")
            output_dims, input_dims = raw_weight.shape
            if leaf == "weight":
                value = mx.array(pack_q8(raw_weight))
            else:
                row_scale = np.asarray(
                    reader.array(source_base + ".weight_scale"), dtype=np.float32
                ).reshape(output_dims, 1)
                repeated = np.broadcast_to(
                    row_scale, (output_dims, input_dims // GROUP_SIZE)
                ).copy()
                if leaf == "biases":
                    repeated *= np.float32(-128.0)
                value = mx.array(repeated)
        else:
            source_key = source_prefix + target_to_source(target)
            value = reader.mlx_array(source_key)
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"shape mismatch for {target}: loaded {value.shape}, expected {parameter.shape}"
            )
        weights.append((target, value))
    block.load_weights(weights, strict=True)


def load_fixture_tensor(
    fixture: Path,
    manifest: dict[str, Any],
    name: str,
) -> mx.array:
    entry = manifest["tensors"][name]
    shape = tuple(int(value) for value in entry["shape"])
    path = fixture / entry["file"]
    if entry["dtype"] == "bf16":
        raw = np.fromfile(path, dtype="<u2").reshape(shape)
        bits = raw.astype(np.uint32) << np.uint32(16)
        return mx.array(bits.view(np.float32)).astype(mx.bfloat16)
    if entry["dtype"] == "f32":
        return mx.array(np.fromfile(path, dtype="<f4").reshape(shape))
    raise ValueError(f"unsupported fixture dtype {entry['dtype']}")


def metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    left = actual.astype(np.float64).reshape(-1)
    right = expected.astype(np.float64).reshape(-1)
    delta = left - right
    rel_l2 = np.linalg.norm(delta) / max(np.linalg.norm(right), np.finfo(np.float64).tiny)
    cosine = float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), np.finfo(np.float64).tiny))
    return float(rel_l2), cosine, float(np.max(np.abs(delta)))


def random_bf16(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> mx.array:
    value = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
    return mx.array(value).astype(mx.bfloat16)


def video_positions(rows: int) -> mx.array:
    geometry = {1001: (13, 7, 11), 4004: (13, 14, 22)}.get(rows)
    if geometry is None:
        value = np.zeros((rows, 3), dtype=np.float32)
        value[:, 0] = np.arange(rows, dtype=np.float32) % np.float32(20.0)
        return mx.array(value)[None]
    frames, height, width = geometry
    value = np.stack(
        np.meshgrid(
            np.arange(frames, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(rows, 3)
    return mx.array(value)[None]


def main() -> None:
    args = parse_args()
    if (
        args.block < 0
        or args.warmup < 0
        or args.iterations <= 0
        or args.audio_rows <= 0
        or args.text_rows <= 0
        or (args.video_rows is not None and args.video_rows <= 0)
    ):
        raise ValueError("block/warmup/iterations must be nonnegative, nonnegative, positive")
    manifest = json.loads((args.fixture / "manifest.json").read_text(encoding="utf-8"))
    reader = SafetensorsMMap(args.checkpoint)
    source_prefix = f"model.diffusion_model.transformer_blocks.{args.block}."
    layers = quantized_layers(reader, source_prefix)
    if not layers:
        raise ValueError(f"no Comfy INT8 layers found below {source_prefix}")

    started = time.perf_counter()
    block = BasicAVTransformerBlock(
        video_dim=4096,
        audio_dim=2048,
        video_num_heads=32,
        audio_num_heads=32,
        video_head_dim=128,
        audio_head_dim=64,
        av_cross_num_heads=32,
        av_cross_head_dim=64,
        ff_mult=4.0,
        norm_eps=1e-6,
        ff_bias=False,
        audio_ff_bias=True,
    )
    replace_quantized_linears(block, layers)
    load_block_weights(block, reader, source_prefix, layers)
    mx.eval(block.parameters(), HADAMARD_256)
    setup_seconds = time.perf_counter() - started

    fixture_video_rows = int(manifest["tensors"]["video_input"]["shape"][0])
    benchmark_only = args.video_rows is not None
    active_video_rows = args.video_rows or fixture_video_rows
    if benchmark_only:
        rng = np.random.default_rng(args.seed)
        video = random_bf16(rng, (1, active_video_rows, 4096), 0.25)
        audio = random_bf16(rng, (1, args.audio_rows, 2048), 0.25)
        video_text = random_bf16(rng, (1, args.text_rows, 4096), 0.1)
        audio_text = random_bf16(rng, (1, args.text_rows, 2048), 0.1)
        text_mask = mx.zeros((1, 1, 1, args.text_rows), dtype=mx.bfloat16)
    else:
        video = load_fixture_tensor(args.fixture, manifest, "video_input")[None]
        audio = load_fixture_tensor(args.fixture, manifest, "audio_input")[None]
        video_text = load_fixture_tensor(args.fixture, manifest, "video_context")[None]
        audio_text = load_fixture_tensor(args.fixture, manifest, "audio_context")[None]
        text_mask = load_fixture_tensor(args.fixture, manifest, "text_mask").reshape(1, 1, 1, -1)

    def params(name: str) -> mx.array:
        return load_fixture_tensor(args.fixture, manifest, name).reshape(1, -1)

    if benchmark_only:
        active_video_positions = video_positions(active_video_rows)
        active_audio_positions = mx.array(
            (np.arange(args.audio_rows, dtype=np.float32) % np.float32(20.0))[:, None]
        )[None]
    else:
        active_video_positions = load_fixture_tensor(
            args.fixture, manifest, "video_positions"
        )[None]
        active_audio_positions = load_fixture_tensor(
            args.fixture, manifest, "audio_positions"
        )[None]
    video_rope = precompute_rope_freqs(
        active_video_positions, 4096, 32, max_pos=[20, 2048, 2048],
        rope_type="split", double_precision=True,
    )
    audio_rope = precompute_rope_freqs(
        active_audio_positions, 2048, 32, max_pos=[20],
        rope_type="split", double_precision=True,
    )
    video_cross_rope = precompute_rope_freqs(
        active_video_positions[..., :1], 2048, 32, max_pos=[20],
        rope_type="split", double_precision=True,
    )

    call_args = (
        video,
        audio,
        params("video_adaln_params"),
        params("audio_adaln_params"),
        params("video_prompt_params"),
        params("audio_prompt_params"),
        params("av_video_params"),
        params("av_audio_params"),
        params("a2v_gate_params"),
        params("v2a_gate_params"),
    )
    call_kwargs = {
        "video_text_embeds": video_text,
        "audio_text_embeds": audio_text,
        "video_rope_freqs": video_rope,
        "audio_rope_freqs": audio_rope,
        "video_cross_rope_freqs": video_cross_rope,
        "audio_cross_rope_freqs": audio_rope,
        "video_cross_attention_mask": text_mask,
    }

    def run() -> tuple[mx.array, mx.array]:
        output = block(*call_args, **call_kwargs)
        mx.eval(*output)
        return output

    for _ in range(args.warmup):
        run()
    timings: list[float] = []
    output: tuple[mx.array, mx.array] | None = None
    for _ in range(args.iterations):
        begin = time.perf_counter()
        output = run()
        timings.append(time.perf_counter() - begin)
    assert output is not None

    video_metrics = None
    audio_metrics = None
    if not benchmark_only:
        actual_video = np.asarray(output[0].astype(mx.float32))[0]
        actual_audio = np.asarray(output[1].astype(mx.float32))[0]
        expected_video = np.fromfile(
            args.fixture / manifest["tensors"]["video_output_f32"]["file"], dtype="<f4"
        ).reshape(actual_video.shape)
        expected_audio = np.fromfile(
            args.fixture / manifest["tensors"]["audio_output_f32"]["file"], dtype="<f4"
        ).reshape(actual_audio.shape)
        video_metrics = metrics(actual_video, expected_video)
        audio_metrics = metrics(actual_audio, expected_audio)
    timings.sort()
    p50 = timings[(len(timings) - 1) // 2]
    p95 = timings[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)]

    print(f"checkpoint={args.checkpoint}")
    print(f"block={args.block} quantized_linears={len(layers)} setup_seconds={setup_seconds:.6f}")
    print(
        f"video_rows={active_video_rows} audio_rows={audio.shape[1]} "
        f"text_rows={video_text.shape[1]} mode={'benchmark' if benchmark_only else 'fixture'}"
    )
    print(f"warmup={args.warmup} iterations={args.iterations} p50_ms={p50 * 1000.0:.3f} p95_ms={p95 * 1000.0:.3f}")
    if video_metrics is not None and audio_metrics is not None:
        print(
            "video_rel_l2=%.9g video_cosine=%.9f video_max_abs=%.9g"
            % video_metrics
        )
        print(
            "audio_rel_l2=%.9g audio_cosine=%.9f audio_max_abs=%.9g"
            % audio_metrics
        )


if __name__ == "__main__":
    main()
