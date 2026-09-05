#!/usr/bin/env python3
"""Export one fixed-shape LTX-2.5 V-to-A cross-attention Core ML model.

The GPU path applies the checkpoint's ConvRot transform before every INT8
linear.  This exporter folds the transform into dequantized FP16 weights and
embeds fixed Stage-1/Stage-2 RoPE tables, leaving two row-major FP16 inputs:
audio query state [101, 2048] and video key/value state [V, 4096].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from export_ane_mlp_artifact import (
    absorb_convrot,
    compile_model,
    mapped_tensor,
    quantize_int8,
    read_header,
    tensor_info,
)


SCHEMA = "ltx-ane-v2a-v1"
AUDIO_ROWS = 101
AUDIO_DIM = 2048
VIDEO_DIM = 4096
HEADS = 32
HEAD_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--video-rows", type=int, choices=(1001, 4004),
                        required=True)
    parser.add_argument("--variants", nargs="+", choices=("fp16", "int8_pc"),
                        default=("fp16",))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiled-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bf16_tensor(path: Path, data_offset: int, info, shape):
    import numpy as np

    bits = mapped_tensor(path, data_offset, info, np.dtype("<u2"), shape)
    expanded = np.asarray(bits, dtype=np.uint32) << np.uint32(16)
    return expanded.view(np.float32).astype(np.float16)


def video_temporal_positions(video_rows: int):
    import numpy as np

    if video_rows == 1001:
        frames, height, width = 13, 7, 11
    else:
        frames, height, width = 13, 14, 22
    values = []
    for frame in range(frames):
        start = max(frame * 8 + 1 - 8, 0)
        end = max((frame + 1) * 8 + 1 - 8, 0)
        timestamp = (start + end) * 0.5 / 24.0
        values.extend([timestamp] * (height * width))
    return np.asarray(values, dtype=np.float32)


def audio_positions():
    import numpy as np

    values = []
    for token in range(AUDIO_ROWS):
        start = max(token * 4 + 1 - 4, 0)
        end = max((token + 1) * 4 + 1 - 4, 0)
        values.append((start + end) * 0.5 * (160.0 / 16000.0))
    return np.asarray(values, dtype=np.float32)


def round_bf16(values):
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def rope_table(positions):
    import numpy as np

    inner = HEADS * HEAD_DIM
    half_inner = inner // 2
    indices = np.arange(half_inner, dtype=np.float64)
    fraction = indices / float(half_inner - 1)
    grid = (np.power(10000.0, fraction) * (math.pi * 0.5)).astype(np.float32)
    angle = grid[None, :] * (
        positions.astype(np.float32)[:, None] / np.float32(20.0) *
        np.float32(2.0) - np.float32(1.0)
    )
    cosine = round_bf16(np.cos(angle).astype(np.float32)).astype(np.float16)
    sine = round_bf16(np.sin(angle).astype(np.float32)).astype(np.float16)
    shape = (1, HEADS, len(positions), HEAD_DIM // 2)
    return cosine.reshape(shape), sine.reshape(shape)


def projection(path: Path, data_offset: int, header, prefix: str,
               name: str, output_dim: int, input_dim: int):
    import numpy as np

    base = f"{prefix}.{name}"
    weight_info = tensor_info(
        header, f"{base}.weight", "I8", (output_dim, input_dim)
    )
    scale_info = tensor_info(
        header, f"{base}.weight_scale", "F32", (output_dim, 1)
    )
    bias_info = tensor_info(
        header, f"{base}.bias", "BF16", (output_dim,)
    )
    weight = mapped_tensor(
        path, data_offset, weight_info, np.int8, (output_dim, input_dim)
    )
    scale = mapped_tensor(
        path, data_offset, scale_info, np.dtype("<f4"), (output_dim, 1)
    )
    folded = absorb_convrot(weight, scale)
    bias = bf16_tensor(path, data_offset, bias_info, (output_dim,))
    return folded, bias


def load_weights(checkpoint: Path, block: int, video_rows: int):
    data_offset, header = read_header(checkpoint)
    prefix = (
        "model.diffusion_model.transformer_blocks."
        f"{block}.video_to_audio_attn"
    )
    query_weight, query_bias = projection(
        checkpoint, data_offset, header, prefix, "to_q",
        AUDIO_DIM, AUDIO_DIM,
    )
    key_weight, key_bias = projection(
        checkpoint, data_offset, header, prefix, "to_k",
        AUDIO_DIM, VIDEO_DIM,
    )
    value_weight, value_bias = projection(
        checkpoint, data_offset, header, prefix, "to_v",
        AUDIO_DIM, VIDEO_DIM,
    )
    output_weight, output_bias = projection(
        checkpoint, data_offset, header, prefix, "to_out.0",
        AUDIO_DIM, AUDIO_DIM,
    )
    query_norm = bf16_tensor(
        checkpoint, data_offset,
        tensor_info(header, f"{prefix}.q_norm.weight", "BF16", (AUDIO_DIM,)),
        (AUDIO_DIM,),
    )
    key_norm = bf16_tensor(
        checkpoint, data_offset,
        tensor_info(header, f"{prefix}.k_norm.weight", "BF16", (AUDIO_DIM,)),
        (AUDIO_DIM,),
    )
    gate_weight = bf16_tensor(
        checkpoint, data_offset,
        tensor_info(
            header, f"{prefix}.to_gate_logits.weight", "BF16",
            (HEADS, AUDIO_DIM),
        ),
        (HEADS, AUDIO_DIM),
    )
    gate_bias = bf16_tensor(
        checkpoint, data_offset,
        tensor_info(
            header, f"{prefix}.to_gate_logits.bias", "BF16", (HEADS,)
        ),
        (HEADS,),
    )
    query_cosine, query_sine = rope_table(audio_positions())
    key_cosine, key_sine = rope_table(video_temporal_positions(video_rows))
    return {
        "query_weight": query_weight,
        "query_bias": query_bias,
        "key_weight": key_weight,
        "key_bias": key_bias,
        "value_weight": value_weight,
        "value_bias": value_bias,
        "output_weight": output_weight,
        "output_bias": output_bias,
        "query_norm": query_norm,
        "key_norm": key_norm,
        "gate_weight": gate_weight,
        "gate_bias": gate_bias,
        "query_cosine": query_cosine,
        "query_sine": query_sine,
        "key_cosine": key_cosine,
        "key_sine": key_sine,
    }


def build_model(video_rows: int, values):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def conv_weight(name):
        return np.ascontiguousarray(values[name][:, :, None, None])

    def token_major(x):
        squeezed = mb.squeeze(x=x, axes=[2])
        return mb.transpose(x=squeezed, perm=[0, 2, 1])

    def rms_norm(x, weight, name):
        square = mb.mul(x=x, y=x, name=f"{name}_square")
        mean = mb.reduce_mean(
            x=square, axes=[-1], keep_dims=True, name=f"{name}_mean"
        )
        inverse = mb.rsqrt(
            x=mb.add(x=mean, y=np.float16(1.0e-6)),
            name=f"{name}_rsqrt",
        )
        normalized = mb.mul(x=x, y=inverse, name=f"{name}_normalized")
        return mb.mul(
            x=normalized,
            y=np.ascontiguousarray(weight.reshape(1, 1, AUDIO_DIM)),
            name=name,
        )

    def head_major(x, rows, name):
        reshaped = mb.reshape(
            x=x, shape=(1, rows, HEADS, HEAD_DIM),
            name=f"{name}_reshape",
        )
        return mb.transpose(
            x=reshaped, perm=[0, 2, 1, 3], name=f"{name}_heads"
        )

    def apply_rope(x, cosine, sine, name):
        first, second = mb.split(
            x=x, split_sizes=[HEAD_DIM // 2, HEAD_DIM // 2],
            axis=3, name=f"{name}_split",
        )
        rotated_first = mb.sub(
            x=mb.mul(x=first, y=cosine),
            y=mb.mul(x=second, y=sine),
            name=f"{name}_first",
        )
        rotated_second = mb.add(
            x=mb.mul(x=first, y=sine),
            y=mb.mul(x=second, y=cosine),
            name=f"{name}_second",
        )
        return mb.concat(
            values=[rotated_first, rotated_second], axis=3, name=name
        )

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, AUDIO_DIM, 1, AUDIO_ROWS), dtype=types.fp16
            ),
            mb.TensorSpec(
                shape=(1, VIDEO_DIM, 1, video_rows), dtype=types.fp16
            ),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(audio, video):
        query = mb.conv(
            x=audio, weight=conv_weight("query_weight"),
            bias=values["query_bias"], name="query_projection",
        )
        key = mb.conv(
            x=video, weight=conv_weight("key_weight"),
            bias=values["key_bias"], name="key_projection",
        )
        value = mb.conv(
            x=video, weight=conv_weight("value_weight"),
            bias=values["value_bias"], name="value_projection",
        )
        query = rms_norm(token_major(query), values["query_norm"], "q_norm")
        key = rms_norm(token_major(key), values["key_norm"], "k_norm")
        value = token_major(value)
        query = apply_rope(
            head_major(query, AUDIO_ROWS, "query"),
            values["query_cosine"], values["query_sine"], "query_rope",
        )
        key = apply_rope(
            head_major(key, video_rows, "key"),
            values["key_cosine"], values["key_sine"], "key_rope",
        )
        value = head_major(value, video_rows, "value")
        scores = mb.matmul(
            x=query, y=key, transpose_y=True, name="attention_scores"
        )
        scores = mb.mul(
            x=scores, y=np.float16(1.0 / math.sqrt(HEAD_DIM)),
            name="scaled_scores",
        )
        probabilities = mb.softmax(x=scores, axis=-1, name="softmax")
        attended = mb.matmul(
            x=probabilities, y=value, name="attention_value"
        )
        gate = mb.conv(
            x=audio,
            weight=np.ascontiguousarray(
                values["gate_weight"][:, :, None, None]
            ),
            bias=values["gate_bias"], name="gate_logits",
        )
        gate = mb.sigmoid(x=gate, name="gate_sigmoid")
        gate = mb.mul(x=gate, y=np.float16(2.0), name="gate_times_two")
        gate = mb.transpose(x=gate, perm=[0, 1, 3, 2], name="gate_heads")
        attended = mb.mul(x=attended, y=gate, name="gated_attention")
        row_major = mb.transpose(
            x=attended, perm=[0, 2, 1, 3], name="row_major_heads"
        )
        row_major = mb.reshape(
            x=row_major, shape=(1, AUDIO_ROWS, AUDIO_DIM),
            name="row_major",
        )
        channels_first = mb.transpose(
            x=row_major, perm=[0, 2, 1], name="channels_first"
        )
        channels_first = mb.expand_dims(
            x=channels_first, axes=[2], name="channels_first_4d"
        )
        return mb.conv(
            x=channels_first, weight=conv_weight("output_weight"),
            bias=values["output_bias"], name="y",
        )

    model = ct.convert(
        program, convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    specification = model.get_spec()
    specification.description.input[0].name = "audio"
    specification.description.input[1].name = "video"
    model = ct.models.MLModel(specification, weights_dir=model.weights_dir)
    model.user_defined_metadata.update({
        "ltx.schema": SCHEMA,
        "ltx.block": "video_to_audio_attn",
        "ltx.video_rows": str(video_rows),
        "ltx.precision": "fp16",
    })
    return model


def main() -> None:
    args = parse_args()
    if args.block < 0 or args.block >= 48:
        raise SystemExit("block must be in [0, 47]")
    if args.compiled_only and not args.compile:
        raise SystemExit("--compiled-only requires --compile")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} is not empty; pass --force")
    if args.force:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = load_weights(args.checkpoint, args.block, args.video_rows)
    base = build_model(args.video_rows, values)
    artifacts = {}
    compiled_artifacts = {}
    for variant in args.variants:
        model = base if variant == "fp16" else quantize_int8(base)
        package = args.out_dir / f"v2a-{variant}.mlpackage"
        model.save(str(package))
        artifacts[variant] = package.name
        if args.compile:
            compiled = args.out_dir / f"v2a-{variant}.mlmodelc"
            compile_model(package, compiled)
            compiled_artifacts[variant] = compiled.name
            if args.compiled_only:
                shutil.rmtree(package)
                artifacts.pop(variant, None)

    manifest = {
        "schema": SCHEMA,
        "block_index": args.block,
        "shape": {
            "audio_rows": AUDIO_ROWS,
            "audio_dim": AUDIO_DIM,
            "video_rows": args.video_rows,
            "video_dim": VIDEO_DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
        },
        "artifacts": artifacts,
        "compiled_artifacts": compiled_artifacts,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
