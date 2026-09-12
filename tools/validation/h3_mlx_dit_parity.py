#!/usr/bin/env python3
"""Compare TurboCider C++ and FastVideo Python on the same real H3 INT6 DiT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))
from run_fastvideo_mlx_converter import install_fastvideo_namespace  # noqa: E402


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    diff = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / denominator) if denominator else float(np.array_equal(left, right))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine,
    }


def fastvideo_rows(checkpoint: Path) -> tuple[
    np.ndarray, np.ndarray, list[dict[str, np.ndarray]], dict[str, np.ndarray]
]:
    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3 import (
        MINIMAX_H3_AUDIO_SHIFT,
        MINIMAX_H3_VIDEO_SHIFT,
        MiniMaxH3SchedulerState,
        _attention,
        _feed_forward,
        _h3_rms_norm,
        _modulate,
        apply_h3_rotary,
        audio_latent_num_frames,
        build_packed_layout,
        build_row_timesteps,
        load_mlx_h3_checkpoint,
        linear,
        rope_cos_sin,
        video_latent_num_frames,
    )
    from fastvideo.mlx_runtime.fastwan import weight_dtype

    dit = load_mlx_h3_checkpoint(checkpoint)
    layout = build_packed_layout(
        4, video_latent_num_frames(22), 2, 2, audio_latent_num_frames(22),
        patch_size=dit.patch_size,
        text_token_tags=np.ones(4, dtype=np.int64),
    )
    video = mx.zeros((7, 96), dtype=mx.float32)
    audio = mx.zeros((74, 32), dtype=mx.float32)
    text = mx.zeros((4, 5120), dtype=mx.float32)
    video_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_VIDEO_SHIFT, 4)
    audio_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_AUDIO_SHIFT, 4)
    first_unique, first_inverse = build_row_timesteps(
        layout, float(video_scheduler.timesteps[0]), float(audio_scheduler.timesteps[0])
    )
    video_embed = linear(
        video.astype(weight_dtype(dit.weights["proj_in.weight"])),
        dit.weights["proj_in.weight"], dit.weights["proj_in.bias"],
    )
    audio_embed = linear(
        audio.astype(weight_dtype(dit.weights["audio_proj_in.weight"])),
        dit.weights["audio_proj_in.weight"], dit.weights["audio_proj_in.bias"],
    )
    text_embed = dit.refine_text(text)
    packed_embed = mx.concatenate([
        text_embed, audio_embed.astype(text_embed.dtype), video_embed.astype(text_embed.dtype)
    ], axis=0)
    cos, sin = rope_cos_sin(mx.array(layout.position_ids), dit.rope_freq_dim, dit.rope_theta)
    cache = dit._adaln_cache
    assert cache is not None
    positions = mx.array(cache.positions(first_unique))
    adaln_indices = (
        positions[mx.array(first_inverse)] * 3 + mx.array(layout.token_tags)
    ).astype(mx.int32)
    block_weights = dit.blocks[0]
    gather = lambda table: cache.block_tables[0][table][adaln_indices]
    norm1 = _h3_rms_norm(packed_embed, block_weights["norm1.weight"], dit.norm_eps)
    modulated1 = _modulate(norm1, 1.0 + gather(1), gather(0))
    q = linear(
        modulated1.astype(weight_dtype(block_weights["attn.to_q.weight"])),
        block_weights["attn.to_q.weight"],
    ).reshape(layout.sequence_length, dit.num_heads, dit.head_dim)
    k = linear(
        modulated1.astype(weight_dtype(block_weights["attn.to_k.weight"])),
        block_weights["attn.to_k.weight"],
    ).reshape(layout.sequence_length, dit.num_heads, dit.head_dim)
    v = linear(
        modulated1.astype(weight_dtype(block_weights["attn.to_v.weight"])),
        block_weights["attn.to_v.weight"],
    ).reshape(layout.sequence_length, dit.num_heads, dit.head_dim)
    q = apply_h3_rotary(
        _h3_rms_norm(q, block_weights["attn.norm_q.weight"], dit.qk_norm_eps), cos, sin
    )
    k = apply_h3_rotary(
        _h3_rms_norm(k, block_weights["attn.norm_k.weight"], dit.qk_norm_eps), cos, sin
    )
    attention = _attention(
        block_weights, modulated1.astype(weight_dtype(block_weights["attn.to_q.weight"])), cos, sin,
        num_heads=dit.num_heads, head_dim=dit.head_dim, eps=dit.qk_norm_eps, use_rope=True,
    )
    after_attention = packed_embed + gather(2) * attention
    norm2 = _h3_rms_norm(after_attention, block_weights["norm2.weight"], dit.norm_eps)
    modulated2 = _modulate(norm2, 1.0 + gather(4), gather(3))
    feed_forward = _feed_forward(
        block_weights, modulated2.astype(weight_dtype(block_weights["ff.net.0.proj.weight"]))
    )
    first_block = after_attention + gather(5) * feed_forward
    mx.eval(video_embed, audio_embed, text_embed, packed_embed, norm1, modulated1, q, k, v,
            attention, after_attention, norm2, modulated2, feed_forward, first_block)
    def fp32(value):
        return np.asarray(value.astype(mx.float32))
    debug = {
        "video_embed": fp32(video_embed),
        "audio_embed": fp32(audio_embed),
        "text_embed": fp32(text_embed),
        "packed_embed": fp32(packed_embed),
        "first_norm1": fp32(norm1),
        "first_modulated1": fp32(modulated1),
        "first_query": fp32(q),
        "first_key": fp32(k),
        "first_value": fp32(v),
        "first_attention": fp32(attention),
        "first_after_attention": fp32(after_attention),
        "first_norm2": fp32(norm2),
        "first_modulated2": fp32(modulated2),
        "first_feed_forward": fp32(feed_forward),
        "first_block": fp32(first_block),
    }
    steps: list[dict[str, np.ndarray]] = []
    for step in range(4):
        unique, inverse = build_row_timesteps(
            layout,
            float(video_scheduler.timesteps[step]),
            float(audio_scheduler.timesteps[step]),
        )
        video_velocity, audio_velocity = dit.forward_with_cache(
            video, audio, text,
            layout=layout,
            step_timesteps=unique,
            row_timestep_inverse=inverse,
            step_index=step,
        )
        video_velocity_array = fp32(video_velocity)
        audio_velocity_array = fp32(audio_velocity)
        video = video_scheduler.step(video_velocity, step, video)
        audio = audio_scheduler.step(audio_velocity, step, audio)
        mx.eval(video, audio)
        steps.append({
            "video_velocity": video_velocity_array,
            "audio_velocity": audio_velocity_array,
            "video_sample": fp32(video),
            "audio_sample": fp32(audio),
        })
    return fp32(video), fp32(audio), steps, debug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--max-abs", type=float, default=0.06)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-dit-parity-") as directory:
        native_path = Path(directory) / "native.safetensors"
        subprocess.run([str(args.probe.resolve()), str(checkpoint), str(native_path)], check=True)
        import mlx.core as mx
        native = mx.load(str(native_path))
        native_video = np.asarray(native["video_rows"], dtype=np.float32)
        native_audio = np.asarray(native["audio_rows"], dtype=np.float32)
        native_first_video = np.asarray(native["first_video_velocity"].astype(mx.float32))
        native_first_audio = np.asarray(native["first_audio_velocity"].astype(mx.float32))
        reference_video, reference_audio, reference_steps, reference_debug = fastvideo_rows(checkpoint)
        report = {
            "video": metrics(reference_video, native_video),
            "audio": metrics(reference_audio, native_audio),
        }
        for step, reference in enumerate(reference_steps):
            for field, value in reference.items():
                report[f"step{step}_{field}"] = metrics(
                    value,
                    np.asarray(native[f"step{step}_{field}"].astype(mx.float32)),
                )
        report["first_video_velocity"] = metrics(
            reference_steps[0]["video_velocity"], native_first_video)
        report["first_audio_velocity"] = metrics(
            reference_steps[0]["audio_velocity"], native_first_audio)
        for name, reference in reference_debug.items():
            candidate = np.asarray(native[f"debug_{name}"].astype(mx.float32))
            report[f"debug_{name}"] = metrics(reference, candidate)
            if name == "first_block":
                report["debug_first_block_text"] = metrics(reference[:4], candidate[:4])
                report["debug_first_block_audio"] = metrics(reference[4:78], candidate[4:78])
                report["debug_first_block_video"] = metrics(reference[78:], candidate[78:])
        native_positions = np.asarray(native["position_ids"].astype(mx.float32))
        native_tags = np.asarray(native["token_tags"])
        install_fastvideo_namespace()
        from fastvideo.mlx_runtime.minimax_h3 import (
            audio_latent_num_frames, build_packed_layout, video_latent_num_frames,
        )
        layout = build_packed_layout(
            4, video_latent_num_frames(22), 2, 2, audio_latent_num_frames(22),
            text_token_tags=np.ones(4, dtype=np.int64),
        )
        report["position_ids"] = metrics(layout.position_ids.astype(np.float32), native_positions)
        report["token_tags"] = metrics(layout.token_tags.astype(np.float32),
                                         native_tags.astype(np.float32))
        print(json.dumps(report, indent=2, sort_keys=True))
        for name, result in report.items():
            if result["max_abs"] > args.max_abs or result["cosine"] < args.min_cosine:
                raise SystemExit(
                    f"FAIL: {name} parity max_abs={result['max_abs']:.6g}, "
                    f"cosine={result['cosine']:.9f}"
                )
        print("PASS: TurboCider C++/MLX four-step DiT matches FastVideo")


if __name__ == "__main__":
    main()
