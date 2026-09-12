#!/usr/bin/env python3
"""Run FastVideo H3 E2E with an explicit shared noise fixture.

This is a validation-only entrypoint.  It mirrors the checked-in FastVideo
pipeline phases but injects the pre-generated video/audio noise arrays so a
native TurboCider run can be compared without relying on cross-language RNG
implementation details.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_fastvideo_mlx_converter import install_fastvideo_namespace


def dense_layers(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(layer < 0 for layer in result):
        raise argparse.ArgumentTypeError("VSA dense layer indices must be nonnegative")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-tensors", type=Path)
    parser.add_argument("--conditioning-input", type=Path)
    parser.add_argument("--stop-after-denoise", action="store_true")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="validation only: allow aligned clips shorter than the 5-second product minimum",
    )
    parser.add_argument("--vsa", action="store_true")
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--vsa-tile-size", type=int, choices=(64, 256), default=64)
    parser.add_argument("--vsa-prefix-mode", choices=("exempt", "compete"), default="exempt")
    parser.add_argument("--vsa-dense-first-n-steps", type=int, default=0)
    parser.add_argument("--vsa-dense-layers", type=dense_layers, default=())
    parser.add_argument("--vsa-impl", choices=("auto", "reference", "simd"),
                        default="reference")
    parser.add_argument(
        "--vsa-disable-gate",
        action="store_true",
        help="validation-only diagnostic: skip trained to_gate_compress branches",
    )
    args = parser.parse_args()

    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3 import (
        MINIMAX_H3_AUDIO_SHIFT,
        MINIMAX_H3_VIDEO_SHIFT,
        MiniMaxH3SchedulerState,
        audio_latent_num_frames,
        build_packed_layout,
        build_row_timesteps,
        load_mlx_h3_checkpoint,
    )
    from fastvideo.mlx_runtime.minimax_h3_pipeline import (
        MiniMaxH3MLXPipeline,
        _adaln_schedule_union,
        _cleanup_mlx,
        _peak_memory_gib,
        _reset_peak_memory,
    )
    from fastvideo.mlx_runtime.minimax_h3_vsa import MiniMaxH3VSAConfig

    fixture = mx.load(str(args.noise))
    video_noise = mx.astype(fixture["video_noise"], mx.float32)
    audio_noise = mx.astype(fixture["audio_noise"], mx.float32)
    pipeline = MiniMaxH3MLXPipeline(
        model_root=args.model_root,
        mlx_dit_checkpoint=args.checkpoint,
        prompt_cache_dir=None,
        video_decode_backend="h3-vae",
        vae_dtype="fp32",
    )
    timings: dict[str, float] = {}
    peaks: dict[str, float] = {}

    _reset_peak_memory()
    started = time.perf_counter()
    if args.conditioning_input is not None:
        conditioning = mx.load(str(args.conditioning_input))
        text_rows = np.asarray(conditioning["conditioning"].astype(mx.float32))
        token_tags = np.asarray(conditioning["token_tags"], dtype=np.int32).tolist()
    else:
        text_rows, token_tags = pipeline.encode_prompt(args.prompt)
    timings["condition_s"] = time.perf_counter() - started
    peaks["condition_gib"] = _peak_memory_gib()
    _cleanup_mlx()

    dit = load_mlx_h3_checkpoint(args.checkpoint)
    dit.configure_vsa(MiniMaxH3VSAConfig(
        enabled=args.vsa,
        sparsity=args.vsa_sparsity,
        tile_size=args.vsa_tile_size,
        prefix_mode=args.vsa_prefix_mode,
        dense_first_n_steps=args.vsa_dense_first_n_steps,
        dense_layers=args.vsa_dense_layers,
        impl=args.vsa_impl,
    ))
    if args.vsa_disable_gate:
        # This runner is intentionally a validation harness. Keeping the
        # override here (rather than in FastVideo or TurboCider's product
        # profile) lets gate-compression parity be isolated without creating
        # a user-facing model mode that the reference implementation lacks.
        dit._gate_active = [False] * len(dit.blocks)
    geometry = pipeline.resolve_geometry(
        args.height,
        args.width,
        args.frames,
        enforce_duration=not args.allow_short,
    )
    layout = build_packed_layout(
        len(token_tags), geometry["latent_frame_count"], geometry["latent_height"],
        geometry["latent_width"], audio_latent_num_frames(geometry["num_frames"]),
        patch_size=dit.patch_size, text_token_tags=np.asarray(token_tags, dtype=np.int64),
    )
    if args.vsa:
        dit.prepare_vsa_geometry(layout)
    require_shapes = {
        "video_noise": (int(layout.video_indices.shape[0]), dit.patch_dim),
        "audio_noise": (int(layout.audio_indices.shape[0]), dit.audio_in_channels),
    }
    if tuple(video_noise.shape) != require_shapes["video_noise"]:
        raise ValueError(f"video fixture shape {video_noise.shape} != {require_shapes['video_noise']}")
    if tuple(audio_noise.shape) != require_shapes["audio_noise"]:
        raise ValueError(f"audio fixture shape {audio_noise.shape} != {require_shapes['audio_noise']}")
    video_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_VIDEO_SHIFT, args.steps)
    audio_scheduler = MiniMaxH3SchedulerState.create(MINIMAX_H3_AUDIO_SHIFT, args.steps)
    union = _adaln_schedule_union(args.steps)
    cache = getattr(dit, "_adaln_cache", None)
    if cache is None:
        dit.precompute_adaln(union, drop_weights=True)
    elif not np.array_equal(cache.timesteps.astype(np.float32), union):
        raise ValueError("noise fixture runner requires the checkpoint's fixed four-step AdaLN ladder")

    _reset_peak_memory()
    started = time.perf_counter()
    video, audio = video_noise, audio_noise
    text = mx.array(text_rows.astype(np.float32))
    for step in range(args.steps):
        unique, inverse = build_row_timesteps(
            layout, float(video_scheduler.timesteps[step]),
            float(audio_scheduler.timesteps[step]),
        )
        video_velocity, audio_velocity = dit.forward_with_cache(
            video, audio, text,
            layout=layout, step_timesteps=unique,
            row_timestep_inverse=inverse, step_index=step,
        )
        video = video_scheduler.step(video_velocity, step, video)
        audio = audio_scheduler.step(audio_velocity, step, audio)
        mx.eval(video, audio)
    timings["denoise_s"] = time.perf_counter() - started
    peaks["denoise_gib"] = _peak_memory_gib()
    video_rows = np.asarray(video.astype(mx.float32))
    audio_rows = np.asarray(audio.astype(mx.float32))
    if args.dump_tensors is not None:
        args.dump_tensors.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(args.dump_tensors / "h3_mlx_e2e.safetensors"), {
            "conditioning": mx.array(text_rows.astype(np.float32)),
            "token_tags": mx.array(np.asarray(token_tags, dtype=np.int32)),
            "video_rows": video.astype(mx.float32),
            "audio_rows": audio.astype(mx.float32),
        })
    vsa_report = None if dit.last_vsa_stats is None else {
        "enabled": True,
        "configured_sparsity": dit.last_vsa_stats.configured_sparsity,
        "achieved_sparsity": dit.last_vsa_stats.achieved_sparsity,
        "video_keep": dit.last_vsa_stats.video_keep,
        "tile_size": dit.last_vsa_stats.tile_size,
        "prefix_mode": dit.last_vsa_stats.prefix_mode,
        "implementation": dit.last_vsa_stats.impl,
        "attention_calls": dit.last_vsa_stats.attention_calls,
        "sparse_calls": dit.last_vsa_stats.sparse_calls,
        "fallback_reason": dit.last_vsa_stats.dense_fallback_reason,
    }
    del dit, text_rows
    _cleanup_mlx()

    if args.stop_after_denoise:
        timings["generate_s"] = sum(timings.values())
        print(json.dumps({
            "timings_s": timings,
            "peak_memory_gib": peaks,
            "noise_fixture": str(args.noise.resolve()),
            "conditioning_input": (str(args.conditioning_input.resolve())
                                   if args.conditioning_input is not None else None),
            "tensor_dump": (str(args.dump_tensors.resolve())
                            if args.dump_tensors is not None else None),
            "vsa": vsa_report,
            "stopped_after_denoise": True,
        }, indent=2))
        return

    _reset_peak_memory()
    started = time.perf_counter()
    frames = pipeline.decode_video(video_rows, height=args.height, width=args.width,
                                   num_frames=args.frames, tiled=True)
    timings["video_decode_s"] = time.perf_counter() - started
    peaks["video_decode_gib"] = _peak_memory_gib()
    _cleanup_mlx()

    _reset_peak_memory()
    started = time.perf_counter()
    waveform = pipeline.decode_audio(audio_rows, num_frames=args.frames)
    timings["audio_decode_s"] = time.perf_counter() - started
    peaks["audio_decode_gib"] = _peak_memory_gib()
    _cleanup_mlx()

    started = time.perf_counter()
    pipeline.mux(frames, waveform, args.output)
    timings["mux_s"] = time.perf_counter() - started
    timings["generate_s"] = sum(timings.values())
    print(json.dumps({
        "video_path": str(args.output),
        "timings_s": timings,
        "peak_memory_gib": peaks,
        "audio_samples": int(waveform.shape[-1]),
        "noise_fixture": str(args.noise.resolve()),
        "conditioning_input": (str(args.conditioning_input.resolve())
                               if args.conditioning_input is not None else None),
        "tensor_dump": (str(args.dump_tensors.resolve())
                        if args.dump_tensors is not None else None),
        "vsa": vsa_report,
    }, indent=2))


if __name__ == "__main__":
    main()
