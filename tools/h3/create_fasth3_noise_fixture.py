#!/usr/bin/env python3
"""Create a FastVideo-compatible H3 video/audio noise fixture.

The fixture is generated with the same MLX key split and packed row geometry
used by FastVideo.  It is intended for strict C++/MLX parity, not normal user
seed semantics.  No model or network access is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_fastvideo_mlx_converter import install_fastvideo_namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-tokens", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3 import (
        audio_latent_num_frames,
        build_packed_layout,
        load_mlx_h3_checkpoint,
        video_latent_num_frames,
    )

    dit = load_mlx_h3_checkpoint(args.checkpoint)
    layout = build_packed_layout(
        args.text_tokens,
        video_latent_num_frames(args.frames),
        args.height // 16,
        args.width // 16,
        audio_latent_num_frames(args.frames),
        patch_size=dit.patch_size,
        text_token_tags=np.ones(args.text_tokens, dtype=np.int64),
    )
    video_key, audio_key = mx.random.split(mx.random.key(args.seed))
    video = mx.random.normal(
        (int(layout.video_indices.shape[0]), dit.patch_dim),
        dtype=mx.float32,
        key=video_key,
    )
    audio = mx.random.normal(
        (int(layout.audio_indices.shape[0]), dit.audio_in_channels),
        dtype=mx.float32,
        key=audio_key,
    )
    mx.eval(video, audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), {
        "video_noise": video,
        "audio_noise": audio,
    }, metadata={
        "runtime": "FastVideo",
        "model_id": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",
        "seed": str(args.seed),
        "frames": str(args.frames),
        "height": str(args.height),
        "width": str(args.width),
        "text_tokens": str(args.text_tokens),
    })
    print(json.dumps({
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "video_checksum": float(mx.sum(video).item()),
        "audio_checksum": float(mx.sum(audio).item()),
    }, indent=2))


if __name__ == "__main__":
    main()
