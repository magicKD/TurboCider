#!/usr/bin/env python3
"""Encode LTX-2.5 text conditioning with the bundled MLX Gemma-4 encoder.

Replaces the ComfyUI-based ``comfy_bridge.py encode --raw-only`` step. The
output files are consumed by the native ``run_connector`` executable:

    raw_video_context.bf16
    raw_audio_context.bf16
    raw_text_mask.bf16
    raw_conditioning.json

The model directory must be the LTX-2.5 MLX pack containing
``text_encoder.safetensors``, ``connector.safetensors``, tokenizer files, and
the transformer config used by ``LTXModelConfig``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import (
    select_text_encoder,
)
from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import (
    Gemma4TextEncoder,
)
from ltx_core_mlx.text_encoders.gemma.feature_extractor import (
    GemmaFeaturesExtractorV2,
)
from ltx_core_mlx.utils.weights import load_split_safetensors


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--model-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--prompt", required=True)
    root.add_argument("--max-length", type=int, default=1024)
    return root


def write_bf16(path: Path, value: mx.array) -> None:
    if value.ndim != 2:
        raise RuntimeError("expected [rows, channels] tensor, got %s" % (value.shape,))
    path.write_bytes(value.astype(mx.bfloat16).tobytes())


def main() -> None:
    args = parser().parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit("MLX text-encoder model directory does not exist: %s" % model_dir)
    if select_text_encoder(model_dir) != "gemma4":
        raise SystemExit("only the LTX-2.5 gemma4 MLX text pack is supported")

    started = time.perf_counter()
    text_encoder = Gemma4TextEncoder()
    text_encoder.load(model_dir)

    config = LTXModelConfig.from_checkpoint_dir(model_dir)
    extractor = GemmaFeaturesExtractorV2(double_precision_rope=config.double_precision_rope)
    connector_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
    projection_weights = load_split_safetensors(
        model_dir / "text_encoder.safetensors",
        prefix="text_encoder.text_embedding_projection.",
    )
    connector_weights.update(
        {"text_embedding_projection." + key: value for key, value in projection_weights.items()}
    )
    extractor.connector.load_weights(list(connector_weights.items()))

    max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", str(args.max_length)))
    hidden_states, attention_mask = text_encoder.encode_all_layers(
        args.prompt, max_length=max_length
    )
    video_embeds, audio_embeds = extractor(
        hidden_states, attention_mask=attention_mask
    )
    mx.eval(video_embeds, audio_embeds)
    encode_seconds = time.perf_counter() - started

    if video_embeds.ndim != 3 or audio_embeds.ndim != 3:
        raise RuntimeError(
            "unexpected encoder output shapes: %s %s"
            % (video_embeds.shape, audio_embeds.shape)
        )
    video = video_embeds[0]
    audio = audio_embeds[0]
    rows = int(video.shape[0])
    if int(audio.shape[0]) != rows:
        raise RuntimeError("video/audio row counts do not match")
    if video.shape[1] != 4096 or audio.shape[1] != 2048:
        raise RuntimeError(
            "unexpected feature dimensions: video=%d audio=%d"
            % (video.shape[1], audio.shape[1])
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(output_dir / "raw_video_context.bf16", video)
    write_bf16(output_dir / "raw_audio_context.bf16", audio)
    mask = mx.zeros((rows,), dtype=mx.bfloat16)
    write_bf16(output_dir / "raw_text_mask.bf16", mask)

    manifest = {
        "format": "ltx-mac-raw-conditioning-v1",
        "prompt": args.prompt,
        "model_dir": str(model_dir),
        "shape": [1, rows, 4096 + 2048],
        "video_shape": [rows, 4096],
        "audio_shape": [rows, 2048],
        "all_input_rows_valid": True,
        "connector_minimum_rows": 1024,
        "connector_register_period": 128,
        "expected_processed_rows": (
            (max(1024, rows) + 127) // 128
        )
        * 128,
        "text_encode_seconds": encode_seconds,
        "requires_embeddings_connector": True,
        "backend": "turbocider-mlx",
    }
    (output_dir / "raw_conditioning.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("raw_conditioning=" + json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
