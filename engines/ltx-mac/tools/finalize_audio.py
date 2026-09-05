#!/usr/bin/env python3
"""Decode native LTX-2.5 audio latents and optionally mux them into MP4s.

The native denoiser writes patchified BF16 audio tokens with shape ``[B, L,
128]``.  This finalizer reuses the validated MLX Audio VAE and vocoder from
``references/ltx-2-mlx``:

    audio_latent.bf16 -> Audio VAE -> stereo mel -> vocoder -> WAV

The official combined ComfyUI checkpoint stores the base vocoder below an
extra ``vocoder.vocoder.`` prefix.  The loader removes that extra component
while retaining ``vocoder.bwe_generator.`` and ``vocoder.mel_stft.``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np


DEFAULT_AUDIO_WEIGHTS = (
    Path(os.environ["LTX_AUDIO_VAE"])
    if os.environ.get("LTX_AUDIO_VAE")
    else None
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("artifacts", type=Path, nargs="+")
    root.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_AUDIO_WEIGHTS,
        help=(
            "combined LTX-2.5 Audio VAE + vocoder safetensors; defaults to "
            "$LTX_AUDIO_VAE"
        ),
    )
    root.add_argument(
        "--mode",
        choices=("full", "base"),
        default="full",
        help="full is 48 kHz BWE; base is the faster 16 kHz vocoder",
    )
    root.add_argument(
        "--video-root",
        type=Path,
        help="mux VIDEO_ROOT/<artifact-name>.mp4 after decoding",
    )
    root.add_argument(
        "--output-root",
        type=Path,
        help="directory for muxed MP4s (defaults to --video-root)",
    )
    root.add_argument("--suffix", default="_with_audio")
    root.add_argument(
        "--sync-to-video",
        action="store_true",
        help="tempo-stretch audio to the exact source MP4 duration",
    )
    root.add_argument("--audio-bitrate", default="192k")
    root.add_argument(
        "--target-lufs",
        type=float,
        help="normalize the muxed track to this integrated loudness",
    )
    root.add_argument("--overwrite", action="store_true")
    return root


def read_generation(artifacts: Path) -> dict:
    path = artifacts / "generation.json"
    if not path.is_file():
        raise RuntimeError(f"generation metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    audio = metadata.get("audio")
    if not isinstance(audio, dict) or audio.get("layout") != "BLC-token-major":
        raise RuntimeError(f"unsupported audio metadata in {path}")
    shape = audio.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[2] != 128:
        raise RuntimeError(f"unexpected audio shape in {path}: {shape}")
    return metadata


def bf16_file_to_float32(path: Path, shape: list[int]) -> np.ndarray:
    elements = math.prod(shape)
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != elements:
        raise RuntimeError(
            f"{path} contains {bits.size} BF16 values, expected {elements}"
        )
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    return values.reshape(shape)


def split_combined_weights(raw: dict) -> tuple[dict, dict]:
    audio_weights = {}
    vocoder_weights = {}
    for key, value in raw.items():
        if key.startswith("audio_vae.decoder."):
            audio_weights[key[len("audio_vae.decoder.") :]] = value
        elif key.startswith("audio_vae.per_channel_statistics."):
            audio_weights[key[len("audio_vae.") :]] = value
        elif key.startswith("vocoder.vocoder."):
            vocoder_weights[key[len("vocoder.vocoder.") :]] = value
        elif key.startswith("vocoder."):
            vocoder_weights[key[len("vocoder.") :]] = value
    return audio_weights, vocoder_weights


def require_exact_keys(name: str, model, weights: dict, tree_flatten) -> None:
    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(weights)
    missing = sorted(model_keys - weight_keys)
    unexpected = sorted(weight_keys - model_keys)
    if missing or unexpected:
        preview = {
            "missing": missing[:20],
            "unexpected": unexpected[:20],
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
        }
        raise RuntimeError(f"{name} weight mismatch: {json.dumps(preview)}")


def convert_pytorch_convs(name: str, model, weights: dict, tree_flatten) -> dict:
    """Convert PyTorch OI(K...) convolution weights to MLX layouts."""
    parameters = {key: value for key, value in tree_flatten(model.parameters())}
    converted = {}
    failures = []
    for key, value in weights.items():
        target = parameters[key]
        if tuple(value.shape) == tuple(target.shape):
            converted[key] = value
            continue
        candidate = None
        if value.ndim == 4:
            candidate = value.transpose(0, 2, 3, 1)
        elif value.ndim == 3:
            if re.search(r"(?:^|\.)ups\.\d+\.weight$", key):
                candidate = value.transpose(1, 2, 0)
            else:
                candidate = value.transpose(0, 2, 1)
        if candidate is not None and tuple(candidate.shape) == tuple(target.shape):
            converted[key] = candidate
        else:
            failures.append(
                {
                    "key": key,
                    "checkpoint": list(value.shape),
                    "model": list(target.shape),
                }
            )
    if failures:
        raise RuntimeError(
            f"{name} has {len(failures)} unconvertible tensor shapes: "
            + json.dumps(failures[:20])
        )
    return converted


def load_models(weights_path: Path, mode: str):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
    from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
    from ltx_core_mlx.utils.weights import remap_audio_vae_keys

    if not weights_path.is_file():
        raise RuntimeError(f"audio weights do not exist: {weights_path}")

    started = time.perf_counter()
    raw = mx.load(str(weights_path))
    audio_weights, vocoder_weights = split_combined_weights(raw)
    audio_weights = remap_audio_vae_keys(audio_weights)
    audio_weights = {
        key.replace(".mean-of-means", ".mean_of_means").replace(
            ".std-of-means", ".std_of_means"
        ): value
        for key, value in audio_weights.items()
    }

    decoder = AudioVAEDecoder()
    require_exact_keys("Audio VAE", decoder, audio_weights, tree_flatten)
    audio_weights = convert_pytorch_convs(
        "Audio VAE", decoder, audio_weights, tree_flatten
    )
    decoder.update(tree_unflatten(list(audio_weights.items())))

    vocoder = VocoderWithBWE()
    if mode == "base":
        vocoder_weights = {
            key: value
            for key, value in vocoder_weights.items()
            if not key.startswith(("bwe_generator.", "mel_stft."))
        }
        model_keys = {
            key
            for key, _ in tree_flatten(vocoder.parameters())
            if not key.startswith(("bwe_generator.", "mel_stft."))
        }
        if model_keys != set(vocoder_weights):
            raise RuntimeError(
                "base vocoder weight mismatch: "
                f"model={len(model_keys)}, checkpoint={len(vocoder_weights)}"
            )
    else:
        require_exact_keys("vocoder", vocoder, vocoder_weights, tree_flatten)

    vocoder_weights = convert_pytorch_convs(
        "vocoder", vocoder, vocoder_weights, tree_flatten
    )
    vocoder.load_weights(list(vocoder_weights.items()))
    vocoder.upcast_weights_to_fp32()
    mx.eval(decoder.parameters(), vocoder.parameters())
    mx.synchronize()

    del raw, audio_weights, vocoder_weights
    gc.collect()
    return decoder, vocoder, time.perf_counter() - started


def write_pcm16_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> dict:
    if waveform.ndim != 2 or waveform.shape[1] != 2:
        raise RuntimeError(f"waveform must be [samples, 2], got {waveform.shape}")
    non_finite = int((~np.isfinite(waveform)).sum())
    if non_finite:
        raise RuntimeError(f"waveform contains {non_finite} non-finite samples")
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    clipped_samples = int((np.abs(waveform) >= 1.0).sum())
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return {
        "sample_rate": sample_rate,
        "channels": 2,
        "samples": int(waveform.shape[0]),
        "duration_seconds": waveform.shape[0] / sample_rate,
        "peak": peak,
        "rms": rms,
        "clipped_samples": clipped_samples,
        "non_finite": non_finite,
    }


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    value = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()
    return float(value)


def mux_audio(
    video: Path,
    wav: Path,
    output: Path,
    audio_duration: float,
    sync_to_video: bool,
    audio_bitrate: str,
    target_lufs: float | None,
    overwrite: bool,
) -> dict:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for MP4 muxing")
    if not video.is_file():
        raise RuntimeError(f"source video does not exist: {video}")
    if output.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing output: {output}")

    video_duration = probe_duration(video)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-i", str(video), "-i", str(wav)])
    command.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"])
    command.extend(["-c:a", "aac", "-b:a", audio_bitrate])

    tempo = 1.0
    audio_filters = []
    if sync_to_video:
        tempo = audio_duration / video_duration
        if not 0.5 <= tempo <= 2.0:
            raise RuntimeError(
                f"required atempo {tempo:.6f} is outside ffmpeg's safe range"
            )
        audio_filters.append(f"atempo={tempo:.9f}")
    if target_lufs is not None:
        audio_filters.append(
            f"loudnorm=I={target_lufs:g}:TP=-1:LRA=11"
        )
        audio_filters.append("aresample=48000")
    if sync_to_video:
        audio_filters.append("apad")
    if audio_filters:
        command.extend(["-af", ",".join(audio_filters)])
    if sync_to_video:
        command.extend(["-t", f"{video_duration:.9f}"])
    else:
        command.append("-shortest")

    output.parent.mkdir(parents=True, exist_ok=True)
    command.append(str(output))
    started = time.perf_counter()
    subprocess.run(command, check=True)
    mux_seconds = time.perf_counter() - started
    return {
        "video": str(video),
        "output": str(output),
        "video_duration_seconds": video_duration,
        "audio_duration_before_sync_seconds": audio_duration,
        "atempo": tempo,
        "target_lufs": target_lufs,
        "mux_seconds": mux_seconds,
    }


def decode_one(artifacts: Path, decoder, vocoder, mode: str) -> tuple[Path, dict]:
    import mlx.core as mx

    metadata = read_generation(artifacts)
    shape = [int(value) for value in metadata["audio"]["shape"]]
    latent_path = artifacts / str(metadata["audio"]["file"])
    latent_np = bf16_file_to_float32(latent_path, shape)
    latent = mx.array(latent_np, dtype=mx.bfloat16)
    token_count = shape[1]
    latent = latent.reshape(1, token_count, 8, 16).transpose(0, 2, 1, 3)

    vae_started = time.perf_counter()
    mel = decoder.decode(latent)
    mx.eval(mel)
    mx.synchronize()
    vae_seconds = time.perf_counter() - vae_started

    vocoder_started = time.perf_counter()
    if mode == "full":
        waveform = vocoder(mel.astype(mx.float32))
        mx.eval(waveform)
        mx.synchronize()
        waveform_np = np.asarray(waveform[0].transpose(1, 0), dtype=np.float32)
        sample_rate = 48000
    else:
        batch, channels, frames, bins = mel.shape
        mel_concat = mel.transpose(0, 2, 1, 3).reshape(
            batch, frames, channels * bins
        )
        waveform = vocoder._run_base_vocoder(mel_concat.astype(mx.float32))
        mx.eval(waveform)
        mx.synchronize()
        waveform_np = np.asarray(waveform[0], dtype=np.float32)
        sample_rate = 16000
    vocoder_seconds = time.perf_counter() - vocoder_started

    wav_path = artifacts / f"audio_{sample_rate // 1000}k.wav"
    audio_stats = write_pcm16_wav(wav_path, waveform_np, sample_rate)
    result = {
        "format": "ltx-mac-audio-finalize-v1",
        "artifacts": str(artifacts),
        "latent": str(latent_path),
        "latent_shape": shape,
        "mode": mode,
        "mel_shape": [int(value) for value in mel.shape],
        "wav": str(wav_path),
        "audio": audio_stats,
        "audio_vae_seconds": vae_seconds,
        "vocoder_seconds": vocoder_seconds,
        "decode_seconds": vae_seconds + vocoder_seconds,
    }
    (artifacts / "audio_finalize.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    del latent, latent_np, mel, waveform, waveform_np
    gc.collect()
    mx.clear_cache()
    return wav_path, result


def main() -> None:
    args = parser().parse_args()
    if args.weights is None:
        raise RuntimeError("pass --weights or set LTX_AUDIO_VAE")
    if args.output_root is not None and args.video_root is None:
        raise RuntimeError("--output-root requires --video-root")
    for artifacts in args.artifacts:
        if not artifacts.is_dir():
            raise RuntimeError(f"artifact directory does not exist: {artifacts}")

    decoder, vocoder, load_seconds = load_models(args.weights, args.mode)
    summaries = []
    for artifacts in args.artifacts:
        started = time.perf_counter()
        wav_path, result = decode_one(artifacts, decoder, vocoder, args.mode)
        result["model_load_seconds"] = load_seconds
        if args.video_root is not None:
            video = args.video_root / f"{artifacts.name}.mp4"
            output_root = args.output_root or args.video_root
            output = output_root / f"{artifacts.name}{args.suffix}.mp4"
            result["mux"] = mux_audio(
                video,
                wav_path,
                output,
                result["audio"]["duration_seconds"],
                args.sync_to_video,
                args.audio_bitrate,
                args.target_lufs,
                args.overwrite,
            )
        result["wall_seconds"] = time.perf_counter() - started
        (artifacts / "audio_finalize.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summaries.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    del decoder, vocoder
    gc.collect()
    print(
        "audio_finalize_summary="
        + json.dumps(
            {
                "weights": str(args.weights),
                "mode": args.mode,
                "model_load_seconds": load_seconds,
                "jobs": len(summaries),
                "total_decode_seconds": sum(
                    item["decode_seconds"] for item in summaries
                ),
                "total_wall_seconds": sum(item["wall_seconds"] for item in summaries),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
