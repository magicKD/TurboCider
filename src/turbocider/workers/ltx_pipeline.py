#!/usr/bin/env python3
"""Run prompt conditioning, native LTX generation, and media finalization."""

from __future__ import annotations

import argparse
from array import array
import json
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--engine", type=Path, required=True)
    root.add_argument("--ltx-root", type=Path, required=True)
    root.add_argument("--comfy-python", type=Path, required=True)
    root.add_argument("--mlx-python", type=Path, required=True)
    root.add_argument("--comfy-root", type=Path, required=True)
    root.add_argument("--text-encoder", type=Path, required=True)
    root.add_argument("--text-encoder-dir", type=Path)
    root.add_argument("--transformer", type=Path, required=True)
    root.add_argument("--upsampler", type=Path, required=True)
    root.add_argument("--video-vae", type=Path, required=True)
    root.add_argument("--audio-vae", type=Path)
    root.add_argument("--conditioning", type=Path, required=True)
    root.add_argument("--artifact-dir", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--prompt", required=True)
    root.add_argument("--width", type=int, required=True)
    root.add_argument("--height", type=int, required=True)
    root.add_argument("--frames", type=int, required=True)
    root.add_argument("--fps", type=int, required=True)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--first-frame", type=Path)
    root.add_argument("--image-strength", type=float, default=1.0)
    root.add_argument("--image-crf", type=int, default=33)
    root.add_argument("--video-vae-helper", type=Path)
    root.add_argument("--dynamic-conditioning", action="store_true")
    root.add_argument("--engine-arg", action="append", default=[])
    return root


def run(command, *, cwd: Path, environment=None) -> None:
    print("worker_command=" + json.dumps([str(item) for item in command]), flush=True)
    subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=environment,
        check=True,
    )


def capture(command, *, cwd: Path) -> bytes:
    print("worker_command=" + json.dumps([str(item) for item in command]), flush=True)
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def _bf16_lookup() -> array:
    values = array("H")
    for byte in range(256):
        value = byte / 127.5 - 1.0
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits += 0x7FFF + ((bits >> 16) & 1)
        values.append(bits >> 16)
    return values


def preprocess_image_bf16(
    image: Path,
    output: Path,
    *,
    width: int,
    height: int,
    crf: int,
    cwd: Path,
) -> None:
    """Match LTX I2V training preprocessing without a Python image package."""
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise RuntimeError("LTX I2V dimensions must be positive multiples of 32")
    if not 0 <= crf <= 51:
        raise RuntimeError("LTX I2V image CRF must be between 0 and 51")
    with tempfile.TemporaryDirectory(prefix="turbocider-ltx-i2v-") as temporary:
        roundtrip = Path(temporary) / "first-frame.mp4"
        run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", image,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-frames:v", "1", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", str(crf), roundtrip,
            ],
            cwd=cwd,
        )
        raw = capture(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", roundtrip,
                "-vf",
                "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,"
                "crop=%d:%d"
                % (width, height, width, height),
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ],
            cwd=cwd,
        )
    expected = width * height * 3
    if len(raw) != expected:
        raise RuntimeError(
            "preprocessed LTX image has %d bytes; expected %d" % (len(raw), expected)
        )
    lookup = _bf16_lookup()
    planar = array("H")
    for channel in range(3):
        planar.extend(lookup[raw[index]] for index in range(channel, len(raw), 3))
    if os.sys.byteorder != "little":
        planar.byteswap()
    output.write_bytes(planar.tobytes())


def prepare_i2v_latents(args, actual_width: int, actual_height: int) -> Dict[str, Any]:
    if args.first_frame is None:
        return {}
    if not args.first_frame.is_file():
        raise RuntimeError("LTX first-frame image does not exist: %s" % args.first_frame)
    if not 0.0 <= args.image_strength <= 1.0:
        raise RuntimeError("LTX image strength must be between 0 and 1")
    helper = args.video_vae_helper or args.ltx_root / "build" / "bench_mlx_video_vae"
    if not helper.is_file():
        raise RuntimeError("LTX VAE encoder helper does not exist: %s" % helper)

    stage1_width = actual_width // 2
    stage1_height = actual_height // 2
    stages = [
        ("stage1", stage1_width, stage1_height),
        ("stage2", actual_width, actual_height),
    ]
    report: Dict[str, Any] = {
        "mode": "image_to_video",
        "source": str(args.first_frame),
        "strength": args.image_strength,
        "crf": args.image_crf,
        "stages": {},
    }
    for name, width, height in stages:
        pixels = args.artifact_dir / ("i2v-%s-pixels.bf16" % name)
        latent = args.artifact_dir / ("i2v-%s-latent.bf16" % name)
        started = time.perf_counter()
        preprocess_image_bf16(
            args.first_frame, pixels, width=width, height=height,
            crf=args.image_crf, cwd=args.ltx_root,
        )
        preprocess_seconds = time.perf_counter() - started
        started = time.perf_counter()
        try:
            run(
                [
                    helper, "--encode", args.video_vae, "1",
                    str(height), str(width), pixels, latent,
                ],
                cwd=args.ltx_root,
            )
        finally:
            pixels.unlink(missing_ok=True)
        encode_seconds = time.perf_counter() - started
        expected_latent_bytes = (width // 32) * (height // 32) * 128 * 2
        if not latent.is_file():
            raise RuntimeError("LTX VAE encoder did not create %s" % latent)
        actual_latent_bytes = latent.stat().st_size
        if actual_latent_bytes != expected_latent_bytes:
            raise RuntimeError(
                "LTX %s latent has %d bytes; expected %d"
                % (name, actual_latent_bytes, expected_latent_bytes)
            )
        report["stages"][name] = {
            "width": width,
            "height": height,
            "latent": str(latent),
            "latent_rows": (width // 32) * (height // 32),
            "latent_bytes": actual_latent_bytes,
            "preprocess_seconds": preprocess_seconds,
            "encode_seconds": encode_seconds,
        }
    return report


def progress(phase: str, completed: int, total: int = 1) -> None:
    print("phase=%s %d/%d" % (phase, completed, total), flush=True)


def audio_mux_command(video: Path, audio: Path, output: Path) -> List[str]:
    """Build a mux command that never truncates the native 97-frame video."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]


def probe_media(path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries",
            "stream=index,codec_type,duration,nb_frames,nb_read_frames,sample_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def validate_video_frames(path: Path, expected_frames: int) -> Dict[str, Any]:
    report = probe_media(path)
    streams = report.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise RuntimeError("final LTX media has no video stream: %s" % path)
    raw_count = video.get("nb_read_frames") or video.get("nb_frames")
    try:
        frame_count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "could not determine frame count for final LTX media: %s" % path
        ) from error
    if frame_count != expected_frames:
        raise RuntimeError(
            "final LTX media has %d video frames; expected %d"
            % (frame_count, expected_frames)
        )
    report["validated_video_frames"] = frame_count
    return report


def main() -> None:
    worker_started = time.perf_counter()
    args = parser().parse_args()
    if args.frames != 97 or args.fps != 24:
        raise RuntimeError("current native LTX engine supports exactly 97 frames at 24 fps")
    for path in (
        args.engine,
        args.comfy_python,
        args.mlx_python,
        args.text_encoder,
        args.transformer,
        args.upsampler,
        args.video_vae,
    ):
        if not path.exists():
            raise RuntimeError("required LTX path does not exist: %s" % path)
    if not args.comfy_root.is_dir():
        raise RuntimeError("ComfyUI root does not exist: %s" % args.comfy_root)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    conditioning = args.conditioning
    phase_times = {}
    bridge = args.ltx_root / "tools" / "comfy_bridge.py"
    if args.dynamic_conditioning:
        raw = args.artifact_dir / "conditioning-raw"
        conditioning = args.artifact_dir / "conditioning"
        started = time.perf_counter()
        progress("text_conditioning", 0)
        text_encoder_dir = args.text_encoder_dir or (
            Path(os.environ["LTX_TEXT_ENCODER_DIR"])
            if os.environ.get("LTX_TEXT_ENCODER_DIR")
            else None
        )
        if text_encoder_dir:
            mlx_python = args.mlx_python
            mlx_environment = os.environ.copy()
            bundled_python_path = args.ltx_root / "python"
            if bundled_python_path.is_dir():
                existing = mlx_environment.get("PYTHONPATH", "")
                mlx_environment["PYTHONPATH"] = str(bundled_python_path) + (
                    os.pathsep + existing if existing else ""
                )
            run(
                [
                    mlx_python,
                    args.ltx_root / "tools" / "encode_text_mlx.py",
                    "--model-dir", text_encoder_dir,
                    "--output-dir", raw,
                    "--prompt", args.prompt,
                ],
                cwd=args.ltx_root,
                environment=mlx_environment,
            )
        else:
            run(
                [
                    args.comfy_python,
                    bridge,
                    "--comfy-root", args.comfy_root,
                    "encode",
                    args.text_encoder,
                    args.transformer,
                    raw,
                    "--prompt", args.prompt,
                    "--raw-only",
                ],
                cwd=args.ltx_root,
            )
        phase_times["text_conditioning_seconds"] = time.perf_counter() - started
        progress("text_conditioning", 1)
        started = time.perf_counter()
        progress("connector", 0)
        run(
            [args.ltx_root / "build" / "run_connector", args.transformer, raw, conditioning, "1"],
            cwd=args.ltx_root,
        )
        phase_times["connector_seconds"] = time.perf_counter() - started
        progress("connector", 1)
    elif not conditioning.is_dir():
        raise RuntimeError("conditioning directory does not exist: %s" % conditioning)
    else:
        progress("text_conditioning", 1)
        progress("connector", 1)

    actual_width = (args.width // 64) * 64
    actual_height = (args.height // 64) * 64
    i2v_report: Dict[str, Any] = {}
    if args.first_frame:
        started = time.perf_counter()
        progress("image_conditioning", 0)
        i2v_report = prepare_i2v_latents(args, actual_width, actual_height)
        phase_times["image_conditioning_seconds"] = time.perf_counter() - started
        progress("image_conditioning", 1)
    else:
        progress("image_conditioning", 1)

    environment = os.environ.copy()
    environment["LTX_OUTPUT_WIDTH"] = str(args.width)
    environment["LTX_OUTPUT_HEIGHT"] = str(args.height)
    environment.setdefault("LTX_MEDIA_BACKEND", "mlx")
    if i2v_report:
        environment["LTX_I2V_STAGE1_LATENT"] = i2v_report["stages"]["stage1"]["latent"]
        environment["LTX_I2V_STAGE2_LATENT"] = i2v_report["stages"]["stage2"]["latent"]
        environment["LTX_I2V_STRENGTH"] = str(args.image_strength)
    started = time.perf_counter()
    progress("native_generation", 0)
    run(
        [
            args.engine,
            args.transformer,
            "all",
            "--generate",
            args.upsampler,
            args.video_vae,
            conditioning,
            args.artifact_dir,
            str(args.seed),
        ] + args.engine_arg,
        cwd=args.ltx_root,
        environment=environment,
    )
    phase_times["native_generation_seconds"] = time.perf_counter() - started
    progress("native_generation", 1)

    started = time.perf_counter()
    progress("video_encode", 0)
    run(
        [
            args.comfy_python,
            bridge,
            "encode-native-video",
            args.artifact_dir / "video_pixels.bf16",
            args.output,
            "--frames", str(args.frames),
            "--height", str(actual_height),
            "--width", str(actual_width),
            "--fps", str(args.fps),
        ],
        cwd=args.ltx_root,
    )
    phase_times["video_encode_seconds"] = time.perf_counter() - started
    progress("video_encode", 1)
    media_report = validate_video_frames(args.output, args.frames)

    audio_output = None
    if args.audio_vae and args.audio_vae.is_file():
        started = time.perf_counter()
        progress("audio_decode", 0)
        audio_environment = os.environ.copy()
        bundled_python = args.ltx_root / "python"
        existing = audio_environment.get("PYTHONPATH", "")
        if not bundled_python.is_dir():
            raise RuntimeError(
                "bundled LTX audio runtime does not exist: %s" % bundled_python
            )
        audio_environment["PYTHONPATH"] = str(bundled_python) + (
            os.pathsep + existing if existing else ""
        )
        run(
            [
                args.mlx_python,
                args.ltx_root / "tools" / "finalize_audio.py",
                args.artifact_dir,
                "--weights", args.audio_vae,
                "--mode", "full",
                "--overwrite",
            ],
            cwd=args.ltx_root,
            environment=audio_environment,
        )
        phase_times["audio_finalize_seconds"] = time.perf_counter() - started
        progress("audio_decode", 1)
        wav = args.artifact_dir / "audio_48k.wav"
        if wav.is_file():
            audio_output = wav
            muxed = args.output.with_name(args.output.stem + "-with-audio.mp4")
            started = time.perf_counter()
            progress("mux", 0)
            run(audio_mux_command(args.output, wav, muxed), cwd=args.ltx_root)
            os.replace(str(muxed), str(args.output))
            media_report = validate_video_frames(args.output, args.frames)
            phase_times["mux_seconds"] = time.perf_counter() - started
            progress("mux", 1)
    else:
        progress("audio_decode", 1)
        progress("mux", 1)

    report = {
        "format": "turbocider-ltx-worker-v1",
        "output": str(args.output),
        "artifacts": str(args.artifact_dir),
        "audio": str(audio_output) if audio_output else None,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "frames": args.frames,
        "fps": args.fps,
        "media_probe": media_report,
        "phase_times": phase_times,
        "conditioning": i2v_report or {"mode": "text_to_video"},
        "wall_seconds": time.perf_counter() - worker_started,
    }
    (args.artifact_dir / "turbocider.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("turbocider_result=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
