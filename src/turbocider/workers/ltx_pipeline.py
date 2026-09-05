#!/usr/bin/env python3
"""Run prompt conditioning, native LTX generation, and media finalization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

    environment = os.environ.copy()
    environment["LTX_OUTPUT_WIDTH"] = str(args.width)
    environment["LTX_OUTPUT_HEIGHT"] = str(args.height)
    environment.setdefault("LTX_MEDIA_BACKEND", "mlx")
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

    actual_width = (args.width // 64) * 64
    actual_height = (args.height // 64) * 64
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
        reference_source = args.ltx_root.parent / "references" / "ltx-2-mlx" / "packages" / "ltx-core-mlx" / "src"
        existing = audio_environment.get("PYTHONPATH", "")
        if reference_source.is_dir():
            audio_environment["PYTHONPATH"] = str(reference_source) + (os.pathsep + existing if existing else "")
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
        "wall_seconds": time.perf_counter() - worker_started,
    }
    (args.artifact_dir / "turbocider.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("turbocider_result=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
