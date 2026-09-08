#!/usr/bin/env python3
"""Offline frame-level quality gate for paired video outputs.

The runtime deliberately does not render a reference video at request time.
This tool is for validation reports only.  It decodes both videos to RGB24
with ffmpeg, compares aligned frames, and also checks whether frame-to-frame
motion energy changed materially.  It intentionally has no NumPy/Pillow
dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _command(program: str, override: Optional[str]) -> str:
    value = override or program
    resolved = shutil.which(value)
    if resolved is None:
        raise RuntimeError(f"required video tool is unavailable: {value}")
    return resolved


def _probe(path: Path, ffprobe: str) -> dict:
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    streams = value.get("streams", [])
    if not streams:
        raise ValueError(f"video has no video stream: {path}")
    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"video has invalid dimensions: {path}")
    frame_rate = stream.get("avg_frame_rate", "0/0")
    try:
        numerator, denominator = (int(part) for part in frame_rate.split("/", 1))
        fps = numerator / denominator if denominator else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    declared_frames = stream.get("nb_frames")
    duration = stream.get("duration", 0.0)
    try:
        duration_value = float(duration) if duration not in (None, "N/A") else 0.0
    except (TypeError, ValueError):
        duration_value = 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frames": int(declared_frames) if declared_frames not in (None, "N/A") else None,
        "duration": duration_value,
    }


def _decode_process(path: Path, ffmpeg: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_frame(stream, frame_bytes: int, path: Path) -> Optional[bytes]:
    value = bytearray()
    while len(value) < frame_bytes:
        chunk = stream.read(frame_bytes - len(value))
        if not chunk:
            break
        value.extend(chunk)
    if not value:
        return None
    if len(value) != frame_bytes:
        raise ValueError(f"decoded RGB payload has an incomplete frame: {path}")
    return bytes(value)


def _rgb_metrics(reference: bytes, candidate: bytes) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired frames have different or empty RGB payloads")
    count = len(reference)
    absolute = squared = dot = first_squared = second_squared = 0.0
    first_sum = second_sum = 0.0
    for left, right in zip(reference, candidate):
        difference = float(left) - float(right)
        absolute += abs(difference)
        squared += difference * difference
        dot += float(left) * float(right)
        first_squared += float(left) * float(left)
        second_squared += float(right) * float(right)
        first_sum += left
        second_sum += right
    # The centered sums can be derived from the first-pass sufficient
    # statistics. Avoiding a second Python pass matters for 97/121-frame
    # validation videos while preserving the same Pearson definition.
    covariance = dot - first_sum * second_sum / count
    first_variance = max(0.0, first_squared - first_sum * first_sum / count)
    second_variance = max(0.0, second_squared - second_sum * second_sum / count)
    return {
        "correlation": covariance / math.sqrt(first_variance * second_variance)
        if first_variance and second_variance else 1.0,
        "cosine": dot / math.sqrt(first_squared * second_squared)
        if first_squared and second_squared else 1.0,
        "mae_255": absolute / count,
        "rmse_255": math.sqrt(squared / count),
    }


def _motion_energy(previous: bytes, current: bytes) -> float:
    if len(previous) != len(current) or not previous:
        raise ValueError("cannot calculate motion energy for invalid frames")
    return sum(abs(left - right) for left, right in zip(previous, current)) / len(previous)


def _vision_metrics(samples: list[tuple[int, bytes, bytes]], width: int, height: int,
                    helper: str) -> dict:
    helper_path = _command("vision-feature-distance", helper)
    with tempfile.TemporaryDirectory(prefix="turbocider-vision-quality-") as raw:
        directory = Path(raw)
        arguments = [helper_path]
        for frame, reference, candidate in samples:
            reference_path = directory / f"reference-{frame:06d}.ppm"
            candidate_path = directory / f"candidate-{frame:06d}.ppm"
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            reference_path.write_bytes(header + reference)
            candidate_path.write_bytes(header + candidate)
            arguments.extend((str(reference_path), str(candidate_path)))
        completed = subprocess.run(
            arguments, check=True, capture_output=True, text=True
        )
    result = json.loads(completed.stdout)
    if int(result.get("pair_count", 0)) != len(samples):
        raise ValueError("Vision helper returned an unexpected sample count")
    mean_distance = float(result.get("mean_distance", float("nan")))
    maximum_distance = float(result.get("maximum_distance", float("nan")))
    if not math.isfinite(mean_distance) or not math.isfinite(maximum_distance):
        raise ValueError("Vision helper returned a non-finite distance")
    return {
        "metric": result.get("metric", "VNFeaturePrintObservation distance"),
        "vision_request_revision": result.get("vision_request_revision"),
        "image_crop_and_scale": result.get("image_crop_and_scale"),
        "operating_system": result.get("operating_system", "unknown"),
        "sampled_frames": [frame for frame, _, _ in samples],
        "pair_count": len(samples),
        "mean_distance": mean_distance,
        "maximum_distance": maximum_distance,
    }


def compare(reference: Path, candidate: Path, *, ffmpeg: str = "ffmpeg",
            ffprobe: str = "ffprobe", vision_helper: Optional[str] = None,
            vision_sample_stride: int = 24, vision_max_samples: int = 16) -> dict:
    if vision_sample_stride <= 0:
        raise ValueError("vision_sample_stride must be positive")
    if vision_max_samples <= 0:
        raise ValueError("vision_max_samples must be positive")
    ffmpeg_path = _command("ffmpeg", ffmpeg)
    ffprobe_path = _command("ffprobe", ffprobe)
    reference_meta = _probe(reference, ffprobe_path)
    candidate_meta = _probe(candidate, ffprobe_path)
    shape_equal = (
        reference_meta["width"] == candidate_meta["width"] and
        reference_meta["height"] == candidate_meta["height"]
    )
    fps_equal = (
        not reference_meta["fps"] or not candidate_meta["fps"] or
        abs(reference_meta["fps"] - candidate_meta["fps"]) <= 1e-6
    )
    if not shape_equal:
        return {
            "shape_equal": False, "frame_count_equal": False,
            "fps_equal": fps_equal, "finite": False,
            "reference": {**reference_meta, "decoded_frames": 0},
            "candidate": {**candidate_meta, "decoded_frames": 0},
            "frame_count_compared": 0, "frame_metrics": [],
            "mean_correlation": 0.0, "minimum_correlation": 0.0,
            "mean_cosine": 0.0, "minimum_cosine": 0.0,
            "mean_mae_255": float("inf"), "maximum_mae_255": float("inf"),
            "mean_motion_energy_reference": 0.0,
            "mean_motion_energy_candidate": 0.0,
            "maximum_motion_relative_error": float("inf"),
            "vision_feature_print": None,
        }
    frame_bytes = reference_meta["width"] * reference_meta["height"] * 3
    reference_process = _decode_process(reference, ffmpeg_path)
    candidate_process = _decode_process(candidate, ffmpeg_path)
    frame_metrics: list[dict[str, float]] = []
    reference_motion: list[float] = []
    candidate_motion: list[float] = []
    reference_count = candidate_count = 0
    previous_reference = previous_candidate = None
    vision_samples: list[tuple[int, bytes, bytes]] = []
    try:
        assert reference_process.stdout is not None and candidate_process.stdout is not None
        while True:
            reference_frame = _read_frame(reference_process.stdout, frame_bytes, reference)
            candidate_frame = _read_frame(candidate_process.stdout, frame_bytes, candidate)
            if reference_frame is None and candidate_frame is None:
                break
            if reference_frame is not None:
                reference_count += 1
            if candidate_frame is not None:
                candidate_count += 1
            if reference_frame is not None and candidate_frame is not None:
                frame_index = len(frame_metrics)
                frame_metrics.append(_rgb_metrics(reference_frame, candidate_frame))
                if (vision_helper and frame_index % vision_sample_stride == 0 and
                        len(vision_samples) < vision_max_samples):
                    vision_samples.append((frame_index, reference_frame, candidate_frame))
                if previous_reference is not None and previous_candidate is not None:
                    reference_motion.append(_motion_energy(previous_reference, reference_frame))
                    candidate_motion.append(_motion_energy(previous_candidate, candidate_frame))
            if reference_frame is not None:
                previous_reference = reference_frame
            if candidate_frame is not None:
                previous_candidate = candidate_frame
        reference_stderr = reference_process.stderr.read() if reference_process.stderr else b""
        candidate_stderr = candidate_process.stderr.read() if candidate_process.stderr else b""
        if reference_process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {reference}: {reference_stderr.decode(errors='replace')}")
        if candidate_process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {candidate}: {candidate_stderr.decode(errors='replace')}")
    finally:
        for process in (reference_process, candidate_process):
            if process.poll() is None:
                process.terminate()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
    frame_count_equal = reference_count == candidate_count
    if (vision_helper and frame_count_equal and frame_metrics and
            previous_reference is not None and
            previous_candidate is not None):
        final_index = len(frame_metrics) - 1
        sampled_final = vision_samples and vision_samples[-1][0] == final_index
        if not sampled_final and len(vision_samples) < vision_max_samples:
            vision_samples.append((final_index, previous_reference, previous_candidate))
    vision = (
        _vision_metrics(
            vision_samples, reference_meta["width"], reference_meta["height"],
            vision_helper,
        )
        if vision_helper and frame_count_equal and vision_samples else None
    )
    motion_count = min(len(reference_motion), len(candidate_motion))
    motion_relative_errors = [
        abs(left - right) / max(left, right, 1.0)
        for left, right in zip(reference_motion[:motion_count], candidate_motion[:motion_count])
    ]
    mean = lambda key: (
        sum(item[key] for item in frame_metrics) / len(frame_metrics)
        if frame_metrics else float("inf")
    )
    minimum = lambda key: min((item[key] for item in frame_metrics), default=0.0)
    return {
        "shape_equal": shape_equal,
        "frame_count_equal": frame_count_equal,
        "fps_equal": fps_equal,
        "finite": all(math.isfinite(value) for item in frame_metrics for value in item.values()),
        "reference": {**reference_meta, "decoded_frames": reference_count},
        "candidate": {**candidate_meta, "decoded_frames": candidate_count},
        "frame_count_compared": len(frame_metrics),
        "frame_metrics": frame_metrics,
        "mean_correlation": mean("correlation"),
        "minimum_correlation": minimum("correlation"),
        "mean_cosine": mean("cosine"),
        "minimum_cosine": minimum("cosine"),
        "mean_mae_255": mean("mae_255"),
        "maximum_mae_255": max((item["mae_255"] for item in frame_metrics), default=float("inf")),
        "mean_motion_energy_reference": (
            sum(reference_motion[:motion_count]) / motion_count if motion_count else 0.0
        ),
        "mean_motion_energy_candidate": (
            sum(candidate_motion[:motion_count]) / motion_count if motion_count else 0.0
        ),
        "maximum_motion_relative_error": max(motion_relative_errors, default=float("inf")),
        "vision_feature_print": vision,
    }


def contract_passes(metrics: dict) -> bool:
    return bool(
        metrics.get("shape_equal") and metrics.get("frame_count_equal") and
        metrics.get("fps_equal") and metrics.get("finite")
    )


def passes(metrics: dict, *, min_mean_correlation: float = 0.99,
           min_frame_correlation: float = 0.95, min_mean_cosine: float = 0.995,
           max_mean_mae_255: float = 5.0,
           max_motion_relative_error: float = 0.15,
           require_aligned_rgb: bool = True,
           max_vision_distance: Optional[float] = None) -> bool:
    if not contract_passes(metrics):
        return False
    if (float(metrics.get("maximum_motion_relative_error", float("inf"))) >
            max_motion_relative_error):
        return False
    if require_aligned_rgb:
        return bool(
            float(metrics.get("mean_correlation", 0.0)) >= min_mean_correlation and
            float(metrics.get("minimum_correlation", 0.0)) >= min_frame_correlation and
            float(metrics.get("mean_cosine", 0.0)) >= min_mean_cosine and
            float(metrics.get("mean_mae_255", float("inf"))) <= max_mean_mae_255
        )
    vision = metrics.get("vision_feature_print")
    return bool(
        max_vision_distance is not None and isinstance(vision, dict) and
        int(vision.get("pair_count", 0)) > 0 and
        float(vision.get("maximum_distance", float("inf"))) <= max_vision_distance
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--min-mean-correlation", type=float, default=0.99)
    parser.add_argument("--min-frame-correlation", type=float, default=0.95)
    parser.add_argument("--min-mean-cosine", type=float, default=0.995)
    parser.add_argument("--max-mean-mae-255", type=float, default=5.0)
    parser.add_argument("--max-motion-relative-error", type=float, default=0.15)
    parser.add_argument("--vision-helper")
    parser.add_argument("--vision-sample-stride", type=int, default=24)
    parser.add_argument("--vision-max-samples", type=int, default=16)
    parser.add_argument("--allow-aligned-rgb-divergence", action="store_true",
                        help="use calibrated Vision feature distance plus motion "
                             "instead of requiring aligned RGB similarity")
    parser.add_argument("--max-vision-distance", type=float,
                        help="workload-calibrated maximum sampled-frame Vision distance")
    args = parser.parse_args()
    if args.allow_aligned_rgb_divergence:
        if not args.vision_helper or args.max_vision_distance is None:
            parser.error("perceptual divergence requires --vision-helper and "
                         "--max-vision-distance")
        if args.max_vision_distance < 0:
            parser.error("--max-vision-distance must be non-negative")
    metrics = compare(args.reference, args.candidate, ffmpeg=args.ffmpeg or "ffmpeg",
                      ffprobe=args.ffprobe or "ffprobe",
                      vision_helper=args.vision_helper,
                      vision_sample_stride=args.vision_sample_stride,
                      vision_max_samples=args.vision_max_samples)
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "thresholds": {
            "minimum_mean_correlation": args.min_mean_correlation,
            "minimum_frame_correlation": args.min_frame_correlation,
            "minimum_mean_cosine": args.min_mean_cosine,
            "maximum_mean_mae_255": args.max_mean_mae_255,
            "maximum_motion_relative_error": args.max_motion_relative_error,
            "maximum_vision_distance": args.max_vision_distance,
        },
        "metrics": metrics,
        "acceptance_mode": (
            "calibrated_vision_perceptual"
            if args.allow_aligned_rgb_divergence else "aligned_rgb_regression"
        ),
    }
    result["contract_passed"] = contract_passes(metrics)
    result["aligned_rgb_gate_passed"] = passes(
        metrics,
        min_mean_correlation=args.min_mean_correlation,
        min_frame_correlation=args.min_frame_correlation,
        min_mean_cosine=args.min_mean_cosine,
        max_mean_mae_255=args.max_mean_mae_255,
        max_motion_relative_error=args.max_motion_relative_error,
    )
    result["perceptual_gate_passed"] = (
        passes(
            metrics,
            min_mean_correlation=args.min_mean_correlation,
            min_frame_correlation=args.min_frame_correlation,
            min_mean_cosine=args.min_mean_cosine,
            max_mean_mae_255=args.max_mean_mae_255,
            max_motion_relative_error=args.max_motion_relative_error,
            require_aligned_rgb=False,
            max_vision_distance=args.max_vision_distance,
        )
        if args.vision_helper and args.max_vision_distance is not None else None
    )
    result["passed"] = passes(
        metrics,
        min_mean_correlation=args.min_mean_correlation,
        min_frame_correlation=args.min_frame_correlation,
        min_mean_cosine=args.min_mean_cosine,
        max_mean_mae_255=args.max_mean_mae_255,
        max_motion_relative_error=args.max_motion_relative_error,
        require_aligned_rgb=not args.allow_aligned_rgb_divergence,
        max_vision_distance=args.max_vision_distance,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
