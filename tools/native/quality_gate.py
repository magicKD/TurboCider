#!/usr/bin/env python3
"""Offline, deterministic quality gate for paired image benchmarks.

This utility is intentionally outside the runtime. Approximate GPU/ANE routes
are compared with a paired output from the same request; callers choose
thresholds appropriate for the model and workload.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import zlib
from pathlib import Path


def _paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _decode_png_rgb(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        value = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", value
            )
        elif kind == b"IDAT":
            compressed.extend(value)
        elif kind == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bit_depth != 8 or interlace != 0 or channels is None or width is None or height is None:
        raise ValueError(
            f"unsupported PNG encoding in {path}: depth={bit_depth}, "
            f"color={color_type}, interlace={interlace}"
        )
    raw = zlib.decompress(compressed)
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise ValueError(f"PNG payload size mismatch in {path}")
    decoded_rows = bytearray(height * stride)
    previous = bytearray(stride)
    source = 0
    for row in range(height):
        filter_type = raw[source]
        source += 1
        encoded = raw[source:source + stride]
        source += stride
        decoded = bytearray(stride)
        for index, byte in enumerate(encoded):
            left = decoded[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = above
            elif filter_type == 3:
                prediction = (left + above) // 2
            elif filter_type == 4:
                prediction = _paeth(left, above, upper_left)
            else:
                raise ValueError(f"invalid PNG filter {filter_type} in {path}")
            decoded[index] = (byte + prediction) & 255
        begin = row * stride
        decoded_rows[begin:begin + stride] = decoded
        previous = decoded
    rgb = bytearray(width * height * 3)
    for pixel in range(width * height):
        source = pixel * channels
        destination = pixel * 3
        if channels in (1, 2):
            rgb[destination:destination + 3] = bytes([decoded_rows[source]]) * 3
        else:
            rgb[destination:destination + 3] = decoded_rows[source:source + 3]
    return width, height, bytes(rgb)


def image_metrics(reference: Path, candidate: Path) -> dict[str, float | bool]:
    """Return finite RGB pixel metrics for two images."""
    first = _decode_png_rgb(reference)
    second = _decode_png_rgb(candidate)
    if first[:2] != second[:2]:
        return {
            "shape_equal": False,
            "finite": False,
            "correlation": 0.0,
            "cosine": 0.0,
            "mae_255": float("inf"),
            "rmse_255": float("inf"),
        }
    a, b = first[2], second[2]
    count = len(a)
    absolute = squared = dot = first_squared = second_squared = 0.0
    first_sum = second_sum = 0.0
    for left, right in zip(a, b):
        difference = float(left) - float(right)
        absolute += abs(difference)
        squared += difference * difference
        dot += float(left) * float(right)
        first_squared += float(left) * float(left)
        second_squared += float(right) * float(right)
        first_sum += left
        second_sum += right
    first_mean = first_sum / count
    second_mean = second_sum / count
    covariance = first_variance = second_variance = 0.0
    for left, right in zip(a, b):
        first_delta = float(left) - first_mean
        second_delta = float(right) - second_mean
        covariance += first_delta * second_delta
        first_variance += first_delta * first_delta
        second_variance += second_delta * second_delta
    return {
        "shape_equal": True,
        "finite": True,
        "correlation": covariance / math.sqrt(first_variance * second_variance)
        if first_variance and second_variance else 1.0,
        "cosine": dot / math.sqrt(first_squared * second_squared)
        if first_squared and second_squared else 1.0,
        "mae_255": absolute / count,
        "rmse_255": math.sqrt(squared / count),
    }


def vision_metrics(reference: Path, candidate: Path, helper: Path) -> dict:
    """Return the public-Vision feature-print distance for an image pair.

    Vision feature-print values are OS/revision dependent, so this function
    deliberately records a distance without imposing a universal threshold.
    A benchmark must supply a workload-calibrated maximum when it elects to
    accept perceptual divergence from the aligned RGB reference.
    """
    completed = subprocess.run(
        [str(helper), str(reference), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    if result.get("pair_count") != 1:
        raise ValueError("Vision helper did not return exactly one image pair")
    mean_distance = float(result.get("mean_distance", float("nan")))
    maximum_distance = float(result.get("maximum_distance", float("nan")))
    if not math.isfinite(mean_distance) or not math.isfinite(maximum_distance):
        raise ValueError("Vision helper returned a non-finite distance")
    return {
        "metric": result.get("metric", "VNFeaturePrintObservation distance"),
        "vision_request_revision": result.get("vision_request_revision"),
        "image_crop_and_scale": result.get("image_crop_and_scale"),
        "operating_system": result.get("operating_system", "unknown"),
        "pair_count": 1,
        "mean_distance": mean_distance,
        "maximum_distance": maximum_distance,
    }


def contract_passes(metrics: dict[str, float | bool]) -> bool:
    return bool(metrics.get("shape_equal") and metrics.get("finite"))


def passes(metrics: dict[str, float | bool], *, min_correlation: float,
           min_cosine: float, max_mae_255: float,
           require_aligned_rgb: bool = True,
           max_vision_distance: float | None = None) -> bool:
    if not contract_passes(metrics):
        return False
    if require_aligned_rgb:
        return bool(
            float(metrics.get("correlation", 0.0)) >= min_correlation and
            float(metrics.get("cosine", 0.0)) >= min_cosine and
            float(metrics.get("mae_255", float("inf"))) <= max_mae_255
        )
    vision = metrics.get("vision_feature_print")
    return bool(
        max_vision_distance is not None and isinstance(vision, dict) and
        int(vision.get("pair_count", 0)) == 1 and
        float(vision.get("maximum_distance", float("inf"))) <= max_vision_distance
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--max-mae-255", type=float, default=5.0)
    parser.add_argument("--vision-helper", type=Path)
    parser.add_argument("--allow-aligned-rgb-divergence", action="store_true",
                        help="use a calibrated Vision feature distance instead of "
                             "requiring aligned RGB similarity")
    parser.add_argument("--max-vision-distance", type=float,
                        help="workload-calibrated maximum Vision feature distance")
    args = parser.parse_args()
    if args.allow_aligned_rgb_divergence:
        if args.vision_helper is None or args.max_vision_distance is None:
            parser.error("perceptual divergence requires --vision-helper and "
                         "--max-vision-distance")
        if args.max_vision_distance < 0:
            parser.error("--max-vision-distance must be non-negative")
    metrics = image_metrics(args.reference, args.candidate)
    if args.vision_helper is not None:
        metrics["vision_feature_print"] = vision_metrics(
            args.reference, args.candidate, args.vision_helper
        )
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "thresholds": {
            "minimum_correlation": args.min_correlation,
            "minimum_cosine": args.min_cosine,
            "maximum_mae_255": args.max_mae_255,
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
        min_correlation=args.min_correlation,
        min_cosine=args.min_cosine,
        max_mae_255=args.max_mae_255,
    )
    result["perceptual_gate_passed"] = (
        passes(
            metrics,
            min_correlation=args.min_correlation,
            min_cosine=args.min_cosine,
            max_mae_255=args.max_mae_255,
            require_aligned_rgb=False,
            max_vision_distance=args.max_vision_distance,
        )
        if args.vision_helper is not None and args.max_vision_distance is not None
        else None
    )
    result["passed"] = passes(
        metrics,
        min_correlation=args.min_correlation,
        min_cosine=args.min_cosine,
        max_mae_255=args.max_mae_255,
        require_aligned_rgb=not args.allow_aligned_rgb_divergence,
        max_vision_distance=args.max_vision_distance,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
