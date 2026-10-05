"""CPU-only equal-size PNG comparison; numerical similarity is not qualification.

Use after timed inference has ended. No resize, learned weights or GPU work.
RGB SSIM is a channel mean with an 11x11, sigma=1.5 Gaussian valid window;
it is not a luminance-only or perceptual/semantic quality score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from runtime_ane_common import sha256_file


def gaussian_mean(value):
    axis = np.arange(-5, 6, dtype=np.float64)
    weights = np.exp(-(axis * axis) / (2 * 1.5 ** 2))
    weights /= weights.sum()
    height, width, _ = value.shape
    horizontal = np.zeros((height, width - 10, 3), dtype=np.float64)
    for offset, weight in enumerate(weights):
        horizontal += weight * value[:, offset:offset + width - 10, :]
    result = np.zeros((height - 10, width - 10, 3), dtype=np.float64)
    for offset, weight in enumerate(weights):
        result += weight * horizontal[offset:offset + height - 10, :, :]
    return result


def rgb_metrics(reference, candidate):
    if (reference.shape != candidate.shape or reference.ndim != 3 or
            reference.shape[-1] != 3 or min(reference.shape[:2]) < 11 or
            reference.dtype != np.uint8 or candidate.dtype != np.uint8):
        raise ValueError("need equal-size uint8 RGB images, at least 11x11; no resize")
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    delta = a - b
    mse = float(np.mean(delta * delta))
    ma, mb = gaussian_mean(a), gaussian_mean(b)
    va = np.maximum(gaussian_mean(a * a) - ma * ma, 0)
    vb = np.maximum(gaussian_mean(b * b) - mb * mb, 0)
    covariance = gaussian_mean(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * ma * mb + c1) * (2 * covariance + c2) /
            ((ma * ma + mb * mb + c1) * (va + vb + c2)))
    x, y = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return {"pixel_exact": bool(np.array_equal(reference, candidate)),
            "rmse_u8": float(np.sqrt(mse)),
            "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "max_abs_u8": int(np.max(np.abs(delta))),
            "rgb_correlation": float(np.sum(x * y) / denominator) if denominator else None,
            "rgb_ssim_gaussian11_valid": float(np.mean(ssim)),
            "ssim_per_channel": [float(x) for x in np.mean(ssim, axis=(0, 1))]}


def compare_png(reference, candidate):
    paths = [Path(reference), Path(candidate)]
    before = [sha256_file(path) for path in paths]
    images = []
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode not in ("RGB", "RGBA"):
                raise ValueError("need 8-bit RGB/RGBA PNGs; no implicit color conversion")
            images.append(np.array(image))
    a, b = images
    if a.shape != b.shape:
        raise ValueError("PNG geometry/channel mismatch; no resize or alpha dropping")
    result = rgb_metrics(a[:, :, :3], b[:, :, :3])
    if a.shape[-1] == 4:
        delta = a[:, :, 3].astype(np.float64) - b[:, :, 3].astype(np.float64)
        result["alpha"] = {"pixel_exact": bool(np.array_equal(a[:, :, 3], b[:, :, 3])),
                           "rmse_u8": float(np.sqrt(np.mean(delta * delta))),
                           "max_abs_u8": int(np.max(np.abs(delta)))}
    if [sha256_file(path) for path in paths] != before:
        raise ValueError("PNG bytes changed during comparison")
    return {"schema": "tc-runtime-ane-png-comparison-v1", "size": list(a.shape[:2][::-1]),
            "png_pixel_exact": bool(np.array_equal(a, b)),
            "source_sha256": before, "metrics": result,
            "scope": "CPU-only unresized PNG numerical comparison; not latent/perceptual/semantic qualification",
            "qualification_passed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_png(args.reference, args.candidate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
