#!/usr/bin/env python3
"""Compare two Wan2.2 decoder outputs from exactly the same saved latent.

Inputs use [1,3,frames,height,width] and the full VAE's [-1,1]-style range.
The scores compare clipped RGB pixels before video encoding, not source-video
fidelity or a learned perceptual metric.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open


def ssim(a, b):
    """RGB SSIM with an 11x11, sigma-1.5 Gaussian and valid borders."""
    kernel = (11, 11)
    blur = lambda image: cv2.GaussianBlur(image, kernel, 1.5)[5:-5, 5:-5]
    ma, mb = blur(a), blur(b)
    va = np.maximum(0, blur(a * a) - ma * ma)
    vb = np.maximum(0, blur(b * b) - mb * mb)
    cov = blur(a * b) - ma * mb
    return float(np.mean(((2 * ma * mb + .01**2) * (2 * cov + .03**2)) /
                         ((ma * ma + mb * mb + .01**2) * (va + vb + .03**2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path)
    parser.add_argument("super", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--key", default="output")
    parser.add_argument("--full-key", help="override the reference tensor key")
    parser.add_argument("--super-key", help="override the Super tensor key")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    args.output.mkdir(parents=True)
    with safe_open(str(args.full), framework="pt", device="cpu") as fa, \
         safe_open(str(args.super), framework="pt", device="cpu") as fb:
        full = fa.get_slice(args.full_key or args.key)
        tiny = fb.get_slice(args.super_key or args.key)
        shape = full.get_shape()
        if shape != tiny.get_shape() or len(shape) != 5 or shape[:2] != [1, 3] or min(shape[3:]) < 12:
            raise ValueError("inputs must have matching [1,3,F,H,W] geometry")
        indexes = sorted(set([0, (shape[2] - 1) // 4, (shape[2] - 1) // 2,
                              (shape[2] - 1) * 3 // 4, shape[2] - 1]))
        tile, label = 256, 24
        sheets = {name: Image.new("RGB", (tile * len(indexes), (tile + label) * 2), "white")
                  for name in ("overview", "center-crop")}
        draws = {name: ImageDraw.Draw(sheet) for name, sheet in sheets.items()}
        records = []
        for index in range(shape[2]):
            values = []
            for source in (full, tiny):
                pixels = source[:, :, index].float()[0].permute(1, 2, 0).numpy()
                values.append(np.clip((pixels + 1) / 2, 0, 1))
            a, b = values
            difference = a - b
            mse = float(np.mean(difference * difference))
            records.append({"frame": index, "mae": float(np.mean(np.abs(difference))),
                            "mse": mse, "ssim": ssim(a, b)})
            if index in indexes:
                column = indexes.index(index)
                for row, (name, value) in enumerate(zip(("Full VAE", "TAEW2.2-Super"), values)):
                    image = Image.fromarray(np.rint(value * 255).astype(np.uint8))
                    center = image.crop(((image.width - tile) // 2, (image.height - tile) // 2,
                                         (image.width + tile) // 2, (image.height + tile) // 2))
                    for sheet_name, selected in (("overview", image.resize((tile, tile), Image.Resampling.LANCZOS)),
                                                  ("center-crop", center.resize((tile, tile)))):
                        y = row * (tile + label)
                        draws[sheet_name].text((column * tile + 3, y + 4), f"{name}  frame {index}", fill="black")
                        sheets[sheet_name].paste(selected, (column * tile, y + label))
        for name, sheet in sheets.items():
            sheet.save(args.output / f"{name}.png")
        mean_mse = float(np.mean([item["mse"] for item in records]))
        report = {"schema": "wan22-vae-compare-v1", "shape": shape,
                  "scope": "same raw latent; RGB clipped to [0,1] before scores; no MP4 encoding",
                  "mean_mae": float(np.mean([item["mae"] for item in records])),
                  "psnr_db": -10 * math.log10(mean_mse) if mean_mse else None,
                  "mean_ssim": float(np.mean([item["ssim"] for item in records])),
                  "frames": records}
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "frames"}, sort_keys=True))


if __name__ == "__main__":
    main()
