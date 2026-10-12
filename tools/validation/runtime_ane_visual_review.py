#!/usr/bin/env python3
"""Prepare unresized PNG pairs and detail crops for explicit visual review.

This CPU-only evidence tool never infers perceptual acceptance from SSIM or
latent distance. Originals remain untouched; the review starts pending.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from runtime_ane_common import sha256_file


def prepare_review(reference, candidate, output):
    paths = [Path(reference), Path(candidate)]
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("visual review output already exists")
    before = [sha256_file(path) for path in paths]
    images = []
    for path in paths:
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode not in ("RGB", "RGBA"):
                raise ValueError("need RGB/RGBA PNG originals; no implicit conversion")
            images.append(source.copy())
    a, b = images
    if a.size != b.size or a.mode != b.mode or min(a.size) < 16:
        raise ValueError("need equal-size/mode images at least 16x16; no resize")
    width, height = a.size
    # Same source-coordinate crops in both arms, never saliency-cherry-picked.
    crop_width, crop_height = min(256, width), min(256, height)
    boxes = {
        "center": ((width-crop_width)//2, (height-crop_height)//2),
        "top-left": (0, 0),
        "bottom-right": (width-crop_width, height-crop_height),
    }
    pairs = [("whole", images, [0, 0, width, height])]
    for name, (x, y) in boxes.items():
        box = [x, y, x+crop_width, y+crop_height]
        pairs.append((name, [image.crop(box) for image in images], box))
    if [sha256_file(path) for path in paths] != before:
        raise ValueError("PNG bytes changed while loading visual evidence")
    output.mkdir(parents=True)
    sheets = []
    for name, pair, box in pairs:
        w, h = pair[0].size
        gutter, header = 16, 28
        canvas = Image.new(a.mode, (2*w+gutter, h+header), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 8), "GPU reference / "+name, fill="black")
        draw.text((w+gutter+4, 8), "ANE candidate / "+name, fill="black")
        # No mask/alpha blending: preserve original RGBA channels in each arm.
        canvas.paste(pair[0], (0, header))
        canvas.paste(pair[1], (w+gutter, header))
        path = output / (name+".png")
        canvas.save(path)
        sheets.append({"file": path.name, "sha256": sha256_file(path),
                       "source_crop_xyxy": box, "scale": 1,
                       "left_origin": [0, header], "right_origin": [w+gutter, header]})
    if [sha256_file(path) for path in paths] != before:
        raise ValueError("PNG bytes changed during visual evidence publication")
    report = {"schema": "tc-runtime-ane-visual-review-v1",
              "sources": [{"path": str(path.resolve()), "sha256": digest}
                          for path, digest in zip(paths, before)],
              "size": [width, height], "mode": a.mode, "sheets": sheets,
              "review": {"status": "pending", "reviewer": None,
                         "whole_image": None, "detail_crops": None,
                         "artifact_checks": None, "visually_very_similar": None},
              "scope": "unresized whole/detail evidence; manual review required; no performance/latent qualification",
              "qualification_passed": False}
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_review(args.reference, args.candidate, args.output), indent=2))


if __name__ == "__main__":
    main()
