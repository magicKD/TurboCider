#!/usr/bin/env python3
"""Make a visual parity contact sheet from captured VAE decoder tensors."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-key", default="output")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    with safe_open(str(args.reference), framework="pt", device="cpu") as reference, \
         safe_open(str(args.native), framework="pt", device="cpu") as native:
        ref = reference.get_slice("expected")
        out = native.get_slice(args.native_key)
        shape = ref.get_shape()
        if shape != out.get_shape() or len(shape) != 5 or shape[:2] != [1, 3]:
            raise ValueError("incompatible reference and native video tensors")
        frames = sorted(set((0, (shape[2] - 1) // 4, (shape[2] - 1) // 2,
                             (shape[2] - 1) * 3 // 4, shape[2] - 1)))
        tile, label = 224, 24
        sheet = Image.new("RGB", (tile * len(frames), (tile + label) * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for column, index in enumerate(frames):
            a = ref[:, :, index].float()[0].permute(1, 2, 0)
            b = out[:, :, index].float()[0].permute(1, 2, 0)
            a = (a * .5 + .5).clamp(0, 1)
            b = (b * .5 + .5).clamp(0, 1)
            difference = ((a - b).abs() * 8).clamp(0, 1)
            for row, (name, value) in enumerate((("ComfyUI", a), ("Native", b), ("Difference x8", difference))):
                pixels = (value * 255).round().to(torch.uint8).numpy()
                image = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.LANCZOS)
                y = row * (tile + label)
                draw.text((column * tile + 3, y + 4), f"{name}  frame {index}", fill="black")
                sheet.paste(image, (column * tile, y + label))
        sheet.save(args.output)


if __name__ == "__main__":
    main()
