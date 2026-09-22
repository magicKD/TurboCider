"""Decode the exact reference latents used by a native diagnostic edit."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--latent-key", default="reference0")
    args = p.parse_args()
    data = mx.load(str(args.native))
    latents = data[args.latent_key]
    if latents.ndim == 3:
        latents = latents.reshape(1, args.height // 16, args.width // 16, 64).transpose(0, 3, 1, 2)
    assert latents.shape == (1, 64, args.height // 16, args.width // 16)
    with tempfile.TemporaryDirectory(prefix="tc-qwen21-reconstruction-") as temp:
        root = Path(temp)
        mx.save_safetensors(str(root / "input.safetensors"), {"latents": latents})
        subprocess.run([str(args.probe.resolve()), str(args.weights.resolve()),
                        str(root / "input.safetensors"), str(root / "output.safetensors")], check=True)
        pixels = mx.load(str(root / "output.safetensors"))["decoded"].transpose(0, 2, 3, 1)
        unit = mx.clip(pixels / 2 + .5, 0, 1)
        array = np.array(mx.round(unit.astype(mx.float32) * 255).astype(mx.uint8))[0]
        Image.fromarray(array).save(args.output)


if __name__ == "__main__":
    main()
