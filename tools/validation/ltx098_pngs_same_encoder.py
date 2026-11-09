#!/usr/bin/env python3
"""Encode reference PNG frames with the native encoder for a fairer comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--encoder-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.frames.resolve(strict=True).glob("frame_*.png"))
    if len(paths) != 121 or args.output.exists():
        parser.error("need 121 reference frames and a new output directory")
    frames = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths])
    args.output.mkdir(parents=True)
    rgb = args.output / "rgb.safetensors"
    reference = args.output / "reference-native-encoder.mp4"
    save_file({"rgb": torch.from_numpy(frames)}, str(rgb))
    subprocess.run([str(args.encoder_probe.resolve(strict=True)), str(rgb), str(reference), "24"], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("ltx098_video_comparison.py")),
                    "--reference", str(reference), "--candidate", str(args.candidate.resolve(strict=True)),
                    "--output", str(args.output / "comparison.json")], check=True)
    report = json.loads((args.output / "comparison.json").read_text())
    print(json.dumps({"shape": list(frames.shape), "aligned_rgb_gate_passed": report["aligned_rgb_gate_passed"],
                      "metrics": report["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
