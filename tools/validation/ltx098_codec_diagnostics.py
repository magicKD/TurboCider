#!/usr/bin/env python3
"""Compare Comfy and native H.264 encoders on the exact same RGB frames."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ltx098_comfy_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-diagnostic", type=Path, required=True)
    parser.add_argument("--comfy-video", type=Path, required=True)
    parser.add_argument("--encoder-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostic, video, probe = (p.resolve(strict=True) for p in (args.comfy_diagnostic, args.comfy_video, args.encoder_probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    import numpy as np
    from PIL import Image
    import torch
    from safetensors.torch import save_file

    paths = sorted(diagnostic.glob("0-frame_*.png"))
    if len(paths) != 121:
        raise ValueError("all 121 Comfy PNG frames are required")
    frames = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths])
    if frames.shape != (121, 512, 512, 3):
        raise ValueError("unexpected frame geometry")
    output.mkdir(parents=True)
    source, encoded = output / "rgb.safetensors", output / "native-encoded.mp4"
    save_file({"rgb": torch.from_numpy(frames)}, str(source))
    encode = subprocess.run([str(probe), str(source), str(encoded), "24"], capture_output=True, text=True, timeout=600)
    if encode.returncode:
        raise RuntimeError(encode.stderr)
    comparison = output / "comparison.json"
    compare = subprocess.run([sys.executable, str(Path(__file__).with_name("ltx098_video_comparison.py")),
        "--reference", str(video), "--candidate", str(encoded), "--output", str(comparison)],
        capture_output=True, text=True, timeout=600)
    if compare.returncode:
        raise RuntimeError(compare.stderr)
    metrics = json.loads(comparison.read_text())
    report = {"schema": "ltx098-codec-diagnostics-v1", "scope": "Same Comfy decoded PNG frames encoded with native AVFoundation H.264; compared against the Comfy H.264 MP4 from the same run; PNG is 8-bit quantized",
              "source_rgb_sha256": sha256(source), "reference_mp4_sha256": sha256(video),
              "candidate_mp4_sha256": sha256(encoded), "contract_passed": metrics["contract_passed"],
              "aligned_rgb_gate_passed": metrics["aligned_rgb_gate_passed"], "metrics": metrics["metrics"]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(compare.stdout)


if __name__ == "__main__":
    main()
