#!/usr/bin/env python3
"""Compare TurboCider and FastVideo on the released ModelScope H3 audio VAE."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))
from run_fastvideo_mlx_converter import install_fastvideo_namespace  # noqa: E402


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    diff = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": float(np.dot(left, right) / denominator) if denominator else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--max-abs", type=float, default=1e-5)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    args = parser.parse_args()
    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3_audio_vae import (
        mlx_h3_audio_vae_from_dir,
    )

    component = args.component.resolve()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-audio-vae-") as directory:
        output = Path(directory) / "native.safetensors"
        subprocess.run([str(args.probe.resolve()), str(component), str(output)], check=True)
        native = mx.load(str(output))
        vae = mlx_h3_audio_vae_from_dir(component, include_encoder=False)
        normalized = mx.zeros((2, 32, 4), dtype=mx.float32)
        reference_latents = vae.denormalize_latents(normalized)
        reference_waveform = vae.decode(reference_latents)
        mx.eval(reference_latents, reference_waveform)
        report = {
            "latents": metrics(np.asarray(reference_latents),
                                np.asarray(native["latents"])),
            "waveform": metrics(np.asarray(reference_waveform),
                                 np.asarray(native["waveform"])),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        for name, result in report.items():
            if result["max_abs"] > args.max_abs or result["cosine"] < args.min_cosine:
                raise SystemExit(
                    f"FAIL: {name} max_abs={result['max_abs']:.6g} "
                    f"cosine={result['cosine']:.9f}")
        print("PASS: TurboCider H3 audio VAE matches FastVideo")


if __name__ == "__main__":
    main()
