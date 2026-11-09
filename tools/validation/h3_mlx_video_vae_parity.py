#!/usr/bin/env python3
"""Compare TurboCider and FastVideo on the released ModelScope H3 video VAE."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from h3_vae_metrics import metrics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))
from run_fastvideo_mlx_converter import install_fastvideo_namespace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--max-abs", type=float, default=5e-5)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    args = parser.parse_args()
    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3_video_vae import (
        MiniMaxH3VideoVAEConfigView,
        mlx_h3_video_vae_from_dir,
    )

    component = args.component.resolve()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-video-vae-") as directory:
        output = Path(directory) / "native.safetensors"
        subprocess.run([str(args.probe.resolve()), str(component), str(output)], check=True)
        native = mx.load(str(output))
        config = MiniMaxH3VideoVAEConfigView.from_vae_dir(component)
        vae = mlx_h3_video_vae_from_dir(component, include_encoder=False, storage_dtype="fp32")
        normalized = mx.zeros((1, 24, 7, 2, 2), dtype=mx.float32)
        reference_latents = vae.denormalize_latents(normalized)
        reference_pixels = vae.decode(vae.denormalize_latents(normalized), tiled=False)
        mx.eval(reference_latents, reference_pixels)
        report = {
            "config": {
                "spatial_ratio": config.spatial_compression_ratio,
                "temporal_ratio": config.temporal_compression_ratio,
            },
            "latents": metrics(np.asarray(reference_latents),
                                np.asarray(native["latents"])),
            "pixels": metrics(np.asarray(reference_pixels),
                               np.asarray(native["pixels"])),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        for name in ("latents", "pixels"):
            result = report[name]
            if result["max_abs"] > args.max_abs or result["cosine"] < args.min_cosine:
                raise SystemExit(
                    f"FAIL: {name} max_abs={result['max_abs']:.6g} "
                    f"cosine={result['cosine']:.9f}")
        print("PASS: TurboCider H3 video VAE matches FastVideo")


if __name__ == "__main__":
    main()
