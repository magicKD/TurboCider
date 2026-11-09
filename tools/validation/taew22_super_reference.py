#!/usr/bin/env python3
"""Run the pinned upstream TAEW2.2-Super decoder on saved Wan2.2 latents.

This is an isolated development oracle, not an App dependency. Both the
upstream source and safetensors checkpoint are checked against Git blob IDs
from the pinned upstream commit before the source is imported.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


SOURCE_BLOB = "8514281c20318c71ccdf916e4383d9a65f825009"
WEIGHTS_BLOB = "2021b7ee3c9996cb2248fe36b044f18c6fefcb97"


def blob_id(path):
    payload = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "comfy", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=0, help="0 uses all frames")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    source, checkpoint, comfy, latent_path = (p.resolve(strict=True) for p in
                                               (args.source, args.checkpoint, args.comfy, args.input))
    if blob_id(source) != SOURCE_BLOB or blob_id(checkpoint) != WEIGHTS_BLOB:
        raise ValueError("upstream source or model does not match pinned Git blobs")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise ValueError("MPS CPU fallback must be disabled")
    import torch
    from safetensors.torch import load_file, save_file
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    sys.path.insert(0, str(comfy))
    from comfy.latent_formats import Wan22
    spec = importlib.util.spec_from_file_location("taehv_pinned_upstream", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = load_file(str(latent_path))["latents"]
    if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[1] != 48:
        raise ValueError("expected unnormalized [1,48,F,H,W] Wan2.2 VAE latent")
    if args.latent_frames:
        if not 1 <= args.latent_frames <= raw.shape[2]:
            raise ValueError("invalid latent frame count")
        raw = raw[:, :, :args.latent_frames]
    normalized = Wan22().process_in(raw.float()).transpose(1, 2).to(device="mps", dtype=torch.float16)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    model = module.TAEHV(str(checkpoint), arch_name="taew2_2_super").to(device="mps", dtype=torch.float16).eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    with torch.inference_mode():
        started = time.perf_counter()
        pixels = model.decode_video(normalized, parallel=True, show_progress_bar=False)
        torch.mps.synchronize()
        decode_seconds = time.perf_counter() - started
    pixels = pixels.transpose(1, 2).contiguous().cpu()
    if pixels.shape != (1, 3, 4 * raw.shape[2] - 3, raw.shape[3] * 16, raw.shape[4] * 16):
        raise ValueError(f"unexpected decoded shape {pixels.shape}")
    if not torch.isfinite(pixels).all():
        raise ValueError("nonfinite TAEW2.2-Super pixels")
    save_file({"output": pixels}, str(args.output / "reference.safetensors"))
    report = {"schema": "taew22-super-reference-v1", "source_blob": SOURCE_BLOB,
              "weights_blob": WEIGHTS_BLOB, "dtype": "fp16", "device": "mps",
              "latent_shape": list(raw.shape), "pixel_shape": list(pixels.shape),
              "load_seconds": load_seconds, "decode_seconds": decode_seconds,
              "scope": "single upstream PyTorch MPS decode; no warmup or MP4 export"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
