"""Run the native Comfy-to-MLX W4A16 adapter, separately from prior baselines."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from z_image_quantization import ROOT, run
from quantization_quality import metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    folder = a.output.resolve()
    layout = folder / "layout"
    (layout / "split_files/diffusion_models").mkdir(parents=True, exist_ok=False)
    source = a.components.resolve()
    for name in ("vae", "text_encoders"):
        (layout / "split_files" / name).symlink_to(source / "split_files" / name)
    (layout / "tokenizer").symlink_to(source / "tokenizer")
    name = "z_image_turbo_nvfp4.safetensors"
    (layout / "split_files/diffusion_models" / name).symlink_to(source / "split_files/diffusion_models" / name)
    stdout, process = run([sys.executable, ROOT / "tools/native/benchmark_native.py",
        "--library", ROOT / "build/native/libturbocider.dylib", "--model", layout,
        "--model-id", "z-image-turbo", "--request", ROOT / "examples/requests/z-image-turbo.json",
        "--output", folder / "images", "--runs", 4, "--width", 512, "--height", 512,
        "--steps", 9, "--seed", 42, "--execution", "gpu"], folder)
    reference = np.asarray(Image.open(a.reference).convert("RGB")) / 255.
    candidate = np.asarray(Image.open(folder / "images/0.png").convert("RGB")) / 255.
    report = {"method": "Comfy NVFP4 -> lossless nibble/scale repack -> MLX weight-only W4A16; no activation quantization",
              "process": process, "quality_vs_bf16": metrics(reference, candidate),
              "benchmark": json.loads((folder / "images/report.json").read_text())}
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"process": process, "quality": report["quality_vs_bf16"]}, indent=2))
