"""Official PE source cap plus pinned processor vs native input preprocessing."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import torch
import transformers
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    args = parser.parse_args()
    assert transformers.__version__ == "5.4.0" and torch.__version__.startswith("2.10.0")
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True).image_processor
    rng = np.random.default_rng(31)
    cases = []
    with tempfile.TemporaryDirectory(prefix="tc-pe-processor-") as directory:
        root = Path(directory)
        for h, w, channels in [(32,32,3), (47,81,3), (513,777,3), (1200,900,3),
                                (1200,1900,3), (256,256,3), (1,1,3), (16,31,4),
                                (321,417,4), (1600,900,4)]:
            pixels = rng.integers(0, 256, (h,w,channels), dtype=np.uint8)
            if channels == 4:
                pixels[:h//2,:,3] = 0  # Hidden RGB must survive RGB conversion.
            source = Image.fromarray(pixels).convert("RGB")
            if h*w > 1024*1024:
                scale = math.sqrt(1024*1024/(h*w))
                source = source.resize((max(1,int(w*scale)),max(1,int(h*scale))), Image.Resampling.LANCZOS)
            reference = processor.preprocess(source, return_tensors="pt")
            expected = reference["pixel_values"].float()
            # Native receives ORIGINAL pixels, not a Python-capped fixture.
            inp, out = root/"input.safetensors", root/"output.safetensors"
            save_file({"pixels":torch.from_numpy(pixels)[None]}, str(inp))
            subprocess.run([str(args.probe.resolve()), "--vision", str(inp), str(out)], check=True)
            got = load_file(str(out))["patches"].float()
            assert got.shape == expected.shape
            with safe_open(str(out), framework="pt") as tensors:
                metadata = tensors.metadata()
            assert [1, int(metadata["grid_height"]), int(metadata["grid_width"])] == reference["image_grid_thw"][0].tolist()
            error = float((got-expected).abs().max())
            record = dict(input_hwc=[h,w,channels], source_cap_wh=list(source.size),
                grid=reference["image_grid_thw"].tolist(), shape=list(got.shape),
                max_abs=error, rmse=float((got-expected).square().mean().sqrt()))
            cases.append(record)
            report = dict(passed=False, cases=cases, transformers_version=transformers.__version__,
                          torch_version=torch.__version__, scope="decoded pixels to PE patch tensors; not file decoder or generation")
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(record), flush=True)
            assert torch.isfinite(got).all() and error < 1e-6, record
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS exact pinned PE processor parity: {len(cases)} cases")


if __name__ == "__main__":
    main()
