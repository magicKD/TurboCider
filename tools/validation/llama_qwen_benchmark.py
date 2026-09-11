"""Benchmark public llama.cpp callback conditioning against native saved tensors.

GGUF Q4_1/Q8_0 are NOT the same quantizer as MLX affine Q4/Q8. All are derived
from the same BF16 source; embedding stays BF16. No reference source edits.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from safetensors.torch import load_file
from z_image_quantization import run, ROOT
from quantization_quality import metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.models = a.models.resolve()
    a.output = a.output.resolve()
    report = {}
    for label, quant in (("bf16", None), ("q4", "Q4_1"), ("q8", "Q8_0")):
        model = a.models / f"qwen3-{label}.gguf"
        if quant and not model.exists():
            run([ROOT / ".deps/llama-bench-build/bin/llama-quantize", "--token-embedding-type", "bf16",
                 a.models / "qwen3-bf16.gguf", model, quant, 8], a.output / f"llama-convert-{label}")
        for count in (32, 128):
            folder = a.output / f"llama-{label}-{count}"
            stdout, process = run([ROOT / "build/native/llama-qwen-conditioning", model, count, 3,
                                   folder / "conditioning.f32"], folder)
            value = json.loads(stdout)
            tensor = np.fromfile(folder / "conditioning.f32", dtype=np.float32).reshape(1, count, 2560)
            reference = load_file(a.output / f"encoder-bf16-{count}/conditioning.safetensors")["conditioning"].float().numpy()
            report[f"{label}-{count}"] = {**process, **value, "weight_bytes": model.stat().st_size,
                "warm_median_seconds": statistics.median(s["seconds"] for s in value["samples"] if not s["warmup"]),
                "vs_native_bf16": metrics(reference, tensor)}
            (a.output / "llama-summary.json").write_text(json.dumps(report, indent=2))
            print(label, count, report[f"{label}-{count}"]["warm_median_seconds"], report[f"{label}-{count}"]["vs_native_bf16"], flush=True)
