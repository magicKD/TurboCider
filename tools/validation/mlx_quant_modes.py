"""Small GPU format probe, NOT an image-model quality or speed benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import time
import struct
import tempfile


def main():
    import mlx.core as mx
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    mx.random.seed(42)
    w = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
    x = mx.random.normal((128, 2048)).astype(mx.bfloat16)
    reference = x @ w.T
    mx.eval(reference)
    report = {"mlx": mx.__version__, "device": mx.device_info(), "shape": [128, 2048, 2048], "modes": {}}
    # The actual Comfy header uses F8_E4M3 scales. Test that container dtype
    # independently, without downloading gigabytes or pretending to run it.
    with tempfile.TemporaryDirectory(prefix="tc-fp8-format-") as temp:
        file = Path(temp) / "scale.safetensors"
        header = json.dumps({"scale": {"dtype": "F8_E4M3", "shape": [16], "data_offsets": [0, 16]}}).encode()
        file.write_bytes(struct.pack("<Q", len(header)) + header + bytes([56] * 16))
        try:
            loaded = mx.load(str(file))
            mx.eval(*loaded.values())
            report["comfy_fp8_scale_container"] = {"loads": True}
        except Exception as error:
            report["comfy_fp8_scale_container"] = {"loads": False, "error": str(error)}
    for label, mode, bits in (("bf16", None, None), ("q4", "affine", 4), ("q8", "affine", 8), ("nvfp4", "nvfp4", 4)):
        try:
            if mode:
                packed = mx.quantize(w, group_size=16 if mode == "nvfp4" else 32, bits=bits, mode=mode)
                mx.eval(*packed)
                fn = lambda: mx.quantized_matmul(x, *packed, group_size=16 if mode == "nvfp4" else 32, bits=bits, mode=mode)
                size = sum(t.nbytes for t in packed)
            else:
                fn = lambda: x @ w.T
                size = w.nbytes
            samples = []
            for i in range(11):
                mx.reset_peak_memory()
                start = time.perf_counter()
                y = fn()
                mx.eval(y)
                if i: samples.append(time.perf_counter() - start)
            cosine = mx.sum(y.astype(mx.float32) * reference.astype(mx.float32)) / (mx.linalg.norm(y.astype(mx.float32)) * mx.linalg.norm(reference.astype(mx.float32)))
            report["modes"][label] = {"supported": True, "weight_bytes": size,
                "median_seconds": statistics.median(samples), "samples": samples,
                "cosine_vs_bf16": cosine.item(), "mlx_peak_bytes": mx.get_peak_memory()}
        except Exception as error:
            report["modes"][label] = {"supported": False, "error": str(error)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
