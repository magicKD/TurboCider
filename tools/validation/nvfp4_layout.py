"""Verify Comfy NVFP4 nibble order and scale layout against dense source weights.

Reference format: Comfy-Org/comfy-kitchen eager dequantize_nvfp4 and
float_utils.from_blocked. Implement the layout mapping independently with
NumPy indexing; no CUDA/Torch execution semantics are assumed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from quantization_quality import metrics


def main():
    import mlx.core as mx
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nvfp4", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native-layer", type=Path)
    a = p.parse_args()
    results = {}
    for prefix in ("layers.0.attention.qkv", "layers.0.feed_forward.w2", "layers.29.adaLN_modulation.0"):
        with safe_open(a.nvfp4, framework="pt", device="cpu") as f:
            packed = f.get_tensor(prefix + ".weight").numpy()
            scales = f.get_tensor(prefix + ".weight_scale").float().numpy()
            scale2 = f.get_tensor(prefix + ".weight_scale_2").float().item()
        with safe_open(a.bf16, framework="pt", device="cpu") as f:
            dense = f.get_tensor(prefix + ".weight").float().numpy()
        rows, cols = dense.shape
        nr, nc = (rows + 127) // 128, (cols // 16 + 3) // 4
        row = np.arange(rows)[:, None]
        col = np.arange(cols // 16)[None, :]
        # cuBLAS scale tile order: row block, col block, row%32,
        # (row%128)//32, col%4.
        offsets = ((((row // 128 * nc + col // 4) * 32 + row % 32) * 4 + row % 128 // 32) * 4 + col % 4)
        unswizzled = scales.reshape(-1)[offsets]
        lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
        digits = np.stack((packed >> 4, packed & 15), axis=-1).reshape(rows, cols)
        values = lut[digits]
        reference = values * np.repeat(unswizzled * scale2, 16, axis=1)
        raw = values * np.repeat(scales * scale2, 16, axis=1)
        swapped = ((packed << 4) | (packed >> 4)).copy()
        w = mx.array(swapped).view(mx.uint32)
        s = mx.to_fp8(mx.array(unswizzled))
        decoded = mx.dequantize(w, s, group_size=16, bits=4, mode="nvfp4", dtype=mx.float32) * scale2
        mx.eval(decoded)
        results[prefix] = {"comfy_vs_bf16": metrics(dense, reference), "raw_scale_vs_bf16": metrics(dense, raw),
                           "mlx_vs_reference": metrics(reference, np.array(decoded)), "scale2": scale2}
        if results[prefix]["mlx_vs_reference"]["max_abs"] != 0:
            raise ValueError("MLX layout does not match the independent decoder exactly")
        if a.native_layer and prefix == "layers.0.attention.qkv":
            with safe_open(a.native_layer, framework="pt", device="cpu") as f:
                native = f.get_tensor("weight").float().numpy()
            results[prefix]["native_vs_reference"] = metrics(reference, native)
            if not np.array_equal(native, reference):
                raise ValueError("Native C++ adapter does not match independent decoder")
        print(prefix, results[prefix], flush=True)
    a.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
