"""Synthetic ConvRot checkpoint used by the Z-Image streaming tests."""

import json
import struct
from pathlib import Path


def write_convrot_fixture(root: Path, change=None) -> Path:
    import numpy as np

    header, payload = {}, bytearray()

    def add(name, values, dtype):
        data = values.tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape),
                       "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)

    embedder = np.array([[1.00390625, 1.01171875], [-0.01, 0.03]], np.float32)
    add("x_embedder.weight", embedder, "F32")

    def projection(prefix, rows, cols, block):
        values = (np.arange(rows * cols, dtype=np.int32) + block) % 256 - 128
        add(prefix + ".weight", values.astype(np.int8).reshape(rows, cols), "I8")
        # Includes BF16 halfway cases to verify CPU packing matches MLX.
        scales = (np.arange(rows, dtype=np.float32) % 13 + 1.00390625) / 128
        add(prefix + ".weight_scale", scales.reshape(rows, 1), "F32")
        quant = np.frombuffer(b'{"format":"int8","rotate":true}', dtype=np.uint8)
        add(prefix + ".comfy_quant", quant, "U8")

    for kind in ("noise_refiner", "context_refiner", "layers"):
        for block in range(30 if kind == "layers" else 2):
            prefix = f"{kind}.{block}"
            for w, rows, cols in ((1, 512, 256), (2, 256, 512), (3, 512, 256)):
                projection(f"{prefix}.feed_forward.w{w}", rows, cols, block)
            if kind == "layers":
                for name, rows in (("attention.qkv", 24), ("attention.out", 8), ("adaLN_modulation.0", 32)):
                    projection(f"{prefix}.{name}", rows, 256, block)
                for j in range(7):
                    norm = np.array([block + 1.00390625, -0.01], np.float32)
                    add(f"{prefix}.norm{j}.weight", norm, "F32")
    if change:
        change(header)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = root / "convrot.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path
