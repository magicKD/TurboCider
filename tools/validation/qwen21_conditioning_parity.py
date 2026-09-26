"""Independent PyTorch processor + multimodal assembly oracle, no model weights."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F


def processor(pixels, minimum, maximum):
    _, height, width, _ = pixels.shape
    h, w = round(height / 32) * 32, round(width / 32) * 32
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h, w = (max(32, math.floor(side / beta / 32) * 32) for side in (height, width))
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h, w = (math.ceil(side * beta / 32) * 32 for side in (height, width))
    resized = F.interpolate(torch.from_numpy(pixels).permute(0, 3, 1, 2),
                            size=(h, w), mode="bilinear", align_corners=False)[0]
    normalized = (resized - .5) / .5
    patches = normalized.unsqueeze(0).repeat(2, 1, 1, 1)
    patches = patches.reshape(1, 2, 3, h // 32, 2, 16, w // 32, 2, 16)
    return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 1536).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(891)
    for dtype in (mx.float32, mx.bfloat16):
        for count, height, width, maximum in ((0, 32, 64, 12845056), (1, 47, 81, 12845056),
                                              (2, 121, 73, 4096), (10, 13, 19, 12845056),
                                              (2, 80, 112, 12845056)):
            pixels = rng.random((1, height, width, 3), dtype=np.float32)
            ids = [151644, 24, 25, 151645, 151644, 26]
            spans = []
            expanded_ids = ids.copy()
            data = {"pixels": mx.array(pixels), "table": mx.array(rng.standard_normal((151656, 8)), dtype=dtype)}
            meta = dict(references=count, min_pixels=3136, max_pixels=maximum)
            for i in range(count):
                gh, gw = ((4, 6) if i % 2 else (6, 4))
                ids += [30 + i, 151652, 151655, 151653]
                expanded_ids += [30 + i, 151652]
                start, size = len(expanded_ids), gh * gw // 4
                spans.append((start, size, gh, gw))
                expanded_ids += [151655] * size + [151653]
                meta.update({f"ref{i}_h": gh, f"ref{i}_w": gw})
                for suffix in ("", "_deep0", "_deep1"):
                    data[f"ref{i}{suffix}"] = mx.array(rng.standard_normal((size, 8)), dtype=mx.float32)
            ids += [40, 41, 151645, 151644, 42]
            expanded_ids += [40, 41, 151645, 151644, 42]
            data["ids"] = mx.array(ids, mx.int32)
            seq = len(expanded_ids)
            expected_embeddings = np.array(data["table"].astype(mx.float32))[expanded_ids]
            deep = [np.zeros_like(expected_embeddings), np.zeros_like(expected_embeddings)]
            positions = np.tile(np.arange(seq), (3, 1))
            keep = np.ones(seq, dtype=bool)
            keep[:4] = False
            slots = []
            offset = 0
            for i, (start, size, gh, gw) in enumerate(spans):
                end = start + size
                expected_embeddings[start:end] = np.array(data[f"ref{i}"].astype(dtype).astype(mx.float32))
                for level in range(2):
                    deep[level][start:end] = np.array(data[f"ref{i}_deep{level}"].astype(dtype).astype(mx.float32))
                # Comfy qwen2vl_mrope_position_ids contract.
                length = max(1, gh, gw) // 2
                positions[:, end:] = np.arange(length + start + offset, length + start + seq - end + offset)
                positions[0, start:end] = start + offset
                positions[1, start:end] = np.repeat(np.arange(gh // 2), gw // 2) + start + offset
                positions[2, start:end] = np.tile(np.arange(gw // 2), gh // 2) + start + offset
                offset += length - size
                keep[start:end] = False
                slots.append(int(keep[:start].sum()))
            expected = {"embeddings": expected_embeddings[None], "positions": positions,
                        "retained": expected_embeddings[None, keep],
                        "patches": processor(pixels, 3136, maximum)}
            if count:
                expected["slots"] = np.array(slots, dtype=np.int32)
                expected.update({f"deep{i}": v[None] for i, v in enumerate(deep)})
            with tempfile.TemporaryDirectory(prefix="tc-qwen21-conditioning-") as temp:
                root = Path(temp)
                mx.save_safetensors(str(root / "inputs.safetensors"), data, {k: str(v) for k, v in meta.items()})
                subprocess.run([str(args.probe.resolve()), str(root / "inputs.safetensors"), str(root / "outputs.safetensors")], check=True)
                actual = mx.load(str(root / "outputs.safetensors"))
                errors = {}
                for key, value in expected.items():
                    out = np.array(actual[key].astype(mx.float32))
                    assert out.shape == value.shape, (key, out.shape, value.shape)
                    error = float(np.max(np.abs(out - value), initial=0))
                    errors[key] = error
                    assert error < (2e-5 if key == "patches" else 1e-7), (key, error)
                print(json.dumps({"dtype": str(dtype), "references": count, "input_hw": [height, width], "errors": errors}), flush=True)


if __name__ == "__main__":
    main()
