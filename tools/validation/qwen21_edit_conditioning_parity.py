"""Real multimodal conditioning oracle using mflux vision/language blocks.

mflux's Qwen21 pipeline is text-only: this oracle explicitly wires the Comfy
image-span/DeepStack/raw-hidden contract around its unmodified components.
It is a development test, never imported by native inference.
"""
import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from qwen21_conditioning_parity import processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mflux", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", type=Path, action="append", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder

    weights = mx.load(str(args.weights))
    vision = Qwen3VLVisionModel(hidden_size=1152, num_heads=16, intermediate_size=4304,
                               depth=27, out_hidden_size=4096, deepstack_visual_indexes=[8, 16, 24])
    params = []
    for key, _ in tree_flatten(vision.parameters()):
        if key.endswith((".weight", ".bias")):
            value = weights["model.visual." + key]
            if key == "patch_embed.proj.weight":
                value = value.transpose(0, 2, 3, 4, 1)
            params.append((key, value))
    vision.load_weights(params, strict=False)
    features = []
    for path in args.image:
        rgba = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGBA"), dtype=np.float32) / 255.
        h, w = rgba.shape[:2]
        assert h % 32 == w % 32 == 0
        rgb = rgba[..., :3] * rgba[..., 3:] + 1. - rgba[..., 3:]
        patches = mx.array(processor(rgb[None], 3136, 12845056))
        merged, deep = vision(patches, mx.array([[1, h // 16, w // 16]]), return_deepstack=True)
        mx.eval(merged, deep)
        features.append((merged, deep, h // 32, w // 32))
    del vision, params
    mx.clear_cache()

    model = Qwen21TextEncoder()
    model.load_weights([(key, weights["model." + key]) for key, _ in tree_flatten(model.parameters())
                        if key.endswith(".weight")], strict=False)
    del weights
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    refs = " ".join(f"<image{i+1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(features)))
    prompt = system + "<|im_start|>user\n" + refs + args.prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    drop = len(tokenizer.encode(system, add_special_tokens=False).ids)
    chunks, spans, offset = [], [], 0
    for i, token in enumerate(ids):
        if token == 151655:
            merged, deep, h, w = features[len(spans)]
            spans.append((i + offset, merged.shape[0], deep, h, w))
            chunks.append(merged.astype(mx.bfloat16))
            offset += merged.shape[0] - 1
        else:
            chunks.append(model.embed_tokens(mx.array([token])))
    hidden = mx.concatenate(chunks, axis=0)[None]
    seq = hidden.shape[1]
    positions = np.tile(np.arange(seq), (3, 1))
    keep = np.ones(seq, dtype=bool)
    keep[:drop] = False
    residuals = [mx.zeros_like(hidden) for _ in range(3)]
    offset, slots = 0, []
    for start, size, deep, h, w in spans:
        end = start + size
        positions[:, end:] = np.arange(start + max(h, w) + offset, start + max(h, w) + seq - end + offset)
        positions[0, start:end] = start + offset
        positions[1, start:end] = np.repeat(np.arange(h), w) + start + offset
        positions[2, start:end] = np.tile(np.arange(w), h) + start + offset
        offset += max(h, w) - size
        keep[start:end] = False
        slots.append(int(keep[:start].sum()))
        for level in range(3):
            residuals[level][:, start:end] = deep[level].astype(hidden.dtype)
    rope = model.rotary_emb(hidden, mx.array(positions[:, None, :], dtype=mx.int32))
    mask = mx.where(mx.arange(seq)[None, :] > mx.arange(seq)[:, None], -float("inf"), 0.0)[None, None]
    for i, layer in enumerate(model.layers):
        hidden, _ = layer(hidden, mask, rope)
        if i < len(residuals):
            hidden += residuals[i]
        mx.eval(hidden)
    native, metadata = mx.load(str(args.native), return_metadata=True)
    if metadata.get("text_final_norm", "false") == "true":
        hidden = model.norm(hidden)
    expected = hidden[:, mx.array(np.flatnonzero(keep))].astype(mx.float32)
    actual = native["text"].astype(mx.float32)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    delta = actual - expected
    relative = mx.sqrt(mx.mean(delta * delta) / mx.mean(expected * expected)).item()
    cosine = (mx.sum(actual * expected) / mx.sqrt(mx.sum(actual * actual) * mx.sum(expected * expected))).item()
    print(json.dumps({"references": len(features), "expanded_tokens": seq, "retained_tokens": int(keep.sum()),
                      "image_slots": slots, "relative_rmse": relative, "cosine": cosine,
                      "max_abs": mx.max(mx.abs(delta)).item()}), flush=True)
    assert relative < .02 and cosine > .999, "real multimodal conditioning gate failed"


if __name__ == "__main__":
    main()
