"""Official Transformers Qwen3-VL multimodal conditioning, dev-only MPS oracle."""
import argparse
import json
from pathlib import Path
import time

from accelerate import init_empty_weights
import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path, action="append", required=True)
    p.add_argument("--prompt", required=True)
    args = p.parse_args()
    config = Qwen3VLConfig.from_json_file(str(args.root / "text_encoders/config.json"))
    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLModel(config)
    raw = load_file(str(args.root / "text_encoders/qwen3vl_8b_bf16.safetensors"))
    weights = {}
    for key, value in raw.items():
        if key == "lm_head.weight":
            continue # encoder-only model; output vocabulary head is not used
        if key.startswith("model.visual."):
            key = key.removeprefix("model.")
        elif key.startswith("model."):
            key = "language_model." + key.removeprefix("model.")
        weights[key] = value
    model.load_state_dict(weights, strict=True, assign=True)
    del raw, weights
    model = model.eval().to("mps")
    tokenizer = Tokenizer.from_file(str(args.root / "processor/tokenizer.json"))
    prefix = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    refs = " ".join(f"<image{i+1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(args.image)))
    prompt = prefix + "<|im_start|>user\n" + refs + args.prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    patches, grids, lengths = [], [], []
    for path in args.image:
        pixels = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGBA"), dtype=np.float32) / 255.
        h, w = pixels.shape[:2]
        assert h % 32 == w % 32 == 0
        rgb = torch.from_numpy(pixels[..., :3] * pixels[..., 3:] + 1 - pixels[..., 3:]).permute(2, 0, 1)
        x = ((rgb - .5) / .5).unsqueeze(0).repeat(2, 1, 1, 1)
        x = x.reshape(1, 2, 3, h // 32, 2, 16, w // 32, 2, 16)
        patches.append(x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 1536))
        grids.append([1, h // 16, w // 16]); lengths.append(h * w // 1024)
    expanded, index = [], 0
    for token in ids:
        if token == 151655:
            expanded.extend([token] * lengths[index]); index += 1
        else:
            expanded.append(token)
    ids = torch.tensor([expanded], device="mps")
    types = (ids == 151655).long()
    grids = torch.tensor(grids, device="mps")
    positions, _ = model.get_rope_index(ids, types, grids)
    handle = model.language_model.norm.register_forward_hook(lambda module, inputs, output: inputs[0])
    start = time.perf_counter()
    try:
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
                       pixel_values=torch.cat(patches).to("mps"), image_grid_thw=grids,
                       mm_token_type_ids=types, use_cache=False)
    finally:
        handle.remove()
    hidden = result.last_hidden_state
    keep = ids[0] != 151655
    keep[:len(tokenizer.encode(prefix, add_special_tokens=False).ids)] = False
    conditioning = hidden[:, keep].cpu()
    native = load_file(str(args.native))
    a, b = conditioning.float(), native["text"].float()
    delta = a - b
    report = dict(relative_rmse=float((delta.square().mean() / a.square().mean()).sqrt()),
                  cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                  max_abs=float(delta.abs().max()), seconds=time.perf_counter()-start)
    print(json.dumps(report), flush=True)
    native["text"] = conditioning.contiguous()
    save_file(native, str(args.output), {"text_source": "official-transformers-5.17-pre-final-norm", "prompt": args.prompt})


if __name__ == "__main__":
    main()
