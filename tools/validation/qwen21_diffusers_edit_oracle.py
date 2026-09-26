"""Official Diffusers DiT on exactly the native edit's saved conditioning/noise.

Development-only PyTorch/MPS oracle. This deliberately does NOT use native or
mflux blocks, geometry, attention masks or prefix cache implementations.
Pass --single --probe to compare the first real-image-conditioned forward.
Without --single, run all saved sigmas and save latents for native VAE decoding.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusers", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--single", action="store_true")
    p.add_argument("--probe", type=Path)
    p.add_argument("--compare-steps", type=int, default=0,
                   help="Compare native cached trajectory for this many saved steps (requires --probe)")
    p.add_argument("--reference-shape", action="append", help="HxW pixels, in reference order")
    args = p.parse_args()
    if args.compare_steps < 0 or (args.compare_steps and (not args.probe or args.single)):
        p.error("--compare-steps must be positive, requires --probe, and cannot be combined with --single")
    sys.path.insert(0, str(args.diffusers / "src"))
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache, QwenImage21Rope, QwenImage21TemporalTimesteps,
    )

    data = load_file(str(args.native))
    text = data["text"]
    slots = data["image_slots"].tolist()
    # CPU FP32 VAE oracle artifacts preserve their accurate encoding. The
    # actual pipeline casts conditioning latents to the DiT activation dtype.
    refs = [data[f"reference{i}"].to(data["initial"].dtype) for i in range(len(slots))]
    shapes = [(args.height // 16, args.width // 16)] * len(refs)
    if args.reference_shape:
        shapes = [tuple(int(v) // 16 for v in item.split("x")) for item in args.reference_shape]
    assert len(shapes) == len(refs)
    parts, masks, cursor = [], [], 0
    metadata = dict(layers="32", heads="32", head_dim="128", axis0="16", axis1="56", axis2="56",
                    height=str(args.height // 16), width=str(args.width // 16), reference_count=str(len(refs)))
    for i, (slot, ref, (h, w)) in enumerate(zip(slots, refs, shapes)):
        assert h * w == ref.shape[1] and h % 2 == w % 2 == 0
        parts += [text[:, cursor:slot], text.new_zeros(1, h * w // 4, text.shape[-1])]
        masks += [torch.zeros(slot - cursor, dtype=torch.bool), torch.ones(h * w // 4, dtype=torch.bool)]
        cursor = slot
        metadata.update({f"reference{i}_height": str(h), f"reference{i}_width": str(w), f"reference{i}_slot": str(slot)})
    parts.append(text[:, cursor:])
    masks += [torch.zeros(text.shape[1] - cursor, dtype=torch.bool), torch.ones(args.height * args.width // 1024, dtype=torch.bool)]
    native_first = None
    native_trajectory = None
    if (args.single or args.compare_steps) and args.probe:
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-official-") as temp:
            root = Path(temp)
            tensors = {"text": text, "latents": data["initial"], **{f"reference{i}": v for i, v in enumerate(refs)}}
            if args.compare_steps:
                assert args.compare_steps < data["sigmas"].numel(), "not enough saved timesteps"
                tensors["sigmas"] = data["sigmas"][:args.compare_steps + 1].contiguous()
            save_file(tensors, str(root / "input.safetensors"), metadata)
            subprocess.run([str(args.probe.resolve()), str(args.weights.resolve()), str(root / "input.safetensors"),
                            str(root / "output.safetensors"), "trajectory" if args.compare_steps else "single"], check=True)
            result = load_file(str(root / "output.safetensors"))
            if args.compare_steps:
                native_trajectory = result
            else:
                native_first = result["output"].float()

    with torch.device("meta"):
        model = QwenImage21Transformer2DModel()
    # Nonpersistent constant tables are not in the checkpoint. Instantiate
    # their original modules outside meta rather than invent replacement math.
    model.time_text_embed.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
    model.pos_embed = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    state = load_file(str(args.weights))
    for key in list(state):
        if key.endswith(".img_mlp.gate_up.weight"):
            gate, up = state.pop(key).chunk(2, dim=0)
            state[key.replace("gate_up", "gate_layer")] = gate
            state[key.replace("gate_up", "proj")] = up
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.eval().to("mps")
    prompt = torch.cat(parts, dim=1).to("mps")
    mask = torch.cat(masks)[None].to("mps")
    images = [v.to("mps") for v in refs]
    image_shapes = [[(1, h, w) for h, w in shapes] + [(1, args.height // 16, args.width // 16)]]
    latents = data["initial"].to("mps")
    sigmas = data["sigmas"].float()
    cache = QwenImage21KVCache(32)
    first = None
    for step in range(1 if args.single else args.compare_steps or len(sigmas) - 1):
        start = time.perf_counter()
        # The official pipeline rounds t*1000, then divides by 1000 in BF16.
        timestep = (sigmas[step:step+1].to("mps") * 1000).to(latents.dtype) / 1000
        noise = model(hidden_states=torch.cat(images + [latents], dim=1), timestep=timestep,
                      encoder_hidden_states=prompt, img_shapes=image_shapes, img_mask=mask,
                      kv_cache=cache, kv_cache_mode="extract" if step == 0 else "cached", return_dict=False)[0]
        noise = noise[:, -latents.shape[1]:]
        if step == 0:
            first = noise.cpu()
            if native_first is not None:
                a, b = first.float(), native_first
                delta = a - b
                report = dict(relative_rmse=float((delta.square().mean() / a.square().mean()).sqrt()),
                              cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                              max_abs=float(delta.abs().max()))
                print(json.dumps({"official_first_step_vs_native": report}), flush=True)
        dt = (sigmas[step+1] - sigmas[step]).to("mps")
        # Official Euler accumulates in FP32, then converts to model dtype.
        latents = (latents.float() + noise.float() * dt).to(latents.dtype)
        torch.mps.synchronize()
        if native_trajectory is not None:
            comparisons = {}
            for name, tensor in (("noise", noise), ("latent", latents)):
                a, b = tensor.float().cpu(), native_trajectory[f"{name}_{step}"].float()
                comparisons[name] = dict(
                    relative_rmse=float(((a-b).square().mean() / a.square().mean()).sqrt()),
                    cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))
            print(json.dumps({"trajectory_step": step + 1, "comparisons": comparisons}), flush=True)
        print(json.dumps({"step": step + 1, "seconds": time.perf_counter() - start}), flush=True)
    output = {"latents": latents.cpu().transpose(1, 2).reshape(1, 64, args.height // 16, args.width // 16).contiguous(),
              "first": first.contiguous()}
    if native_first is not None:
        output["native_first"] = native_first.contiguous()
    save_file(output, str(args.output))
    del model, cache
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
