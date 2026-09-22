"""Independent official VAE real-reference check; development-only Torch oracle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
import torch

from qwen21_vae_parity import canonical


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusers", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--native", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--decode", type=Path, help="Decode saved NCHW latents to an RGBA PNG using the official VAE")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    p.add_argument("--probe", type=Path, help="Compare a fresh native encode on identical float pixels instead of the saved reference")
    args = p.parse_args()
    if args.device == "mps":
        # Torch 2.14 MPS returned all zeros for this large temporal padding on
        # the validation machine. Do not silently bless a broken oracle.
        source = torch.ones(1, 96, 1, 256, 256)
        expected = torch.nn.functional.pad(source, (0, 0, 0, 0, 1, 0))
        actual = torch.nn.functional.pad(source.to("mps"), (0, 0, 0, 0, 1, 0)).cpu()
        if not torch.equal(expected, actual):
            raise RuntimeError("MPS temporal padding is incorrect at real-image sizes; use --device cpu")
    sys.path.insert(0, str(args.diffusers / "src"))
    from diffusers import AutoencoderKLQwenImage21

    if not args.decode and (not args.native or not args.image):
        p.error("encoding requires --native and --image")
    pixels = None if args.decode else np.asarray(ImageOps.exif_transpose(Image.open(args.image)).convert("RGBA"), dtype=np.float32) / 255.
    native_fresh = None
    native_trace = {}
    if args.probe:
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-official-vae-") as temp:
            root = Path(temp)
            input_image = (torch.from_numpy(pixels).permute(2, 0, 1)[None] * 2 - 1).contiguous()
            save_file({"image": input_image}, str(root / "input.safetensors"), {"trace": "true"})
            subprocess.run([str(args.probe.resolve()), str((args.root / "vae/qwen_image_2.1_vae_bf16.safetensors").resolve()),
                            str(root / "input.safetensors"), str(root / "output.safetensors")], check=True)
            native_trace = load_file(str(root / "output.safetensors"))
            native_fresh = native_trace["encoded"].flatten(2).transpose(1, 2)
    config = json.loads((args.root / "vae/config.json").read_text())
    with torch.device("meta"):
        model = AutoencoderKLQwenImage21.from_config(config)
    expected = model.state_dict()
    state = {}
    for key, value in load_file(str(args.root / "vae/qwen_image_2.1_vae_bf16.safetensors")).items():
        key = canonical(key)
        if value.ndim == 5 and value.shape[2] == 1 and expected[key].ndim == 4:
            value = value.squeeze(2)
        state[key] = value
    model.load_state_dict(state, strict=True, assign=True)
    dtype = getattr(torch, args.dtype)
    model = model.eval().to(args.device, dtype=dtype)
    if args.decode:
        latent = load_file(str(args.decode))["latents"].unsqueeze(2).to(args.device, dtype=dtype)
        mean = torch.tensor(config["latents_mean"], device=args.device, dtype=dtype)[None, :, None, None, None]
        std = torch.tensor(config["latents_std"], device=args.device, dtype=dtype)[None, :, None, None, None]
        decoded = model.decode(latent * std + mean).sample
        rgba = ((decoded[0, :, 0].float().permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        Image.fromarray(rgba).save(args.output)
        return
    image = (torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None] * 2 - 1).to(args.device, dtype=dtype)
    handles = []
    def trace_hook(name):
        def hook(module, inputs, output):
            a = output.detach().float().cpu().squeeze(2)
            b = native_trace[name].float()
            delta = a - b
            print(json.dumps({"stage": name, "relative_rmse": float((delta.square().mean() / a.square().mean()).sqrt()),
                              "max_abs": float(delta.abs().max())}), flush=True)
        return hook
    modules = [("conv_in", model.encoder.conv_in),
                         *[(f"down_{i}", m) for i, m in enumerate(model.encoder.down_blocks)],
                         ("middle", model.encoder.mid_block)]
    for i, block in enumerate(model.encoder.down_blocks):
        modules.extend((f"down_{i}_res_{j}", m) for j, m in enumerate(block.resnets))
        modules.append((f"down_{i}_shortcut", block.avg_shortcut))
        if block.downsampler is not None:
            modules.append((f"down_{i}_main", block.downsampler))
    for name, module in modules:
        if name in native_trace:
            handles.append(module.register_forward_hook(trace_hook(name)))
    try:
        latent = model.encode(image).latent_dist.mode()
    finally:
        for handle in handles:
            handle.remove()
    mean = torch.tensor(config["latents_mean"], device=args.device, dtype=latent.dtype)[None, :, None, None, None]
    std = torch.tensor(config["latents_std"], device=args.device, dtype=latent.dtype)[None, :, None, None, None]
    latent = ((latent - mean) / std).squeeze(2).flatten(2).transpose(1, 2).cpu().contiguous()
    data = load_file(str(args.native))
    a, b = latent.float(), (native_fresh if native_fresh is not None else data["reference0"]).float()
    delta = a - b
    report = dict(relative_rmse=float((delta.square().mean() / a.square().mean()).sqrt()),
                  cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                  max_abs=float(delta.abs().max()), comparison="fresh-native" if args.probe else "saved-native")
    print(json.dumps(report), flush=True)
    data["reference0"] = latent
    save_file(data, str(args.output), {"reference_source": "official-diffusers-vae", "comparison": json.dumps(report)})


if __name__ == "__main__":
    main()
