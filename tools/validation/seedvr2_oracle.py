#!/usr/bin/env python3
"""Independent pinned-ComfyUI intermediate oracle, never production inference.

Explicit paths keep user installations and services unchanged. MPS fallback
is disabled: unsupported operators must fail visibly, not benchmark the CPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_report(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    import numpy as np
    from PIL import Image
    import torch
    from safetensors.torch import save_file
    if args.output.exists():
        raise ValueError("a fresh fixture directory is required")
    args.output.mkdir(parents=True)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    image = Image.open(args.image).convert("RGB")
    if args.source_size:
        image = image.resize((args.source_size, args.source_size), Image.Resampling.LANCZOS)
        image.save(args.output / "source.png")
    image = image.resize((args.size, args.size), Image.Resampling.BICUBIC)
    pixels = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    frames = [np.roll(pixels, i, axis=1) for i in range(args.frames)]
    pixels = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    latent_shape = (1, 16, (args.frames - 1) // 4 + 1, args.size // 8, args.size // 8)
    fixture = dict(pixels=pixels, noise=torch.randn(latent_shape, generator=rng),
                   condition=torch.randn(latent_shape, generator=rng) * .3)
    path = args.output / "input.safetensors"
    save_file(fixture, str(path))
    write_report(args.output / "fixture.json", dict(schema="seedvr2-component-fixture-v1",
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_image=str(args.image),
        source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        size=args.size, source_size=args.source_size, frames=args.frames, seed=args.seed,
        scope="component parity fixture; rolling image is not real video acceptance evidence"))


def oracle(args):
    import torch
    from safetensors.torch import load_file, save_file
    if args.output.exists():
        raise ValueError("a fresh oracle output directory is required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise ValueError("automatic MPS CPU fallback is forbidden in the oracle")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; grant Metal access rather than enabling CPU fallback")
    sys.path.insert(0, str(args.comfy.resolve(strict=True)))
    sys.path.insert(0, str(args.comfy_extra.resolve(strict=True)))
    # Configure the stock model-management module before importing it.
    sys.argv = [sys.argv[0], "--cpu"] if args.device == "cpu" else [sys.argv[0], "--gpu-only", "--use-pytorch-cross-attention"]
    import comfy.model_management
    import comfy.ops
    from comfy.ldm.seedvr.model import NaDiT
    from comfy.ldm.seedvr.vae import VideoAutoencoderKLWrapper
    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    synchronize = torch.mps.synchronize if args.device == "mps" else lambda: None
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    inputs = load_file(str(args.input), device="cpu")
    started = time.perf_counter()
    with torch.inference_mode():
        if args.stage in ("dit", "restore"):
            if args.stage == "restore":
                vae = VideoAutoencoderKLWrapper()
                vae_state = load_file(str(args.model / "vae/seedvr2_ema_vae_fp16.safetensors"), device="cpu")
                vae.load_state_dict(vae_state, strict=True, assign=True)
                del vae_state
                vae.eval().to(device=device, dtype=torch.float16)
                pixels = inputs["pixels"].to(device=device, dtype=torch.float16)
                condition = vae.comfy_format_encoded(vae.encode(pixels))
                synchronize()
                encoded = time.perf_counter()
            model = NaDiT(norm_eps=1e-5, num_layers=32, mlp_type="swiglu", vid_out_norm=True,
                          dtype=torch.float16, device="cpu", operations=comfy.ops.manual_cast)
            checkpoint_path = args.model / "diffusion_models/seedvr2_3b_fp16.safetensors"
            state = load_file(str(checkpoint_path), device="cpu")
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.eval().to(device)
            synchronize()
            loaded = time.perf_counter()
            def save(name, tensors):
                synchronize()
                save_file({k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()},
                          str(args.output / f"{name}.safetensors"))
            embeddings = {}
            model.vid_in.register_forward_hook(lambda module, args_, output: embeddings.update(video=output[0].clone()))
            model.txt_in.register_forward_hook(lambda module, args_, output: embeddings.update(text=output.clone()))
            model.emb_in.register_forward_hook(lambda module, args_, output: embeddings.update(time=output.clone()))
            for index in (0, 1, 9, 10, 31):
                def capture(module, args_, output, index=index):
                    if index == 0:
                        save("embedding", embeddings)
                    save(f"block-{index}", dict(video=output[0], text=output[1]))
                    print(f"captured block {index}", flush=True)
                model.blocks[index].register_forward_hook(capture)
            noise = inputs["noise"].to(device=device, dtype=dtype)
            raw_condition = condition if args.stage == "restore" else inputs["condition"].to(device)
            condition = raw_condition.to(dtype=dtype)
            condition = torch.cat((condition, torch.ones_like(condition[:, :1])), dim=1)
            velocity = model(noise, torch.tensor([1000.0], device=device), model.positive_conditioning.to(dtype).unsqueeze(0),
                             condition=condition, transformer_options={"cond_or_uncond": [0]})
            synchronize()
            finished = time.perf_counter()
            save("velocity", dict(velocity=velocity))
            save("sampling", dict(restored=inputs["noise"].to(device).float() - velocity.float()))
            if args.stage == "restore":
                latent = inputs["noise"].to(device).float() - velocity.float()
                decoded = vae.decode(latent.to(dtype=torch.float16))
                synchronize()
                decoded_at = time.perf_counter()
                save("restoration", dict(pixels=decoded, condition=raw_condition,
                                         velocity=velocity, latent=latent))
        else:
            checkpoint_path = args.model / "vae/seedvr2_ema_vae_fp16.safetensors"
            model = VideoAutoencoderKLWrapper()
            state = load_file(str(checkpoint_path), device="cpu")
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.eval().to(device=device, dtype=torch.float16)
            synchronize()
            loaded = time.perf_counter()
            pixels = inputs["pixels"].to(device=device, dtype=torch.float16)
            latent = model.comfy_format_encoded(model.encode(pixels))
            synchronize()
            encoded = time.perf_counter()
            decoded = model.decode(latent)
            synchronize()
            finished = time.perf_counter()
            save_file(dict(latent=latent.detach().cpu().contiguous(), decoded=decoded.detach().cpu().contiguous()),
                      str(args.output / "vae.safetensors"))
    report = dict(schema="seedvr2-independent-component-oracle-v1", stage=args.stage,
        device=args.device, dtype=args.dtype if args.stage in ("dit", "restore") else "fp16",
        torch_version=torch.__version__, comfy_commit=subprocess.check_output(
            ["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip(),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        load_seconds=loaded-started, inference_seconds_with_dumps=finished-loaded,
        scope="component parity with intermediate dumps, not a complete ComfyUI workflow speed benchmark")
    if args.stage == "vae":
        report.update(encode_seconds=encoded-loaded, decode_seconds=finished-encoded)
    if args.stage == "restore":
        report.update(decode_seconds=decoded_at-finished,
                      scope="complete component restoration with intermediate dumps; loading/stages differ, not a ComfyUI workflow speed benchmark")
    write_report(args.output / "report.json", report)
    print(json.dumps(report), flush=True)


def compare(args):
    import numpy as np
    from safetensors.torch import load_file
    results = []
    for source in sorted(args.reference.glob(args.files)):
        candidate = args.candidate / source.name
        reference = load_file(str(source))
        actual = load_file(str(candidate))
        if set(reference) != set(actual):
            raise ValueError(f"tensor names differ: {source.name}")
        for name, target in reference.items():
            value = actual[name]
            if value.shape != target.shape:
                raise ValueError(f"shape differs: {source.name}:{name}")
            x, y = target.float().numpy().astype(np.float64).ravel(), value.float().numpy().astype(np.float64).ravel()
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            norm = np.linalg.norm(x)
            result = dict(file=source.name, tensor=name, finite=finite,
                relative_l2=float(np.linalg.norm(x-y) / max(norm, 1e-20)),
                cosine=float(np.dot(x, y) / max(norm * np.linalg.norm(y), 1e-20)),
                max_abs=float(np.max(np.abs(x-y))))
            results.append(result)
    if not results:
        raise ValueError("no matching intermediate tensors found")
    if args.output:
        write_report(args.output, dict(schema="seedvr2-component-parity-v1",
            reference=str(args.reference), candidate=str(args.candidate), tensors=results,
            passed=all(r["finite"] and r["cosine"] >= args.minimum_cosine and
                       r["relative_l2"] <= args.maximum_relative_l2 for r in results),
            thresholds=dict(minimum_cosine=args.minimum_cosine, maximum_relative_l2=args.maximum_relative_l2)))
    print(json.dumps(results, indent=2))
    if not all(r["finite"] and r["cosine"] >= args.minimum_cosine and
               r["relative_l2"] <= args.maximum_relative_l2 for r in results):
        raise ValueError("component parity thresholds failed")


def render(args):
    import numpy as np
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file
    source = load_file(str(args.input))["pixels"].float().numpy()[0].transpose(1, 2, 3, 0)
    restored = load_file(str(args.restoration))["pixels"].float().numpy()[0].transpose(1, 2, 3, 0)
    if source.shape != restored.shape or not np.isfinite(restored).all():
        raise ValueError("matching finite restored RGB frames required")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = sorted({0, len(source) // 2, len(source) - 1})
    h, w = source.shape[1:3]
    sheet = Image.new("RGB", (w * 2, (h + 28) * len(frames)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(frames):
        for column, pixels in enumerate((source[index], restored[index])):
            data = np.round(np.clip((pixels + 1) * .5, 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(data)
            if column:
                image.save(args.output / f"restored-{index}.png")
            else:
                image.save(args.output / f"input-{index}.png")
            sheet.paste(image, (column * w, row * (h + 28) + 28))
            label = f"Frame {index}: input (bicubic)" if column == 0 else f"Frame {index}: SeedVR2 (raw, no color correction)"
            draw.text((column * w + 8, row * (h + 28) + 8), label, fill="black")
    sheet.save(args.output / "comparison.png")
    print(args.output / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("prepare")
    fixture.add_argument("--image", type=Path, required=True)
    fixture.add_argument("--size", type=int, default=128)
    fixture.add_argument("--frames", type=int, default=1)
    fixture.add_argument("--source-size", type=int,
                         help="Downsample source before bicubic upscaling to --size; create a meaningful SR fixture.")
    fixture.add_argument("--seed", type=int, default=42)
    fixture.add_argument("--output", type=Path, required=True)
    reference = subparsers.add_parser("run")
    reference.add_argument("--stage", choices=("dit", "vae", "restore"), required=True)
    reference.add_argument("--comfy", type=Path, required=True)
    reference.add_argument("--comfy-extra", type=Path, required=True)
    reference.add_argument("--model", type=Path, required=True)
    reference.add_argument("--input", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--device", choices=("mps", "cpu"), default="mps")
    reference.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--minimum-cosine", type=float, default=.999)
    comparison.add_argument("--maximum-relative-l2", type=float, default=.05)
    comparison.add_argument("--files", default="*.safetensors", help="Declared tensor-file subset; defaults to every reference artifact.")
    rendering = subparsers.add_parser("render")
    rendering.add_argument("--input", type=Path, required=True)
    rendering.add_argument("--restoration", type=Path, required=True)
    rendering.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.size < 16 or args.size % 16 or args.frames < 1 or args.frames % 4 != 1:
            parser.error("size must be a positive multiple of sixteen, frames must be 4n+1")
        if args.source_size is not None and not 16 <= args.source_size <= args.size:
            parser.error("source-size must be between sixteen and the output size")
        prepare(args)
    elif args.command == "run":
        oracle(args)
    elif args.command == "compare":
        compare(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
