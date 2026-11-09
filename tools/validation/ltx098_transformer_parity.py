#!/usr/bin/env python3
"""Compare the real LTX 0.9.8 transformer against stock ComfyUI on MPS.

The models run sequentially on a small synthetic latent and T5 embedding.
This validates one denoiser evaluation, not the full generation pipeline.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=(1, 28), default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()
    checkpoint, comfy, probe = (path.resolve(strict=True) for path in (args.checkpoint, args.comfy, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")
    sys.path.insert(0, str(comfy))
    sys.argv = sys.argv[:1]
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import comfy.ops
    from comfy.ldm.lightricks.model import LTXVModel

    prefix = "model.diffusion_model."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        config = json.loads(source.metadata()["config"])["transformer"]
        state = {}
        for name in source.keys():
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            if key.startswith("transformer_blocks.") and int(key.split(".")[1]) >= args.blocks:
                continue
            state[key] = source.get_tensor(name)
    config["num_layers"] = args.blocks
    with torch.device("meta"):
        model = LTXVModel(**config, dtype=torch.bfloat16, device="meta", operations=comfy.ops.disable_weight_init)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device="mps", dtype=torch.bfloat16).eval()
    torch.manual_seed(101)
    frames, height, width, text_rows = 2, 2, 2, 8
    latents = (torch.randn((1, frames, height, width, 128), dtype=torch.float32) * .3).to(torch.bfloat16)
    text = (torch.randn((1, text_rows, 4096), dtype=torch.float32) * .3).to(torch.bfloat16)
    text_valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
    additive = torch.tensor([0, 0, 0, 0, 0, 0, -10000, -10000], dtype=torch.bfloat16).reshape(1, 1, 1, text_rows)
    timestep = torch.full((1, frames * height * width, 1), .7, dtype=torch.float32, device="mps")
    timestep[:, :height * width] = 0
    started = time.perf_counter()
    with torch.inference_mode():
        result = model(latents.permute(0, 4, 1, 2, 3).to("mps"), timestep,
                       text.to("mps"), text_valid.to("mps"), frame_rate=24)
        reference = result.permute(0, 2, 3, 4, 1).contiguous().cpu().float()
    reference_seconds = time.perf_counter() - started
    del model, result, timestep
    gc.collect()
    torch.mps.empty_cache()
    output.mkdir(parents=True)
    save_file({"latents": latents, "text": text, "text_mask": additive}, str(output / "input.safetensors"))
    save_file({"output": reference}, str(output / "reference.safetensors"))
    started = time.perf_counter()
    run = subprocess.run([str(probe), str(checkpoint), str(output / "input.safetensors"),
                          str(output / "native.safetensors"), str(args.blocks), str(args.eval_every)],
                         capture_output=True, text=True, timeout=600)
    native_seconds = time.perf_counter() - started
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    native_bf16 = load_file(str(output / "native.safetensors"))["output"]
    if native_bf16.dtype != torch.bfloat16:
        raise ValueError(f"native transformer changed BF16 output dtype: {native_bf16.dtype}")
    native = native_bf16.float()
    if native.shape != reference.shape:
        raise ValueError(f"output shape mismatch: {native.shape} != {reference.shape}")
    delta = native - reference
    report = {"schema": "ltx098-transformer-parity-v1", "blocks": args.blocks,
              "latent_shape": list(latents.shape), "text_rows": text_rows,
              "relative_l2": (delta.norm() / reference.norm()).item(),
              "cosine": F.cosine_similarity(native.flatten(), reference.flatten(), dim=0).item(),
              "max_abs": delta.abs().max().item(), "all_finite": bool(torch.isfinite(native).all()),
              "reference_seconds_excluding_load": reference_seconds,
              "native_process_seconds_including_load": native_seconds,
              "probe_stdout": run.stdout, "probe_stderr": run.stderr}
    report["passed"] = report["all_finite"] and report["relative_l2"] < .05 and report["cosine"] > .998
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
