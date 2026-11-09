#!/usr/bin/env python3
"""Compare one real LTX 0.9.8 BF16 block against independent PyTorch math.

This isolates block numerics, including image-conditioning token timesteps;
it is not a full-video quality or throughput test.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0, choices=range(28))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    checkpoint, probe, output = args.checkpoint.resolve(strict=True), args.probe.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    torch.manual_seed(42)
    dim, heads, rows, text_rows = 2048, 32, 12, 8
    dtype = torch.bfloat16
    prefix = f"model.diffusion_model.transformer_blocks.{args.block}."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        weights = {name[len(prefix):]: source.get_tensor(name).to(args.device)
                   for name in source.keys() if name.startswith(prefix)}
    if len(weights) != 25:
        raise ValueError(f"expected 25 block tensors, found {len(weights)}")
    hidden = (torch.randn((1, rows, dim), dtype=torch.float32) * .3).to(dtype)
    context = (torch.randn((1, text_rows, dim), dtype=torch.float32) * .3).to(dtype)
    # Two distinct timestep modulation rows, as in first-frame I2V, expanded
    # over their respective tokens. This catches scalar-only implementations.
    two_rows = (torch.randn((1, 2, 6 * dim), dtype=torch.float32) * .1).to(dtype)
    modulation = torch.cat((two_rows[:, :1].expand(1, 3, -1), two_rows[:, 1:].expand(1, rows - 3, -1)), dim=1).contiguous()
    angles = torch.randn((1, heads, rows, dim // heads // 2), dtype=torch.float32)
    cosine, sine = angles.cos().to(dtype), angles.sin().to(dtype)
    mask = torch.tensor([0, 0, 0, 0, 0, 0, -10000, -10000], dtype=dtype).reshape(1, 1, 1, text_rows)
    values = {"hidden": hidden, "context": context, "modulation": modulation,
              "cosine": cosine, "sine": sine, "text_mask": mask}
    gpu = {name: value.to(args.device) for name, value in values.items()}

    def linear(value, name):
        return F.linear(value, weights[name + ".weight"], weights[name + ".bias"])

    def norm(value):
        return F.rms_norm(value.float(), (dim,), eps=1e-6).to(dtype)

    def attention(x, context_input, name, apply_rope, text_mask):
        q = F.rms_norm(linear(x, name + ".to_q"), (dim,), weights[name + ".q_norm.weight"], 1e-5)
        k = F.rms_norm(linear(context_input, name + ".to_k"), (dim,), weights[name + ".k_norm.weight"], 1e-5)
        v = linear(context_input, name + ".to_v")
        q = q.reshape(1, q.shape[1], heads, dim // heads).transpose(1, 2)
        k = k.reshape(1, k.shape[1], heads, dim // heads).transpose(1, 2)
        v = v.reshape(1, v.shape[1], heads, dim // heads).transpose(1, 2)
        if apply_rope:
            def rotate(value):
                pairs = value.float().reshape(1, heads, rows, dim // heads // 2, 2)
                a, b = pairs.unbind(-1)
                c, s = gpu["cosine"].float(), gpu["sine"].float()
                return torch.stack((a * c - b * s, a * s + b * c), -1).reshape_as(value).to(dtype)
            q, k = rotate(q), rotate(k)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=text_mask)
        return linear(result.transpose(1, 2).reshape(1, x.shape[1], dim), name + ".to_out.0")

    started = time.perf_counter()
    with torch.inference_mode():
        x, txt = gpu["hidden"], gpu["context"]
        params = (gpu["modulation"].reshape(1, rows, 6, dim) +
                  weights["scale_shift_table"].reshape(1, 1, 6, dim)).unbind(2)
        self_input = norm(x) * (1 + params[1]) + params[0]
        x = x + attention(self_input, self_input, "attn1", True, None) * params[2]
        x = x + attention(x, txt, "attn2", False, gpu["text_mask"])
        feed = norm(x) * (1 + params[4]) + params[3]
        x = x + linear(F.gelu(linear(feed, "ff.net.0.proj"), approximate="tanh"), "ff.net.2") * params[5]
        reference = x.cpu().float().contiguous()
    reference_seconds = time.perf_counter() - started
    del weights, gpu, x, txt, params, self_input, feed
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    output.mkdir(parents=True)
    save_file(values, str(output / "input.safetensors"))
    save_file({"output": reference}, str(output / "reference.safetensors"))
    started = time.perf_counter()
    run = subprocess.run([str(probe), str(checkpoint), str(output / "input.safetensors"),
                          str(output / "native.safetensors"), str(args.block), str(heads)],
                         capture_output=True, text=True, timeout=300)
    native_seconds = time.perf_counter() - started
    if run.returncode != 0:
        raise RuntimeError(run.stdout + run.stderr)
    native_bf16 = load_file(str(output / "native.safetensors"))["output"]
    if native_bf16.dtype != dtype:
        raise ValueError(f"native block changed BF16 output dtype: {native_bf16.dtype}")
    native = native_bf16.float()
    delta = native - reference
    report = {"schema": "ltx098-real-block-parity-v1", "block": args.block,
              "reference_device": args.device, "rows": rows, "text_rows": text_rows,
              "relative_l2": (delta.norm() / reference.norm()).item(),
              "cosine": F.cosine_similarity(native.flatten(), reference.flatten(), dim=0).item(),
              "max_abs": delta.abs().max().item(), "all_finite": bool(torch.isfinite(native).all()),
              "reference_seconds_excluding_weight_load": reference_seconds,
              "native_process_seconds_including_weight_load": native_seconds,
              "probe_stdout": run.stdout, "probe_stderr": run.stderr}
    report["passed"] = report["all_finite"] and report["relative_l2"] < .04 and report["cosine"] > .999
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
