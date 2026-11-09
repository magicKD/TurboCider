#!/usr/bin/env python3
"""Compare the real LTX 0.9.8 T5-XXL encoder to Transformers BF16.

The reference and native probes execute sequentially to avoid retaining two
full 9.8 GB model copies on the GPU at the same time. This tests text encoder
numerics only; it does not establish whole-video quality or speed.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="directory containing the T5 safetensors")
    parser.add_argument("--tokenizer", type=Path, required=True, help="local T5 tokenizer directory")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--prompt", default="A lighthouse stands on dark rocks as waves roll gently beneath a warm evening sky.")
    args = parser.parse_args()
    weights = args.weights.resolve(strict=True)
    tokenizer_dir = args.tokenizer.resolve(strict=True)
    probe = args.probe.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tokenized = tokenizer(args.prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = tokenized.input_ids, tokenized.attention_mask
    valid = int(mask.sum().item())
    if valid <= 0 or valid > 128:
        raise ValueError("invalid T5 token count")
    files = sorted(weights.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("one T5 safetensors checkpoint is required")
    config = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240,
                      num_layers=24, num_heads=64, dropout_rate=0.0,
                      feed_forward_proj="gated-gelu")
    with torch.device("meta"):
        model = T5EncoderModel(config)
    state = load_file(str(files[0]), device="cpu")
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"unexpected T5 checkpoint keys: {incompatible}")
    del state
    model = model.to(device=args.device, dtype=torch.bfloat16).eval()
    layer_zero = []
    handle = model.encoder.block[0].register_forward_hook(
        lambda module, inputs, result: layer_zero.append(result[0].detach().cpu().float().contiguous()))
    started = time.perf_counter()
    with torch.inference_mode():
        reference = model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device)).last_hidden_state
        reference = reference.detach().cpu().float().contiguous()
    reference_seconds = time.perf_counter() - started
    handle.remove()
    del model, handle
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    output.mkdir(parents=True)
    save_file({"ids": ids[0].to(torch.int32).contiguous(), "valid": torch.tensor(valid, dtype=torch.int32)},
              str(output / "input.safetensors"))
    save_file({"block_0": layer_zero[0], "output": reference}, str(output / "reference.safetensors"))
    started = time.perf_counter()
    result = subprocess.run([str(probe), str(weights), str(output / "input.safetensors"),
                             str(output / "native.safetensors"), "--t5-xxl"],
                            capture_output=True, text=True, timeout=600)
    native_seconds = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    native = load_file(str(output / "native.safetensors"))
    metrics = {}
    for name, expected in (("block_0", layer_zero[0]), ("output", reference)):
        actual = native[name][:, :valid].float()
        expected = expected[:, :valid].float()
        diff = actual - expected
        metrics[name] = {"relative_l2": (diff.norm() / expected.norm()).item(),
                         "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
                         "max_abs": diff.abs().max().item(), "all_finite": bool(torch.isfinite(actual).all())}
    report = {"schema": "ltx098-real-t5-parity-v1", "prompt": args.prompt,
              "reference_device": args.device, "length": int(ids.shape[1]), "valid_tokens": valid,
              "metrics": metrics, "reference_seconds_excluding_load": reference_seconds,
              "native_process_seconds_including_load": native_seconds, "probe_stdout": result.stdout,
              "probe_stderr": result.stderr}
    report["passed"] = all(value["all_finite"] and value["relative_l2"] < .05 and value["cosine"] > .998
                           for value in metrics.values())
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
