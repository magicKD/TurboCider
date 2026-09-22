"""CPU-only Qwen3.5 language-layer oracle. No production Python dependency."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(420)
    reports = []
    with tempfile.TemporaryDirectory(prefix="tc-qwen35-language-") as temp:
        root = Path(temp)
        for dtype in (torch.float32, torch.bfloat16):
            for types in (["linear_attention"], ["full_attention"],
                          ["linear_attention"] * 3 + ["full_attention"]):
                config = Qwen3_5TextConfig(
                    vocab_size=64, hidden_size=32, intermediate_size=80,
                    num_hidden_layers=len(types), num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                    layer_types=types, attn_implementation="eager",
                    rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
                                     "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]})
                model = Qwen3_5TextModel(config).eval().to(dtype)
                # Nontrivial learned norm/decay parameters test more than the
                # all-zero/unit initialization of a freshly constructed model.
                for name, param in model.named_parameters():
                    if name.endswith("A_log"):
                        param.copy_(torch.linspace(-1, 0.5, param.numel()).reshape_as(param))
                    elif "norm" in name:
                        param.add_(torch.randn_like(param) * 0.02)
                save_file({"model.language_model." + k: v.contiguous() for k, v in model.state_dict().items()},
                          str(root / "weights.safetensors"))
                for length in (1, 9, 67):
                    ids = torch.randint(0, 64, (1, length))
                    positions = torch.stack([torch.arange(length), torch.arange(length) // 2,
                                             torch.arange(length) % 4]).to(torch.int32)
                    expected = model(input_ids=ids, use_cache=False).last_hidden_state
                    mrope = model(input_ids=ids, position_ids=positions[:, None].long(), use_cache=False).last_hidden_state
                    if length > 1:
                        # A deliberately all-linear fixture has no attention
                        # cache from which HF can infer absolute positions.
                        extra = {"attention_mask": {"linear_attention": None, "full_attention": None}} if "full_attention" not in types else {}
                        prefill = model(input_ids=ids[:, :-1], use_cache=True,
                                        position_ids=torch.arange(length-1)[None], **extra)
                        decoded = model(input_ids=ids[:, -1:], past_key_values=prefill.past_key_values,
                                        position_ids=torch.tensor([[length-1]]), use_cache=True, **extra).last_hidden_state
                        split = torch.cat([prefill.last_hidden_state, decoded], 1)
                    else:
                        split = expected
                    metadata = dict(hidden="32", layers=str(len(types)), heads="4", kv_heads="2", head_dim="16",
                                    rotary_dim="8", key_heads="2", value_heads="4", key_dim="8", value_dim="8",
                                    conv="4", section0="2", section1="1", section2="1",
                                    full_attention="".join("1" if t == "full_attention" else "0" for t in types))
                    save_file({"ids": ids.int(), "positions": positions}, str(root / "inputs.safetensors"), metadata)
                    proc = subprocess.run([str(args.probe.resolve()), str(root / "weights.safetensors"),
                                           str(root / "inputs.safetensors"), str(root / "outputs.safetensors")],
                                          capture_output=True, text=True)
                    assert proc.returncode == 0, proc.stderr
                    result = load_file(str(root / "outputs.safetensors"))
                    tol = 2e-4 if dtype == torch.float32 else 0.04
                    torch.testing.assert_close(result["all"], expected, atol=tol, rtol=tol)
                    torch.testing.assert_close(result.get("split", result["all"]), split, atol=tol, rtol=tol)
                    torch.testing.assert_close(result["mrope"], mrope, atol=tol, rtol=tol)
                    torch.testing.assert_close(result["reset"], result["all"], atol=0, rtol=0)
                    reports.append(dict(dtype=str(dtype), layers=types, length=length,
                                        max_abs=float((result["all"].float()-expected.float()).abs().max()),
                                        rmse=float((result["all"].float()-expected.float()).square().mean().sqrt())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True, cases=reports,
        scope="small random-weight CPU language parity; not full PE generation or actual-checkpoint parity"), indent=2) + "\n")
    print(f"PASS {len(reports)} Qwen35 language cases: linear/full/mixed, cached decode, mRoPE, reset, cancellation")


if __name__ == "__main__":
    main()
