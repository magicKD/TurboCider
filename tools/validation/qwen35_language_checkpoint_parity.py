"""Official CPU oracle on downloaded Qwen3.5 PE weights and native saved outputs."""
import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding


def metric(native, reference):
    a, b = native.float(), reference.float()
    return dict(relative_rmse=float((a-b).square().mean().sqrt()/b.square().mean().sqrt()),
                max_abs=float((a-b).abs().max()),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = Qwen3_5TextConfig(**json.loads((args.weights / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5TextModel(config)
    model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    index = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    state, head = {}, None
    for shard in sorted(set(index.values())):
        with safe_open(args.weights / shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = weights.get_tensor(key)
                elif key == "lm_head.weight":
                    head = weights.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    model.eval()
    inputs = load_file(str(args.input))
    native = load_file(str(args.native))
    ids = inputs["ids"].long()
    reference = model(input_ids=ids, use_cache=False).last_hidden_state
    report = dict(scope="real BF16 PE checkpoint language component only; not prompt generation/vision acceptance",
                  token_count=ids.shape[1], hidden=metric(native["all"], reference))
    if ids.shape[1] > 1:
        prefill = model(input_ids=ids[:, :-1], use_cache=True)
        decode = model(input_ids=ids[:, -1:], past_key_values=prefill.past_key_values,
                       use_cache=True).last_hidden_state
        report["decode_hidden"] = metric(native["split"][:, -1:], decode)
    if head is not None:
        logits = torch.nn.functional.linear(reference, head)
        report["logits"] = metric(native["logits"], logits)
        report["last_argmax_equal"] = bool((native["logits"][:, -1].argmax(-1) == logits[:, -1].argmax(-1)).all())
        report["native_last_argmax"] = native["logits"][:, -1].argmax(-1).tolist()
        report["reference_last_argmax"] = logits[:, -1].argmax(-1).tolist()
    if "positions" in inputs:
        mrope = model(input_ids=ids, position_ids=inputs["positions"][:, None].long(), use_cache=False).last_hidden_state
        report["mrope_hidden"] = metric(native["mrope"], mrope)
    metrics = [v for v in report.values() if isinstance(v, dict) and "relative_rmse" in v]
    report["passed"] = all(v["relative_rmse"] < 0.02 and v["cosine"] > 0.999 for v in metrics) and report.get("last_argmax_equal", False)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PE language checkpoint parity thresholds not met")


if __name__ == "__main__":
    main()
