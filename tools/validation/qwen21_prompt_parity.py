"""Validate native real-prompt tokenizer and full 36-layer text conditioning."""
import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I.format(args.prompt), add_special_tokens=False).ids
    prefix = tokenizer.encode(Qwen21PromptEncoder.SYSTEM_PREFIX, add_special_tokens=False).ids
    native, metadata = mx.load(str(args.native), return_metadata=True)
    assert native["ids"].tolist() == [ids], "native tokenizer differs"
    assert int(metadata["system_prefix_tokens"]) == len(prefix), "system-prefix boundary differs"
    model = Qwen21TextEncoder()
    weights = mx.load(str(args.weights))
    params = []
    for key, _ in tree_flatten(model.parameters()):
        if key.endswith(".weight"):
            source = "model.language_model." + key if "model.language_model." + key in weights else "model." + key
            params.append((key, weights[source]))
    model.load_weights(params, strict=False)
    del weights, params
    output = model(mx.array([ids], dtype=mx.int32))[:, len(prefix):].astype(mx.float32)
    actual = native["text"].astype(mx.float32)
    delta = output - actual
    rmse = mx.sqrt(mx.mean(delta * delta) / mx.mean(output * output)).item()
    cosine = (mx.sum(actual * output) / mx.sqrt(mx.sum(actual * actual) * mx.sum(output * output))).item()
    print(json.dumps({"tokens": len(ids), "drop": len(prefix), "relative_rmse": rmse,
                      "cosine": cosine, "max_abs": mx.max(mx.abs(delta)).item()}), flush=True)
    assert rmse < 0.02 and cosine > 0.999, "real text encoder parity gate failed"


if __name__ == "__main__":
    main()
