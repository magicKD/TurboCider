"""Convert Qwen3-4B for TurboCider, retaining dense embeddings and norm weights.

Example: Python/bin/python3 tools/convert/qwen3_affine.py --source MODEL_ROOT
  --weights MODEL_ROOT/split_files/text_encoders/qwen_3_4b.safetensors
  --bits 4 --output models/qwen3-4b-tc-q4

Output is a reusable text_encoder/ + tokenizer/ component, not a full image model.
Does not accept GGUF or Comfy FP4/FP8 mixed weights. Never overwrites a directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


def convert(source, weights, output, bits):
    import mlx.core as mx

    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    tokenizer = source / "tokenizer"
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (tokenizer / name).is_file():
            raise ValueError(f"Missing tokenizer: {name}")
    tensors = mx.load(str(weights))
    if tensors["model.embed_tokens.weight"].shape != (151936, 2560):
        raise ValueError("Expected Qwen3-4B dense embedding")
    if "model.layers.35.input_layernorm.weight" not in tensors:
        raise ValueError("Expected 36 Qwen3 layers")
    if any(v.dtype not in (mx.bfloat16, mx.float16, mx.float32) for v in tensors.values()):
        raise ValueError("Input must be dense floating-point safetensors")
    started = time.perf_counter()
    result = {}
    for name, value in tensors.items():
        # The conditioning implementation gathers dense embedding rows. Keep
        # the embedding unquantized; skip unused output head to save storage.
        if name.startswith("lm_head."):
            continue
        value = value.astype(mx.bfloat16)
        if name.startswith("model.layers.") and name.endswith(".weight") and value.ndim == 2:
            packed, scales, biases = mx.quantize(value, group_size=32, bits=bits)
            prefix = name[:-7]
            result[name] = packed
            result[prefix + ".scales"] = scales
            result[prefix + ".biases"] = biases
            mx.eval(packed, scales, biases)
        else:
            result[name] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-convert-", dir=output.parent) as temp:
        staging = Path(temp) / "component"
        encoder = staging / "text_encoder"
        encoder.mkdir(parents=True)
        mx.save_safetensors(str(encoder / "model.safetensors"), result)
        config = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
                  "vocab_size": 151936, "num_attention_heads": 32, "num_key_value_heads": 8,
                  "quantization": {"bits": bits, "group_size": 32, "mode": "affine"},
                  "turbocider_dense_embedding": True}
        (encoder / "config.json").write_text(json.dumps(config, indent=2))
        shutil.copytree(tokenizer, staging / "tokenizer")
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        record = {"schema": "tc-qwen3-affine-v1", "source_sha256": digest,
                  "mlx": mx.__version__, "bits": bits, "group_size": 32,
                  "dense_embedding": True, "seconds": time.perf_counter() - started,
                  "weight_bytes": (encoder / "model.safetensors").stat().st_size}
        (staging / "conversion.json").write_text(json.dumps(record, indent=2))
        staging.rename(output)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, choices=(4, 8), required=True)
    a = p.parse_args()
    convert(a.source.resolve(), a.weights.resolve(), a.output.resolve(), a.bits)
