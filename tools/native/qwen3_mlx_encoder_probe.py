#!/usr/bin/env python3
"""Reference MLX Qwen3 encoder probe used by the native prefill sweep.

The native encoder intentionally exposes intermediate residual taps rather than
the language-model head.  This probe follows the same contract using the
upstream ``mlx_lm`` Qwen3 module and the *same* safetensors directory.  It is
kept as a separate process so loading and peak-memory accounting cannot alter
the native probe's samples.

This is a benchmark/reference adapter, not a production text-generation
entrypoint.  It accepts deterministic token IDs so tokenization and prompt
formatting are outside the timing comparison.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path


OUTPUT_LAYERS = {
    "flux_klein": (8, 17, 26),
    "z_image": (34,),
}
MODE_DEFAULTS = {
    "flux_klein": {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "rope_scaling": None,
    },
    "z_image": {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
        "rope_scaling": None,
    },
}


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def qwen3_eval_interval(token_count: int) -> int:
    if os.environ.get("TURBOCIDER_QWEN3_DEFER_LAYER_EVAL"):
        return 0
    setting = os.environ.get("TURBOCIDER_QWEN3_EVAL_INTERVAL")
    if setting is None:
        return max(1, min(4, 1024 // token_count))
    try:
        interval = int(setting)
    except ValueError as error:
        raise ValueError("invalid TURBOCIDER_QWEN3_EVAL_INTERVAL") from error
    if interval < 0:
        raise ValueError("TURBOCIDER_QWEN3_EVAL_INTERVAL must be nonnegative")
    return interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("tokens", type=int)
    parser.add_argument("runs", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=tuple(OUTPUT_LAYERS))
    args = parser.parse_args()
    if args.tokens < 1 or args.tokens > 2048:
        parser.error("tokens must be in 1...2048")
    if args.runs < 1 or args.runs > 50:
        parser.error("runs must be in 1...50")
    return args


def model_location(path: Path):
    path = path.resolve()
    if path.is_dir():
        return path, None
    if path.is_file() and path.name.endswith(".safetensors.index.json"):
        return path.parent, None
    if path.is_file() and path.suffix == ".safetensors":
        return path.parent, path
    raise ValueError(f"weights must be a Qwen3 directory or safetensors file: {path}")


def load_reference_model(model_path: Path, mode: str, exact_weight: Path | None):
    # Import only after argument/path validation.  This keeps the failure mode
    # useful on hosts where MLX is not installed or Metal is unavailable.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3 import Model, ModelArgs

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Qwen3 config.json missing under {model_path}")
    config = json.loads(config_path.read_text())
    defaults = MODE_DEFAULTS[mode]
    for key, value in defaults.items():
        config.setdefault(key, value)
    for key in (
        "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "rms_norm_eps", "vocab_size", "max_position_embeddings", "rope_theta",
        "tie_word_embeddings",
    ):
        if config[key] != defaults[key]:
            raise ValueError(
                f"Qwen3 config {key}={config[key]!r} does not match {mode} "
                f"geometry {defaults[key]!r}"
            )
    if config["num_hidden_layers"] <= OUTPUT_LAYERS[mode][-1]:
        raise ValueError(f"Qwen3 checkpoint has too few layers for {mode}")
    args = ModelArgs.from_dict(config)
    model = Model(args)

    weight_files = [exact_weight] if exact_weight else sorted(
        model_path.glob("model*.safetensors")
    )
    if not weight_files:
        raise ValueError(f"no model*.safetensors files under {model_path}")
    weights = {}
    for weight_file in weight_files:
        weights.update(mx.load(str(weight_file)))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    if quantization := config.get("quantization"):
        required = ("group_size", "bits")
        if any(name not in quantization for name in required):
            raise ValueError("Qwen3 quantization config is missing bits/group_size")

        def class_predicate(path, module):
            return hasattr(module, "to_quantized") and f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    # Materialize the resident checkpoint before timing the first encoder pass;
    # this mirrors native Weights::materialize() and avoids charging lazy weight
    # faults to the first encoder sample.
    mx.eval(model.parameters())
    mx.synchronize()
    return mx, model, config, weight_files


def conditioning(mx, model, mode: str, token_count: int):
    """Run Qwen3 blocks through the consumer's final hidden-state tap."""

    ids = mx.array(
        [[100 + (index * 7919) % 100000 for index in range(token_count)]],
        dtype=mx.int32,
    )
    is_z_image = mode == "z_image"
    x = model.model.embed_tokens(ids)
    # TurboCider's Z-Image contract keeps residual accumulation in FP32 and
    # casts the tapped result back to BF16.  FLUX/Klein remains in checkpoint
    # activation dtype (normally BF16).
    if is_z_image:
        x = x.astype(mx.float32)

    mask = "causal" if token_count > 1 else None
    output_layers = OUTPUT_LAYERS[mode]
    eval_interval = qwen3_eval_interval(token_count)
    outputs = []
    for layer_index, block in enumerate(model.model.layers):
        if layer_index > output_layers[-1]:
            break

        normalized = block.input_layernorm(x)
        attention = block.self_attn
        queries = attention.q_proj(normalized)
        keys = attention.k_proj(normalized)
        values = attention.v_proj(normalized)

        batch, length, _ = queries.shape
        queries = attention.q_norm(
            queries.reshape(batch, length, attention.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = attention.k_norm(
            keys.reshape(batch, length, attention.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, attention.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries = attention.rope(queries)
        keys = attention.rope(keys)

        # Native Qwen3 expands GQA below 128 rows and leaves MLX's grouped
        # attention path intact for longer prefills.  Keep that geometry here
        # so the reference observes the same dispatch boundary.
        if token_count < 128:
            repeats = attention.n_heads // attention.n_kv_heads
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)

        # Native attend(..., fp32=true) promotes SDPA inputs, then casts the
        # result back to the query dtype.  Explicitly preserve that boundary
        # instead of relying on the MLX primitive's dtype heuristic.
        query_dtype = queries.dtype
        attended = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32),
            keys.astype(mx.float32),
            values.astype(mx.float32),
            scale=attention.scale,
            mask=mask,
        ).astype(query_dtype)
        # The transposed attention result is [B, L, H, D].  Flattening the
        # final two dimensions is the same layout consumed by o_proj; avoid
        # deriving an integer head dimension from the floating-point scale.
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        x = x + attention.o_proj(attended)

        normalized = block.post_attention_layernorm(x)
        x = x + block.mlp(normalized)
        if eval_interval > 0 and (layer_index + 1) % eval_interval == 0:
            mx.eval(x)
        if layer_index in output_layers:
            outputs.append(x)

    result = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=-1)
    if is_z_image:
        result = result.astype(mx.bfloat16)
    return result


def main() -> int:
    args = parse_args()
    model_path, exact_weight = model_location(args.weights)
    start = time.perf_counter()
    mx, model, config, weight_files = load_reference_model(
        model_path, args.mode, exact_weight
    )
    load_seconds = time.perf_counter() - start

    samples = []
    result = None
    for index in range(args.runs + 1):
        mx.reset_peak_memory()
        started = time.perf_counter()
        result = conditioning(mx, model, args.mode, args.tokens)
        mx.eval(result)
        mx.synchronize()
        seconds = time.perf_counter() - started
        samples.append(
            {
                "warmup": index == 0,
                "seconds": seconds,
                "mlx_peak_bytes": mx.get_peak_memory(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), {"conditioning": result})
    warm = [sample["seconds"] for sample in samples if not sample["warmup"]]
    report = {
        "backend": "mlx_reference",
        "mode": args.mode,
        "tokens": args.tokens,
        "runs": args.runs,
        "model_path": str(model_path),
        "weight_files": [str(path) for path in weight_files],
        "model_type": config.get("model_type"),
        "output_layers": list(OUTPUT_LAYERS[args.mode]),
        "token_pattern": "100 + (index * 7919) % 100000",
        "attention_compute_dtype": "float32",
        "residual_compute_dtype": "float32" if args.mode == "z_image" else "checkpoint",
        "conditioning_dtype": str(result.dtype),
        "eval_interval": qwen3_eval_interval(args.tokens),
        "mlx_version": package_version("mlx"),
        "mlx_lm_version": package_version("mlx-lm"),
        "load_seconds": load_seconds,
        "first_encoder_seconds": samples[0]["seconds"],
        "warm_seconds": warm,
        "warm_median_seconds": statistics.median(warm),
        "samples": samples,
        "output": str(args.output),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"qwen3 MLX reference probe failed: {error}", file=sys.stderr)
        raise
