#!/usr/bin/env python3
"""Compare TurboCider's streamed Qwen3-VL conditioner with FastVideo."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))
from run_fastvideo_mlx_converter import install_fastvideo_namespace  # noqa: E402


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = reference.astype(np.float64).reshape(-1)
    right = candidate.astype(np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    nonfinite_equal = bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    if not nonfinite_equal:
        mismatch = np.flatnonzero(
            (np.isnan(left) != np.isnan(right))
            | (np.isposinf(left) != np.isposinf(right))
            | (np.isneginf(left) != np.isneginf(right))
        )
        maximum = int(mismatch[0])
        return {
            "max_abs": float("inf"),
            "max_flat_index": maximum,
            "reference_at_max": float(left[maximum]),
            "candidate_at_max": float(right[maximum]),
            "count_over_1e_4": int(mismatch.size),
            "count_over_1e_3": int(mismatch.size),
            "mean_abs": float("inf"),
            "rmse": float("inf"),
            "cosine": 0.0,
            "nonfinite_equal": False,
        }
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size == 0:
        return {
            "max_abs": 0.0,
            "max_flat_index": 0,
            "reference_at_max": float(left[0]) if left.size else 0.0,
            "candidate_at_max": float(right[0]) if right.size else 0.0,
            "count_over_1e_4": 0,
            "count_over_1e_3": 0,
            "mean_abs": 0.0,
            "rmse": 0.0,
            "cosine": 1.0,
            "nonfinite_equal": True,
        }
    finite_left = left[finite]
    finite_right = right[finite]
    difference = finite_left - finite_right
    absolute = np.abs(difference)
    local_maximum = int(np.argmax(absolute))
    maximum = int(finite_indices[local_maximum])
    denominator = np.linalg.norm(finite_left) * np.linalg.norm(finite_right)
    return {
        "max_abs": float(absolute[local_maximum]),
        "max_flat_index": maximum,
        "reference_at_max": float(left[maximum]),
        "candidate_at_max": float(right[maximum]),
        "count_over_1e_4": int(np.count_nonzero(absolute > 1e-4)),
        "count_over_1e_3": int(np.count_nonzero(absolute > 1e-3)),
        "mean_abs": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(finite_left, finite_right) / denominator) if denominator else 1.0,
        "nonfinite_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--prompt", default="A red fox runs through fresh snow.")
    # The native and FastVideo implementations use the same MLX GPU kernels,
    # but their streamed layer/reduction scheduling can differ slightly.  The
    # production contract therefore combines a bounded absolute error with a
    # near-unit cosine and exact token tags instead of claiming bit parity.
    parser.add_argument("--max-abs", type=float, default=1e-3)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--intermediates", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    # The requested H3 asset policy is ModelScope-only.  Transformers is used
    # solely as the local tokenizer API and must never attempt hub access.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    install_fastvideo_namespace()
    import mlx.core as mx
    from fastvideo.mlx_runtime.minimax_h3_conditioner import (
        StreamedMiniMaxH3TextConditioner,
    )

    component = args.component.resolve()
    tokenizer = args.tokenizer.resolve()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-conditioner-") as directory:
        output = Path(directory) / "native.safetensors"
        native_command = [str(args.probe.resolve()), str(component), str(tokenizer),
                          str(output), args.prompt]
        if args.per_layer:
            native_command.append("--layers")
        if args.intermediates:
            native_command.append("--intermediates")
        native_process = subprocess.run(
            native_command,
            check=True,
            text=True,
            capture_output=True,
        )
        native = mx.load(str(output))
        mx.eval(native["hidden_states"], native["token_tags"])

        reference_started = time.perf_counter()
        reference_conditioner = StreamedMiniMaxH3TextConditioner(component, tokenizer)
        layer_reports = []
        intermediate_reports = {}
        if args.per_layer or args.intermediates:
            from fastvideo.mlx_runtime.minimax_h3_conditioner import (
                TEXT_ENCODER_LAYER,
                _apply_mrope,
                _linear,
                _mrope_cos_sin,
                _rms_norm,
            )
            token_ids = reference_conditioner.tokenize(args.prompt)
            positions = np.stack([
                np.arange(len(token_ids), dtype=np.float64),
                np.arange(len(token_ids), dtype=np.float64),
                np.arange(len(token_ids), dtype=np.float64),
            ])
            cosine, sine = _mrope_cos_sin(positions, reference_conditioner.config)
            rows = [reference_conditioner.index.get_row(
                "model.language_model.embed_tokens.weight", token)
                    for token in token_ids]
            reference_value = mx.array(np.stack(rows).astype(np.float32))
            del rows
            first_layer = 0
            if args.intermediates:
                cfg = reference_conditioner.config
                prefix = "model.language_model.layers.0."

                def w(name):
                    return reference_conditioner.index.get_mlx(prefix + name)

                def record(name, value):
                    mx.eval(value)
                    intermediate_reports[name] = metrics(
                        np.asarray(value).astype(np.float32),
                        np.asarray(native[f"debug_{name}"]).astype(np.float32),
                    )

                record("embedding", reference_value)
                record("mrope_cosine", cosine)
                record("mrope_sine", sine)
                residual = reference_value
                normed = _rms_norm(reference_value, w("input_layernorm.weight"),
                                   cfg.rms_norm_eps)
                record("layer0_input", reference_value)
                record("layer0_input_norm", normed)
                query = _linear(normed, w("self_attn.q_proj.weight")).reshape(
                    len(token_ids), cfg.num_attention_heads, cfg.head_dim)
                key = _linear(normed, w("self_attn.k_proj.weight")).reshape(
                    len(token_ids), cfg.num_key_value_heads, cfg.head_dim)
                value = _linear(normed, w("self_attn.v_proj.weight")).reshape(
                    len(token_ids), cfg.num_key_value_heads, cfg.head_dim)
                record("layer0_q_proj", query)
                record("layer0_k_proj", key)
                record("layer0_v_proj", value)
                query = _rms_norm(query, w("self_attn.q_norm.weight"),
                                  cfg.rms_norm_eps)
                key = _rms_norm(key, w("self_attn.k_norm.weight"),
                                cfg.rms_norm_eps)
                record("layer0_q_norm", query)
                record("layer0_k_norm", key)
                query = _apply_mrope(query, cosine, sine)
                key = _apply_mrope(key, cosine, sine)
                record("layer0_q_rope", query)
                record("layer0_k_rope", key)
                repeats = cfg.num_attention_heads // cfg.num_key_value_heads
                key = mx.repeat(key, repeats, axis=1)
                value = mx.repeat(value, repeats, axis=1)
                record("layer0_k_repeat", key)
                record("layer0_v_repeat", value)
                scores = (query.transpose(1, 0, 2) @ key.transpose(1, 2, 0)) * cfg.head_dim**-0.5
                record("layer0_scores_unmasked", scores)
                mask = mx.triu(mx.ones((len(token_ids), len(token_ids)),
                                       dtype=mx.bool_), k=1)
                scores = mx.where(mask[None], mx.array(-np.inf, dtype=scores.dtype), scores)
                record("layer0_scores_masked", scores)
                probabilities = mx.softmax(scores, axis=-1)
                record("layer0_probabilities", probabilities)
                attended_heads = probabilities @ value.transpose(1, 0, 2)
                record("layer0_attended_heads", attended_heads)
                attended = attended_heads.transpose(1, 0, 2).reshape(len(token_ids), -1)
                record("layer0_attended", attended)
                attention_output = _linear(attended, w("self_attn.o_proj.weight"))
                record("layer0_attention_output", attention_output)
                reference_value = residual + attention_output
                record("layer0_post_attention", reference_value)
                residual = reference_value
                normed = _rms_norm(reference_value,
                                   w("post_attention_layernorm.weight"),
                                   cfg.rms_norm_eps)
                record("layer0_post_attention_norm", normed)
                gate = _linear(normed, w("mlp.gate_proj.weight"))
                up = _linear(normed, w("mlp.up_proj.weight"))
                activated = gate * mx.sigmoid(gate) * up
                record("layer0_gate", gate)
                record("layer0_up", up)
                record("layer0_activated", activated)
                down = _linear(activated, w("mlp.down_proj.weight"))
                record("layer0_down", down)
                reference_value = residual + down
                record("layer0_output", reference_value)
                if args.per_layer:
                    layer_reports.append(intermediate_reports["layer0_output"])
                first_layer = 1
            for layer in range(first_layer, TEXT_ENCODER_LAYER):
                reference_value = reference_conditioner._decoder_layer(
                    layer, reference_value, cosine, sine)
                mx.eval(reference_value)
                gc.collect()
                if args.per_layer:
                    layer_reports.append(metrics(
                        np.asarray(reference_value).astype(np.float32),
                        np.asarray(native[f"layer{layer}"]).astype(np.float32),
                    ))
            reference_hidden = np.asarray(reference_value).astype(np.float32)
            reference_tags = np.full((len(token_ids),), 1, dtype=np.int64)
        else:
            reference_hidden, reference_tags = reference_conditioner.encode_prompt(args.prompt)
        reference_seconds = time.perf_counter() - reference_started
        reference_conditioner.close()

        candidate_hidden = np.asarray(native["hidden_states"])
        candidate_tags = np.asarray(native["token_tags"])
        report = {
            "native": json.loads(native_process.stdout),
            "reference_seconds": reference_seconds,
            "hidden_states": metrics(reference_hidden, candidate_hidden),
            "shape": list(candidate_hidden.shape),
            "tags_equal": bool(np.array_equal(reference_tags, candidate_tags)),
        }
        if args.per_layer:
            report["first_nonzero_layer"] = next(
                (index for index, value in enumerate(layer_reports)
                 if value["max_abs"] != 0), None)
            report["layer_max_abs"] = max(value["max_abs"] for value in layer_reports)
            if not args.summary_only:
                report["layers"] = layer_reports
        if args.intermediates:
            report["first_nonzero_intermediate"] = next(
                (name for name, value in intermediate_reports.items()
                 if value["max_abs"] != 0), None)
            report["intermediate_max_abs"] = max(
                value["max_abs"] for value in intermediate_reports.values())
            if not args.summary_only:
                report["intermediates"] = intermediate_reports
        print(json.dumps(report, indent=2, sort_keys=True))
        result = report["hidden_states"]
        if (not report["tags_equal"] or
                result["max_abs"] > args.max_abs or
                result["cosine"] < args.min_cosine):
            raise SystemExit(
                "FAIL: conditioner "
                f"tags_equal={report['tags_equal']} "
                f"max_abs={result['max_abs']:.6g} "
                f"cosine={result['cosine']:.9f}"
            )
        print("PASS: TurboCider H3 conditioner matches FastVideo")


if __name__ == "__main__":
    main()
