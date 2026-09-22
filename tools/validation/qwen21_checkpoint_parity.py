"""Compare native DiT outputs against mflux using the same Comfy BF16 tensors.

This is a developer oracle with synthetic conditioning, not a generation test.
Run the native transformer probe first, then this script to avoid overlapping
GPU benchmarks. The mflux checkpoint mapping splits Comfy's fused gate/up rows.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    ModelConfig.precision = mx.bfloat16
    model = Qwen21Transformer()
    original = mx.load(str(args.weights))
    params = []
    for key, tensor in original.items():
        if key.endswith(".img_mlp.gate_up.weight"):
            gate, up = mx.split(tensor, 2, axis=0)
            params.extend([(key.replace("gate_up", "gate_layer"), gate),
                           (key.replace("gate_up", "proj"), up)])
        else:
            params.append((key.replace("modulation.1", "modulation.layers.1"), tensor))
    model.load_weights(params, strict=False)
    mx.eval(model.parameters())
    del original, params
    inputs, metadata = mx.load(str(args.inputs), return_metadata=True)
    native = mx.load(str(args.native))
    height, width = int(metadata["height"]), int(metadata["width"])
    cos, sin = model.pos_embed(inputs["text"].shape[1], height, width)
    if args.trace:
        from mlx import nn
        trace = mx.load(str(args.trace))
        def compare(name, value):
            mx.eval(value)
            ref, actual = value.astype(mx.float32), trace[name].astype(mx.float32)
            delta = ref - actual
            print(json.dumps({"stage": name, "max_abs": mx.max(mx.abs(delta)).item(),
                              "relative_rmse": mx.sqrt(mx.mean(delta * delta) / mx.maximum(mx.mean(ref * ref), 1e-20)).item()}), flush=True)
        text_len = inputs["text"].shape[1]
        temb = model.time_text_embed(mx.array([0.8, 0.0]))
        compare("temb", temb[:1])
        modulation = model.modulation(temb)
        compare("modulation", modulation[:1])
        mods = [model._select_modulation_rows(v, text_len, height * width) for v in mx.split(modulation, 4, axis=-1)]
        for i, mod in enumerate(mods):
            compare(f"mod{i}", mod)
        compare("cosine", cos)
        compare("sine", sin)
        hidden = mx.concatenate([model.txt_in(inputs["text"]), model.img_in(inputs["latents"])], axis=1)
        compare("input", hidden)
        m1, m2 = mx.concatenate(mods[:2], axis=-1), mx.concatenate(mods[2:], axis=-1)
        block = model.transformer_blocks[0]
        attention_input = block.img_norm1(hidden) * (1 + mods[0])
        compare("attention_input", attention_input)
        attn = block.attn
        q, k, v = [mx.reshape(projection(attention_input), (1, text_len + height * width, 32, 128))
                   for projection in (attn.to_q, attn.to_k, attn.to_v)]
        q = mx.transpose(attn._apply_rope(attn.norm_q(q), cos, sin), (0, 2, 1, 3))
        k = mx.transpose(attn._apply_rope(attn.norm_k(k), cos, sin), (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        for name, value in (("q", q), ("k", k), ("v", v)):
            compare(name, value)
        text_out = mx.fast.scaled_dot_product_attention(q[:, :, :text_len], k[:, :, :text_len], v[:, :, :text_len], scale=128**-0.5, mask="causal")
        image_out = mx.fast.scaled_dot_product_attention(q[:, :, text_len:], k, v, scale=128**-0.5)
        attention = mx.transpose(mx.concatenate([text_out, image_out], axis=2), (0, 2, 1, 3)).reshape(hidden.shape)
        compare("attention", attention)
        projected = attn.to_out[0](attention)
        compare("projected", projected)
        after_attention = hidden + mx.tanh(mods[1]) * projected
        compare("after_attention", after_attention)
        mlp_input = block.img_norm2(after_attention) * (1 + mods[2])
        compare("mlp_input", mlp_input)
        compare("ff", nn.silu(block.img_mlp.gate_layer(mlp_input)) * block.img_mlp.proj(mlp_input))
        for i, block in enumerate(model.transformer_blocks):
            hidden = mx.compile(lambda x: block(x, m1, m2, cos, sin, None, text_len))(hidden)
            compare(f"block{i}", hidden)
        scale = model._select_modulation_rows(model.norm_out.linear(nn.silu(temb)), text_len, height * width)
        output = model.proj_out(model.norm_out(hidden, scale))[:, text_len:]
        compare("output", output)
        return
    compiled = mx.compile(model._forward)
    comparisons = {}
    for name, text, timestep in [("first", inputs["text"], 0.8), ("cached", inputs["text"], 0.3),
                                 ("uncached", inputs["text"], 0.3), ("changed", inputs["text_changed"], 0.3)]:
        expected = compiled(inputs["latents"], text, mx.array([timestep, 0.0]), cos, sin, None)
        mx.eval(expected)
        ref = expected.astype(mx.float32)
        actual = native[name].astype(mx.float32)
        delta = actual - ref
        comparisons[name] = {
            "max_abs": mx.max(mx.abs(delta)).item(),
            "relative_rmse": mx.sqrt(mx.mean(delta * delta) / mx.mean(ref * ref)).item(),
            "cosine": (mx.sum(ref * actual) / (mx.sqrt(mx.sum(ref * ref)) * mx.sqrt(mx.sum(actual * actual)))).item(),
        }
    durations = []
    rows = mx.array([0.3, 0.0])
    for iteration in range(-2, args.iterations):
        start = time.perf_counter()
        output = compiled(inputs["latents"], inputs["text"], rows, cos, sin, None)
        mx.eval(output)
        if iteration >= 0:
            durations.append(time.perf_counter() - start)
    report = {"scope": "DiT only; synthetic conditioning; not an end-to-end image test",
              "comparisons": comparisons, "mflux_step_seconds": durations,
              "mflux_median_seconds": statistics.median(durations)}
    print(json.dumps(report, indent=2), flush=True)
    assert all(x["relative_rmse"] < 0.02 and x["cosine"] > 0.999 for x in comparisons.values()), report


if __name__ == "__main__":
    main()
