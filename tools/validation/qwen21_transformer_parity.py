"""Development-only mflux oracle; never imported by production inference.

Run with --mflux pointing at an unmodified mflux checkout and --probe at the
native binary. This uses a small, deterministic transformer, not model weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    results = []
    for dtype in (mx.float32, mx.bfloat16):
        ModelConfig.precision = dtype
        mx.random.seed(731)
        model = Qwen21Transformer(in_channels=4, out_channels=4, num_layers=2,
                                 num_attention_heads=2, attention_head_dim=8,
                                 context_in_dim=16, axes_dims_rope=(2, 2, 4))
        # Exercise nonzero scales and gates, including zero-centered text norm.
        params = [(k, (mx.random.normal(v.shape) * 0.1).astype(dtype))
                  for k, v in tree_flatten(model.parameters()) if k.endswith(".weight")]
        model.load_weights(params, strict=False)
        weights = {k.replace("modulation.layers.1", "modulation.1"): v for k, v in params}
        for height, width, text_len in ((2, 3, 5), (3, 2, 1)):
            checkpoint = dict(weights)
            if text_len == 1:
                for block in range(2):
                    prefix = f"transformer_blocks.{block}.img_mlp."
                    checkpoint[prefix + "gate_up.weight"] = mx.concatenate(
                        [checkpoint.pop(prefix + "gate_layer.weight"), checkpoint.pop(prefix + "proj.weight")], axis=0)
            latents = mx.random.normal((1, height * width, 4)).astype(dtype)
            text = mx.random.normal((1, text_len, 16)).astype(dtype)
            changed = mx.random.normal(text.shape).astype(dtype)
            cos, sin = model.pos_embed(text_len, height, width)

            def oracle(condition, t):
                result = model._forward(latents, condition, mx.array([t, 0.0]), cos, sin, None)
                mx.eval(result)
                return result.astype(mx.float32)

            expected = {"first": oracle(text, 0.8), "cached": oracle(text, 0.3),
                        "uncached": oracle(text, 0.3), "changed": oracle(changed, 0.3)}
            with tempfile.TemporaryDirectory(prefix="tc-qwen21-") as directory:
                root = Path(directory)
                mx.save_safetensors(str(root / "weights.safetensors"), checkpoint)
                metadata = {"layers": "2", "heads": "2", "head_dim": "8", "axis0": "2",
                            "axis1": "2", "axis2": "4", "height": str(height), "width": str(width)}
                mx.save_safetensors(str(root / "inputs.safetensors"),
                                    {"latents": latents, "text": text, "text_changed": changed}, metadata)
                subprocess.run([str(args.probe.resolve()), str(root / "weights.safetensors"),
                                str(root / "inputs.safetensors"), str(root / "outputs.safetensors")], check=True)
                actual = mx.load(str(root / "outputs.safetensors"))
                errors = {name: mx.max(mx.abs(actual[name].astype(mx.float32) - ref)).item()
                          for name, ref in expected.items()}
                cache_error = mx.max(mx.abs(actual["cached"].astype(mx.float32) - actual["uncached"].astype(mx.float32))).item()
                split_error = mx.max(mx.abs(actual["cached"].astype(mx.float32) - actual["split_cached"].astype(mx.float32))).item()
                changed_cache_error = mx.max(mx.abs(actual["changed"].astype(mx.float32) - actual["changed_reference"].astype(mx.float32))).item()
                tolerance = 3e-5 if dtype == mx.float32 else 0.004
                assert all(error <= tolerance for error in errors.values()), errors
                assert cache_error <= tolerance, cache_error
                assert split_error <= tolerance, split_error
                assert changed_cache_error <= tolerance, changed_cache_error
                schedule = mx.array([1., .8, .3, 0.], dtype=mx.float32)
                mx.save_safetensors(str(root / "trajectory-input.safetensors"),
                                    {"latents": latents, "text": text, "sigmas": schedule}, metadata)
                subprocess.run([str(args.probe.resolve()), str(root / "weights.safetensors"),
                                str(root / "trajectory-input.safetensors"),
                                str(root / "trajectory.safetensors"), "trajectory"], check=True)
                trajectory = mx.load(str(root / "trajectory.safetensors"))
                state, trajectory_errors = latents, []
                for step in range(3):
                    t = (schedule[step] * 1000).astype(dtype) / mx.array(1000., dtype=dtype)
                    noise = model._forward(state, text, mx.array([t.item(), 0.0]), cos, sin, None)
                    state = (state.astype(mx.float32) + noise.astype(mx.float32) *
                             (schedule[step + 1] - schedule[step])).astype(dtype)
                    error = mx.max(mx.abs(state.astype(mx.float32) -
                                         trajectory[f"latent_{step}"].astype(mx.float32))).item()
                    assert error <= tolerance, (step, error)
                    trajectory_errors.append(error)
                results.append({"dtype": str(dtype), "geometry": [height, width, text_len],
                                "oracle_max_abs": errors, "cache_max_abs": cache_error,
                                "trajectory_max_abs": trajectory_errors})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
