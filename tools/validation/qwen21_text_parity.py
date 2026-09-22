"""Native Qwen3-VL text stack oracle; development only, no production imports."""
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
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder

    for dtype in (mx.float32, mx.bfloat16):
        mx.random.seed(194)
        model = Qwen21TextEncoder(vocab_size=64, hidden_size=16, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1,
                                 intermediate_size=48, head_dim=8, mrope_section=[2, 1, 1])
        params = [(key, (mx.random.normal(value.shape) * 0.15).astype(dtype))
                  for key, value in tree_flatten(model.parameters()) if key.endswith(".weight")]
        model.load_weights(params, strict=False)
        for valid in (7, 5):
            ids = mx.array([[1, 2, 3, 4, 5, 6, 7]], dtype=mx.int32)
            positions = mx.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 3, 1, 2, 3]], dtype=mx.int32)
            mask = (mx.arange(7)[None, :] < valid).astype(mx.int32)
            expected = model(ids, mask)
            hidden = model.embed_tokens(ids)
            rope = model.rotary_emb(hidden, positions[:, None, :])
            causal_mask = mx.where((mx.arange(7)[None, :] > mx.arange(7)[:, None]) | (mx.arange(7)[None, :] >= valid), -float("inf"), 0.0)[None, None]
            for layer in model.layers:
                hidden, _ = layer(hidden, causal_mask, rope)
            expected_positions = model.norm(hidden)
            deltas = [(mx.random.normal(hidden.shape) * .1).astype(dtype) *
                      mx.array([0,0,1,1,0,0,0],dtype=dtype)[None,:,None] for _ in range(2)]
            hidden = model.embed_tokens(ids)
            for i, layer in enumerate(model.layers):
                hidden, _ = layer(hidden, causal_mask, rope)
                hidden = hidden + deltas[i]
            expected_deepstack, expected_raw = model.norm(hidden), hidden
            mx.eval(expected, expected_positions)
            with tempfile.TemporaryDirectory(prefix="tc-qwen21-text-") as directory:
                root = Path(directory)
                mx.save_safetensors(str(root / "weights.safetensors"),
                                    {("model.language_model." if valid == 7 else "model.") + key: value for key, value in params})
                mx.save_safetensors(str(root / "inputs.safetensors"), {"ids": ids, "positions": positions, "deep0": deltas[0], "deep1": deltas[1]},
                                    {"layers": "2", "heads": "2", "kv_heads": "1", "head_dim": "8",
                                     "section0": "2", "section1": "1", "section2": "1", "valid": str(valid)})
                subprocess.run([str(args.probe.resolve()), str(root / "weights.safetensors"),
                                str(root / "inputs.safetensors"), str(root / "outputs.safetensors")], check=True)
                actual = mx.load(str(root / "outputs.safetensors"))
                errors = {"text": mx.max(mx.abs(actual["text"].astype(mx.float32) - expected.astype(mx.float32))).item(),
                          "positions": mx.max(mx.abs(actual["positions"].astype(mx.float32) - expected_positions.astype(mx.float32))).item()}
                for name, value in (("deepstack", expected_deepstack), ("raw_deepstack", expected_raw)):
                    errors[name] = mx.max(mx.abs(actual[name].astype(mx.float32) - value.astype(mx.float32))).item()
                print(json.dumps({"dtype": str(dtype), "valid_tokens": valid, "max_abs": errors}), flush=True)
                assert max(errors.values()) < (2e-6 if dtype == mx.float32 else 0.003), errors


if __name__ == "__main__":
    main()
