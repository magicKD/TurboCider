"""Check an experimental base-only fused Z-Image FFN against runtime LoRA math.

This is a one-block numerical probe, not a quality or speed qualification.
Run on the source .mlpackage before exporting the other 31 blocks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora", type=Path,
                        help="optional real runtime adapter; never added to base weights")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=0,
                        help="optional warm Core ML-only call timing, not full GPU/ANE timing")
    args = parser.parse_args()
    import coremltools as ct
    import numpy as np
    from tools.coreml.export_z_image import DEFAULT_BLOCKS, source_from_model

    rng = np.random.default_rng(120)
    rows, hidden = 1056, 3840
    model = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    width = int(model.get_spec().description.output[0].type.multiArrayType.shape[1]) - hidden
    if width not in (4096, 6144, 8192):
        raise ValueError(f"unsupported runtime LoRA probe width: {width}")
    # Sparse held-out synthetic rows bound CPU reference cost; the full Core ML
    # ABI still sees 1056 tokens and the runtime delta changes at every call.
    x = np.zeros((rows, hidden), dtype=np.float16)
    x[:8] = rng.normal(0, 0.12, (8, hidden)).astype(np.float16)
    base = DEFAULT_BLOCKS[args.block] + ".feed_forward."
    with source_from_model(args.checkpoint) as source:
        gate = source.tensor(base + "w1.weight", (10240, hidden), np)[:width].astype(np.float32)
        up = source.tensor(base + "w3.weight", (10240, hidden), np)[:width].astype(np.float32)
        down = source.tensor(base + "w2.weight", (hidden, 10240), np)[:, :width].astype(np.float32)
    def projection(lhs, rhs):
        # Accelerate's large-stride sgemm can emit spurious FP warnings on
        # these sparse 8-row inputs. Direct contraction is deterministic and
        # tests numerical correctness independently of that BLAS path.
        return np.einsum("ni,oi->no", lhs, rhs, optimize=False)
    if args.lora:
        from tools.coreml.lora import load
        bundle = load([str(args.lora)], [1.0], ["transformer"], np)[0]
        def pair(name):
            value = bundle["pairs"][f"diffusion_model.{base}{name}"]
            scale = bundle["strength"]
            if "alpha" in value:
                scale *= float(value["alpha"].reshape(-1)[0]) / value["down"].shape[0]
            return (value["down"].astype(np.float32),
                    value["up"].astype(np.float32), scale)
        def update(value, name, start=0, end=width):
            a, b, scale = pair(name)
            return projection(projection(value, a[:, start:end]), b[:width] if name != "w2" else b) * scale
        delta = np.concatenate((update(x[:8].astype(np.float32), "w1"),
                                update(x[:8].astype(np.float32), "w3")), -1).astype(np.float16)
    else:
        # Synthetic dynamic adapter for standalone single-block testing.
        delta = rng.normal(0, 0.075, (8, 2 * width)).astype(np.float16)
    zeros = np.zeros((rows, 2 * width), dtype=np.float16)
    nonzero = zeros.copy()
    nonzero[:8] = delta
    def predict(value):
        result = model.predict({
            "x": x.T[None, :, None, :].copy(),
            "lora_gate_up": value.T[None, :, None, :].copy(),
        })["y"]
        if result.shape != (1, hidden + width, 1, rows):
            raise ValueError(f"invalid Core ML output shape: {result.shape}")
        return result[0, :, 0, :8].T.astype(np.float32)
    baseline = predict(zeros)
    adapted = predict(nonzero)
    def reference(value):
        g = projection(x[:8].astype(np.float32), gate) + value[:, :width].astype(np.float32)
        u = projection(x[:8].astype(np.float32), up) + value[:, width:].astype(np.float32)
        hidden_value = g / (1 + np.exp(-np.clip(g, -80, 80))) * u
        return projection(hidden_value, down), hidden_value
    base_down, _ = reference(zeros[:8])
    lora_down, lora_hidden = reference(delta)
    # The base output is divided by the export output_scale (default 32).
    # Compare the adapter *effect* too: omitting gate/up LoRA must fail here.
    relative = lambda actual, expected: float(np.linalg.norm(actual - expected) /
                                                max(np.linalg.norm(expected), 1e-6))
    base_error = relative(baseline[:, :hidden] * 32, base_down)
    adapted_error = relative(adapted[:, :hidden] * 32, lora_down)
    hidden_error = relative(adapted[:, hidden:], lora_hidden)
    delta_error = relative((adapted[:, :hidden] - baseline[:, :hidden]) * 32,
                           lora_down - base_down)
    if args.lora:
        down_gpu = update(adapted[:, hidden:], "w2")
        down_expected = update(lora_hidden, "w2")
    else:
        rank = 16
        factor_a = rng.normal(0, 0.001, (rank, width)).astype(np.float32)
        factor_b = rng.normal(0, 0.01, (hidden, rank)).astype(np.float32)
        down_gpu = projection(projection(adapted[:, hidden:], factor_a), factor_b)
        down_expected = projection(projection(lora_hidden, factor_a), factor_b)
    down_delta_error = relative(down_gpu, down_expected)
    expected_finite = all(np.isfinite(value).all() for value in
                          (base_down, lora_down, lora_hidden))
    result = {"ane_width": width, "base_relative_l2": base_error,
              "adapted_relative_l2": adapted_error,
              "hidden_relative_l2": hidden_error, "gate_up_effect_relative_l2": delta_error,
              "runtime_down_effect_relative_l2": down_delta_error,
              "changed_output": bool(np.any(adapted != baseline)),
              "finite": bool(np.isfinite(adapted).all() and expected_finite)}
    print(json.dumps(result, sort_keys=True))
    if not result["finite"] or not result["changed_output"] or not all(np.isfinite(
            value) for value in (base_error, adapted_error, hidden_error,
                                  delta_error, down_delta_error)) or max(
            base_error, adapted_error, hidden_error, delta_error, down_delta_error) > 0.14:
        raise SystemExit("runtime LoRA fused layer did not pass the numerical probe")
    if args.timing_repeats:
        if args.timing_repeats < 2:
            raise SystemExit("timing requires at least two repeats")
        features = {"x": x.T[None, :, None, :].copy(),
                    "lora_gate_up": nonzero.T[None, :, None, :].copy()}
        timings = []
        for _ in range(args.timing_repeats):
            started = time.perf_counter()
            model.predict(features)
            timings.append((time.perf_counter() - started) * 1000)
        print(json.dumps({"ane_width": width, "coreml_median_ms": float(np.median(timings)),
                          "scope": "Core ML prediction API only; excludes GPU delta/suffix, session, and image"}))


if __name__ == "__main__":
    main()
