"""Compare a real-calibrated Z-Image ANE partial FFN with BF16 source weights.

This checks numerical quality, not physical ANE placement or image quality.
Use a held-out capture (different prompt/seed) for meaningful qualification.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
from export_z_image import SafetensorsSource
from z_image_smoothquant import load_calibration, route_channel_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--image-rows", type=int,
                        help="split image and conditioning rows (1024 for 512x512 Z-Image)")
    parser.add_argument("--debug-intermediates", action="store_true")
    parser.add_argument("--compare-adaptive-manifest", type=Path,
                        help="compare adaptive post-Q/DQ image activations, diagnostic only")
    args = parser.parse_args()
    import numpy as np
    import coremltools as ct

    manifest = json.loads(args.manifest.read_text())
    record = manifest["artifacts"][str(args.block)]
    shape = manifest["shape"]
    rows, hidden, width = shape["buckets"][0], shape["K"], shape["ane_mlp_end"]
    if args.block < 0 or args.block >= 32 or shape["ane_mlp_start"] != 0:
        raise ValueError("invalid Z-Image ANE prefix geometry")
    if args.sample.is_dir():
        raise ValueError("select exactly one held-out capture sample")
    samples, _ = load_calibration(args.sample, rows, hidden, np, pad_rows=args.block < 2)
    if len(samples) != 1:
        raise ValueError("sample must contain exactly one FFN input")
    x = np.asarray(samples[0], dtype=np.float32)
    prefix = (f"noise_refiner.{args.block}" if args.block < 2
              else f"layers.{args.block - 2}") + ".feed_forward"
    with SafetensorsSource.from_model(args.model) as reader:
        routing = manifest["export_identity"].get("channel_routing")
        ane = (route_channel_indices(routing["ane_group_indexes"][str(args.block)],
                                     routing["group_width"], 10240, width, np)[0]
               if routing else np.arange(width))
        g = reader.tensor(prefix + ".w1.weight", (10240, hidden), np)[ane].astype(np.float32)
        u = reader.tensor(prefix + ".w3.weight", (10240, hidden), np)[ane].astype(np.float32)
        d = reader.tensor(prefix + ".w2.weight", (hidden, 10240), np)[:, ane].astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gate = x @ g.T
        up = x @ u.T
        expected = ((gate / (1 + np.exp(-np.clip(gate, -80, 80)))) * up) @ d.T
    if not np.isfinite(expected).all():
        raise ValueError("source partial FFN produced nonfinite values")
    package = args.manifest.parent / record["int8_pc"]
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    provided = np.ascontiguousarray(samples[0].T[None, :, None, :])
    output = model.predict({"x": provided})["y"]
    if args.compare_adaptive_manifest is not None:
        from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger
        other = json.loads(args.compare_adaptive_manifest.read_text())
        other_package = args.compare_adaptive_manifest.parent / other["artifacts"][
            str(args.block)]["int8_pc"]
        other_model = ct.models.MLModel(str(other_package),
                                        compute_units=ct.ComputeUnit.CPU_ONLY)
        comparisons = {}
        compared_values = {}
        for label, current, remote in (
            ("post_a8", "a8_hidden_adaptive_restore", "a8_hidden_adaptive_select_0"),
            ("pre_a8", "a8_hidden_adaptive_image", "a8_hidden_adaptive_image"),
        ):
            a = ModelDebugger(model).predict_intermediate_outputs(
                {"x": provided}, [current])[current].astype(np.float32)
            b = ModelDebugger(other_model).predict_intermediate_outputs(
                {"x": provided}, [remote])[remote].astype(np.float32)
            comparisons[label] = {
                "relative_l2": float(np.linalg.norm((a - b).astype(np.float64).ravel()) /
                                     max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-12)),
                "a_abs_max": float(np.max(np.abs(a))),
                "b_abs_max": float(np.max(np.abs(b))),
            }
            compared_values[label] = (a, b)
        debug_names = ("a8_hidden_peak", "a8_hidden_ratio_select_0",
                       "a8_hidden_inverse_select_0", "a8_hidden_adaptive_prepare",
                       "a8_hidden_adaptive_dq_shared")
        debugger = ModelDebugger(model)
        debug = debugger.predict_intermediate_outputs({"x": provided}, debug_names)
        comparisons["shared_intermediates"] = {
            key: {"shape": list(value.shape),
                  "abs_max": float(np.abs(value).max()),
                  "first_values": [float(v) for v in value.ravel()[:4]]}
            for key, value in debug.items()}
        print(json.dumps({"adaptive_intermediate_comparison": comparisons}))
    if args.debug_intermediates:
        from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger
        debugger = ModelDebugger(model)
        names = ("smoothquant_input", "a8_input_dequantized", "projected",
                 "split_0_0", "split_0_1", "silu_0", "mul_0", "mul_1",
                 "mul_2", "dequantize_0")
        names = tuple(name for name in names if name in debugger.block_info.operations)
        values = debugger.predict_intermediate_outputs({"x": provided}, names)
        print(json.dumps({"intermediates": {
            name: {"abs_max": float(np.max(np.abs(value))),
                   "nonzero": int(np.count_nonzero(value)),
                   "finite": bool(np.isfinite(value).all())}
            for name, value in values.items()}}))
    actual = output[0, :, 0, :].T.astype(np.float32) * float(shape["output_scale"])
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("quantized FFN output shape or finite check failed")
    delta = actual - expected
    regions = {}
    if args.image_rows is not None:
        if not 0 < args.image_rows < rows:
            raise ValueError("image-rows must split the input bucket")
        for name, left, right in (("image", 0, args.image_rows),
                                  ("conditioning", args.image_rows, rows)):
            reference = expected[left:right].astype(np.float64)
            estimate = actual[left:right].astype(np.float64)
            error = estimate - reference
            regions[name] = {
                "rows": right - left,
                "relative_l2": float(np.linalg.norm(error.ravel()) /
                                     max(np.linalg.norm(reference.ravel()), 1e-12)),
                "reference_abs_max": float(np.max(np.abs(reference))),
                "actual_nonzero": int(np.count_nonzero(estimate)),
                "mean_abs_error": float(np.mean(np.abs(error))),
            }
    norm = np.linalg.norm(expected.ravel().astype(np.float64))
    relative_l2 = float(np.linalg.norm(delta.ravel().astype(np.float64)) / max(norm, 1e-12))
    cosine = float(np.dot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) /
                   max(np.linalg.norm(actual.ravel().astype(np.float64)) * norm, 1e-12))
    print(json.dumps({"block": args.block, "rows": rows, "ane_width": width,
                      "relative_l2": relative_l2, "cosine": cosine,
                      "actual_nonzero": int(np.count_nonzero(actual)),
                      "actual_abs_max": float(np.max(np.abs(actual))),
                      "mean_abs_error": float(np.mean(np.abs(delta))),
                      "max_abs_error": float(np.max(np.abs(delta))),
                      "reference_abs_max": float(np.max(np.abs(expected))),
                      **({"regions": regions} if regions else {}),
                      "scope": "partial_ffn_cpu_only; not ANE placement or E2E image quality"}))


if __name__ == "__main__":
    main()
