"""Standalone MLX benchmark for the GPU half of a Z-Image hybrid FFN.

Reports warmed, synchronized suffix latency for MLX affine W8A16 (optionally
MXFP8) against BF16 matmul, on-demand dequantization, and pre-dequantized
W8 storage. With --model and --capture,
uses the same BF16 suffix slices and a captured FFN input as the native path.
This is an isolated GPU measurement, not an end-to-end speedup or proof of
ANE overlap. Dequantized modes keep dense weights resident and are diagnostic.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def read_bf16_suffix(reader, name, shape, suffix_start, down, np):
    """Read source BF16 without the exporter's BF16 -> FP16 conversion."""
    _, memory, meta, offset = reader._tensor_record(name)
    if tuple(meta["shape"]) != shape or meta["dtype"] != "BF16":
        raise ValueError(f"expected original BF16 Z-Image weight: {name}")
    begin, end = meta["data_offsets"]
    if end - begin != 2 * shape[0] * shape[1]:
        raise ValueError(f"invalid BF16 weight length: {name}")
    bits = np.frombuffer(memory, dtype="<u2", count=shape[0] * shape[1],
                         offset=offset + begin).reshape(shape)
    sliced = bits[:, suffix_start:] if down else bits[suffix_start:]
    return (sliced.astype(np.uint32) << 16).view(np.float32)


def load_real_inputs(args, mx):
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
    from export_z_image import SafetensorsSource

    if not 0 <= args.block < 32 or not 0 <= args.suffix_start < 10240:
        raise ValueError("invalid Z-Image block or suffix boundary")
    if args.suffix_start % args.group or (10240 - args.suffix_start) % args.group:
        raise ValueError("suffix boundary must align with the W8 group")
    captured = np.load(args.capture, allow_pickle=False)
    if captured.ndim != 2 or captured.shape[1] != 3840 or captured.shape[0] < 1:
        raise ValueError("expected a nonempty [rows, 3840] captured FFN input")
    prefix = (f"noise_refiner.{args.block}" if args.block < 2 else
              f"layers.{args.block - 2}") + ".feed_forward"
    with SafetensorsSource.from_model(args.model) as reader:
        gate = read_bf16_suffix(reader, prefix + ".w1.weight", (10240, 3840),
                                args.suffix_start, False, np)
        up = read_bf16_suffix(reader, prefix + ".w3.weight", (10240, 3840),
                              args.suffix_start, False, np)
        down = read_bf16_suffix(reader, prefix + ".w2.weight", (3840, 10240),
                                args.suffix_start, True, np)
    end = args.row_end if args.row_end is not None else captured.shape[0]
    if not 0 <= args.row_start < end <= captured.shape[0]:
        raise ValueError("invalid FFN input row slice")
    x = mx.array(captured[None, args.row_start:end]).astype(mx.bfloat16)
    weights = [mx.array(w).astype(mx.bfloat16) for w in (gate, up, down)]
    return x, weights


def relative_l2(reference, candidate, mx):
    delta = reference.astype(mx.float32) - candidate.astype(mx.float32)
    numerator = mx.sum(delta * delta)
    denominator = mx.sum(reference.astype(mx.float32) ** 2)
    mx.eval(numerator, denominator)
    return (float(numerator.item()) / max(float(denominator.item()), 1e-30)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="BF16 Z-Image model root")
    parser.add_argument("--capture", type=Path, help="held-out block FFN input .npy")
    parser.add_argument("--block", type=int, default=18)
    parser.add_argument("--suffix-start", type=int, default=4096)
    parser.add_argument("--image-rows", type=int, default=1024,
                        help="report image/caption suffix errors separately for real captures")
    parser.add_argument("--row-start", type=int, default=0,
                        help="first captured FFN input row for isolated timing")
    parser.add_argument("--row-end", type=int,
                        help="exclusive captured FFN input row for isolated timing")
    parser.add_argument("--rows", type=int, default=1056)
    parser.add_argument("--hidden", type=int, default=3840)
    parser.add_argument("--width", type=int, default=6144)
    parser.add_argument("--group", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--quant-mode", choices=("affine", "mxfp8"), default="affine")
    parser.add_argument("--fp16-down-scale", type=float, default=64.,
                        help="experimental hidden rescaling before on-demand FP16 down GEMM")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=12)
    args = parser.parse_args()
    if (args.model is None) != (args.capture is None):
        parser.error("--model and --capture must be supplied together")
    if args.quant_mode == "mxfp8" and args.group != 32:
        parser.error("MXFP8 requires group 32")
    if min(args.rows, args.hidden, args.width, args.warmup, args.repetitions) < 1 or (
            args.hidden % args.group or args.width % args.group):
        parser.error("positive aligned geometry and nonzero iterations required")
    if not 1 <= args.fp16_down_scale <= 1024:
        parser.error("FP16 down scale must be 1...1024")

    import mlx.core as mx
    if args.model is not None:
        x, weights = load_real_inputs(args, mx)
    else:
        mx.random.seed(42)
        x = mx.random.normal((1, args.rows, args.hidden)).astype(mx.bfloat16) * 0.03
        weights = [mx.random.normal((out, inn)).astype(mx.bfloat16) * 0.02
                   for out, inn in ((args.width, args.hidden), (args.width, args.hidden),
                                    (args.hidden, args.width))]
    rows, hidden, width = x.shape[1], x.shape[2], weights[0].shape[0]
    if any(w.shape != shape for w, shape in zip(
            weights, ((width, hidden), (width, hidden), (hidden, width)))):
        raise ValueError("inconsistent suffix projection shapes")
    quantized = [mx.quantize(w.astype(mx.float16), args.group, 8, args.quant_mode)
                 for w in weights]
    def operands(item):
        return item[0], item[1], item[2] if len(item) == 3 else None

    mx.eval(x, *weights, *(v for item in quantized for v in item))
    fused_pair = tuple(mx.concatenate((left, right), axis=0)
                       for left, right in zip(quantized[0], quantized[1]))
    fused_operands = operands(fused_pair)
    mx.eval(*fused_pair)
    dense_q8 = [mx.dequantize(*operands(item), args.group, 8, args.quant_mode,
                              dtype=mx.bfloat16)
                for item in quantized]
    dense_q8_f16 = [mx.dequantize(*operands(item), args.group, 8, args.quant_mode,
                                  dtype=mx.float16)
                    for item in quantized]
    mx.eval(*dense_q8, *dense_q8_f16)

    def suffix(inp, matrices, mode):
        def project(value, matrix, index):
            if mode in ("w8_dequant_each", "w8_dequant_each_fp16",
                        "w8_dequant_fused_gate_up_bf16"):
                dtype = mx.float16 if mode.endswith("fp16") else mx.bfloat16
                dense = mx.dequantize(*operands(quantized[index]), args.group, 8,
                                      args.quant_mode, dtype=dtype)
                return mx.matmul(value.astype(dtype), dense.T)
            if (mode == "w8_fp16_gate_up_direct_down" and index < 2) or (
                    mode == "w8_fp16_scaled_all"):
                dense = mx.dequantize(*operands(quantized[index]), args.group, 8,
                                      args.quant_mode, dtype=mx.float16)
                if index == 2:
                    value = (value / args.fp16_down_scale).astype(mx.float16)
                    return mx.matmul(value, dense.T).astype(mx.bfloat16) * args.fp16_down_scale
                return mx.matmul(value.astype(mx.float16), dense.T).astype(mx.bfloat16)
            if mode not in ("direct_q8", "fused_q8_gate_up",
                            "w8_fp16_gate_up_direct_down"):
                return mx.matmul(value, matrix.T)
            packed, scales, biases = operands(quantized[index])
            return mx.quantized_matmul(value, packed, scales, biases, True,
                                       args.group, 8, args.quant_mode)
        if mode == "fused_q8_gate_up":
            gate, up = mx.split(mx.quantized_matmul(inp, *fused_operands, True,
                                                   args.group, 8, args.quant_mode), 2, -1)
        elif mode == "w8_dequant_fused_gate_up_bf16":
            fused_dense = mx.dequantize(*fused_operands, args.group, 8,
                                        args.quant_mode, dtype=mx.bfloat16)
            gate, up = mx.split(mx.matmul(inp, fused_dense.T), 2, -1)
        else:
            gate = project(inp, matrices[0], 0)
            up = project(inp, matrices[1], 1)
        return project(mx.sigmoid(gate) * gate * up, matrices[2], 2)

    choices = {
        "bf16": (x, weights),
        "direct_q8": (x, quantized),
        "w8_dequant_each": (x, quantized),
        "w8_dequant_fused_gate_up_bf16": (x, quantized),
        "w8_dequant_each_fp16": (x.astype(mx.float16), quantized),
        "w8_fp16_gate_up_direct_down": (x, quantized),
        "w8_fp16_scaled_all": (x, quantized),
        "fused_q8_gate_up": (x, quantized),
        "w8_dequant_bf16": (x, dense_q8),
        "w8_dequant_fp16": (x.astype(mx.float16), dense_q8_f16),
    }
    def project(value, matrix, index, mode):
        if mode in ("w8_dequant_each", "w8_dequant_each_fp16",
                    "w8_dequant_fused_gate_up_bf16"):
            dtype = mx.float16 if mode.endswith("fp16") else mx.bfloat16
            dense = mx.dequantize(*operands(quantized[index]), args.group, 8,
                                  args.quant_mode, dtype=dtype)
            return mx.matmul(value.astype(dtype), dense.T)
        if mode in ("w8_fp16_gate_up_direct_down", "w8_fp16_scaled_all") and (
                index < 2 or mode == "w8_fp16_scaled_all"):
            dense = mx.dequantize(*operands(quantized[index]), args.group, 8,
                                  args.quant_mode, dtype=mx.float16)
            if index == 2:
                value = (value / args.fp16_down_scale).astype(mx.float16)
                return mx.matmul(value, dense.T).astype(mx.bfloat16) * args.fp16_down_scale
            return mx.matmul(value.astype(mx.float16), dense.T).astype(mx.bfloat16)
        if mode in ("direct_q8", "fused_q8_gate_up", "w8_fp16_gate_up_direct_down"):
            return mx.quantized_matmul(value, *operands(quantized[index]), True,
                                       args.group, 8, args.quant_mode)
        return mx.matmul(value, matrix.T)

    # Isolate the expensive projections from SwiGLU, using the *same* hidden
    # activation for all down-projection modes. No setup is included in timing.
    gate = mx.matmul(x, weights[0].T)
    up = mx.matmul(x, weights[1].T)
    hidden_input = mx.sigmoid(gate) * gate * up
    mx.eval(hidden_input)
    samples = {stage: {name: [] for name in choices}
               for stage in ("suffix", "gate_up", "down")}
    schedule = tuple(choices) + tuple(reversed(tuple(choices)))
    for round_index in range(args.warmup + args.repetitions):
        for stage in samples:
            for name in schedule:
                inp, matrices = choices[name]
                start = time.perf_counter()
                if stage == "suffix":
                    result = suffix(inp, matrices, name)
                elif stage == "gate_up":
                    if name == "fused_q8_gate_up":
                        result = mx.quantized_matmul(inp, *fused_operands, True,
                                                     args.group, 8, args.quant_mode)
                    elif name == "w8_dequant_fused_gate_up_bf16":
                        fused_dense = mx.dequantize(*fused_operands, args.group, 8,
                                                    args.quant_mode, dtype=mx.bfloat16)
                        result = mx.matmul(inp, fused_dense.T)
                    else:
                        result = (project(inp, matrices[0], 0, name),
                                  project(inp, matrices[1], 1, name))
                else:
                    value = (hidden_input.astype(mx.float16)
                             if name in ("w8_dequant_fp16", "w8_dequant_each_fp16")
                             else hidden_input)
                    result = project(value, matrices[2], 2, name)
                mx.eval(result)
                mx.synchronize()
                if round_index >= args.warmup:
                    samples[stage][name].append((time.perf_counter() - start) * 1000)
    reference = suffix(*choices["bf16"], "bf16")
    mx.eval(reference)
    errors = {}
    region_errors = {}
    for name, pair in choices.items():
        if name == "bf16":
            continue
        candidate = suffix(*pair, name)
        errors[name] = relative_l2(reference, candidate, mx)
        if args.model is not None and 0 < args.image_rows < rows:
            region_errors[name] = {
                "image": relative_l2(reference[:, :args.image_rows],
                                     candidate[:, :args.image_rows], mx),
                "caption": relative_l2(reference[:, args.image_rows:],
                                       candidate[:, args.image_rows:], mx),
            }
    print(json.dumps({"rows": rows, "hidden": hidden, "width": width,
                      "group": args.group,
                      "fp16_down_scale": args.fp16_down_scale,
                      "quant_mode": args.quant_mode,
                      "source": ({"model": str(args.model), "capture": str(args.capture),
                                  "block": args.block, "suffix_start": args.suffix_start,
                                  "row_start": args.row_start, "row_end": args.row_end}
                                 if args.model else "seeded_synthetic"),
                      "median_milliseconds": {stage: {name: statistics.median(values)
                                                       for name, values in modes.items()}
                                              for stage, modes in samples.items()},
                      "relative_l2_vs_bf16": errors,
                      "relative_l2_by_region": region_errors,
                      "packed_weight_bytes": sum(item.nbytes for group in quantized for item in group),
                      "dense_bf16_weight_bytes": sum(w.nbytes for w in weights),
                      "repetitions_per_mode": args.repetitions * 2,
                      "scope": "isolated MLX GPU suffix; w8_dequant_each includes "
                               "per-projection expansion, pre-dequantized dense modes "
                               "exclude setup and are performance diagnostics"}))


if __name__ == "__main__":
    main()
