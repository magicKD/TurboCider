"""Executable Q-format, W8A8 and SmoothQuant references; standard library only."""
import math
import random
import struct

try:
    from .common import (bf16, dequantize_rows, fp16, linear, matrix, metrics, quantize_rows,
                         report, rotate, silu, swiglu, transpose)
except ImportError:
    from common import (bf16, dequantize_rows, fp16, linear, matrix, metrics, quantize_rows,
                        report, rotate, silu, swiglu, transpose)


def pack_codes(codes, bits):
    if bits not in (4, 8) or any(type(x) is not int or not 0 <= x < 1 << bits for x in codes):
        raise ValueError("Q4/Q8 unsigned integer codes required")
    per_word, words = 32 // bits, []
    for first in range(0, len(codes), per_word):
        words.append(sum(value << (offset * bits) for offset, value in enumerate(codes[first:first + per_word])))
    return b"".join(struct.pack("<I", word) for word in words)


def unpack_codes(data, bits, count):
    if bits not in (4, 8) or type(count) is not int or count < 0 or len(data) != ((count + 32 // bits - 1) // (32 // bits)) * 4:
        raise ValueError("packed byte length/count mismatch")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    return [(words[index // (32 // bits)] >> ((index % (32 // bits)) * bits)) & ((1 << bits) - 1) for index in range(count)]


def ggml_q4_0_block(data):
    """Independent 32-value oracle: FP16 scale, 16 low then 16 high nibbles."""
    if len(data) != 18:
        raise ValueError("Q4_0 requires exactly 18 bytes per 32 elements")
    scale = struct.unpack("<e", data[:2])[0]
    if not math.isfinite(scale):
        raise ValueError("nonfinite Q4_0 scale")
    codes = [value & 15 for value in data[2:]] + [value >> 4 for value in data[2:]]
    return [(value - 8) * scale for value in codes]


def ggml_q8_0_block(data):
    """Independent 32-value oracle: FP16 scale followed by signed INT8 codes."""
    if len(data) != 34:
        raise ValueError("Q8_0 requires exactly 34 bytes per 32 elements")
    scale = struct.unpack("<e", data[:2])[0]
    if not math.isfinite(scale):
        raise ValueError("nonfinite Q8_0 scale")
    return [value * scale for value in struct.unpack("<32b", data[2:])]


def affine_slice(data, *, rows, physical_columns, bits, first, count, scales, offsets, group):
    if (physical_columns % group or not 0 <= first < physical_columns or count <= 0 or first + count > physical_columns or
            len(scales) != rows or len(offsets) != rows or
            any(len(x) != physical_columns // group for x in scales + offsets)):
        raise ValueError("invalid affine physical geometry/metadata")
    codes = unpack_codes(data, bits, rows * physical_columns)
    return [[codes[row * physical_columns + col] * scales[row][col // group] + offsets[row][col // group]
             for col in range(first, first + count)] for row in range(rows)]


def formats(seed=17):
    rng = random.Random(seed)
    comparisons = []
    for bits in (4, 8):
        rows, width, group = 3, 24, 8
        codes = [rng.randrange(1 << bits) for _ in range(rows * width)]
        packed = pack_codes(codes, bits)
        assert unpack_codes(packed, bits, len(codes)) == codes
        scales = [[.02 * (1 + row + col) for col in range(width // group)] for row in range(rows)]
        offsets = [[-.1 * (1 + row) for _ in range(width // group)] for row in range(rows)]
        selected = affine_slice(packed, rows=rows, physical_columns=width, bits=bits, first=8,
                                count=8, scales=scales, offsets=offsets, group=group)
        witness = [[codes[row * width + col] * scales[row][col // group] + offsets[row][col // group]
                    for col in range(8, 16)] for row in range(rows)]
        assert selected == witness
        # Deliberately wrong local pitch: same in-bounds bytes, different matrix.
        wrong = [[codes[row * 8 + col] * scales[row][1] + offsets[row][1] for col in range(8)] for row in range(rows)]
        comparisons.append({"bits": bits, "bytes": len(packed), "physical_pitch": width,
                            "selected_columns": [8, 16], "correct": metrics(witness, selected),
                            "wrong_local_pitch": metrics(witness, wrong)})
    payload = bytes(index | ((15 - index) << 4) for index in range(16))
    q4 = struct.pack("<e", .125) + payload
    correct = [(index - 8) * .125 for index in range(16)] + [(7 - index) * .125 for index in range(16)]
    assert ggml_q4_0_block(q4) == correct
    wrong = [(code - 8) * .125 for value in payload for code in (value & 15, value >> 4)]
    q8 = struct.pack("<e", .25) + struct.pack("<32b", *range(-16, 16))
    assert ggml_q8_0_block(q8) == [value * .25 for value in range(-16, 16)]
    return report("quant-formats", {"comparisons": comparisons,
                  "ggml_controls": {"q4_0_bytes": len(q4), "q8_0_bytes": len(q8),
                                    "wrong_interleaved_q4": metrics(correct, wrong),
                                    "equal_size_is_not_layout_equivalence": {"ggml_two_q8_0_blocks": 68, "mlx_affine_q8_group64_fp16_metadata": 68}},
                  "note": "Affine controls are unsigned low-bit-first; separate GGML block controls do not translate formats or qualify K-quants/ConvRot/NVFP4."})


def w8a8(seed=17, rows=5, hidden=16, width=32):
    rng = random.Random(seed)
    if hidden < 2 or hidden & (hidden - 1) or width < 2 or width & (width - 1):
        raise ValueError("this reference uses power-of-two H/F to expose complete rotation blocks")
    x = matrix(rows, hidden, rng, .7)
    wg, wu, wd = matrix(width, hidden, rng, .2), matrix(width, hidden, rng, .2), matrix(hidden, width, rng, .2)
    reference, _ = swiglu(x, wg, wu, wd)
    comparisons = []
    for rotation in (False, True):
        transform = lambda source, block: [rotate(row, block) for row in source] if rotation else source
        xr, gr, ur, dr = transform(x, hidden), transform(wg, hidden), transform(wu, hidden), transform(wd, width)
        qx, sx = quantize_rows(xr, normalize=True, half_scales=True)
        qg, sg = quantize_rows(gr, normalize=True, half_scales=True)
        qu, su = quantize_rows(ur, normalize=True, half_scales=True)
        qd, sd = quantize_rows(dr, normalize=True, half_scales=True)
        xx = dequantize_rows(qx, sx, normalize=True)
        gate = linear(xx, dequantize_rows(qg, sg, normalize=True))
        up = linear(xx, dequantize_rows(qu, su, normalize=True))
        corrected = [[silu(g) * u for g, u in zip(a, b)] for a, b in zip(gate, up)]
        rotated_hidden = transform(corrected, width)
        qh, sh = quantize_rows(rotated_hidden, normalize=True, half_scales=True)
        # Integer dot + both scales is an independent witness of DQ matmul.
        integer_dot = linear(qh, qd)
        actual = [[value * sh[row] * sd[col] / (128 * 128) for col, value in enumerate(values)]
                  for row, values in enumerate(integer_dot)]
        dq = linear(dequantize_rows(qh, sh, normalize=True), dequantize_rows(qd, sd, normalize=True))
        assert metrics(dq, actual)["max_abs"] < 1e-12
        comparisons.append({"rotation": rotation, "source_to_candidate": metrics(reference, actual),
                            "integer_dot_vs_dq": metrics(dq, actual),
                            "code_range": [min(sum(qh, [])), max(sum(qh, []))],
                            "logical_io_bytes": {"fp16_weights": 3 * hidden * width * 2,
                                                 "q8_codes": 3 * hidden * width,
                                                 "f16_weight_scales": (2 * width + hidden) * 2},
                            "hidden_quantization_is_after_silu": True})
    exact_rotated, _ = swiglu(x, wg, wu, wd)
    # Rotate only across linear boundaries, never commute rotation through SiLU.
    xr, gr, ur = [rotate(row, hidden) for row in x], [rotate(row, hidden) for row in wg], [rotate(row, hidden) for row in wu]
    gate, up = linear(xr, gr), linear(xr, ur)
    hh = [[silu(g) * u for g, u in zip(a, b)] for a, b in zip(gate, up)]
    rotated_result = linear([rotate(row, width) for row in hh], [rotate(row, width) for row in wd])
    assert metrics(exact_rotated, rotated_result)["max_abs"] < 1e-12
    return report("w8a8-reference", {"shape": [rows, hidden, width], "comparisons": comparisons,
                  "unquantized_rotation_witness": metrics(exact_rotated, rotated_result),
                  "boundaries": "Real arithmetic CPU witness. Device FP16 reductions/headroom/Comfy H256 need separate recipes."})


def smoothquant(seed=17):
    rng = random.Random(seed)
    calibration, holdout = matrix(12, 16, rng), matrix(7, 16, rng)
    for row in calibration + holdout:
        row[3] *= 24
    weight = matrix(9, 16, rng, .2)
    activation_peak = [max(abs(row[col]) for row in calibration) for col in range(16)]
    weight_peak = [max(abs(row[col]) for row in weight) for col in range(16)]
    reference = linear(holdout, weight)
    sweep = []
    for alpha in (0, .25, .5, .75, 1):
        scales = [max(a, 1e-8) ** alpha / max(w, 1e-8) ** (1 - alpha) for a, w in zip(activation_peak, weight_peak)]
        scaled_x = [[x / scale for x, scale in zip(row, scales)] for row in holdout]
        scaled_w = [[w * scale for w, scale in zip(row, scales)] for row in weight]
        exact = linear(scaled_x, scaled_w)
        qx, sx = quantize_rows(scaled_x)
        qw, sw = quantize_rows(scaled_w)
        candidate = linear(dequantize_rows(qx, sx), dequantize_rows(qw, sw))
        assert metrics(reference, exact)["max_abs"] < 1e-12
        sweep.append({"alpha": alpha, "exact_transform": metrics(reference, exact),
                      "holdout_q8": metrics(reference, candidate), "scale_range": [min(scales), max(scales)]})
    return report("smoothquant", {"calibration_rows": len(calibration), "holdout_rows": len(holdout), "sweep": sweep,
                  "revisit": "Choose on calibration, validate held-out layers/steps; an alpha winning this toy is not a model policy."})


def w4a8(seed=17):
    """Complete toy FFN: U4 storage -> shared signed-I8 LUT, per-token A8."""
    rng, rows, hidden, width = random.Random(seed), 5, 16, 32
    x = matrix(rows, hidden, rng, .7)
    wg, wu, wd = matrix(width, hidden, rng, .2), matrix(width, hidden, rng, .2), matrix(hidden, width, rng, .2)
    reference, _ = swiglu(x, wg, wu, wd)
    lookup = list(range(-8, 8))
    def weight(source):
        codes, scales = [], []
        for row in source:
            peak = max(abs(value) for value in row)
            scale = fp16(max(peak / 7, 2 ** -24)) if peak else 1.
            codes.append([max(-7, min(7, round(value / scale))) + 8 for value in row])
            scales.append(scale)
        packed = pack_codes(sum(codes, []), 4)
        decoded = unpack_codes(packed, 4, len(source) * len(source[0]))
        lut_rows = [[lookup[decoded[row * len(source[0]) + col]] for col in range(len(source[0]))] for row in range(len(source))]
        return lut_rows, scales, len(packed)
    qg, sg, bg = weight(wg)
    qu, su, bu = weight(wu)
    qd, sd, bd = weight(wd)
    qx, sx = quantize_rows(x)
    gate_int, up_int = linear(qx, qg), linear(qx, qu)
    gate = [[value * sx[row] * sg[col] for col, value in enumerate(values)] for row, values in enumerate(gate_int)]
    up = [[value * sx[row] * su[col] for col, value in enumerate(values)] for row, values in enumerate(up_int)]
    hh = [[silu(g) * u for g, u in zip(a, b)] for a, b in zip(gate, up)]
    qh, sh = quantize_rows(hh)
    down_int = linear(qh, qd)
    actual = [[value * sh[row] * sd[col] for col, value in enumerate(values)] for row, values in enumerate(down_int)]
    witness = linear(dequantize_rows(qh, sh), dequantize_rows(qd, sd))
    assert metrics(witness, actual)["max_abs"] < 1e-12
    return report("w4a8-reference", {"shape": [rows, hidden, width],
                  "source_to_w4a8": metrics(reference, actual), "integer_vs_lut_dq": metrics(witness, actual),
                  "storage_unsigned_code_zero_point": 8, "signed_quantizer_range": [-7, 7],
                  "shared_int8_lookup": lookup, "packed_weight_bytes": bg + bu + bd,
                  "weight_scale_f16_bytes": (2 * width + hidden) * 2,
                  "expanded_i8_bank_if_materialized_bytes": 3 * hidden * width,
                  "limits": "Uniform symmetric LUT reference, not arbitrary MLX affine/GGUF/NVFP4 or a CoreML INT4-MAC implementation. Stage expansion/lookup cost must be measured."})


def precision(seed=17):
    source = [0, .1, -.1, 2 ** -24, 65504, 70000, 1e12]
    table = []
    for value in source:
        try:
            half = fp16(value)
        except ValueError:
            half = "overflow-rejected"
        table.append({"source": value, "fp16": half, "bf16": bf16(value)})
    rng = random.Random(seed)
    different = None
    for _ in range(2000):
        scale, code, offset = fp16(rng.uniform(.001, .1)), rng.randrange(256), fp16(rng.uniform(-1, 1))
        separate = fp16(fp16(scale * code) + offset)
        widen_first = fp16(scale * code + offset)
        if separate != widen_first:
            different = {"scale": scale, "code": code, "offset": offset,
                         "rounded_product_then_add": separate, "widen_then_one_round": widen_first}
            break
    assert different is not None
    return report("precision-boundaries", {"range_table": table, "affine_rounding_counterexample": different,
                  "note": "An explicit two-round witness, not a claim about every compiler's FMA lowering."})
