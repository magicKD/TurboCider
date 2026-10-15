"""Optional original Metal affine-Q4/Q8 GEMM; all operand strides explicit."""
from pathlib import Path

GUARD = 17.0


def geometry(m, n, k, bits, group, bm, pad=0):
    if (any(type(value) is not int for value in (m, n, k, bits, group, bm, pad))
            or not 1 <= m <= 2048 or not 1 <= n <= 24576 or not 1 <= k <= 12288
            or bits not in (4, 8) or group not in (32, 64, 128) or k % group
            or bm not in (32, 64) or pad not in (0, 8)):
        raise ValueError("bounded affine FP16 geometry, group-aligned K and BM32/64 required")
    return {"m": m, "n": n, "k": k, "bm": bm, "bn": 32, "bk": 32,
            "threads": (bm // 16) * 2 * 32, "logical_input_weight_bytes": n * k * bits // 8,
            "declared_threadgroup_bytes": (bm * 32 + 32 * (32 + pad)) * 2 + bm * 32 * 4}


def factory(mx, bits, group, bm, coalesced=False, pad=0):
    if type(coalesced) is not bool or pad not in (0, 8):
        raise ValueError("bounded load-order/padding candidates required")
    source = Path(__file__).with_suffix(".metal").read_text()
    kernel = mx.fast.metal_kernel(name=f"tc_affine_sg_b{bits}_g{group}_m{bm}_c{int(coalesced)}_p{pad}",
                                 input_names=["X", "Q", "S", "B"], output_names=["Y"],
                                 source=source, header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
                                 ensure_row_contiguous=False, compile_options={"math_mode": "safe"})

    def apply(x, q, scale, bias):
        m, k = x.shape
        n = scale.shape[0]
        spec = geometry(m, n, k, bits, group, bm, pad)
        if (x.dtype != mx.float16 or q.dtype != mx.uint32 or scale.dtype != mx.float16
                or bias.dtype != mx.float16 or q.shape != (n, k * bits // 32)
                or scale.shape != (n, k // group) or bias.shape != scale.shape):
            raise ValueError("affine kernel physical storage contract differs")
        raw = kernel(inputs=[x, q, scale, bias], template=[("BM", bm), ("BN", 32), ("BITS", bits), ("GROUP_SIZE", group),
                                                      ("COALESCED", coalesced), ("PAD", pad)],
                     grid=(((n + 31) // 32) * spec["threads"], (m + bm - 1) // bm, 1),
                     threadgroup=(spec["threads"], 1, 1), output_shapes=[(m * n + 32,)],
                     output_dtypes=[mx.float16], init_value=GUARD)[0]
        return raw[:m * n].reshape(m, n), raw[m * n:]
    return apply


def actual_strides(mx, value):
    kernel = mx.fast.metal_kernel(name="tc_actual_2d_strides", input_names=["X"], output_names=["Y"],
                                 source="Y[0] = X_strides[0]; Y[1] = X_strides[1];", ensure_row_contiguous=False)
    result = kernel(inputs=[value], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                    output_shapes=[(2,)], output_dtypes=[mx.int64])[0]
    mx.eval(result)
    return result.tolist()
