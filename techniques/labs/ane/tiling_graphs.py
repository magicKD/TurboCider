"""Original public MIL graph variants inspired by ccv/vpipe, not bundle emitters.

One prediction contains all tiles. Gate/up stay paired, reduction completes
before SiLU, and the down reduction is summed before the output is returned.
K sums change finite-precision rounding; this is not a bitwise rewrite.
"""


def spans(size, tile):
    if type(size) is not int or size <= 0 or type(tile) is not int or tile < 0:
        raise ValueError("positive dimension and nonnegative tile required")
    step = tile or size
    return [(first, min(first + step, size)) for first in range(0, size, step)]


def geometry(rows, hidden, width, tile_k, tile_n):
    if any(type(value) is not int or value <= 0 for value in (rows, hidden, width)):
        raise ValueError("positive integer geometry required")
    if not 1 <= rows <= 2048 or not 1 <= hidden <= 4096 or not 1 <= width <= 12288:
        raise ValueError("bounded FFN geometry exceeded")
    up = len(spans(hidden, tile_k)) * len(spans(2 * width, tile_n))
    down = len(spans(width, tile_k)) * len(spans(hidden, tile_n))
    if up + down > 256 or (3 * hidden * width + rows * hidden) * 2 > 512 * 1024 * 1024:
        raise ValueError("graph operation or raw FP16 buffer admission exceeded")
    return {"shape_mhf": [rows, hidden, width], "projection_matmuls_before_passes": up + down,
            "tile_k": tile_k, "tile_n": tile_n}


def program(ct, np, gu, down, rows, tile_k=0, tile_n=0, runtime=False, layout="rows", silu="sigmoid"):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    width, hidden = gu.shape[0] // 2, gu.shape[1]
    geometry(rows, hidden, width, tile_k, tile_n)
    if (gu.shape != (2 * width, hidden) or down.shape != (hidden, width)
            or layout not in ("rows", "channels") or silu not in ("sigmoid", "exp")):
        raise ValueError("paired gate/up, down and supported layout required")
    channels = layout == "channels"
    input_shape = (1, 1, hidden, rows) if channels else (rows, hidden)
    gu_shape = (1, 1, 2 * width, hidden) if channels else gu.shape
    down_shape = (1, 1, hidden, width) if channels else down.shape

    def projection(x, weight, frozen):
        outputs, reduction = frozen.shape
        columns = []
        for n0, n1 in spans(outputs, tile_n):
            products = []
            for k0, k1 in spans(reduction, tile_k):
                if channels:
                    a = mb.slice_by_index(x=x, begin=[0, 0, k0, 0], end=[1, 1, k1, rows])
                    b = (mb.slice_by_index(x=weight, begin=[0, 0, n0, k0], end=[1, 1, n1, k1]) if runtime
                         else np.ascontiguousarray(frozen[n0:n1, k0:k1])[None, None, :, :])
                    products.append(mb.matmul(x=b, y=a))
                else:
                    a = mb.slice_by_index(x=x, begin=[0, k0], end=[rows, k1])
                    b = (mb.slice_by_index(x=weight, begin=[n0, k0], end=[n1, k1]) if runtime
                         else np.ascontiguousarray(frozen[n0:n1, k0:k1]))
                    products.append(mb.matmul(x=a, y=b, transpose_y=True))
            # Balanced tree, unlike a host loop with one predict per tile.
            while len(products) > 1:
                products = [mb.add(x=products[i], y=products[i + 1]) if i + 1 < len(products) else products[i]
                            for i in range(0, len(products), 2)]
            columns.append(products[0])
        return columns[0] if len(columns) == 1 else mb.concat(values=columns, axis=2 if channels else 1)

    def body(x, weight_gu=None, weight_down=None):
        projected = projection(x, weight_gu, gu)
        gate, up = mb.split(x=projected, num_splits=2, axis=2 if channels else 1)
        if silu == "sigmoid":
            value = mb.mul(x=gate, y=mb.sigmoid(x=gate))
        else:
            value = mb.real_div(x=gate, y=mb.add(x=np.float16(1), y=mb.exp(x=mb.mul(x=gate, y=np.float16(-1)))))
        activated = mb.mul(x=value, y=up)
        return mb.identity(x=projection(activated, weight_down, down), name="y")

    specs = [mb.TensorSpec(shape=input_shape, dtype=types.fp16)]
    if runtime:
        specs += [mb.TensorSpec(shape=gu_shape, dtype=types.fp16), mb.TensorSpec(shape=down_shape, dtype=types.fp16)]
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def result(x, weight_gu, weight_down):
            return body(x, weight_gu, weight_down)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def result(x):
            return body(x)
    return result
