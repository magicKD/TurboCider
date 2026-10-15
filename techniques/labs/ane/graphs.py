"""Self-contained public Core ML graphs, without product-runtime imports.

The FP16 graph has checkpoint-independent weights. INT8 isolates the shared
feature ABI with one normalized matmul, not the product's W8A8 SwiGLU/LoRA.
"""


def runtime_fp16(rows, hidden, width):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16) for shape in
                            ((rows, hidden), (width, hidden), (width, hidden), (hidden, width))],
                opset_version=ct.target.macOS15)
    def program(x, wg, wu, wd):
        gate = mb.matmul(x=x, y=wg, transpose_y=True)
        up = mb.matmul(x=x, y=wu, transpose_y=True)
        activated = mb.real_div(x=gate, y=mb.add(x=np.float16(1),
            y=mb.exp(x=mb.mul(x=gate, y=np.float16(-1)))))
        return mb.identity(x=mb.matmul(x=mb.mul(x=activated, y=up), y=wd, transpose_y=True), name="y")
    return program


def shared_int8(rows=33, hidden=128, outputs=64):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if not hasattr(ct.target, "macOS26"):
        raise RuntimeError("INT8 model features require the macOS26 target")

    @mb.program(input_specs=[mb.TensorSpec(shape=(hidden, rows), dtype=types.int8),
                            mb.TensorSpec(shape=(outputs, hidden), dtype=types.int8)],
                opset_version=ct.target.macOS26)
    def program(x, w):
        a = mb.dequantize(input=x, scale=np.float16(1 / 128))
        b = mb.dequantize(input=w, scale=np.float16(1 / 128))
        return mb.identity(x=mb.matmul(x=b, y=a), name="y")
    return program
