"""Independent full FP64 oracle for bounded synthetic frozen-FP16 subgraphs.

No ANE/MLX imports or candidate implementations. NumPy is passed explicitly;
einsum without BLAS/tuning keeps the reference independent of GPU recipes.
"""


def reference(case, value, weights, np):
    x = value.astype(np.float64)
    if case == "reduction":
        return np.sum(x, axis=-1, keepdims=True, dtype=np.float64)
    def dot(a, b):
        return np.einsum("mk,kn->mn", a, b.astype(np.float64), optimize=False)
    gate = x if case == "activation" else dot(x, weights[0])
    if case == "matmul":
        return gate
    activation = gate / (1. + np.exp(-gate))
    if case in ("activation", "projection-silu"):
        return activation
    if case != "ffn":
        raise ValueError("unknown reference case")
    return dot(activation * dot(x, weights[1]), weights[2])
