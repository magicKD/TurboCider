"""Independent selected-row/full-K,F Qwen witness; no backend/runner imports."""
def selected(x, gu, down, np):
    rows = sorted({0, x.shape[0] // 3, 2 * x.shape[0] // 3, x.shape[0] - 1})
    cols = sorted({0, down.shape[0] // 3, 2 * down.shape[0] // 3, down.shape[0] - 1})
    width = gu.shape[0] // 2
    projection = np.einsum("mk,nk->mn", x[rows].astype(np.float64), gu.astype(np.float64), optimize=False)
    gate, up = np.split(projection, 2, axis=1)
    small = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + small), small / (1 + small))
    out = np.einsum("mk,nk->mn", (gate * sigmoid) * up, down[cols].astype(np.float64), optimize=False)
    return rows, cols, out, {"sample_gate_min": float(gate.min()), "sample_gate_max": float(gate.max()),
                             "sample_gate_abs_over_11": int(np.sum(np.abs(gate) > 11)), "sample_gate_elements": int(gate.size)}
