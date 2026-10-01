"""CPU-only independent numerical contracts for complete dynamic SwiGLU."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/coreml"))
from runtime_ane_swiglu_int8_candidate import hadamard, lora_delta, prepare_inputs, reference, specification


def dense_hadamard(group):
    result = np.ones((1, 1), dtype=np.float64)
    while result.shape[0] < group:
        result = np.block([[result, result], [result, -result]])
    return result / np.sqrt(group)


def delta(x, a, b, scale):
    # Independent float64 oracle, not the candidate's preparation helper.
    return np.einsum("ir,or->io", np.einsum("ik,rk->ir", x.astype(np.float64), a), b) * scale


def relative(actual, expected):
    return np.linalg.norm(actual.astype(np.float64) - expected) / max(np.linalg.norm(expected), 1e-12)


def fixture():
    rng = np.random.default_rng(179)
    rows, hidden, width, rank = 5, 64, 96, 4
    x = rng.normal(0, .6, (rows, hidden)).astype(np.float32)
    weights = [rng.normal(0, .15, shape).astype(np.float32)
               for shape in ((width, hidden), (width, hidden), (hidden, width))]
    adapters = [(rng.normal(0, .12, (rank, n)).astype(np.float32),
                 rng.normal(0, .20, (m, rank)).astype(np.float32), scale)
                for n, m, scale in ((hidden, width, .9), (hidden, width, -.6), (width, hidden, .7))]
    return x, weights, adapters


class SwiGluTests(unittest.TestCase):
    def test_fwht_order_and_same_operand_basis(self):
        rng = np.random.default_rng(103)
        for group in (1, 2, 16, 32):
            x = rng.normal(size=(5, 64)).astype(np.float32)
            w = rng.normal(size=(7, 64)).astype(np.float32)
            dense = dense_hadamard(group)
            rotated = (x.reshape(5, -1, group).astype(np.float64) @ dense).reshape(x.shape)
            np.testing.assert_allclose(hadamard(x, group), rotated, atol=5e-7, rtol=2e-6)
            expected = np.einsum("ik,jk->ij", x.astype(np.float64), w.astype(np.float64), optimize=False)
            np.testing.assert_allclose(np.einsum("ik,jk->ij", hadamard(x, group), hadamard(w, group), optimize=False),
                                       expected, atol=5e-6, rtol=2e-5)
            if group > 1:
                self.assertGreater(relative(np.einsum("ik,jk->ij", hadamard(x, group), w, optimize=False), expected), .1)

    def test_complete_lora_before_nonlinearity_and_original_hidden_down(self):
        x, weights, adapters = fixture()
        wg, wu, wd = weights
        dg = delta(x, *adapters[0])
        du = delta(x, *adapters[1])
        gate = np.einsum("ik,jk->ij", x.astype(np.float64), wg.astype(np.float64), optimize=False) + dg
        up = np.einsum("ik,jk->ij", x.astype(np.float64), wu.astype(np.float64), optimize=False) + du
        exact_h = gate / (1 + np.exp(-gate)) * up
        exact_y = np.einsum("ik,jk->ij", exact_h, wd.astype(np.float64), optimize=False) + delta(exact_h, *adapters[2])
        for precision in ("fp16", "qdq_int8"):
            for group in (1, 2, 16, 32):
                with self.subTest(precision=precision, group=group):
                    spec = specification(5, 64, 96, precision, group, 4)
                    inputs = prepare_inputs(x, *weights, spec, dg=dg, du=du,
                                            down_a=adapters[2][0], down_b=adapters[2][1], down_scale=adapters[2][2])
                    actual = reference(inputs, spec)
                    limit = .003 if precision == "fp16" else .035
                    self.assertLess(relative(actual["h"], exact_h), limit)
                    self.assertLess(relative(actual["y"], exact_y), limit)
                    base_inputs = {**inputs, "down_b": np.zeros_like(inputs["down_b"])}
                    base = reference(base_inputs, spec)
                    correction = delta(actual["h"], inputs["down_a"], inputs["down_b"], inputs["down_scale"][0])
                    expected = (base["y"].astype(np.float64) + correction).astype(np.float16)
                    np.testing.assert_allclose(actual["y"], expected, rtol=.001, atol=.001)
                    if group > 1:
                        wrong_down = delta(hadamard(actual["h"], group), *adapters[2])
                        self.assertGreater(relative(wrong_down, delta(actual["h"], *adapters[2])), .1)
        # This counterexample detects adding the gate delta after SiLU.
        base_gate = np.einsum("ik,jk->ij", x.astype(np.float64), wg.astype(np.float64), optimize=False)
        wrong_h = (base_gate / (1 + np.exp(-base_gate)) + dg) * up
        self.assertGreater(relative(wrong_h, exact_h), .05)
        np.testing.assert_allclose(lora_delta(x, *adapters[0]), dg, rtol=2e-5, atol=2e-7)

    def test_hidden_scale_is_fp32_and_zero_rows_are_safe(self):
        for precision in ("fp16", "qdq_int8"):
            spec = specification(2, 4, 4, precision, 4, 1)
            x = np.zeros((2, 4), np.float32)
            wg, wu = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32)
            # Corrected h=60000. Its FWHT maximum=120000 exceeds FP16
            # although both original h and the identity down output are finite.
            inputs = prepare_inputs(x, wg, wu, np.eye(4, dtype=np.float32), spec,
                                    dg=np.full((2, 4), 20., np.float32), du=np.full((2, 4), 3000., np.float32))
            result = reference(inputs, spec)
            np.testing.assert_array_equal(result["h"], np.full((2, 4), 60000, np.float16))
            np.testing.assert_array_equal(result["y"], result["h"])
            zero = reference(prepare_inputs(x, wg, wu, np.eye(4, dtype=np.float32), spec), spec)
            self.assertTrue(all(np.all(value == 0) for value in zero.values()))

    def test_zero_lora_base_control_and_input_ownership(self):
        x, weights, _ = fixture()
        originals = [value.copy() for value in (x, *weights)]
        for precision in ("fp16", "qdq_int8"):
            spec = specification(5, 64, 96, precision, 32, 4)
            implicit = prepare_inputs(x, *weights, spec)
            explicit = prepare_inputs(x, *weights, spec,
                                      dg=np.zeros((5, 96), np.float32), du=np.zeros((5, 96), np.float32),
                                      down_a=np.zeros((4, 96), np.float32), down_b=np.zeros((64, 4), np.float32))
            self.assertTrue(all(value.dtype == np.float16 and value.flags.c_contiguous for value in implicit.values()))
            for name in implicit:
                np.testing.assert_array_equal(implicit[name], explicit[name])
            for name, result in reference(implicit, spec).items():
                np.testing.assert_array_equal(result, reference(explicit, spec)[name])
        for actual, original in zip((x, *weights), originals):
            np.testing.assert_array_equal(actual, original)

    def test_rejects_invalid_shapes_scales_and_nonfinite_output(self):
        for arguments in ((2, 64, 95, "fp16", 32, 4), (2, 64, 96, "fp16", 3, 4),
                          (2, 64, 96, "fp16", 32, 0), (2, 64, 96, "int8", 32, 4)):
            with self.assertRaises(ValueError):
                specification(*arguments)
        x, weights, _ = fixture()
        spec = specification(5, 64, 96, "fp16", 32, 4)
        for scale in (True, float("inf"), "1", 1e9):
            with self.assertRaises(ValueError):
                prepare_inputs(x, *weights, spec, down_scale=scale)
        for invalid in (np.full_like(x, np.nan), np.full_like(x, 1e-12), np.ones((5, 63), np.float32)):
            with self.assertRaises(ValueError):
                prepare_inputs(invalid, *weights, spec)
        inputs = prepare_inputs(x, *weights, spec)
        for invalid in (np.zeros_like(inputs["x_scale"]), -inputs["x_scale"]):
            with self.assertRaises(ValueError):
                reference({**inputs, "x_scale": invalid}, spec)
        with self.assertRaises(ValueError):
            reference({**inputs, "dg": np.full((5, 96), 60000, np.float16),
                       "du": np.full((5, 96), 60000, np.float16)}, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
