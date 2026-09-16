import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from analyze_ltx_fine_pooling import fine_output, summaries
from analyze_ltx_route_recall import approximate_output, block_means, block_sums, round_bf16, softmax


class FinePoolingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.q = round_bf16(rng.normal(size=(3, 8)))
        self.k = round_bf16(rng.normal(size=(129, 8)))
        self.v = round_bf16(rng.normal(size=(129, 4)))

    def test_64_token_summaries_match_existing_diagnostic(self):
        kc, vs, counts = summaries(self.k, self.v, 64)
        np.testing.assert_array_equal(kc, round_bf16(block_means(self.k)))
        np.testing.assert_array_equal(vs, round_bf16(block_sums(self.v)))
        np.testing.assert_array_equal(counts, [64, 64, 1])
        for selected in ([], [0], [1, 2]):
            out, _ = fine_output(self.q, self.k, self.v, selected, 64)
            expected = approximate_output(self.q, self.k, self.v, kc, vs, selected)
            np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-12)

    def test_unit_width_reconstructs_dense(self):
        dense = softmax(self.q.astype(np.float64) @ self.k.astype(np.float64).T / np.sqrt(8)) @ self.v
        for selected in ([], [0], [1, 2]):
            out, work = fine_output(self.q, self.k, self.v, selected, 1)
            np.testing.assert_allclose(out, dense, atol=1e-12, rtol=1e-12)
            self.assertEqual(work["represented_keys"], 129)

    def test_selected_parent_blocks_remain_exact_at_every_width(self):
        dense = softmax(self.q.astype(np.float64) @ self.k.astype(np.float64).T / np.sqrt(8)) @ self.v
        for width in (64, 32, 16, 8, 4, 2, 1):
            out, work = fine_output(self.q, self.k, self.v, [0, 1, 2], width)
            np.testing.assert_allclose(out, dense, atol=1e-12, rtol=1e-12)
            self.assertEqual(work["remote_summaries"], 0)

    def test_tail_and_parent_mapping(self):
        _, work = fine_output(self.q, self.k, self.v, [0], 16)
        self.assertEqual(work, dict(exact_tokens=64, remote_summaries=5, represented_keys=69))
        _, work = fine_output(self.q, self.k, self.v, [2], 16)
        self.assertEqual(work, dict(exact_tokens=1, remote_summaries=8, represented_keys=9))

    def test_split_recovers_two_constant_subblocks(self):
        k = np.ones((64, 1), np.float32); k[32:] *= -1
        v = k.copy(); q = np.ones((1, 1), np.float32)
        coarse, _ = fine_output(q, k, v, [], 64)
        fine, _ = fine_output(q, k, v, [], 32)
        dense = softmax(q @ k.T) @ v
        self.assertAlmostEqual(float(coarse[0, 0]), 0.)
        np.testing.assert_allclose(fine, dense, atol=1e-7)

    def test_invalid_inputs(self):
        for width in (0, 3, 128, True):
            with self.assertRaises(ValueError):
                summaries(self.k, self.v, width)
        for selected in ([-1], [3], [1.5], [[0]]):
            with self.assertRaises(ValueError):
                fine_output(self.q, self.k, self.v, selected, 16)
