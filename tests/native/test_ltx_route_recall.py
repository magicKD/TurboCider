"""CPU checks for the sampled exact-mass routing diagnostic."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/native"))
from analyze_ltx_route_recall import (block_means, round_bf16, rotate, score_routes,
                                     approximate_output, softmax, gated_output_energy)
from analyze_ltx_route_recall import centroid_distribution, mass_budget, block_variances


class RouteRecallTests(unittest.TestCase):
    def test_pooled_tail_matches_scalar_weighted_formula(self):
        q = np.array([[1., 0.], [-1., .5]])
        k = np.zeros((65, 2)); k[:32, 0] = 1; k[32:64, 0] = -1; k[64] = [2, 1]
        v = np.arange(65, dtype=float).reshape(-1, 1)
        kc = block_means(k); vs = np.array([[v[:64].sum()], [v[64:].sum()]])
        for selected in ([], [0], [1]):
            reference = []
            for query in q:
                numerator = denominator = 0.
                for block in range(2):
                    begin, end = block * 64, min(65, (block + 1) * 64)
                    if block in selected:
                        for token in range(begin, end):
                            weight = np.exp(np.dot(query, k[token]) / np.sqrt(2))
                            denominator += weight
                            numerator += weight * v[token, 0]
                    else:
                        weight = np.exp(np.dot(query, kc[block]) / np.sqrt(2))
                        denominator += weight * (end - begin)
                        numerator += weight * vs[block, 0]
                reference.append([numerator / denominator])
            np.testing.assert_allclose(approximate_output(q, k, v, kc, vs, selected), reference)

    def test_gate_and_zero_energy_accounting(self):
        ref = np.ones((2, 4)); candidate = ref * 2
        error, norm = gated_output_energy(ref, candidate, np.array([0., .5]))
        np.testing.assert_array_equal(error, [0, 1])
        np.testing.assert_array_equal(norm, [0, 1])

    def test_approximate_output_all_exact_matches_dense(self):
        rng = np.random.default_rng(3)
        q, k, v = rng.normal(size=(2, 8)), rng.normal(size=(65, 8)), rng.normal(size=(65, 4))
        kc, vs = block_means(k), np.array([v[:64].sum(0), v[64:].sum(0)])
        out = approximate_output(q, k, v, kc, vs, [0, 1])
        ref = softmax(q @ k.T / np.sqrt(8)) @ v
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    def test_key_variance_distinguishes_equal_means(self):
        q = np.ones((64, 4), np.float32)
        k = np.zeros((128, 4), np.float32)
        k[64::2, 0], k[65::2, 0] = 2, -2
        kc = block_means(k)
        kv = block_variances(k)
        self.assertEqual(kv[1, 0], 4)
        mass, ranks = score_routes(q, q[:2], k, kc, kv)
        np.testing.assert_array_equal(ranks['centroid'], [0, 1])
        np.testing.assert_array_equal(ranks['diagonal_key_variance'], [1, 0])
        self.assertTrue(np.all(mass[:, 1] > mass[:, 0]))
        estimate = centroid_distribution(q, kc, 128, kv)
        self.assertGreater(estimate[1], estimate[0])
        np.testing.assert_allclose(centroid_distribution(q, kc, 128, np.zeros_like(kv)),
                                   centroid_distribution(q, kc, 128))
        with self.assertRaises(ValueError):
            centroid_distribution(q, kc, 128, -np.ones_like(kv))

    def test_mass_budget_minimal_and_stable(self):
        p = np.array([.4, .4, .15, .05])
        np.testing.assert_array_equal(mass_budget(p, .75), [0, 1])
        np.testing.assert_array_equal(mass_budget(p, 1), [0, 1, 2, 3])
        np.testing.assert_array_equal(mass_budget([1.], .99), [0])
        for invalid in ([0., 0.], [-.1, 1.1], [float('nan')], [[1.]]):
            with self.assertRaises(ValueError):
                mass_budget(invalid, .9)
        for target in (0, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                mass_budget(p, target)

    def test_centroid_distribution_counts_tail(self):
        q = np.zeros((2, 4), np.float32)
        kc = np.zeros((2, 4), np.float32)
        np.testing.assert_allclose(centroid_distribution(q, kc, 65), [64 / 65, 1 / 65])
        with self.assertRaises(ValueError):
            centroid_distribution(q, kc, 129)

    def test_rope_identity_and_quarter_turn(self):
        x = np.array([[1, 2, 3, 4]], dtype=np.float32)
        ones = np.ones((1, 2), np.float32)
        zeros = np.zeros_like(ones)
        np.testing.assert_array_equal(rotate(x, ones, zeros), x)
        np.testing.assert_array_equal(rotate(x, zeros, ones), [[-3, -4, 1, 2]])

    def test_round_to_nearest_even(self):
        x = np.array([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256], np.float32)
        np.testing.assert_array_equal(round_bf16(x), [1, 1 + 1 / 64, -1])

    def test_tail_mass_and_stable_ties(self):
        key = np.zeros((65, 4), np.float32)
        q = np.zeros((2, 4), np.float32)
        mass, ranks = score_routes(q, q, key, block_means(key))
        np.testing.assert_allclose(mass, [[64 / 65, 1 / 65]] * 2)
        for rank in ranks.values():
            np.testing.assert_array_equal(rank, [0, 1])

    def test_oracle_aggregate_upper_bound(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(64, 8)).astype(np.float32)
        k = rng.normal(size=(257, 8)).astype(np.float32)
        mass, ranks = score_routes(q, q[::8], k, round_bf16(block_means(k)))
        np.testing.assert_allclose(mass.sum(axis=1), 1)
        for keep in (1, 2, 5):
            optimum = mass[:, ranks['in_sample_oracle'][:keep]].sum(axis=1).mean()
            for policy in ('centroid', 'sample_logmeanexp', 'diagonal_key_variance'):
                actual = mass[:, ranks[policy][:keep]].sum(axis=1).mean()
                self.assertLessEqual(actual, optimum + 1e-12)


if __name__ == '__main__':
    unittest.main()
