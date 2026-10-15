"""Independent exactness and counterexample checks for the model-stage samples."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
import model_stages as stages
from common import metrics


class EncoderAndDit(unittest.TestCase):
    def test_padding_and_bidirectional_prefix(self):
        for seed in (0, 17, 91):
            details = stages.encoder_buckets(seed)["details"]
            self.assertLess(details["masked_bucket"]["max_abs"], 1e-12)
            for key in ("missing_padding_mask", "bidirectional_prefix_kv_is_not_causal"):
                self.assertGreater(details[key]["max_abs"], 1e-4)

    def test_adaln_cross_kv_and_cfg_counterexamples(self):
        for seed in (0, 17, 91):
            details = stages.dit_conditioning(seed)["details"]
            self.assertLess(details["adaln_boundary"]["max_abs"], 1e-12)
            self.assertTrue(all(row["max_abs"] < 1e-12 for row in details["cross_kv_hoisting"]))
            self.assertGreater(details["stale_condition_kv"]["max_abs"], 1e-4)
            self.assertGreater(details["guidance_before_nonlinearity"]["max_abs"], .1)

    def test_adaln_width_checked(self):
        with self.assertRaises(ValueError):
            stages.adaln([[1., 2.]], [1.], [1.])


class CausalAttention(unittest.TestCase):
    def test_gqa_rejects_mismatched_rows_dimensions_and_global_offset(self):
        good = [[[1., 2.]]]
        for q, k, v, offset in ((good, good, [[]], 0), (good, good, [[[1.]]], 0),
                                 (good, good, good, 1), (good, good, good, True)):
            with self.assertRaises(ValueError):
                stages.gqa(q, k, v, offset)

    def test_chunk_prefill_and_single_token_decode(self):
        for seed in (0, 17, 91):
            details = stages.llm_prefill(seed)["details"]
            self.assertTrue(all(row["metrics"]["max_abs"] < 1e-12 for row in details["prefill_chunked_and_decode"]))
            self.assertGreater(details["wrong_chunk_causal_offset"]["max_abs"], 1e-4)
            self.assertEqual(details["gqa_kv_elements"] * 2, details["expanded_mha_kv_elements"])

    def test_explicit_gqa_mapping_equals_repeated_kv(self):
        rng = random.Random(19)
        q = [[[rng.random() for _ in range(4)] for _ in range(3)] for _ in range(4)]
        k, v = [[[[rng.random() for _ in range(4)] for _ in range(3)] for _ in range(2)] for _ in range(2)]
        compact = stages.gqa(q, k, v)
        expanded = stages.gqa(q, [k[i // 2] for i in range(4)], [v[i // 2] for i in range(4)])
        self.assertEqual(compact, expanded)

    def test_identity_position_and_head_errors_do_not_append(self):
        cache = stages.CausalKV("source-A:recipe-A", 1)
        value = [[[1., 2.]]]
        for identity, start in (("source-B", 0), (cache.identity, 1), (cache.identity, True)):
            with self.subTest(identity=identity, start=start), self.assertRaises(ValueError):
                cache.append(value, value, value, identity=identity, start=start)
            self.assertEqual(cache.length, 0)
        cache.append(value, value, value, identity=cache.identity, start=0)
        before = [row[:] for row in cache.k[0]]
        for bad in ([[[1., 2., 3., 4.]]], [[[float("nan"), 2.]]], [[]]):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                cache.append(bad, bad, bad, identity=cache.identity, start=1)
            self.assertEqual(cache.length, 1)
            self.assertEqual(cache.k[0], before)

    def test_cache_owns_appended_arrays(self):
        cache, value = stages.CausalKV("A", 1), [[[1., 2.]]]
        cache.append(value, value, value, identity="A", start=0)
        value[0][0][0] = 99.
        self.assertEqual(cache.k[0][0], [1., 2.])
        self.assertEqual(cache.v[0][0], [1., 2.])

    def test_rope_uses_global_positions(self):
        values = [[[1., 2., 3., 4.], [2., 3., 4., 5.]]]
        all_rows = stages.rope_heads(values, 7)
        self.assertEqual(all_rows[0][1], stages.rope_heads([[values[0][1]]], 8)[0][0])
        self.assertNotEqual(all_rows[0][1], stages.rope_heads([[values[0][1]]], 0)[0][0])
        with self.assertRaises(ValueError):
            stages.rope_heads([[[1., 2., 3.]]], 0)


class SpeculativeProbability(unittest.TestCase):
    def test_exact_law_for_matching_disjoint_and_partial_support(self):
        for p, q in (([.1, .6, .3], [.4, .2, .4]), ([.5, .5], [.5, .5]),
                     ([1., 0.], [0., 1.]), ([0., .5, .5], [.4, .6, 0.])):
            result = stages.speculative_distribution(p, q)
            self.assertLess(metrics(p, result["distribution"])["max_abs"], 1e-12)
            self.assertTrue(0 <= result["acceptance_probability"] <= 1)

    def test_invalid_distribution_rejected(self):
        for p, q in (([], []), ([.5], [1.]), ([float("nan")], [1.]), ([-.1, 1.1], [.5, .5]), ([1.], [.5, .5])):
            with self.subTest(p=p, q=q), self.assertRaises(ValueError):
                stages.speculative_distribution(p, q)


class VaeGeometry(unittest.TestCase):
    def scatter_oracle(self, x, kernel, stride, padding):
        length = (len(x) - 1) * stride - 2 * padding + len(kernel)
        result = [0.] * length
        for index, value in enumerate(x):
            for tap, weight in enumerate(kernel):
                out = index * stride + tap - padding
                if 0 <= out < length:
                    result[out] += value * weight
        return result

    def test_transpose_tiles_against_independent_scatter(self):
        rng = random.Random(31)
        for size, width, stride, padding in ((9, 3, 2, 1), (7, 5, 3, 2), (4, 2, 1, 0), (1, 3, 2, 0)):
            x, kernel = [rng.random() for _ in range(size)], [rng.random() for _ in range(width)]
            expected = self.scatter_oracle(x, kernel, stride, padding)
            for tile in (1, 4, 6):
                output = []
                for begin in range(0, len(expected), tile):
                    output.extend(stages.transpose_conv1d(x, kernel, stride=stride, padding=padding,
                                                         begin=begin, end=min(begin + tile, len(expected))))
                self.assertLess(metrics(expected, output)["max_abs"], 1e-12)

    def test_welford_groups_with_large_offset_and_uneven_tail(self):
        values = [1e6 + i * .125 for i in range(17)]
        state = (0, 0., 0.)
        for begin in range(0, len(values), 3):
            state = stages.merge_moments(state, stages.moments(values[begin:begin + 3]))
        expected = stages.moments(values)
        self.assertEqual(state[0], expected[0])
        self.assertAlmostEqual(state[1], expected[1], places=8)
        self.assertAlmostEqual(state[2], expected[2], places=8)
        self.assertEqual(stages.merge_moments(expected, (0, 0., 0.)), expected)

    def test_bad_tiles_and_nonfinite_stats_rejected(self):
        for options in ({"stride": 0}, {"padding": -1}, {"begin": 3, "end": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                stages.transpose_conv1d([1., 2.], [1., 2., 3.], **options)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                stages.moments(values)


if __name__ == "__main__":
    unittest.main()
