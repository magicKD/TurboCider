import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_w8a8_routing", ROOT / "tools/validation/z_image_w8a8_routing.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ZImageW8A8RoutingTests(unittest.TestCase):
    def test_greedy_route_accounts_for_canceling_output_errors(self):
        errors = np.array([[[1., 0.]], [[-1.1, 0.]], [[0., 2.]]], np.float32)
        self.assertEqual(MODULE.route_groups(errors, 2, np), [0, 1])
        with self.assertRaisesRegex(ValueError, "routing error"):
            MODULE.route_groups(np.array([[[np.nan]]], np.float32), 1, np)

    def test_calibration_row_selection_excludes_caption(self):
        sample = np.ones((1056, 2), np.float16)
        sample[1025, :] = 1000
        sampled = MODULE.select_image_rows([sample], 8, 1024, np)
        self.assertGreater(sampled.shape[0], 0)
        self.assertLessEqual(sampled.shape[0], 8)
        self.assertEqual(sampled.shape[1], 2)
        self.assertEqual(float(sampled.max()), 1.)

    def test_caption_weight_balances_relative_error_not_raw_activation_size(self):
        image = np.array([[[0.]], [[1.]], [[2.]]], np.float32)
        caption = np.array([[[200.]], [[0.]], [[0.]]], np.float32)
        combined = MODULE.combine_region_errors(
            image, caption, np.array([[1.]], np.float32),
            np.array([[100.]], np.float32), 1., np)
        self.assertEqual(MODULE.route_groups(combined, 1, np), [1])
        self.assertTrue(np.array_equal(MODULE.combine_region_errors(
            image, caption, np.array([[1.]], np.float32),
            np.array([[100.]], np.float32), 0., np), image))
        with self.assertRaisesRegex(ValueError, "reference energy"):
            MODULE.combine_region_errors(image, caption, np.zeros((1, 1), np.float32),
                                         np.ones((1, 1), np.float32), 1., np)

    def test_constrained_route_preserves_image_training_budget(self):
        image = np.array([[[1.]], [[1.02]], [[3.]]], np.float32)
        caption = np.array([[[10.]], [[0.]], [[0.]]], np.float32)
        chosen, swaps = MODULE.constrained_caption_route(image, caption, [0], .05, np)
        self.assertEqual(chosen, [1])
        self.assertEqual([(swap["old"], swap["new"]) for swap in swaps], [(0, 1)])
        self.assertLessEqual(swaps[0]["image_sse_vs_initial"], 1.05)
        self.assertEqual(MODULE.constrained_caption_route(image, caption, [0], 0., np)[0], [0])
        with self.assertRaisesRegex(ValueError, "constrained caption"):
            MODULE.constrained_caption_route(image, caption, [0], float("nan"), np)


if __name__ == "__main__":
    unittest.main()
