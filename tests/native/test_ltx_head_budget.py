import sys
import unittest
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from analyze_ltx_head_budget import optimize, evaluate


class HeadBudgetTests(unittest.TestCase):
    def test_allocation_respects_budget(self):
        allocation, error, used = optimize([1, 2, 4], [[9, 4, 1], [8, 7, 0]], 5)
        self.assertEqual(sum(allocation), used)
        self.assertEqual(allocation, [4, 1])
        self.assertEqual(error, 9.)

    def test_optimum_beats_uniform_when_curves_differ(self):
        allocation, error, used = optimize([2, 4], [[10, 1], [9, 8]], 6)
        self.assertEqual(allocation, [4, 2])
        self.assertLess(error, 10 + 8)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            optimize([1, 2], [[1], [1]], 0)
        with self.assertRaises(ValueError):
            optimize([1, 2], [[1], [1]], 5)

    def test_dynamic_program_matches_brute_force_with_head_specific_costs(self):
        keeps = [1, 2, 4]
        errors = [[7, 6, 2], [10, 3, 2], [4, 5, 0]]
        costs = [[2, 5, 9], [3, 4, 7], [1, 2, 8]]
        for budget in (6, 8, 11, 15, 24):
            best = min((sum(errors[h][i] for h, i in enumerate(indices)),
                        sum(costs[h][i] for h, i in enumerate(indices)))
                       for indices in itertools.product(range(3), repeat=3)
                       if sum(costs[h][i] for h, i in enumerate(indices)) <= budget)
            allocation, error, used = optimize(keeps, errors, budget, costs)
            self.assertEqual((error, used), best)
            self.assertEqual(used, sum(costs[h][keeps.index(k)] for h, k in enumerate(allocation)))

    def test_frozen_evaluation_does_not_reoptimize(self):
        out = evaluate([1, 2], [[1, 0], [9, 0]], [10, 10], [[1, 2], [1, 2]], [2, 1], 1)
        self.assertAlmostEqual(out["relative_l2"], (9 / 20) ** .5)
        self.assertEqual(out["selected_cell_ratio"], 1.5)
        with self.assertRaises(ValueError):
            evaluate([1, 2], [[1, 0]], [1], [[1, 2]], [4], 1)
