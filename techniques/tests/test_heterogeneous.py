"""Topological equivalence, ABI index and optional-device negative controls."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
import heterogeneous as h


class TopologyAndLayout(unittest.TestCase):
    def test_parallel_branch_projection_and_wrong_sequential_rewrite(self):
        for seed in (0, 17, 91):
            details = h.parallel_branches(seed)["details"]
            self.assertLess(details["concat_output_projection_vs_partial_sum"]["max_abs"], 1e-12)
            self.assertGreater(details["wrong_sequential_rewrite"]["max_abs"], .01)

    def test_complete_head_partials_and_missing_head(self):
        for seed in (0, 17, 91):
            details = h.attention_head_shards(seed)["details"]
            self.assertLess(details["complete_heads_through_output_projection"]["max_abs"], 1e-12)
            self.assertGreater(details["missing_head_despite_same_output_shape"]["max_abs"], .01)

    def test_strided_address_maps_for_uneven_shapes(self):
        for rows, channels in ((1, 1), (3, 7), (7, 3), (5, 11)):
            offsets = h.strided_offsets([1, channels, 1, rows], [rows * channels, 1, rows * channels, channels])
            self.assertEqual(offsets, [row * channels + col for col in range(channels) for row in range(rows)])
            self.assertEqual(set(offsets), set(range(rows * channels)))

    def test_aliased_negative_or_unbounded_strides_rejected(self):
        for shape, strides in (([2, 3], [1, 1]), ([2], [0]), ([2], [-1]), ([5000], [1]), ([True], [1]), ([2, 3], [1])):
            with self.assertRaises(ValueError):
                h.strided_offsets(shape, strides)

    def test_segmented_pairing_and_gate_order(self):
        for seed in (0, 17, 91):
            details = h.segmented_ffn(seed)["details"]
            self.assertLess(details["paired_segments_through_down_partials"]["max_abs"], 1e-12)
            self.assertGreater(details["swapped_gate_up_abi"]["max_abs"], .01)

    def test_default_producer_layout_is_not_consumer_layout(self):
        details = h.consumer_strides()["details"]
        self.assertEqual(details["producer_written_row_major"]["max_abs"], 0.)
        self.assertGreater(details["default_channel_major_read_as_row_major"]["max_abs"], 1.)


class OptionalPlacement(unittest.TestCase):
    def profiles(self):
        return {"gpu": h.DeviceCost(.1, 1.), "ane": h.DeviceCost(.8, .55), "cpu": h.DeviceCost(.4, 1.8, 1.8)}

    def test_optional_cpu_and_full_gpu_fallback(self):
        details = h.optional_devices()["details"]
        self.assertEqual(details["optional_cpu_declined"]["selected"]["widths"]["cpu"], 0)
        self.assertTrue(all(width > 0 for width in details["profitable_three_way_scenario"]["selected"]["widths"].values()))
        self.assertEqual(details["all_accelerator_options_declined"]["selected"]["widths"], {"gpu": 8, "ane": 0, "cpu": 0})

    def test_all_enumerated_selected_widths_cover_exactly(self):
        for width in (2, 4, 8):
            plan = h.choose_partition(width, 1, self.profiles())
            self.assertEqual(sum(plan["selected"]["widths"].values()), width)
            self.assertIsNone(plan["selected"]["rejected"])
            self.assertLessEqual(plan["selected"]["wall_ms"], plan["gpu_baseline_ms"])

    def test_expensive_join_can_make_full_gpu_optimal(self):
        result = h.choose_partition(8, 1, self.profiles(), join_ms=100.)
        self.assertEqual(result["selected"]["widths"], {"gpu": 8, "ane": 0, "cpu": 0})

    def test_invalid_geometry_and_nonfinite_profiles(self):
        for width, quantum in ((0, 1), (5, 2), (100, 1), (True, 1)):
            with self.assertRaises(ValueError):
                h.choose_partition(width, quantum, self.profiles())
        for values in ((-.1, 1., 1.), (.1, 0., 1.), (.1, 1., .9), (.1, float("nan"), 1.)):
            with self.assertRaises(ValueError):
                h.DeviceCost(*values)
        with self.assertRaises(ValueError):
            h.DeviceCost(.1, 1e308).cost(8)


class HazardsAndPrecision(unittest.TestCase):
    def test_overlap_must_respect_physical_aliases(self):
        a = {"allocation": "A", "offset": 0, "bytes": 64, "write": True}
        self.assertFalse(h.independent_regions(a, {**a, "offset": 32}))
        self.assertTrue(h.independent_regions(a, {**a, "offset": 64}))
        self.assertTrue(h.independent_regions(a, {**a, "allocation": "B"}))
        self.assertTrue(h.independent_regions({**a, "write": False}, {**a, "write": False}))
        with self.assertRaises(ValueError):
            h.independent_regions(a, {**a, "bytes": -1})

    def test_more_overlap_is_not_necessarily_less_wall_time(self):
        value = h.resource_hazards()["details"]
        serial = value["coarse_identity_serial_scenario"]["makespan"]
        self.assertLess(value["independent_identity_scenario"]["makespan"], serial)
        self.assertEqual(value["saturated_bandwidth_scenario_no_wall_gain"]["makespan"], serial)

    def test_state_and_parameter_precision_both_matter(self):
        value = h.precision_islands()["details"]
        self.assertLess(value["shadow_error"], 1e-4)
        self.assertGreater(value["low_state_error"], .01)
        self.assertGreater(value["state_only_error"], .001)
        self.assertLess(value["shadow_error"], value["state_only_error"])


if __name__ == "__main__":
    unittest.main()
