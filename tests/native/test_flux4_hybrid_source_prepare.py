import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_hybrid_source_prepare import ANE_WEIGHT_BYTES, RESERVE, plan, prefixes, tensor_bytes


class Flux4HybridSourcePlanTests(unittest.TestCase):
    def headers(self):
        result = {}
        for prefix in prefixes():
            rows, columns = (27648, 384) if prefix.endswith("to_qkv_mlp_proj") else (3072, 1536)
            for suffix, dtype, shape, size in (
                (".weight", "U32", [rows, columns], rows * columns * 4),
                (".scales", "BF16", [rows, columns // 8], rows * columns // 8 * 2),
                (".biases", "BF16", [rows, columns // 8], rows * columns // 8 * 2),
            ):
                result[prefix + suffix] = dict(dtype=dtype, shape=shape, data_offsets=[0, size])
        for index in range(267):
            result[f"unchanged.{index}"] = dict(dtype="BF16", shape=[64], data_offsets=[0, 128])
        return result

    def test_exact_forty_projections_and_untouched_tensor_coverage(self):
        result = plan(self.headers(), 20 << 30)
        self.assertEqual(len(set(prefixes())), 40)
        self.assertEqual((result["derived_tensors"], result["unchanged_tensors"]), (307, 267))
        self.assertEqual(result["dense_replacements_bytes"], 4907335680)
        self.assertEqual(result["replaced_original_bytes"], 1380188160)
        self.assertEqual(result["conservative_total_budget_bytes"],
                         result["derived_payload_bytes"] + 3 * ANE_WEIGHT_BYTES + (1 << 30))

    def test_reserve_is_checked_before_creating_outputs(self):
        required = plan(self.headers(), 20 << 30)["conservative_total_budget_bytes"] + RESERVE
        plan(self.headers(), required)
        with self.assertRaisesRegex(ValueError, "violate2GiB reserve"):
            plan(self.headers(), required - 1)

    def test_wrong_source_dtype_group_geometry_or_partial_model_rejected(self):
        for change in (
            lambda h: h.pop("unchanged.0"),
            lambda h: h[prefixes()[0] + ".weight"].update(dtype="BF16"),
            lambda h: h[prefixes()[0] + ".scales"].update(shape=[27648, 96]),
            lambda h: h[prefixes()[0] + ".biases"].update(data_offsets=[0, 1]),
        ):
            headers = copy.deepcopy(self.headers())
            change(headers)
            with self.assertRaises((ValueError, KeyError)):
                plan(headers, 20 << 30)

    def test_truncated_payload_or_noninteger_dimensions_rejected(self):
        for item in (dict(dtype="U32", shape=[1, 1], data_offsets=[0, 3]),
                     dict(dtype="BF16", shape=[True], data_offsets=[0, 2]),
                     dict(dtype="BF16", shape=[0], data_offsets=[0, 0])):
            with self.assertRaises(ValueError):
                tensor_bytes(item)


if __name__ == "__main__":
    unittest.main()
