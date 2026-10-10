import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_gguf_ffn_dense_screen import ROWS, validate_record


def record(component):
    return dict(schema="tc-qwen21-real-ffn-dense-screen-v1", qualification_passed=False,
        component=component, source_sha256="a" * 64, verification_bytes=0, registration_cache_hits=1,
        selected_tensor_count=2 if component == "dit" else 3, dense_blocks_retained=1, group_size=32,
        fresh_packed_arguments_dynamic=True, K=4096, intermediate=12288, gate_bits=4, down_bits=8,
        direct_u4_mpp=False,
        dense_block_bytes=2 * (24576 * 4096 + 4096 * 12288), source_read_bytes=1, packed_capacity_bytes=1,
        mlx_logical_peak_bytes=1, initial_dequantize_seconds=.01, warmups=3, hot_samples=9,
        cases=[dict(M=m, complete_output_elements=m * 4096, fresh_relative_l2=.01, cached_relative_l2=.01,
            fresh_cached_relative_l2=0, packed_seconds=[.02] * 9, fresh_dequant_dense_seconds=[.01] * 9,
            cached_dense_seconds=[.009] * 9) for m in ROWS])


class FfnDenseScreenTests(unittest.TestCase):
    def test_direct_u4_has_actual_q4_planes_errors_and_hot_spans(self):
        data = record("dit"); data.update(direct_u4_mpp=True, down_bits=4)
        for case in data["cases"]:
            case.update(direct_u4_relative_l2=.01, direct_u4_mpp_seconds=[.02] * 9)
        validate_record(data, "dit", "a" * 64, True)
        for field, value in (("direct_u4_relative_l2", float("nan")), ("direct_u4_relative_l2", .051),
                              ("direct_u4_mpp_seconds", [.02] * 8)):
            invalid = copy.deepcopy(data); invalid["cases"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_record(invalid, "dit", "a" * 64, True)
        data["down_bits"] = 8
        with self.assertRaises(ValueError):
            validate_record(data, "dit", "a" * 64, True)

    def test_actual_source_bounded_block_and_full_output_contract(self):
        for component in ("dit", "encoder"):
            validate_record(record(component), component, "a" * 64)
            for name, value in (("verification_bytes", 1), ("registration_cache_hits", True),
                ("selected_tensor_count", 4), ("source_sha256", "b" * 64), ("dense_blocks_retained", 32),
                ("group_size", 64), ("fresh_packed_arguments_dynamic", False), ("dense_block_bytes", 1),
                ("gate_bits", 6), ("intermediate", 0), ("initial_dequantize_seconds", float("nan"))):
                invalid = record(component); invalid[name] = value
                with self.subTest(component=component, name=name), self.assertRaises(ValueError):
                    validate_record(invalid, component, "a" * 64)

    def test_sampled_or_nonfinite_outputs_and_missing_hot_spans_rejected(self):
        for name, value in (("complete_output_elements", 8), ("M", True), ("fresh_relative_l2", .051),
            ("cached_relative_l2", float("nan")), ("fresh_cached_relative_l2", .001),
            ("packed_seconds", [.02] * 8), ("fresh_dequant_dense_seconds", [0] * 9),
            ("cached_dense_seconds", [float("inf")] * 9)):
            invalid = record("dit"); invalid["cases"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_record(invalid, "dit", "a" * 64)
        invalid = copy.deepcopy(record("dit")); invalid["cases"] = invalid["cases"][:1]
        with self.assertRaises(ValueError):
            validate_record(invalid, "dit", "a" * 64)


if __name__ == "__main__":
    unittest.main()
