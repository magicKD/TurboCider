import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_current_framework_screen import ORDER, ROUTES, validate_report


def reference(route="mflux-bf16", count=4):
    bits = 4 if route == "mflux-q4" else None
    reduced = route == "mflux-fp16"
    coverage = {name: dict(required_targets=10, covered_targets=10, missing=[])
                for name in ("transformer", "text_encoder", "vae")}
    if bits:
        for name, linears in (("transformer", 153), ("text_encoder", 252)):
            coverage[name].update(bits=4, group_size=32, quantized_linears=linears)
    return dict(engine=route, precision="mlx.core.bfloat16", quantization=bits,
        compute_precision_name="float16" if reduced else None,
        transformer_floating_parameter_dtypes={"mlx.core.bfloat16": 10, "mlx.core.float16": 300} if reduced else {"mlx.core.bfloat16": 310},
        text_embedding_dtype="mlx.core.bfloat16", reference_cache_limit_bytes=0,
        required_weight_coverage=coverage,
        runs=[dict(index=index, request_wall_including_export=10. + index, actual_steps=[0, 1, 2, 3],
                   actual_quantization=bits, loras=[], prompt_cache_hit=False, prompt_cache_entries=0,
                   generation_parameters={"compute_precision": "float16"} if reduced else {})
              for index in range(count)])


class CurrentFluxFrameworkTests(unittest.TestCase):
    def test_full_seven_route_palindrome_has_both_sources_and_actual_hybrid(self):
        self.assertEqual(len(ORDER), 14)
        self.assertEqual(ORDER, tuple(reversed(ORDER)))
        self.assertTrue(all(ORDER.count(route) == 2 for route in ROUTES))
        self.assertEqual(set(ROUTES), {"native-bf16", "native-hybrid", "native-q4", "native-predecoded",
                                      "mflux-bf16", "mflux-fp16", "mflux-q4"})

    def test_all_declared_reference_recipes_and_short_prechecks(self):
        for route in ("mflux-bf16", "mflux-fp16", "mflux-q4"):
            validate_report(reference(route), route)
            validate_report(reference(route, 1), route, 1)
        with self.assertRaises(ValueError):
            validate_report(reference(count=1), "mflux-bf16", 4)

    def test_bf16_control_cannot_silently_enable_partial_float16(self):
        report = reference()
        report["compute_precision_name"] = "float16"
        with self.assertRaisesRegex(ValueError, "actual upstream compute precision"):
            validate_report(report, "mflux-bf16")

    def test_float16_label_requires_actual_parameters_and_each_output_metadata(self):
        for field, value in (("transformer_floating_parameter_dtypes", {"mlx.core.bfloat16": 310}),
                             ("text_embedding_dtype", "mlx.core.float16")):
            report = reference("mflux-fp16")
            report[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "partial FP16"):
                validate_report(report, "mflux-fp16")
        report = reference("mflux-fp16")
        report["runs"][3]["generation_parameters"] = {}
        with self.assertRaisesRegex(ValueError, "output compute metadata"):
            validate_report(report, "mflux-fp16")

    def test_quantized_reference_needs_both_complete_g32_components(self):
        for name, field, value in (("transformer", "quantized_linears", 152),
                                  ("text_encoder", "quantized_linears", 251),
                                  ("text_encoder", "group_size", 64)):
            report = reference("mflux-q4")
            report["required_weight_coverage"][name][field] = value
            with self.subTest(name=name, field=field), self.assertRaisesRegex(ValueError, "complete matched"):
                validate_report(report, "mflux-q4")

    def test_reference_cache_steps_base_and_memory_policy_are_actual_not_labels(self):
        for field, value in (("prompt_cache_hit", True), ("prompt_cache_entries", 1),
                             ("actual_steps", [0]), ("loras", ["adapter"]),
                             ("request_wall_including_export", float("nan"))):
            report = reference()
            report["runs"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(report, "mflux-bf16")
        report = reference()
        report["reference_cache_limit_bytes"] = 1024 << 20
        with self.assertRaisesRegex(ValueError, "actual retained cache0"):
            validate_report(report, "mflux-bf16")

    def test_predecode_requires_initial_decode_and_every_full_compiled_graph(self):
        report = dict(engine="native-predecoded", runs=[dict(index=index,
            request_wall_including_export=1., actual_compiled_single_blocks=96,
            native_event_counts=dict(transformer_block=128, affine_predecoded_gpu_route=1,
                                    affine_predecode_projection=153 if index == 0 else 0),
            metrics=dict(model="flux2-klein-9b", operation="image.generate", width=512, height=512,
                         steps=4, seed=42, text_tokens=512, prompt_cache_hit=index == 3, hybrid=None))
            for index in range(4)])
        validate_report(report, "native-predecoded")
        for event, value in (("affine_predecode_projection", 153), ("transformer_block", 127)):
            bad = copy.deepcopy(report)
            bad["runs"][1]["native_event_counts"][event] = value
            with self.subTest(event=event), self.assertRaises(ValueError):
                validate_report(bad, "native-predecoded")


if __name__ == "__main__":
    unittest.main()
