"""Host-only paired decoder policy, cache lifecycle and full-work contracts."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_vae_precision_screen as screen


class VaePrecisionScreenTests(unittest.TestCase):
    def report(self, recipe="fp16", control="original"):
        runs = []
        for i, (case, half, seed) in enumerate(screen.points(73)):
            selected = "gpu; joint BF16 LoRA A/B operands"
            actual = recipe if half else control
            fp16 = actual in ("fp16", "combined")
            fused = actual in ("fused", "combined")
            if fp16:
                selected += "; " + screen.MARKER
            if fused:
                selected += "; " + screen.FUSED_MARKER
            if i >= 9:
                selected += "; edit prefix KV snapshot " + ("hit" if i >= 10 else "miss")
            runs.append(dict(request_wall_including_export=7., metrics=dict(model="qwen-image-2.1",
                operation="image."+case, width=512, height=512, steps=6, actual_denoise_steps=6, seed=seed,
                lora_strategy="inference_time", lora_applied_projections=227, acceleration_selection=selected,
                runtime_precision="bf16" + ("+vae_fp16_conv_fp32_state" if fp16 else "") +
                    ("+vae_fp32_fused_norm_silu_nhwc" if fused else ""),
                prompt_cache_hit=i not in (0, 8))))
        return dict(runs=runs)

    def test_fused_and_combined_are_not_relabelled_fp16_controls(self):
        for recipe in ("fp16", "fused", "combined"):
            screen.validate(self.report(recipe), 73, recipe)
            for wrong in ("fp16", "fused", "combined"):
                if wrong != recipe:
                    with self.subTest(recipe=recipe, wrong=wrong), self.assertRaises(ValueError):
                        screen.validate(self.report(recipe), 73, wrong)
        strong = self.report("combined", "fp16")
        screen.validate(strong, 73, "combined", "fp16")
        with self.assertRaises(ValueError):
            screen.validate(strong, 73, "combined", "original")

    def test_recipe_and_strong_edit_prefix_are_observed(self):
        screen.validate(self.report(), 73)
        for key, value in (("actual_denoise_steps", 5), ("lora_applied_projections", 226),
                           ("runtime_precision", "bf16"), ("prompt_cache_hit", False),
                           ("hybrid", {"active": True})):
            bad = self.report(); bad["runs"][11]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(bad, 73)
        bad = self.report()
        bad["runs"][11]["metrics"]["acceleration_selection"] = bad["runs"][11]["metrics"]["acceleration_selection"].replace("snapshot hit", "snapshot miss")
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)

    def test_dump_only_setup_and_edit_geometry_never_leak_to_generation(self):
        args = SimpleNamespace(seed=73, output=Path("/tmp/vae-results"), lora=Path("/tmp/r128.safetensors"),
                               reference=Path("/tmp/reference.png"))
        values = screen.requests(args)
        self.assertEqual(len(values), 16)
        self.assertEqual([i for i, value in enumerate(values) if "dump_tensors" in value], [0, 8])
        self.assertTrue(all("dump" not in value for value in values))
        for i, value in enumerate(values):
            self.assertEqual(value["steps"], 6)
            self.assertEqual(value["execution"], "gpu")
            self.assertEqual(value["qwen21_dit_cache"], "off")
            self.assertEqual("inputs" in value, i >= 8)
            self.assertEqual("qwen21_reference_size" in value, i >= 8)

    def test_controls_and_candidates_keep_matching_seed_pairs(self):
        points = screen.points(73)
        for a, b in ((2, 3), (5, 4), (10, 11), (13, 12)):
            self.assertEqual(points[a][0], points[b][0])
            self.assertEqual(points[a][2], points[b][2])
            self.assertFalse(points[a][1]); self.assertTrue(points[b][1])

    def test_incomplete_or_nonfinite_requests_rejected(self):
        bad = self.report(); bad["runs"].pop()
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)
        for value in (True, 0, -1, float("nan"), float("inf")):
            bad = copy.deepcopy(self.report()); bad["runs"][3]["request_wall_including_export"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                screen.validate(bad, 73)


if __name__ == "__main__":
    unittest.main()
