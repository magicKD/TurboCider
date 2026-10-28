"""Host-only pinned six-step schedule and actual-receipt validation."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_student_prefix_screen as screen


class StudentPrefixScreenTests(unittest.TestCase):
    def test_rank_pins_match_native_and_unknown_names_cannot_select_schedule(self):
        native = (Path(__file__).resolve().parents[2] / "native/models/qwen21/viggle_adapter.hpp").read_text()
        for filename, pin in screen.ADAPTER_PINS.items():
            self.assertIn(filename, native)
            self.assertIn(pin["sha256"], native)
            self.assertEqual(screen.adapter_pin(Path("/tmp") / filename), pin)
        with self.assertRaises(ValueError):
            screen.adapter_pin(Path("/tmp/renamed.safetensors"))

    def report(self, precision="original"):
        runs = []
        for i, (case, _, snapshot, seed, hit, conditioning_hit) in enumerate(screen.schedule(73, precision)):
            selection = "gpu"
            if screen.actual_precision(precision, i) == "joint":
                selection += "; experimental joint BF16 LoRA A/B operands, F32 ranks"
            if snapshot:
                selection += "; edit prefix KV snapshot " + ("hit" if hit else "miss")
            m = dict(model="qwen-image-2.1", operation="image." + case, width=512, height=512,
                steps=6, actual_denoise_steps=6, seed=seed, prompt_cache_hit=conditioning_hit,
                lora_strategy="inference_time", lora_applied_projections=227, acceleration_selection=selection)
            runs.append(dict(metrics=m, request_wall_including_export=8.))
        return dict(runs=runs)

    def test_requests_never_apply_edit_only_resize_to_generation(self):
        args = SimpleNamespace(seed=73, output=Path("/tmp/results"), lora=Path("/tmp/adapter"),
                               reference=Path("/tmp/ref.png"), reference_size=512)
        values = screen.requests(args)
        self.assertEqual(len(values), 11)
        for i, value in enumerate(values):
            self.assertEqual(value["steps"], 6)
            self.assertEqual(value["qwen21_dit_cache"], "off")
            self.assertEqual(value["loras"], [dict(path="/tmp/adapter", strength=1., role="transformer")])
            if i < 4:
                self.assertNotIn("qwen21_reference_size", value)
                self.assertNotIn("inputs", value)
            else:
                self.assertEqual(value["qwen21_reference_size"], 512)
                self.assertEqual(value["inputs"][0]["path"], "/tmp/ref.png")

    def test_matching_precision_and_full_student_work_are_required(self):
        for precision in ("original", "joint", "switch"):
            screen.validate(self.report(precision), 73, precision)
        with self.assertRaises(ValueError):
            screen.validate(self.report("original"), 73, "joint")
        with self.assertRaises(ValueError):
            screen.validate(self.report("joint"), 73, "original")
        for key, value in (("actual_denoise_steps", 5), ("steps", 40), ("seed", 74),
                           ("lora_strategy", "in_memory_merge"), ("lora_applied_projections", 226),
                           ("prompt_cache_hit", False), ("hybrid", {"enabled": True}),
                           ("student_ffn_reuse", {"enabled": True}), ("qwen21_dbcache", {"cached_steps": 1})):
            report = self.report()
            report["runs"][7]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(report, 73, "original")

    def test_precision_switch_invalidates_before_replay_and_restores_original(self):
        seq = screen.schedule(73, "switch")
        self.assertEqual([p[4] for p in seq], [False, True, False, True, False, True])
        self.assertEqual([screen.actual_precision("switch", i) for i in range(6)],
                         ["original", "original", "joint", "joint", "original", "original"])
        bad = self.report("switch")
        bad["runs"][2]["metrics"]["acceleration_selection"] = "joint BF16 LoRA A/B operands; edit prefix KV snapshot hit"
        with self.assertRaisesRegex(ValueError, "prefix hit"):
            screen.validate(bad, 73, "switch")

    def test_prefix_hit_is_observed_not_inferred_from_requested_flag(self):
        report = self.report()
        report["runs"][7]["metrics"]["acceleration_selection"] = "gpu; edit prefix KV snapshot miss"
        with self.assertRaises(ValueError):
            screen.validate(report, 73, "original")
        report = self.report()
        report["runs"][6]["metrics"]["acceleration_selection"] = "gpu"
        with self.assertRaises(ValueError):
            screen.validate(report, 73, "original")

    def test_missing_rows_or_nonfinite_wall_fail_closed(self):
        report = self.report()
        report["runs"].pop()
        with self.assertRaises(ValueError):
            screen.validate(report, 73, "original")
        for value in (0, -1, True, float("nan"), float("inf")):
            report = copy.deepcopy(self.report())
            report["runs"][7]["request_wall_including_export"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                screen.validate(report, 73, "original")


if __name__ == "__main__":
    unittest.main()
