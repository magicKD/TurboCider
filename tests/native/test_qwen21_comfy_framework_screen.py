import copy
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_comfy_framework_screen import ORDER, ROUTES, sigmas, workflow, launch_command, validate_comfy_row, validate_trial
from qwen21_comfy_framework_evidence import validate_cohort, validate_reference_cohort, existing_native_best


class QwenComfyProtocolTests(unittest.TestCase):
    def row(self, index=0):
        return dict(index=index, seed=30 if index == 3 else 29, request_wall_including_export=1.,
            actual_steps=list(range(1, 41)), executed_nodes=["4", "8", "9", "10"], cached_nodes=[], history_status="success")

    def test_base_schedule_has40_euler_steps_and_original_terminal_stretch(self):
        values = sigmas()
        self.assertEqual(len(values), 41)
        self.assertEqual((values[0], values[-1]), (1., 0.))
        self.assertAlmostEqual(values[-2], .02, places=6)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        text = workflow("prompt", 0)["12"]["inputs"]["sigmas"]
        parsed = np.array([float(v) for v in re.findall(r"[-+]?(?:\d*\.*\d+)", text)], dtype=np.float32)
        np.testing.assert_array_equal(parsed, np.array(values, dtype=np.float32))

    def test_stock_graph_full_source_lossless_cache_and_repeat_new_seed(self):
        graph = workflow("prompt", 0)
        self.assertEqual(graph["8"]["inputs"]["cfg"], 1.)
        self.assertEqual(graph["7"]["inputs"], dict(model=["1", 0], device="auto", dtype="default"))
        self.assertEqual(graph["11"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["8"]["inputs"]["noise_seed"], 29)
        self.assertEqual(workflow("prompt", 3)["8"]["inputs"]["noise_seed"], 30)
        self.assertFalse(any("Lora" in n["class_type"] for n in graph.values()))

    def test_gpu_residency_and_sdpa_are_separate_presets(self):
        for route in ROUTES[2:]:
            cmd = launch_command(Path("/comfy"), Path("/python"), Path("/output"), route, 8190)
            self.assertEqual("--gpu-only" in cmd, route != "comfy-default")
            self.assertEqual("--use-pytorch-cross-attention" in cmd, route == "comfy-sdpa-gpu")
            self.assertIn("--disable-all-custom-nodes", cmd)
            self.assertIn("sqlite:///:memory:", cmd)

    def test_cached_sampler_or_fresh_encoder_and_partial_steps_rejected(self):
        validate_comfy_row(self.row(), 0)
        for change in (dict(actual_steps=list(range(1, 40))), dict(actual_steps=[]), dict(seed=30), dict(history_status="error")):
            with self.assertRaises(ValueError):
                validate_comfy_row(dict(self.row(), **change), 0)
        for node in ("4", "8", "9", "10"):
            row = self.row()
            row["cached_nodes"].append(node)
            with self.assertRaises(ValueError):
                validate_comfy_row(row, 0)

    def test_repeat_may_cache_only_conditioning_not_sampling(self):
        row = self.row(3)
        row["executed_nodes"].remove("4")
        row["cached_nodes"].append("4")
        validate_comfy_row(row, 3)
        row["executed_nodes"].remove("8")
        with self.assertRaises(ValueError):
            validate_comfy_row(row, 3)

    def test_success_and_owned_exit_complete_memory_required(self):
        trial = dict(route="comfy-gpu", memory=dict(verified=dict(complete=True)),
            report=dict(status="complete", server_terminal=True, runs=[self.row(i) for i in range(4)]))
        validate_trial(trial)
        for edit in (lambda t: t["memory"]["verified"].update(complete=False),
                     lambda t: t["report"].update(server_terminal=False), lambda t: t["report"]["runs"].pop()):
            wrong = copy.deepcopy(trial)
            edit(wrong)
            with self.assertRaises(ValueError):
                validate_trial(wrong)

    def test_full_palindrome_schedule_and_actual_four_fresh_samples_required(self):
        data = dict(status="complete_diagnostic", count=4, steps=40, order=list(ORDER),
            schedule=dict(sigmas=sigmas(), rule="40Euler,mu(.5-.9 at256-8192),terminal.02 then0", cfg=1, seed=29, repeat_seed=30),
            trials=[dict(index=i, route=r, report=dict(runs=[dict(request_wall_including_export=1.)] * 4)) for i, r in enumerate(ORDER)],
            medians={r: dict(samples=[1.] * 4, fresh_warm_seconds=1.) for r in ROUTES})
        with patch("qwen21_comfy_framework_evidence.validate_trial"):
            validate_cohort(data)
            for edit in (lambda d: d.update(count=1), lambda d: d.update(steps=4), lambda d: d["trials"].pop(),
                         lambda d: d["schedule"].update(sigmas=[1., 0.]),
                         lambda d: d["medians"][ROUTES[0]].update(fresh_warm_seconds=.001)):
                wrong = copy.deepcopy(data)
                edit(wrong)
                with self.assertRaises(ValueError):
                    validate_cohort(wrong)

    def test_reference_only_has_no_native_execution_or_fake_pair(self):
        path = ROOT / "docs/design/validation/qwen21-comfy039-reference-current-20261012.json"
        data = json.loads(path.read_text())
        self.assertIs(data["reference_only"], True)
        self.assertIs(data["same_window_native_pair"], False)
        self.assertEqual(data["order"], list(ROUTES[2:]))
        self.assertEqual(len(data["trials"]), 3)
        self.assertEqual({len(r["fresh_warm_seconds"]) for r in data["routes"]}, {2})
        self.assertEqual(data["comparisons"]["best_measured_comfy_route"], "comfy-gpu")
        self.assertNotIn("/Users/", path.read_text())
        raw = ROOT / data["evidence"]
        if raw.is_file():
            validate_reference_cohort(json.loads(raw.read_text()))

    def test_existing_native_best_rejects_different_source_components(self):
        data = json.loads((ROOT / "docs/design/validation/qwen21-comfy039-reference-current-20261012.json").read_text())
        hashes = {str(ROOT / k): v for k, v in data["model_sha256"].items()}
        baseline = existing_native_best(hashes)
        self.assertAlmostEqual(baseline["native_frozen_seconds"], 31.058807333465666)
        self.assertEqual(len(baseline["native_samples"]["native-frozen"]), 4)
        hashes[next(iter(hashes))] = "0" * 64
        with self.assertRaises(ValueError):
            existing_native_best(hashes)


if __name__ == "__main__":
    unittest.main()
