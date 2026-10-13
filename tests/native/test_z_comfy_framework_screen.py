import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_comfy_framework_screen import ORDER, ROUTES, launch_command, validate_comfy_row, validate_trial, workflow
from z_comfy_framework_evidence import event_spans, validate_cohort
from base_model_benchmark_evidence import z_comfy_campaign


class ComfyProtocolTests(unittest.TestCase):
    def row(self, index=0):
        return dict(index=index, seed=43 if index == 3 else 42, request_wall_including_export=1.,
            actual_steps=list(range(1, 9)), executed_nodes=["4", "8", "9", "10"], cached_nodes=[], history_status="success")

    def test_repeat_prompt_changes_seed_and_still_requires_sampling(self):
        self.assertEqual(workflow("prompt", 2)["8"]["inputs"]["seed"], 42)
        self.assertEqual(workflow("prompt", 3)["8"]["inputs"]["seed"], 43)
        row = self.row(3)
        row["executed_nodes"].remove("4")
        row["cached_nodes"] = ["4"]
        validate_comfy_row(row, 3)
        row["actual_steps"] = []
        with self.assertRaises(ValueError):
            validate_comfy_row(row, 3)

    def test_cached_sampler_decode_or_fresh_encoder_is_not_inference(self):
        for node in ("4", "8", "9", "10"):
            row = self.row()
            row["cached_nodes"].append(node)
            with self.assertRaises(ValueError):
                validate_comfy_row(row, 0)
            row = self.row()
            row["executed_nodes"].remove(node)
            with self.assertRaises(ValueError):
                validate_comfy_row(row, 0)

    def test_success_and_all_steps_are_required(self):
        for edit in (dict(history_status="error"), dict(actual_steps=list(range(8))),
                     dict(actual_steps=list(range(1, 8))), dict(seed=43)):
            with self.assertRaises(ValueError):
                validate_comfy_row(dict(self.row(), **edit), 0)
        for wall in (0, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_comfy_row(dict(self.row(), request_wall_including_export=wall), 0)

    def test_owned_instance_and_explicit_gpu_recipe(self):
        cmd = launch_command(Path("/comfy"), Path("/python"), Path("/output"), "comfy-default", 8190)
        for flag in ("--disable-all-custom-nodes", "--disable-api-nodes", "--user-directory", "--database-url"):
            self.assertIn(flag, cmd)
        self.assertIn("sqlite:///:memory:", cmd)
        self.assertNotIn("--gpu-only", cmd)
        cmd = launch_command(Path("/comfy"), Path("/python"), Path("/output"), "comfy-sdpa-gpu", 8190)
        for flag in ("--gpu-only", "--use-pytorch-cross-attention", "--bf16-unet", "--bf16-text-enc", "--bf16-vae"):
            self.assertIn(flag, cmd)

    def test_owned_server_exit_and_complete_memory_required(self):
        trial = dict(route="comfy-default", memory=dict(verified=dict(complete=True)),
            report=dict(status="complete", server_terminal=True, runs=[self.row(i) for i in range(4)]))
        validate_trial(trial)
        for edit in (lambda t: t["memory"]["verified"].update(complete=False),
                     lambda t: t["report"].update(server_terminal=False),
                     lambda t: t["report"].update(status="failed")):
            invalid = copy.deepcopy(trial)
            edit(invalid)
            with self.assertRaises(ValueError):
                validate_trial(invalid)

    def test_complete_palindrome_and_actual_medians_required(self):
        data = dict(status="complete_diagnostic", count=4, order=list(ORDER),
            trials=[dict(index=i, route=r, report=dict(runs=[dict(request_wall_including_export=1.)] * 4)) for i, r in enumerate(ORDER)],
            medians={r: dict(samples=[1.] * 4, fresh_warm_seconds=1.) for r in ROUTES})
        with patch("z_comfy_framework_evidence.validate_trial"):
            validate_cohort(data)
            for change in (dict(status="complete_precheck_only"), dict(count=1), dict(order=list(ROUTES)), dict(trials=data["trials"][:-1])):
                with self.assertRaises(ValueError):
                    validate_cohort(dict(data, **change))
            data["medians"][ROUTES[0]]["fresh_warm_seconds"] = .001
            with self.assertRaises(ValueError):
                validate_cohort(data)

    def test_node_envelopes_are_host_event_spans(self):
        events = [dict(type="executing", seconds=t, data=dict(node=n)) for n, t in (("4", 1.), ("8", 2.), ("9", 10.))]
        events.append(dict(type="execution_success", seconds=11., data={}))
        self.assertEqual(event_spans(events), {"4": 1., "8": 8., "9": 1.})

    def evidence(self):
        return json.loads((ROOT / "docs/design/validation/z-comfy039-current-20261012.json").read_text())

    def test_published_same_window_media_memory_and_versions(self):
        path = "docs/design/validation/z-comfy039-current-20261012.json"
        campaign = z_comfy_campaign(path)
        self.assertEqual(campaign["best_measured_comfy_route"], "comfy-default")
        self.assertEqual({r["warm_sample_count"] for r in campaign["routes"]}, {4})
        data = self.evidence()
        self.assertEqual(len(data["output_checks"]), 16)
        self.assertTrue(all(c["rgba_equal"] is True for c in data["output_checks"]))
        self.assertNotIn("/Users/", json.dumps(data))
        self.assertTrue(all(t["memory"]["verified"]["complete"] is True for t in data["trials"]))

    def test_old_version_or_mismatched_package_cannot_be_current_headline(self):
        for edit in (lambda d: d.update(comfy_commit="old"),
                     lambda d: d.update(comfy_local_changes=[" M model.py"]),
                     lambda d: d["trials"][2]["report"]["system_stats"]["system"].update(comfyui_version="0.32.0"),
                     lambda d: d["trials"][2]["report"]["system_stats"]["system"]["comfy_package_versions"][0].update(installed="old")):
            data = self.evidence()
            edit(data)
            with patch("base_model_benchmark_evidence.source", return_value=(data, {})), self.assertRaises(ValueError):
                z_comfy_campaign("ignored")

    def test_slow_preset_or_cross_cohort_samples_cannot_be_headline(self):
        for edit in (lambda d: d["comparisons"].update(best_measured_comfy_route="comfy-sdpa-gpu"),
                     lambda d: d["routes"][0].update(fresh_warm_median=.001),
                     lambda d: d["routes"][0]["fresh_warm_seconds"].__setitem__(0, .001)):
            data = self.evidence()
            edit(data)
            with patch("base_model_benchmark_evidence.source", return_value=(data, {})), self.assertRaises(ValueError):
                z_comfy_campaign("ignored")


if __name__ == "__main__":
    unittest.main()
