import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_comfy_reference_screen import workflow, validate_events
from ltx_comfy_reference_evidence import native_baselines
from base_model_benchmark_evidence import best_known_comfy_comparisons, matched_comparison, ltx_comfy_reference_campaign, qwen_comfy_reference_campaign


class ComfyReferenceTests(unittest.TestCase):
    def events(self):
        events = [dict(type="executing", data=dict(node=n)) for n in ("5", "8", "18", "21", "24", "25", "27")]
        for node, count in (("8", 8), ("18", 3)):
            events += [dict(type="progress", data=dict(node=node, value=i, max=count)) for i in range(1, count + 1)]
        return events

    def test_full_base_two_stage_av_graph_and_flat_dynamic_combo(self):
        graph = workflow(0)
        self.assertEqual(graph["9"]["inputs"], dict(width=352, height=224, length=97, batch_size=1))
        self.assertEqual(graph["10"]["inputs"]["frames_number"], 97)
        self.assertEqual(graph["12"]["inputs"]["video_cfg"], 1.)
        self.assertEqual(graph["12"]["inputs"]["audio_cfg"], 1.)
        self.assertEqual(graph["27"]["inputs"]["format"], "mp4")
        self.assertEqual(graph["27"]["inputs"]["format.codec"], "h264")
        self.assertNotIsInstance(graph["27"]["inputs"]["format"], dict)
        self.assertEqual(workflow(1)["11"]["inputs"]["noise_seed"], 43)
        self.assertEqual(workflow(1)["19"]["inputs"]["noise_seed"], 43)
        self.assertFalse(any("Lora" in n["class_type"] or "Prompt" in n["class_type"] for n in graph.values()))

    def test_all8_plus3_steps_upscale_decoders_export_must_execute(self):
        validate_events(self.events(), 0)
        for node in ("8", "18", "21", "24", "25", "27"):
            events = self.events()
            events.append(dict(type="execution_cached", data=dict(nodes=[node])))
            with self.assertRaises(ValueError):
                validate_events(events, 0)
        events = self.events()
        events.pop()
        with self.assertRaises(ValueError):
            validate_events(events, 0)

    def test_portable_complete_media_memory_and_one_warm_not_median(self):
        path = ROOT / "docs/design/validation/ltx25-comfy039-reference-current-20261012.json"
        data = json.loads(path.read_text())
        self.assertNotIn("/Users/", path.read_text())
        self.assertIs(data["same_window_native_pair"], False)
        self.assertEqual(data["warm_sample_count"], 1)
        self.assertEqual(len(data["output_checks"]), 2)
        self.assertTrue(all(c["video_frames"] == 97 and c["audio_finite"] is True and c["complete_decode"] is True for c in data["output_checks"]))
        self.assertIs(data["memory"]["verified"]["complete"], True)
        self.assertGreater(data["comparisons"]["comfy_over_native_quality"], 1.75)
        self.assertGreater(data["comparisons"]["comfy_over_native_fastest"], 1.81)

    def test_native_best_preserves_large_approximation_and_same_source(self):
        data = json.loads((ROOT / "docs/design/validation/ltx25-comfy039-reference-current-20261012.json").read_text())
        baseline = native_baselines(data["model_sha256"])
        self.assertGreater(baseline["fastest_quality"]["mae_255"], 25.)
        hashes = dict(data["model_sha256"])
        hashes[next(iter(hashes))] = "0" * 64
        with self.assertRaises(ValueError):
            native_baselines(hashes)

    def test_best_record_reuse_is_never_promoted_to_paired_headline(self):
        values = best_known_comfy_comparisons()
        self.assertEqual(len(values), 4)
        self.assertTrue(all(v["same_window_pair"] is False and v["user_requested_baseline_reuse"] is True for v in values))
        self.assertAlmostEqual(values[0]["speedup"], 1.4766817273882067)
        self.assertAlmostEqual(values[1]["speedup"], 1.5567139452865246)
        for campaign in (qwen_comfy_reference_campaign("docs/design/validation/qwen21-comfy039-reference-current-20261012.json"),
                         ltx_comfy_reference_campaign("docs/design/validation/ltx25-comfy039-reference-current-20261012.json")):
            campaign["in_current_scope"] = True
            self.assertIs(campaign["native_comparison_valid"], False)
            with self.assertRaises(ValueError):
                matched_comparison(campaign, "native-frozen", "comfy-gpu", "not paired")

    def test_flux_whole_index_is_not_a_single_shard_or_runtime_proof(self):
        data = json.loads((ROOT / "docs/design/validation/comfy-flux4-local-format-admission-20261012.json").read_text())
        self.assertEqual(data["required_shards"], ["0.safetensors", "1.safetensors"])
        self.assertEqual(data["required_parameters"], 387)
        self.assertIs(data["index_admitted"], False)
        self.assertIs(data["compatible_complete_generation_verified"], False)
        self.assertIs(data["latency_comparison_valid"], False)


if __name__ == "__main__":
    unittest.main()
