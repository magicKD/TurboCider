import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from base_model_benchmark_evidence import measured


class BaseBenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "docs/benchmark/results-20261011.json").read_text())

    def test_expanded_goal_remains_active_without_promoting_previous_stage(self):
        self.assertEqual(self.report["status"], "partial_expanded_evidence_catalog")
        self.assertIs(self.report["goal_complete"], False)
        self.assertTrue((ROOT / self.report["completion_audit"]).is_file())
        self.assertIn("not release", self.report["goal_completion_scope"])
        self.assertIs(self.report["qualification_passed"], False)
        self.assertTrue((ROOT / self.report["expansion_status"]).is_file())
        self.assertTrue(self.report["missing_current_comparisons"])
        self.assertTrue(self.report["unmeasured_or_inapplicable"])
        self.assertGreaterEqual(len(self.report["campaigns"]), 46)

    def test_complete_actual_sample_arrays_and_medians(self):
        for campaign in self.report["campaigns"]:
            self.assertTrue(campaign["timing_scope"] if "timing_scope" in campaign else campaign["note"])
            for route in campaign["routes"]:
                samples = route["warm_seconds"]
                self.assertEqual(route["warm_sample_count"], len(samples))
                self.assertTrue(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in samples))
                self.assertAlmostEqual(route["warm_median_seconds"], statistics.median(samples), places=12)

    def test_bound_raw_evidence_when_locally_available(self):
        for campaign in self.report["campaigns"]:
            binding = campaign["evidence"]
            path = ROOT / binding["path"]
            self.assertRegex(binding["sha256"], r"^[a-f0-9]{64}$")
            if path.is_file():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), binding["sha256"], str(path))
            elif binding["path"].startswith("docs/"):
                self.fail("committed evidence missing: " + str(path))
            # Raw outputs are deliberately not distributed with the repo.

    def test_failed_quality_and_busy_campaign_not_promoted(self):
        ltx = next(c for c in self.report["campaigns"] if c["model"] == "ltx-2.5-distilled")
        self.assertEqual({r["quality_status"] for r in ltx["routes"] if r["route"] != "gpu"},
                         {"experimental-quality-fail"})
        busy = next(c for c in self.report["campaigns"] if c["evidence"]["status"] == "complete_busy_diagnostic")
        self.assertIn("both CPU load checks failed", busy["note"])
        current = next(c for c in self.report["campaigns"] if c["model"] == "qwen-image-2.1-current-BF16-native")
        self.assertEqual({row["route"] for row in current["routes"]}, {"native-gpu", "native-frozen"})
        self.assertIn("rendered noise", current["note"])

    def test_docs_local_links_and_no_machine_specific_paths(self):
        for doc in sorted((ROOT / "docs/benchmark").glob("*.md")):
            for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
                if "://" not in link:
                    self.assertTrue((doc.parent / link.split("#")[0]).exists(), (str(doc), link))
            self.assertNotIn("/Users/", doc.read_text())
        for path in (ROOT / "docs/benchmark/results-20261011.json",
                     ROOT / "docs/benchmark/local-model-inventory-20261011.json",
                     ROOT / "docs/design/validation/ltx-base-current-20261011.json"):
            self.assertNotIn("/Users/", path.read_text())

    def test_inventory_component_availability_never_grants_runtime_proof(self):
        data = json.loads((ROOT / "docs/benchmark/local-model-inventory-20261011.json").read_text())
        self.assertEqual(len(data["rows"]), 15)
        for row in data["rows"]:
            self.assertIs(row["runtime_support_verified"], False)
            if row["missing_components"]:
                self.assertIs(row["assets_present"], False)

    def test_missing_or_invalid_latency_is_not_a_success_sample(self):
        for cold, warm in (([], [1]), ([1], []), ([True], [1]), ([1], [float("nan")]), ([1], [0])):
            with self.subTest(cold=cold, warm=warm), self.assertRaises(ValueError):
                measured("test", cold, warm)

    def test_new_valid_framework_cohorts_keep_distinct_precision_and_scope(self):
        specs = (
            ("qwen-image-2.1-valid-framework-cohort", 5, 40),
            ("z-image-turbo-valid-framework-cohort", 7, 8),
            ("flux2-klein-9b-stock-reference-precisions", 3, 4),
            ("qwen-image-2.1-cache0-framework-cohort", 3, 40),
        )
        for model, count, steps in specs:
            campaign = next(c for c in self.report["campaigns"] if c["model"] == model)
            self.assertEqual((len(campaign["routes"]), campaign["steps"]), (count, steps))
            self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
            evidence = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
            self.assertIs(evidence["qualification_passed"], False)
            self.assertIs(evidence["goal_complete"], False)
        flux = next(c for c in self.report["campaigns"] if c["model"] == "flux2-klein-9b-stock-reference-precisions")
        self.assertIs(flux["native_comparison_valid"], False)
        self.assertIn("no cross-cohort native ranking", flux["note"])
        cached = json.loads((ROOT / "docs/design/validation/qwen21-cache0-framework-current-20261011.json").read_text())
        self.assertEqual(cached["reference_cache_mib"], 0)
        for trial in cached["trials"]:
            if trial["route"].startswith("mflux-"):
                self.assertEqual(trial["report"]["reference_cache_limit_bytes"], 0)

    def test_latest_flux_pair_uses_actual_coverage_and_current_same_cohort_ratio(self):
        campaign = next(c for c in self.report["campaigns"] if c["model"] == "flux2-klein-9b-current-BF16-native-mflux")
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertIs(data["reference_required_weight_coverage_verified"], True)
        self.assertEqual(data["order"], ["native", "mflux", "mflux", "native"])
        self.assertAlmostEqual(data["fresh_warm_speedup"],
            data["medians"]["mflux"]["fresh_warm_seconds"] / data["medians"]["native"]["fresh_warm_seconds"])
        self.assertLess(data["fresh_warm_speedup"], 1.2)
        cached = next(c for c in self.report["campaigns"] if c["model"] == "flux2-klein-9b-cache0-BF16-native-mflux")
        proof = json.loads((ROOT / cached["evidence"]["path"]).read_text())
        self.assertEqual(proof["reference_cache_mib"], 0)
        self.assertLess(proof["fresh_warm_speedup"], 1)
        self.assertIn("retained negative result", cached["note"])

    def test_ltx_full_media_modes_and_clock_boundaries_are_not_pooled(self):
        batch = next(c for c in self.report["campaigns"] if c["model"] == "ltx-2.5-full-config-persistent-batch")
        clean = next(c for c in self.report["campaigns"] if c["model"] == "ltx-2.5-full-config-clean-exec")
        self.assertEqual((len(batch["routes"]), len(clean["routes"])), (6, 2))
        self.assertTrue(all(row["warm_sample_count"] == 2 for row in (*batch["routes"], *clean["routes"])))
        self.assertTrue(all(row["native_timing_boundary"] == "native request_wall" for row in batch["routes"]))
        self.assertTrue(all("estimate" in row["native_timing_boundary"] for row in clean["routes"]))
        proof = json.loads((ROOT / batch["evidence"]["path"]).read_text())
        self.assertEqual(sum(len(row["memory"]) for row in proof["trials"]), 20)
        self.assertTrue(all(row["metrics"]["frame_count_compared"] == 97 for row in proof["quality"]))
        self.assertTrue(all(item["complete"] for row in proof["trials"] for item in row["memory"]))

    def test_flux_compiled_candidates_need_actual_branches_not_just_request_flag(self):
        for model in ("flux2-klein-9b-compiled-GPU-cohort", "flux2-klein-9b-compiled-deferred-encoder-cohort", "flux2-klein-9b-compiled-fused-encoder-cohort"):
            campaign = next(c for c in self.report["campaigns"] if c["model"] == model)
            data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
            compiled = [trial for trial in data["trials"] if trial["engine"] == "native-compiled"]
            self.assertEqual(len(compiled), 2)
            self.assertTrue(all(row["actual_compiled_single_blocks"] == 96 for trial in compiled for row in trial["report"]["runs"]))
            self.assertTrue(all(route["warm_sample_count"] == 4 for route in campaign["routes"]))
            self.assertGreater(data["compiled_gpu_comparison"]["native_eager_over_compiled"], 1)

    def test_complete_z_private_grid_uses_candidate_own_controls(self):
        campaigns = [row for row in self.report["campaigns"] if row["model"].startswith("z-image-turbo-private-a")]
        self.assertEqual(len(campaigns), 14)
        self.assertTrue(all({row["route"] for row in campaign["routes"]} == {"native-gpu", "native-frozen", "native-private"} for campaign in campaigns))
        data = json.loads((ROOT / campaigns[0]["evidence"]["path"]).read_text())
        self.assertEqual((data["best_tested_private"]["channels"], data["best_tested_private"]["join"]), (4096, "f32"))
        self.assertLess(data["best_tested_private"]["frozen_over_private"], 1)
        self.assertTrue(all(row["warm_sample_count"] == 4 for campaign in campaigns for row in campaign["routes"]))

    def test_flux9_complete_hybrid_cohort_retains_no_fresh_win_and_source_receipts(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "flux2-klein-9b-experimental-hybrid-four-route-cohort")
        self.assertEqual(campaign["date"], "2026-10-12")
        self.assertEqual({row["route"] for row in campaign["routes"]}, {"native", "native-compiled", "native-hybrid", "mflux"})
        self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((len(data["trials"]), len(data["rgb_comparisons"])), (8, 24))
        self.assertTrue(all(trial["memory"]["complete"] for trial in data["trials"]))
        self.assertLess(abs(data["hybrid_comparison"]["hybrid_reduction_vs_compiled_percent"]), 0.1)
        self.assertLess(data["hybrid_comparison"]["reference_over_hybrid"], 1)
        self.assertTrue(data["reference_required_weight_coverage_verified"])
        for trial in data["trials"]:
            if trial["engine"] == "native-hybrid":
                for row in trial["report"]["runs"]:
                    actual = row["metrics"]["hybrid"]
                    self.assertEqual(actual["runtime_calls_session_total"], 96 * (row["index"]+1))
                    self.assertEqual((actual["hidden"], actual["block_count"], actual["bucket"]), (4096,24,1536))
                    self.assertEqual(actual["ane_mlp_range"], [0,4096])
                    self.assertTrue(actual["checkpoint_sha256_verified"])
                    self.assertFalse(actual["runtime_failed"])
        self.assertNotIn("/Users/", json.dumps(data))

    def test_flux9_hybrid_candidate_does_not_replace_proven_gpu_selection(self):
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        chosen = next(row for row in configs["configurations"] if row["model"] == "flux2-klein-9b")
        self.assertEqual(chosen["request"]["execution"], "gpu")
        candidate = configs["experimental_configurations"][0]
        data = json.loads((ROOT / candidate["evidence"]).read_text())
        self.assertEqual(candidate["fresh_warm_median_seconds"], data["medians"]["native-hybrid"]["fresh_warm_seconds"])
        self.assertEqual(candidate["compiled_gpu_control_seconds"], data["medians"]["native-compiled"]["fresh_warm_seconds"])
        self.assertEqual(candidate["request"]["execution"], "gpu_ane")
        self.assertTrue(candidate["request"]["allow_approximation"])
        self.assertIn("not a new best/default", candidate["selection_note"])
        self.assertEqual(candidate["request"]["ane_manifest"], candidate["compiled_manifest"])

    def test_flux9_matched_q4_has_all_planes_actual_groups_and_no_false_speedup(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "flux2-klein-9b-matched-dit-q4-g32-cohort")
        self.assertEqual({row["route"] for row in campaign["routes"]},{"native-bf16","native-q4","mflux-bf16","mflux-q4"})
        self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((len(data["trials"]),len(data["quality"])),(8,32))
        self.assertEqual(data["conversion"]["converted_components"],["transformer"])
        self.assertTrue(data["conversion"]["components"]["transformer"]["full_compressed_payload_equal"])
        self.assertLess(data["comparison"]["native_q4_reduction_percent"],0)
        self.assertLess(data["comparison"]["mflux_q4_reduction_percent"],0)
        for trial in data["trials"]:
            self.assertTrue(trial["memory"]["complete"])
            if trial["route"] == "mflux-q4":
                loaded = trial["report"]["required_weight_coverage"]["transformer"]
                self.assertEqual((loaded["required_targets"],loaded["quantized_linears"],loaded["bits"],loaded["group_size"]),(539,153,4,32))
        self.assertNotIn("/Users/",json.dumps(data))

    def test_flux9_memory_candidate_remains_distinct_from_fastest_gpu(self):
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        candidate = configs["memory_configurations"][0]
        data = json.loads((ROOT / candidate["evidence"]).read_text())
        self.assertEqual(candidate["fresh_warm_median_seconds"],data["medians"]["native-q4"]["fresh_warm_seconds"])
        self.assertEqual(candidate["request"]["execution"],"gpu")
        self.assertFalse(candidate["request"]["compile_gpu"])
        self.assertTrue(candidate["request"]["allow_approximation"])
        self.assertIn("not fastest/default",candidate["selection_note"])
        if (ROOT / "models/FLUX.2-klein-9B").exists():
            self.assertTrue((ROOT / candidate["model_root"] / "transformer/0.safetensors").is_file())

    def test_retained_winners_are_from_their_actual_cohort(self):
        data = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        self.assertIs(data["goal_complete"], False)
        self.assertIs(data["qualification_passed"], False)
        for configuration in data["configurations"]:
            evidence = json.loads((ROOT / configuration["evidence"]).read_text())
            if configuration.get("selected_route", "").startswith("native"):
                self.assertEqual(configuration["request"]["execution"], "gpu")
                self.assertIs(configuration["request"]["allow_approximation"], False)
                self.assertEqual(configuration["fresh_warm_median_seconds"], evidence["medians"][configuration["selected_route"]]["fresh_warm_seconds"])
                self.assertTrue((ROOT / configuration["library"]).is_file() or not (ROOT / configuration["model_root"]).exists())
                continue
            routes = {row["route"]: row for row in evidence["routes"]}
            self.assertEqual(configuration["fresh_warm_median_seconds"], routes["native-frozen"]["fresh_warm_median"])
            self.assertEqual(configuration["gpu_control_seconds"], routes["native-gpu"]["fresh_warm_median"])
            self.assertTrue(configuration["request"]["allow_approximation"])
            self.assertEqual(configuration["request"]["ane_manifest"], configuration["compiled_manifest"])
            self.assertNotIn("/Users/", json.dumps(configuration))
            if (ROOT / configuration["model_root"]).exists():
                self.assertTrue((ROOT / configuration["library"]).is_file())
                self.assertTrue((ROOT / configuration["compiled_manifest"]).is_file())
                self.assertTrue((ROOT / configuration["source_manifest"]).is_file())


if __name__ == "__main__":
    unittest.main()
