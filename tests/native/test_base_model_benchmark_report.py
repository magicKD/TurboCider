import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from base_model_benchmark_evidence import COMPARISON_POLICY, matched_comparison, measured, verified_advantages, flux_upstream_campaign, qwen_cache_campaign, qwen_upstream_campaign, z_upstream_campaign


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
        self.assertGreaterEqual(len(self.report["campaigns"]), 53)

    def test_complete_actual_sample_arrays_and_medians(self):
        for campaign in self.report["campaigns"]:
            self.assertTrue(campaign["timing_scope"] if "timing_scope" in campaign else campaign["note"])
            for route in campaign["routes"]:
                samples = route["warm_seconds"]
                self.assertEqual(route["warm_sample_count"], len(samples))
                self.assertTrue(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in samples))
                self.assertAlmostEqual(route["warm_median_seconds"], statistics.median(samples), places=12)

    def test_current_focus_is_four_requested_families_without_promoting_flux9_history(self):
        self.assertEqual(self.report["active_model_families"], ["flux2-klein-4b", "z-image-turbo", "qwen-image-2.1", "ltx-2.5-distilled"])
        self.assertEqual(self.report["deferred_model_families"], ["flux2-klein-9b"])
        active = [row for row in self.report["campaigns"] if row["in_current_scope"]]
        deferred = [row for row in self.report["campaigns"] if not row["in_current_scope"]]
        self.assertEqual((len(active), len(deferred)), (self.report["active_campaign_count"], self.report["deferred_campaign_count"]))
        self.assertTrue(all(row["model"].startswith("flux2-klein-9b") for row in deferred))
        self.assertTrue(all(not row["model"].startswith("flux2-klein-9b") for row in active))
        self.assertTrue((ROOT / self.report["historical_report"]).is_file())
        summary = (ROOT / "docs/benchmark/SUMMARY.md").read_text()
        self.assertNotIn("| 10-12 [FLUX9B", summary)
        self.assertIn("FLUX4B", summary)

    def test_compatible_only_policy_and_headlines_use_own_actual_cohort(self):
        self.assertEqual(self.report["comparison_policy"], COMPARISON_POLICY)
        self.assertIs(COMPARISON_POLICY["adapt_unsupported_frameworks"], False)
        self.assertIs(COMPARISON_POLICY["preserve_reference_wins"], True)
        self.assertEqual(self.report["verified_advantages"], verified_advantages(self.report["campaigns"]))
        self.assertEqual(len(self.report["verified_advantages"]), 5)
        for row, expected in zip(self.report["verified_advantages"], (3.40, 3.146, 1.319, 1.338)):
            self.assertAlmostEqual(row["speedup"], expected, delta=.005)
            self.assertIn("approximate", row["precision_note"])
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        self.assertEqual(configs["comparison_policy"], COMPARISON_POLICY)

    def test_prechecks_invalid_and_deferred_routes_cannot_be_speed_headlines(self):
        campaign = dict(self.report["campaigns"][0])
        for change in (dict(in_current_scope=False), dict(reference_comparison_valid=False),
                       dict(reference_comparison_valid=0), dict(native_comparison_valid=False),
                       dict(evidence=dict(status="complete_precheck_only"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                matched_comparison(dict(campaign, **change), "mixed", "original-unsloth-gpu", "test")

    def test_duplicate_route_or_label_only_median_cannot_create_advantage(self):
        campaign = self.report["campaigns"][0]
        rows = [dict(row) for row in campaign["routes"]]
        rows[0]["warm_median_seconds"] = .001
        with self.assertRaises(ValueError):
            matched_comparison(dict(campaign, routes=rows), "gpu", "original-unsloth-gpu", "test")
        with self.assertRaises(ValueError):
            matched_comparison(dict(campaign, routes=[*campaign["routes"], campaign["routes"][0]]),
                               "gpu", "original-unsloth-gpu", "test")

    def test_reference_win_is_not_filtered_out_of_a_valid_comparison(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "qwen-image-2.1-mflux022-bf16-fp16-cohort")
        row = matched_comparison(campaign, "native-gpu", "mflux-bf16", "strict BF16 controls")
        self.assertLess(row["speedup"], 1)
        self.assertLess(row["reduction_percent"], 0)

    def test_consolidated_report_uses_fastest_measured_unsloth_and_keeps_boundaries(self):
        report=(ROOT / self.report["primary_report"]).read_text()
        self.assertIn("3.40×",report)
        self.assertIn("3.15×",report)
        self.assertIn("18.735s",report)
        self.assertIn("22.600s",report)
        self.assertIn("不再扩大参数网格",report)
        self.assertIn("未形成当前配对排名",report)
        self.assertIn("Qwen纯GPU43.309s略输mflux41.952s",report)
        campaign=next(row for row in self.report["campaigns"] if row["model"]=="z-image-turbo-unsloth-mflux-bf16-cohort")
        values={row["route"]:row["warm_median_seconds"] for row in campaign["routes"]}
        self.assertLess(values["unsloth-gpu-text"],values["unsloth-cpu-text"])
        self.assertAlmostEqual(values["unsloth-gpu-text"]/values["native-frozen"],3.1459257859863987)
        configs=json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        self.assertFalse(configs["additional_experiments_scheduled"])

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

    def test_new_qwen_cache_cohorts_keep_actual_limits_samples_and_pressure(self):
        for cache_mib in (1024, 4096):
            campaign = next(row for row in self.report["campaigns"]
                if row["model"] == f"qwen-image-2.1-cache{cache_mib}-framework-cohort")
            self.assertEqual((campaign["date"], campaign["reference_cache_mib"]), ("2026-10-12", cache_mib))
            evidence = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
            self.assertTrue(campaign["reference_comparison_valid"])
            self.assertEqual(evidence["order"], ["native-gpu", "native-frozen", "mflux-bf16", "mflux-bf16", "native-frozen", "native-gpu"])
            self.assertEqual(sum(len(trial["pngs"]) for trial in evidence["trials"]), 24)
            for route in campaign["routes"]:
                self.assertEqual(route["warm_sample_count"], 4)
                self.assertEqual(len(route["cold_seconds"]), 2)
                self.assertEqual(len(route["repeated_prompt_seconds"]), 2)
                trials = [trial for trial in evidence["trials"] if trial["route"] == route["route"]]
                self.assertEqual(route["physical_peaks"], [trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials])
                self.assertEqual(route["compression_bytes"], [trial["memory"]["compression_bytes"] for trial in trials])
                self.assertEqual(route["swap_out_bytes"], [0, 0])
                if route["route"] == "mflux-bf16":
                    self.assertTrue(all(trial["report"]["reference_cache_limit_bytes"] == cache_mib << 20 for trial in trials))
            self.assertIn("no causal cache-speed ranking", campaign["note"])

    def test_qwen_cache_export_rejects_wrong_actual_limit_or_incomplete_memory(self):
        path = "docs/design/validation/qwen21-cache1024-framework-current-20261012.json"
        for corruption, message in (("limit", "actual reference cache limit"), ("memory", "complete cache candidate memory")):
            evidence = json.loads((ROOT / path).read_text())
            if corruption == "limit":
                evidence["trials"][2]["report"]["reference_cache_limit_bytes"] = 0
            else:
                evidence["trials"][0]["memory"]["complete"] = False
            with mock.patch("base_model_benchmark_evidence.source", return_value=(evidence, {})):
                with self.assertRaisesRegex(ValueError, message):
                    qwen_cache_campaign(path, 1024)

    def test_retained_qwen_reference_uses_own_native_controls_and_preserves_other_limits(self):
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        reference = configs["reference_configurations"][0]
        evidence = json.loads((ROOT / reference["evidence"]).read_text())
        routes = {row["route"]: row for row in evidence["routes"]}
        self.assertEqual(reference["fresh_warm_median_seconds"], routes["mflux-bf16"]["fresh_warm_median"])
        self.assertEqual(reference["native_gpu_control_seconds"], routes["native-gpu"]["fresh_warm_median"])
        self.assertEqual(reference["native_frozen_control_seconds"], routes["native-frozen"]["fresh_warm_median"])
        self.assertEqual(reference["free_cache_limit_bytes"], 1024 << 20)
        self.assertIn("not a causal/global speed winner", reference["selection_note"])
        for candidate in reference["other_cache_candidates"]:
            other = json.loads((ROOT / candidate["evidence"]).read_text())
            actual = next(row for row in other["routes"] if row["route"] == "mflux-bf16")
            self.assertEqual(candidate["free_cache_limit_bytes"], other["reference_cache_mib"] << 20)
            self.assertEqual(candidate["fresh_warm_median_seconds"], actual["fresh_warm_median"])

    def test_current_z_reference_preserves_version_math_memory_and_smaller_advantage(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "z-image-turbo-mflux022-bf16-fp16-cohort")
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((data["date"], data["mflux_package_version"], data["reference_cache_mib"]), ("2026-10-12", "0.22.0", 1024))
        self.assertEqual(sum(len(trial["pngs"]) for trial in data["trials"]), 32)
        self.assertEqual(len(data["reference_compute_quality"]), 8)
        self.assertTrue(all(trial["memory"]["complete"] for trial in data["trials"]))
        self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
        comparison = data["comparisons"]
        self.assertGreater(comparison["reference_bf16_over_native_gpu"], 1)
        self.assertLess(comparison["reference_bf16_over_native_frozen"], 1.4)
        self.assertLess(comparison["reference_fp16_reduction_vs_bf16_percent"], 0)
        for trial in data["trials"]:
            if trial["route"] == "mflux-fp16":
                self.assertEqual(trial["report"]["transformer_floating_parameter_dtypes"],
                                 {"mlx.core.bfloat16": 215, "mlx.core.float16": 306})
                self.assertTrue(all(run["generation_parameters"]["compute_precision"] == "float16" for run in trial["report"]["runs"]))
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        reference = next(row for row in configs["reference_configurations"] if row["model"] == "z-image-turbo")
        routes = {row["route"]: row for row in data["routes"]}
        self.assertEqual(reference["fresh_warm_median_seconds"], routes["mflux-bf16"]["fresh_warm_median"])
        self.assertEqual(reference["native_frozen_control_seconds"], routes["native-frozen"]["fresh_warm_median"])
        self.assertLess(max(reference["physical_peak_bytes"]), min(routes["native-gpu"]["physical_peaks"]))

    def test_current_z_export_rejects_label_only_float16_candidate(self):
        path = "docs/design/validation/z-mflux022-bf16-fp16-current-20261012.json"
        data = json.loads((ROOT / path).read_text())
        data["trials"][3]["report"]["runs"][0]["generation_parameters"] = {}
        with mock.patch("base_model_benchmark_evidence.source", return_value=(data, {})):
            with self.assertRaisesRegex(ValueError, "actual parameters and generation metadata"):
                z_upstream_campaign(path)

    def test_current_qwen_short_precheck_is_not_a_full_benchmark_campaign(self):
        path = "docs/design/validation/qwen21-mflux022-precheck-20261012.json"
        data = json.loads((ROOT / path).read_text())
        self.assertEqual((data["status"], data["steps"], data["requests_per_process"]), ("complete_precheck_only", 4, 1))
        self.assertEqual(data["mflux_package_version"], "0.22.0")
        self.assertEqual(data["routes"], [])
        self.assertTrue(all(trial["memory"]["complete"] for trial in data["trials"]))
        candidate = data["trials"][1]
        self.assertEqual(candidate["report"]["transformer_floating_parameter_dtypes"],
                         {"mlx.core.bfloat16": 9, "mlx.core.float16": 288})
        self.assertNotIn(path, [campaign["evidence"]["path"] for campaign in self.report["campaigns"]])

    def test_current_qwen_reference_gpu_win_and_no_stable_float16_gain_are_retained(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "qwen-image-2.1-mflux022-bf16-fp16-cohort")
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((data["steps"], data["requests_per_process"], data["mflux_package_version"]), (40, 4, "0.22.0"))
        self.assertEqual(sum(len(trial["pngs"]) for trial in data["trials"]), 32)
        self.assertTrue(all(trial["memory"]["complete"] for trial in data["trials"]))
        self.assertEqual(len(data["reference_compute_quality"]), 8)
        values = {row["route"]: row["warm_median_seconds"] for row in campaign["routes"]}
        self.assertLess(values["mflux-bf16"], values["native-gpu"])
        self.assertLess(values["native-frozen"], values["mflux-bf16"])
        self.assertLess(abs(values["mflux-fp16"] / values["mflux-bf16"] - 1), .002)
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        selected = next(row for row in configs["reference_configurations"] if row["model"] == "qwen-image-2.1")
        self.assertEqual(selected["package_version_field"], "0.22.0")
        self.assertEqual(selected["fresh_warm_median_seconds"], values["mflux-bf16"])
        self.assertEqual(selected["native_gpu_control_seconds"], values["native-gpu"])
        self.assertEqual(selected["native_frozen_control_seconds"], values["native-frozen"])
        self.assertEqual(selected["evidence"], campaign["evidence"]["path"])

    def test_current_qwen_export_rejects_wrong_actual_limit(self):
        path = "docs/design/validation/qwen21-mflux022-bf16-fp16-current-20261012.json"
        data = json.loads((ROOT / path).read_text())
        data["trials"][2]["report"]["reference_cache_limit_bytes"] = 0
        with mock.patch("base_model_benchmark_evidence.source", return_value=(data, {})):
            with self.assertRaisesRegex(ValueError, "actual current Qwen reference cache"):
                qwen_upstream_campaign(path)

    def test_current_flux_seven_routes_preserve_reference_wins_and_predecode_tradeoff(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "flux2-klein-9b-mflux022-seven-route-cohort")
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((len(data["trials"]), len(data["quality"])), (14, 64))
        self.assertEqual(sum(len(trial["png_sha256"]) for trial in data["trials"]), 56)
        self.assertTrue(all(trial["memory"]["complete"] for trial in data["trials"]))
        self.assertTrue(all(all(check["exact_requests"]) for check in data["output_checks"]))
        values = {row["route"]: row["warm_median_seconds"] for row in campaign["routes"]}
        self.assertLess(values["mflux-bf16"], values["native-bf16"])
        self.assertLess(values["mflux-bf16"], values["native-hybrid"])
        self.assertLess(values["mflux-q4"], values["native-q4"])
        self.assertGreater(values["mflux-fp16"], values["mflux-bf16"])
        self.assertLess(values["native-predecoded"], values["native-q4"])
        self.assertLess(data["comparisons"]["native_hybrid_reduction_vs_compiled_percent"], 5)
        self.assertGreater(data["comparisons"]["native_predecode_reduction_vs_packed_percent"], 5)
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        reference = next(row for row in configs["reference_configurations"] if row["model"] == "flux2-klein-9b")
        self.assertEqual(reference["fresh_warm_median_seconds"], values["mflux-bf16"])
        self.assertEqual(reference["native_predecoded_control_seconds"], values["native-predecoded"])
        self.assertEqual(configs["performance_configurations"][0]["current_replication"]["fresh_warm_median_seconds"], values["native-predecoded"])
        self.assertEqual(configs["experimental_configurations"][0]["current_replication"]["fresh_warm_median_seconds"], values["native-hybrid"])

    def test_current_flux_export_requires_actual_calls_not_hybrid_label(self):
        path = "docs/design/validation/flux9-mflux022-current-20261012.json"
        data = json.loads((ROOT / path).read_text())
        data["trials"][1]["report"]["runs"][0]["metrics"]["hybrid"]["runtime_calls_session_total"] = 0
        with mock.patch("base_model_benchmark_evidence.source", return_value=(data, {})):
            with self.assertRaisesRegex(ValueError, "actual complete96"):
                flux_upstream_campaign(path)

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

    def test_full_encoder_dit_recipe_has_all36_layers_and_does_not_claim_latency_win(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "flux2-klein-9b-matched-encoder-dit-q4-g32-cohort")
        self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual(data["conversion"]["converted_components"],["transformer","text_encoder"])
        self.assertEqual(data["conversion"]["components"]["text_encoder"]["stored_tensors"],902)
        self.assertEqual(data["conversion"]["components"]["text_encoder"]["quantized_linears"],252)
        self.assertLess(data["comparison"]["native_q4_reduction_percent"],0)
        for trial in data["trials"]:
            self.assertTrue(trial["memory"]["complete"])
            if trial["route"] == "mflux-q4":
                covered = trial["report"]["required_weight_coverage"]["text_encoder"]
                self.assertEqual((covered["required_targets"],covered["covered_targets"],covered["quantized_linears"],covered["group_size"]),(902,902,252,32))
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        candidate = configs["memory_configurations"][1]
        self.assertEqual(candidate["fresh_warm_median_seconds"],data["medians"]["native-q4"]["fresh_warm_seconds"])
        self.assertIn("not fastest/default",candidate["selection_note"])
        self.assertNotIn("/Users/",json.dumps(data))

    def test_derived_mlxserve_rejection_is_not_original_format_or_latency_denominator(self):
        data = json.loads((ROOT / "docs/design/validation/flux9-full-q4-compatibility-20261012.json").read_text())
        probe = data["mlxserve"]
        self.assertTrue(probe["derived_layout"])
        self.assertTrue(probe["server_terminal"])
        self.assertEqual((probe["status"],probe["exit_code"],probe["requests"]),("startup_rejected",1,[]))
        self.assertFalse(probe["latency_comparison_valid"])
        self.assertIn("embed_tokens.scales",probe["rejection"])
        self.assertEqual(data["retained_conversion_failure"]["status"],"failed")

    def test_predecoded_candidate_has_actual_initial_decode_retention_and_own_controls(self):
        campaign = next(row for row in self.report["campaigns"] if row["model"] == "flux2-klein-9b-affine-predecoded-gpu-cohort")
        self.assertEqual(len(campaign["routes"]),5)
        self.assertTrue(all(row["warm_sample_count"] == 4 for row in campaign["routes"]))
        data = json.loads((ROOT / campaign["evidence"]["path"]).read_text())
        self.assertEqual((len(data["trials"]),len(data["quality"])),(10,48))
        self.assertGreater(data["predecode_comparison"]["predecode_reduction_vs_packed_percent"],5)
        for trial in data["trials"]:
            self.assertTrue(trial["memory"]["complete"])
            if trial["route"] == "native-predecoded":
                for row in trial["report"]["runs"]:
                    self.assertEqual(row["actual_compiled_single_blocks"],96)
                    self.assertEqual(row["native_event_counts"].get("affine_predecode_projection",0),153 if row["index"]==0 else 0)
        configs = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
        saved = configs["performance_configurations"][0]
        self.assertEqual(saved["fresh_warm_median_seconds"],data["medians"]["native-predecoded"]["fresh_warm_seconds"])
        self.assertTrue(saved["request"]["allow_approximation"])
        self.assertTrue(saved["request"]["compile_gpu"])
        self.assertEqual(saved["environment"]["TURBOCIDER_FLUX9_PREDECODE_AFFINE"],"1")

    def test_same_session_packed_decoded_switch_restores_each_pixels(self):
        data = json.loads((ROOT / "docs/design/validation/flux9-predecode-transition-20261012.json").read_text())
        self.assertTrue(data["pixel_restoration_verified"])
        self.assertEqual(data["order"],["packed","predecoded","packed","predecoded"])
        self.assertEqual([row["events"].get("affine_predecode_projection",0) for row in data["runs"]],[0,153,0,153])
        self.assertEqual([row["events"].get("compiled_single_gpu_block",0) for row in data["runs"]],[0,96,0,96])

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
