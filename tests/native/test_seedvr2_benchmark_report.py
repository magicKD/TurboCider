"""Validate published retained evidence without running inference or a server."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/design/validation/seedvr2-comfy-benchmark-current-20261015.json"
DETAIL = ROOT / "docs/benchmark/SEEDVR2_COMFYUI.md"


class SeedVR2BenchmarkReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads(EVIDENCE.read_text())

    def test_export_is_speed_only_and_preserves_protocol(self):
        e = self.evidence
        self.assertEqual(e["schema"], "seedvr2-comfy-benchmark-export-v1")
        self.assertEqual(e["measurement_date"], "2026-10-14")
        self.assertEqual(e["published_date"], "2026-10-15")
        self.assertIs(e["new_inference_experiments_run"], False)
        self.assertIs(e["overall_qualification_passed"], False)
        self.assertEqual(e["hardware"]["ram_bytes"], 64 * 1024**3)
        p = e["protocol"]
        self.assertEqual(p["order"], ["native-gpu", "reference", "reference", "native-gpu"])
        self.assertEqual(p["seeds_per_process"], [42, 43, 44, 45])
        self.assertEqual(p["warm_requests_per_route"], 6)
        self.assertEqual((p["steps"], p["sampler"], p["scheduler"], p["cfg"]),
                         (1, "euler", "simple", 1))
        self.assertIs(p["fresh_conditioning"], True)
        self.assertIn("export", p["timing_scope"])
        self.assertEqual(e["native"]["route"], "native-gpu")
        self.assertEqual(e["native"]["weight_residency"], "resident")
        self.assertEqual(e["native"]["numerical_recipe"], "seedvr2-stable-fp16-fp32-v1")
        self.assertIn("torch.bfloat16", e["reference"]["observed_dit_dtype_log"])

    def test_all_eight_cohorts_keep_individual_samples_and_exact_arithmetic(self):
        rows = self.evidence["comparisons"]
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["id"] for r in rows}, {
            workload + "-" + preset
            for workload in ("image512", "image1024", "video512", "video1024")
            for preset in ("default", "sdpa")
        })
        for row in rows:
            with self.subTest(comparison=row["id"]):
                route = self.evidence["reference"]["presets"][row["reference_preset"]]["route"]
                self.assertEqual([c["index"] for c in row["cohorts"]], [0, 1, 2, 3])
                self.assertEqual([c["route"] for c in row["cohorts"]],
                                 ["native-gpu", route, route, "native-gpu"])
                for cohort in row["cohorts"]:
                    self.assertEqual(len(cohort["warm_seconds"]), 3)
                    for value in [cohort["cold_request_seconds"], *cohort["warm_seconds"]]:
                        self.assertIs(type(value), float)
                        self.assertTrue(math.isfinite(value) and value > 0)
                medians = [statistics.median([
                    v for c in row["cohorts"] if c["route"] == selected
                    for v in c["warm_seconds"]
                ]) for selected in ("native-gpu", route)]
                native, reference = medians
                self.assertEqual(native, row["native_warm_median_seconds"])
                self.assertEqual(reference, row["reference_warm_median_seconds"])
                self.assertEqual(reference / native, row["speedup"])
                self.assertEqual(reference - native, row["saved_seconds"])
                self.assertEqual(100 * (1 - native / reference), row["time_saved_percent"])
                self.assertIs(row["speed_gate_passed"], True)
                self.assertIs(row["overall_qualification_passed"], False)
                self.assertEqual(row["outputs_checked"], 16)
                self.assertIs(row["fresh_conditioning_checked"], True)
                self.assertIs(row["output_geometry_checked"], True)
                w = row["workload"]
                self.assertEqual(w["width"], 2 * w["source_width"])
                self.assertEqual(w["height"], 2 * w["source_height"])
                self.assertEqual((w["frames"], w["fps"]),
                                 (25, 24) if w["kind"] == "video" else (1, None))

    def test_headlines_select_faster_reference_and_its_own_native(self):
        rows = self.evidence["comparisons"]
        expected = [min((r for r in rows if r["id"].startswith(workload + "-")),
                        key=lambda r: r["reference_warm_median_seconds"])["id"]
                    for workload in ("image512", "image1024", "video512", "video1024")]
        self.assertEqual(expected, self.evidence["headline_comparison_ids"])
        self.assertEqual(expected, ["image512-sdpa", "image1024-default",
                                    "video512-default", "video1024-default"])
        labels = ["图像256→512", "图像512→1024", "视频256→512", "视频512→1024"]
        for relative in ("docs/benchmark/SEEDVR2_COMFYUI.md",
                         "docs/benchmark/PERFORMANCE_REPORT.md",
                         "docs/benchmark/COMFYUI_MODEL_COMPARISON.md"):
            document = (ROOT / relative).read_text()
            for label, selected in zip(labels, expected):
                row = next(r for r in rows if r["id"] == selected)
                line = next(line for line in document.splitlines()
                            if line.startswith("| " + label))
                for value in (f"{row['native_warm_median_seconds']:.3f}",
                              f"{row['reference_warm_median_seconds']:.3f}",
                              f"{row['speedup']:.2f}×", f"{row['saved_seconds']:.3f}s",
                              f"{row['time_saved_percent']:.1f}%"):
                    self.assertIn(value, line, (relative, selected))

    def test_default_sdpa_controls_and_cold_values_are_published(self):
        detail = DETAIL.read_text()
        public = (ROOT / "docs/public/BENCHMARKS.md").read_text()
        for row in self.evidence["comparisons"]:
            line = next(line for line in detail.splitlines()
                        if line.endswith("| `" + row["id"] + "` |"))
            for c in row["cohorts"]:
                self.assertIn(f"{c['cold_request_seconds']:.3f}", line)
            self.assertIn(f"{row['speedup']:.2f}×", line)
            self.assertIn(f"{row['native_warm_median_seconds']:.6f}", public)
            self.assertIn(f"{row['reference_warm_median_seconds']:.6f}", public)
            self.assertIn(row["record"]["path"].removeprefix("outputs/"), detail)
            self.assertIn(row["native_library_sha256"][:12], detail)

    def test_machine_paths_not_exported_and_bound_files_match_when_available(self):
        self.assertNotIn("/Users/", EVIDENCE.read_text())

        def check(value):
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"}:
                    relative = Path(value["path"])
                    self.assertFalse(relative.is_absolute())
                    self.assertNotIn("..", relative.parts)
                    self.assertRegex(value["sha256"], r"^[a-f0-9]{64}$")
                    path = ROOT / relative
                    if path.is_file():
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), value["sha256"])
                    else:
                        self.assertTrue(value["path"].startswith("outputs/"), str(path))
                for key, item in value.items():
                    if key.endswith("sha256"):
                        self.assertRegex(item, r"^[a-f0-9]{64}$")
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(self.evidence)

    def test_samples_match_raw_reports_when_locally_available(self):
        for row in self.evidence["comparisons"]:
            paired_path = ROOT / row["record"]["path"]
            if paired_path.is_file():
                raw = json.loads(paired_path.read_text())
                self.assertEqual(raw["status"], "complete")
                self.assertIs(raw["overall_qualification_passed"], False)
                self.assertEqual(raw["library_sha256"], row["native_library_sha256"])
                self.assertEqual(raw["source_sha256"], row["source_sha256"])
                self.assertEqual(raw["native_runtime_build_manifest_sha256"],
                                 row["native_runtime_build_manifest_sha256"])
                for key in ("native_warm_median_seconds", "reference_warm_median_seconds", "speedup"):
                    self.assertEqual(raw[key], row[key])
                for raw_cohort, cohort in zip(raw["cohorts"], row["cohorts"]):
                    self.assertEqual(raw_cohort["warm_seconds"], cohort["warm_seconds"])
                    self.assertEqual(raw_cohort["cold_request_seconds"], cohort["cold_request_seconds"])
            for cohort in row["cohorts"]:
                child_path = ROOT / cohort["record"]["path"]
                if not child_path.is_file():
                    continue  # Raw local artifacts are intentionally not distributed.
                child = json.loads(child_path.read_text())
                self.assertEqual(child["model_provenance"], self.evidence["model_provenance"])
                self.assertEqual(child["status"], "complete")
                samples = [r["request_wall_including_export"] for r in child["runs"]]
                self.assertEqual(samples, [cohort["cold_request_seconds"], *cohort["warm_seconds"]])
                if cohort["route"].startswith("comfy-"):
                    self.assertEqual(child["comfy_commit"], self.evidence["reference"]["comfy_commit"])
                    self.assertEqual(child["comfy_seedvr2_vae_sha256"], row["comfy_vae_source_sha256"])
                    self.assertEqual(hashlib.sha256(child["comfy_local_changes"].encode()).hexdigest(),
                                     row["comfy_local_changes_sha256"])
                    log = ROOT / cohort["dtype_log_record"]["path"]
                    if log.is_file():
                        self.assertIn(self.evidence["reference"]["observed_dit_dtype_log"], log.read_text())

    def test_benchmark_entry_points_and_comparison_boundaries(self):
        for relative in ("docs/benchmark/README.md", "docs/benchmark/SUMMARY.md",
                         "docs/benchmark/PERFORMANCE_REPORT.md", "docs/benchmark/COMFYUI_MODEL_COMPARISON.md",
                         "docs/public/PERFORMANCE.md", "docs/public/BENCHMARKS.md"):
            document = (ROOT / relative).read_text()
            self.assertIn("SEEDVR2_COMFYUI.md", document)
            for multiplier in ("4.81×", "4.43×", "3.46×", "3.48×"):
                self.assertIn(multiplier, document)
        detail = DETAIL.read_text()
        for phrase in ("overall_qualification_passed=false", "manual cast: torch.bfloat16",
                       "不发布跨框架内存倍率", "不是未修改stock视频", "不外推"):
            self.assertIn(phrase, detail)
        for relative in (DETAIL, ROOT / "docs/public/PERFORMANCE.md", ROOT / "docs/public/BENCHMARKS.md"):
            for target in re.findall(r"\]\(([^)]+)\)", relative.read_text()):
                if "://" not in target:
                    self.assertTrue((relative.parent / target.split("#")[0]).exists(), (relative, target))


if __name__ == "__main__":
    unittest.main()
