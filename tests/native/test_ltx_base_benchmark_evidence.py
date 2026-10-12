import copy
import json
from pathlib import Path
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_base_benchmark_evidence import collect, validate_quality, validate_request, validate_sources


class LtxBaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "docs/design/validation/ltx-base-current-20261011.json").read_text())

    def test_complete_current_media_and_actual_sample_arrays(self):
        self.assertEqual(self.data["status"], "complete_diagnostic")
        self.assertIs(self.data["qualification_passed"], False)
        self.assertEqual([sweep["order"] for sweep in self.data["sweeps"]],
                         [["gpu", "int8", "fp16"], ["int8", "gpu"]])
        for route in self.data["routes"]:
            trials = [trial for sweep in self.data["sweeps"] for trial in sweep["trials"]
                      if trial["mode"] == route["mode"]]
            self.assertEqual(route["cold_seconds"], [trial["rows"][0]["timings_seconds"]["request_wall"] for trial in trials])
            self.assertEqual(route["warm_seconds"], [trial["rows"][1]["timings_seconds"]["request_wall"] for trial in trials])
            self.assertEqual(route["warm_median"], statistics.median(route["warm_seconds"]))
            self.assertEqual(len(trials), 1 if route["mode"] == "fp16" else 2)
            self.assertEqual(route["physical_peaks"], [trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials])
            for trial in trials:
                self.assertIs(trial["memory"]["complete"], True)
                for index, row in enumerate(trial["rows"]):
                    media = row["media"]
                    self.assertEqual((media["width"], media["height"], media["frames"], media["fps"]), (704, 448, 97, "24/1"))
                    self.assertEqual((media["video_codec"], media["audio_codec"], media["channels"], media["sample_rate"]),
                                     ("h264", "aac", 2, 48000))
                    self.assertIs(row["conditioning_cache_hit"], index == 1)

    def test_failed_window_and_framework_are_not_performance_denominators(self):
        probe = self.data["mlxserve"]["result"]
        self.assertEqual(probe["status"], "startup_rejected")
        self.assertIs(probe["server_terminal"], True)
        self.assertEqual(probe["exit_code"], 1)
        self.assertEqual(probe["requests"], [])
        self.assertIs(probe["latency_comparison_valid"], False)
        self.assertIn("memory preflight", self.data["mlxserve"]["reason"])
        self.assertIn("unknown to sampler", self.data["retained_failed_window"]["reason"])
        self.assertNotIn("/Users/", json.dumps(self.data))

    def test_all_97_frames_and_drift_remain_visible(self):
        quality = self.data["video_quality"]
        validate_quality(quality)
        for comparison in quality.values():
            self.assertGreater(comparison["mean_mae_255"], 19)
            self.assertLess(comparison["mean_correlation"], .9)
        self.assertIn("not broad perceptual", self.data["video_quality_scope"])

    def test_sources_require_full_matching_sets_not_just_an_intersection(self):
        common = {"build/test/libturbocider.dylib": "1" * 64,
                  "build/test/ltx-video-vae-decode": "2" * 64,
                  "models/LTX-2.5/connector.safetensors": "3" * 64}
        for index in range(48):
            common[f"models/coreml/ltx_mlp_a6912_dual_int8_20260913/block-{index}/manifest.json"] = "4" * 64
        first = dict(directory="outputs/forward", order=["gpu", "int8", "fp16"], source_sha256=dict(common))
        second = dict(directory="outputs/reverse", order=["int8", "gpu"], source_sha256=dict(common))
        for index in range(48):
            first["source_sha256"][f"models/coreml/ltx_mlp_a6912_dual_fp16_20260913/block-{index}/manifest.json"] = "5" * 64
        first["source_sha256"].update({"outputs/forward/int8-profile.json": "6" * 64,
                                       "outputs/forward/fp16-profile.json": "7" * 64})
        second["source_sha256"]["outputs/reverse/int8-profile.json"] = "8" * 64
        validate_sources(first, second)
        for mutation in ("missing", "changed", "extra"):
            altered = copy.deepcopy(second)
            if mutation == "missing":
                del altered["source_sha256"]["build/test/ltx-video-vae-decode"]
            elif mutation == "changed":
                altered["source_sha256"]["models/LTX-2.5/connector.safetensors"] = "9" * 64
            else:
                altered["source_sha256"]["models/another-checkpoint.safetensors"] = "a" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_sources(first, altered)

    def test_request_cannot_use_short_media_adapter_wrong_route_or_false_warm(self):
        row = copy.deepcopy(self.data["sweeps"][0]["trials"][0]["rows"][1])
        row["denoiser_implementation"] = "c_metal"
        validate_request(row["request"], row, "gpu", 1)
        for field, value in (("frames", 9), ("fps", 30), ("steps", 4), ("audio", False),
                             ("loras", ["adapter"]), ("execution", "gpu_ane")):
            request = dict(row["request"], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(request, row, "gpu", 1)
        changed = dict(row, conditioning_cache_hit=False)
        with self.assertRaises(ValueError):
            validate_request(row["request"], changed, "gpu", 1)
        for wall in (True, 0, float("nan")):
            changed = copy.deepcopy(row)
            changed["timings_seconds"]["request_wall"] = wall
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                validate_request(row["request"], changed, "gpu", 1)

    def test_partial_or_resized_quality_cannot_be_promoted(self):
        for field, value in (("frame_count_compared", 1), ("finite", False), ("shape_equal", False), ("frame_metrics", [])):
            quality = copy.deepcopy(self.data["video_quality"])
            quality["int8"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_quality(quality)
        quality = copy.deepcopy(self.data["video_quality"])
        quality["fp16"]["candidate"]["width"] = 512
        with self.assertRaises(ValueError):
            validate_quality(quality)

    def test_bound_local_raw_evidence_without_rehashing_models(self):
        paths = [ROOT / sweep["evidence"] for sweep in self.data["sweeps"]]
        if not all(path.is_file() for path in paths):
            self.skipTest("raw run directories intentionally not distributed")
        raw = []
        for path, sweep in zip(paths, self.data["sweeps"]):
            source, actual = collect(path.parent, sweep["order"])
            self.assertEqual(actual, sweep)
            raw.append(source)
        validate_sources(*raw)


if __name__ == "__main__":
    unittest.main()
