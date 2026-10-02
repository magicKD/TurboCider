"""Fail-closed local BF16 comparison, including cache/tuning/lifecycle mismatch."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
spec = importlib.util.spec_from_file_location("bf16_screen", ROOT / "tools/validation/screen_quantized_against_bf16.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def report(baseline, mode, seconds, peak):
    result = {"status": "completed", "measurement": mode, "baseline_bf16": baseline,
              "hardware": {"gpu":"Apple M4 Max", "physical_memory_bytes":64<<30, "os":"Version 26.6.2", "mlx_version":"0.32.0"},
              "dit_source_sha256": ("8" if baseline else "4")*64,
              "component_binding": {r: [{"sha256": "1"*64, "bytes": 1024}] for r in ("encoder", "vae", "tokenizer")},
              "library_sha256": "2"*64, "loaded_libraries": {"libturbocider.dylib": {"sha256": "2"*64},
              "libmlx.dylib": {"sha256": "3"*64}}, "prompt_cache_policy": "hit",
              "runtime_environment": {"TURBOCIDER_Z_MPP_PROJECTIONS": "1"}, "runs": []}
    for i in range(4 if mode == "timing" else 1):
        metrics = {"text_tokens": 34, "prompt_cache_hit": True, "runtime_precision": "bf16" if baseline else "q8",
                   "runtime_backend": "mlx_cpp_metal" if baseline else "gguf_cpu_direct",
                   "plan": {"gpu_graph": "compiled_fused_blocks"}, "timings_seconds": {"denoise": seconds-.2}}
        if not baseline:
            metrics["gguf_import"] = {"source_sha256": "4"*64, "plan_digest": "5"*64,
                                      "consumer_revision": "release", "gpu_graph_recipe": "native"}
        result["runs"].append({"status": 0, "error": None, "measurement": mode, "warmup": False,
            "request": {"inputs": [{"text": "fixed prompt"}], "sampling": {"seed": 42, "steps": 4},
                        "operation": "image.generate", "parameters": {"dynamic_text": True},
                        "execution": {"policy": "gpu"}, "outputs": [{"path": "different.png", "width": 512, "height": 512}]},
            "metrics": metrics, "wall_seconds": seconds, "png_sha256": ("6" if baseline else "7")*64,
            "memory_samples": [] if mode == "timing" else [{"time": t, "phys_footprint_bytes": peak,
                "lifetime_max_phys_footprint_bytes": peak} for t in (1.,1.01,1.02)],
            "memory_sample_max_gap_seconds": .01, "sampling_errors": [], "vm_deltas": {"Swapouts": 0}})
    return result


class Bf16ScreenTests(unittest.TestCase):
    def inputs(self, candidate_seconds=4.4):
        return [report(True,"timing",4.,16_000_000_000), report(True,"memory",4.,16_000_000_000),
                report(False,"timing",candidate_seconds,10_000_000_000), report(False,"memory",candidate_seconds,10_000_000_000)]

    def test_thresholds_and_no_certification(self):
        result = module.screen(*self.inputs())
        self.assertTrue(result["hard_20pct_latency_pass"])
        self.assertAlmostEqual(result["observed_process_memory_ratio"], .625)
        self.assertFalse(result["production_qualified"])
        self.assertEqual(result["whole_request_memory"], "unknown")
        self.assertFalse(module.screen(*self.inputs(4.81))["hard_20pct_latency_pass"])
        self.assertFalse(module.screen(*self.inputs(4.5))["preferred_10pct_latency_pass"])

    def test_binding_cache_and_route_negative_cases(self):
        for fault in ("library","component","empty_hash","seed","prompt","cache","tuning","precision","route","graph","lifecycle","source","baseline_source","device"):
            reports = copy.deepcopy(self.inputs()); target = reports[2]; row = target["runs"][0]
            if fault=="library": target["library_sha256"]="9"*64
            if fault=="component": target["component_binding"]["encoder"][0]["sha256"]="9"*64
            if fault=="empty_hash": target["component_binding"]["tokenizer"][0]["sha256"]=""
            if fault=="seed": row["request"]["sampling"]["seed"]=1234
            if fault=="prompt": row["request"]["inputs"][0]["text"]="different"
            if fault=="cache": row["metrics"]["prompt_cache_hit"]=False
            if fault=="tuning": target["runtime_environment"]["TURBOCIDER_Z_MPP_PROJECTIONS"]="attention_out"
            if fault=="precision": reports[0]["runs"][0]["metrics"]["runtime_precision"]="fp16"
            if fault=="route": row["request"]["execution"]["policy"]="gpu_ane"
            if fault=="graph": reports[0]["runs"][0]["metrics"]["plan"]["gpu_graph"]="eager"
            if fault=="lifecycle": row["metrics"]["gguf_import"]["consumer_revision"]="session"
            if fault=="source": row["metrics"]["gguf_import"]["source_sha256"]="9"*64
            if fault=="baseline_source": reports[1]["dit_source_sha256"]="9"*64
            if fault=="device": reports[1]["hardware"]["gpu"]="Apple M5"
            with self.subTest(fault=fault), self.assertRaises(ValueError): module.screen(*reports)

    def test_legacy_unbound_device_never_passes_target(self):
        reports = self.inputs()
        for report in reports: report.pop("hardware")
        result = module.screen(*reports)
        self.assertFalse(result["empirical_speed_memory_target_pass"])
        self.assertEqual(result["device_binding"], "unknown_legacy_reports")

    def test_bad_samples_memory_and_observers(self):
        for fault in ("missing","gap","lie_gap","swap","swap_unknown","observer","nan","warmup","png","failed","source_validation"):
            reports = copy.deepcopy(self.inputs()); row = reports[3]["runs"][0]
            if fault=="missing": row["memory_samples"]=[]
            if fault=="gap": row["memory_samples"][1]["time"]=1.2
            if fault=="lie_gap": row["memory_sample_max_gap_seconds"]=.001
            if fault=="swap": row["vm_deltas"]["Swapouts"]=1
            if fault=="swap_unknown": row["vm_deltas"].clear()
            if fault=="observer": reports[2]["runs"][0]["memory_samples"]=row["memory_samples"]
            if fault=="nan": reports[2]["runs"][0]["wall_seconds"]=float("nan")
            if fault=="warmup": reports[2]["runs"][0]["warmup"]=1
            if fault=="png": row["png_sha256"]="9"*64
            if fault=="failed": reports[2]["runs"][0]["status"]=1
            if fault=="source_validation": reports[2]["runs"][0]["metrics"]["quantized_source_validation"]={"not_timing":True}
            with self.subTest(fault=fault), self.assertRaises(ValueError): module.screen(*reports)


if __name__ == "__main__": unittest.main()
