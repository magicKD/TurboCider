"""Fail-closed ConvRot speed/memory screen; mocks never qualify hardware."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

from test_bf16_quantized_screen import report

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("conv_screen",ROOT/"tools/validation/screen_convrot_against_bf16.py")
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def candidate(mode):
    d=report(False,mode,4.4,10_000_000_000)
    d.update(route="gpu",gpu_recipe="compiled_dense",source_sha256="4"*64,controls={})
    for row in d["runs"]:
        m=row["metrics"];m.pop("gguf_import")
        m.update(runtime_backend="mlx_cpp_metal_convrot_compiled_experimental",runtime_precision="convrot-legacy-packed-bf16-scale-v1:compiled_dense")
        m["plan"]["gpu_graph"]="convrot_parameterized_packed_bf16_blocks"
        m["convrot_experiment"]={"source_profile":"convrot-legacy-packed-bf16-scale-v1","dense_scope":"none","activation_quantization":"none",
            "execution_recipe":m["runtime_precision"],"allocator_cache_limit_bytes":None,"allocator_cache_retention":"request-cleanup-v1"}
    return d


class ConvRotBf16ScreenTests(unittest.TestCase):
    def inputs(self):return [report(True,"timing",4.,16_000_000_000),report(True,"memory",4.,16_000_000_000),candidate("timing"),candidate("memory")]
    def test_scope_and_targets(self):
        result=MODULE.screen(*self.inputs())
        self.assertTrue(result["empirical_speed_memory_target_pass"])
        self.assertFalse(result["production_qualified"]);self.assertEqual(result["quality"],"not_run")
        data=self.inputs()
        for r in data[2]["runs"]:r["wall_seconds"]=4.81
        self.assertFalse(MODULE.screen(*data)["hard_20pct_latency_pass"])
    def test_binding_and_observer_fail_closed(self):
        for fault in ("library","source","component","device","recipe","cache","a8","graph","observer","validation","gap","swap","failed"):
            data=copy.deepcopy(self.inputs());d=data[3];r=d["runs"][0];m=r["metrics"]
            if fault=="library":d["library_sha256"]="9"*64
            if fault=="source":d["source_sha256"]="9"*64
            if fault=="component":d["component_binding"]["vae"][0]["sha256"]="9"*64
            if fault=="device":d["hardware"]["gpu"]="Apple M5"
            if fault=="recipe":d["gpu_recipe"]="compiled_butterfly"
            if fault=="cache":m["convrot_experiment"]["allocator_cache_limit_bytes"]=1
            if fault=="a8":m["convrot_experiment"]["activation_quantization"]="int8"
            if fault=="graph":m["plan"]["gpu_graph"]="eager"
            if fault=="observer":data[2]["runs"][0]["memory_samples"]=r["memory_samples"]
            if fault=="validation":m["quantized_source_validation"]={"pass":True}
            if fault=="gap":r["memory_samples"][1]["time"]=1.2
            if fault=="swap":r["vm_deltas"]["Swapouts"]=1
            if fault=="failed":r["status"]=1
            with self.subTest(fault=fault),self.assertRaises(ValueError):MODULE.screen(*data)


if __name__=="__main__":unittest.main()
