"""Explicit GGUF retention/read evidence; no checkpoint or device required."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
with mock.patch.object(sys,"path",[str(ROOT/"tools/validation"),*sys.path]):
    import runtime_ane_gguf as GGUF


def row(route="gpu",source="affine",warm=False,retain=True):
    imported=dict(recipe="gguf-mlx-compat-affine-packed-bank-v1",experimental=True,
        whole_request_bounded_certified=False,gpu_graph_recipe="native-compat-eager-v1",
        session_packed_retention=retain,reused_packed_bank=warm and retain,released_before_vae=not retain,
        source_sha256="a"*64,plan_digest="b"*64,affine_packing_recipe="fused-affine-one-pass-v1",
        float_import_recipe="direct-bf16-read-inplace-f16-v1",output_bytes=1024,packed_capacity_bytes=16384,
        allocator_cache_limit_bytes=0,
        request_source_read_bytes=0 if warm and retain else 1024,logical_source_bytes=1024,
        request_load_seconds=0 if warm and retain else .1,raw_window_budget_bytes=256<<20 if source=="raw" else 0,
        raw_window_peak_bytes=16384 if route=="runtime" and source=="raw" else 0,
        raw_window_live_bytes=0,raw_window_entries=0,
        raw_window_source_read_bytes_session_total=(2048 if warm and retain else 1024) if route=="runtime" and source=="raw" else 0,
        raw_window_misses_session_total=3 if route=="runtime" and source=="raw" else 0,
        ane_weight_source="bounded-raw-ggml-window-v1" if source=="raw" else "mlx-affine-import")
    return dict(model="z-image-turbo-gguf",runtime_backend="mlx_cpp_metal_gguf_cpu_direct_packed" if route=="gpu" else
        "mlx_cpp_metal_gguf+coreml_runtime_weight",runtime_precision="z-mlx-compat-affine-v1",gguf_import=imported)


class GgufScreenTests(unittest.TestCase):
    def test_explicit_policy_environment_and_wrong_model_routes(self):
        self.assertEqual(GGUF.gguf_screen_environment("z-image-turbo",["gpu"]),{})
        env=GGUF.gguf_screen_environment("z-image-turbo-gguf",["gpu","runtime"],"cpu_direct","raw",True)
        self.assertEqual(env["TURBOCIDER_Z_GGUF_RETAIN_PACKED"],"1")
        self.assertEqual(GGUF.gguf_screen_environment("z-image-turbo-gguf",["gpu","runtime"],"cpu_direct","affine",True,1<<30)
            ["TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES"],str(1<<30))
        for invalid in (True,-1,(1<<30)+1):
            with self.assertRaises(ValueError):GGUF.gguf_screen_environment("z-image-turbo-gguf",["gpu"],"cpu_direct","affine",False,invalid)
        for args in (("qwen-image-2.1",["gpu"],"cpu_direct","affine",False),
                ("z-image-turbo-gguf",["frozen"],"cpu_direct","affine",False),
                ("z-image-turbo-gguf",["runtime"],"mlx","raw",False),
                ("z-image-turbo-gguf",[],"cpu_direct","affine",False)):
            with self.assertRaises(ValueError):GGUF.gguf_screen_environment(*args)

    def test_original_cpu_label_preserved_and_zero_warm_load_required(self):
        rows=[row(),row(warm=True)]
        normalized,identity=GGUF.validate_gguf_screen(rows,"gpu","affine",True)
        self.assertEqual(normalized[0]["runtime_backend"],"mlx_cpp_metal_gguf")
        self.assertEqual(rows[0]["runtime_backend"],"mlx_cpp_metal_gguf_cpu_direct_packed")
        self.assertEqual(identity["source_sha256"],"a"*64)
        for change in (dict(reused_packed_bank=False),dict(request_load_seconds=.1),dict(request_source_read_bytes=1024),
                dict(released_before_vae=True),dict(source_sha256="c"*64),dict(plan_digest="c"*64),
                dict(raw_window_live_bytes=1),dict(packed_capacity_bytes=0),dict(request_load_seconds=True),
                dict(whole_request_bounded_certified=True),dict(gpu_graph_recipe="compiled_affine_gpu_decode_fp16")):
            bad=copy.deepcopy(rows);bad[1]["gguf_import"].update(change)
            with self.assertRaises(ValueError):GGUF.validate_gguf_screen(bad,"gpu","affine",True)
        hinted=copy.deepcopy(rows)
        for value in hinted:value["gguf_import"]["allocator_cache_limit_bytes"]=1<<30
        GGUF.validate_gguf_screen(hinted,"gpu","affine",True,1<<30)
        with self.assertRaises(ValueError):GGUF.validate_gguf_screen(hinted,"gpu","affine",True)

    def test_raw_consumption_and_per_bank_reset_vs_retained_monotonicity(self):
        rows=[row("runtime","raw"),row("runtime","raw",warm=True)]
        GGUF.validate_gguf_screen(rows,"runtime","raw",True)
        for change in (dict(raw_window_source_read_bytes_session_total=0),dict(raw_window_source_read_bytes_session_total=1024),
                dict(raw_window_peak_bytes=0),dict(raw_window_peak_bytes=(256<<20)+1),dict(raw_window_entries=1),
                dict(raw_window_misses_session_total=True),dict(ane_weight_source="mlx-affine-import")):
            bad=copy.deepcopy(rows);bad[1]["gguf_import"].update(change)
            with self.assertRaises(ValueError):GGUF.validate_gguf_screen(bad,"runtime","raw",True)
        released=[row("runtime","raw",retain=False),row("runtime","raw",retain=False)]
        GGUF.validate_gguf_screen(released,"runtime","raw",False)
        bad=[row("gpu","raw")];bad[0]["gguf_import"]["raw_window_source_read_bytes_session_total"]=1
        with self.assertRaises(ValueError):GGUF.validate_gguf_screen(bad,"gpu","raw",True)


if __name__=="__main__":unittest.main()
