#!/usr/bin/env python3
"""Public SeedVR2 contract checks. No weights or GPU work are required."""
import ctypes as C
import json
import os
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
lib = C.CDLL(str(NATIVE / "libturbocider.dylib"))
lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
lib.tc_string_free.argtypes = [C.c_void_p]
lib.tc_models_json.restype = C.c_void_p


def consume(pointer):
    if not pointer.value:
        return None
    value = C.string_at(pointer).decode()
    lib.tc_string_free(pointer)
    return value


def plan(request):
    result, error = C.c_void_p(), C.c_void_p()
    code = lib.tc_plan_json(json.dumps(request).encode(), C.byref(result), C.byref(error))
    value, message = consume(result), consume(error)
    return code, json.loads(value) if value else None, message


def request(video=False):
    return dict(schema_version=1, model="seedvr2-3b", operation="video.upscale" if video else "image.upscale",
        inputs=[dict(kind="video" if video else "image", role="source", path="source.mp4" if video else "source.png")],
        output="result.mp4" if video else "result.png", width=1024, height=1024,
        frames=25 if video else 1, fps=24, steps=1, seed=42, execution="gpu", audio=False)


class SeedVR2Contracts(unittest.TestCase):
    def test_registry_declares_real_operations_and_manual_hybrid_capability(self):
        models = json.loads(consume(C.c_void_p(lib.tc_models_json())))["models"]
        model = next(m for m in models if m["id"] == "seedvr2-3b")
        self.assertTrue(model["executor"])
        self.assertEqual(model["executor_operations"], ["image.upscale", "video.upscale"])
        self.assertEqual(model["output"], "image+video")
        self.assertEqual(model["runtime_dependency"], "native")
        self.assertTrue(model["supports_gpu_ane"])

    def test_both_output_kinds_and_sizes_are_plannable(self):
        for video in (False, True):
            for size in (512, 1024):
                with self.subTest(video=video, size=size):
                    r = {**request(video), "width": size, "height": size}
                    code, value, error = plan(r)
                    self.assertEqual(code, 0, error)
                    self.assertEqual(value["model"], "seedvr2-3b")
                    self.assertEqual(value["precision"],
                        "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_vae+fp32_euler")
                    self.assertIsNotNone(value.get("memory_estimate_bytes"))
                    self.assertEqual(value["encoder_execution"], "not_applicable")
                    self.assertEqual(value["executor_operations"], ["image.upscale", "video.upscale"])

    def test_schema_two_output_kind_tracks_the_operation(self):
        for video in (False, True):
            r = request(video)
            v2 = dict(schema_version=2, model=r["model"], operation=r["operation"], inputs=r["inputs"],
                outputs=[dict(kind="video" if video else "image", path=r["output"], width=r["width"], height=r["height"],
                              frames=r["frames"], fps=r["fps"], audio=False)],
                sampling=dict(seed=42, steps=1), execution=dict(policy="gpu", residency="resident"))
            self.assertEqual(plan(v2)[0], 0, plan(v2)[2])
            v2["outputs"][0]["kind"] = "image" if video else "video"
            self.assertNotEqual(plan(v2)[0], 0)

    def test_staged_residency_preserves_operation_and_precision(self):
        for video in (False, True):
            for residency in ("resident", "component_staged"):
                code, value, error = plan({**request(video), "residency": residency})
                self.assertEqual(code, 0, error)
                self.assertEqual(value["residency"], residency)
                self.assertEqual(value["precision"],
                    "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_vae+fp32_euler")
                self.assertEqual(value["memory_estimate_kind"], "conservative_heuristic_not_hard_limit")
                r = request(video)
                v2 = dict(schema_version=2, model=r["model"], operation=r["operation"], inputs=r["inputs"],
                    outputs=[dict(kind="video" if video else "image", path=r["output"], width=r["width"], height=r["height"],
                                  frames=r["frames"], fps=r["fps"], audio=False)],
                    sampling=dict(seed=42, steps=1), execution=dict(policy="gpu", residency=residency))
                code, value, error = plan(v2)
                self.assertEqual(code, 0, error)
                self.assertEqual(value["residency"], residency)

    def test_schema_one_default_is_image_upscale(self):
        r = request()
        del r["operation"]
        code, value, error = plan(r)
        self.assertEqual(code, 0, error)
        self.assertEqual(value["operation"], "image.upscale")

    def test_request_local_vae_controls_are_visible_in_both_schema_plans(self):
        for decoder in ("buffered", "streaming"):
            for convolution in ("singleton2d", "conv3d", "mixed3d", "tiled3d"):
                options = dict(decoder=decoder, convolution=convolution, tile_rows=4096, allocator_cache_bytes=1 << 30)
                r = {**request(True), "seedvr2_vae": options}
                code, value, error = plan(r)
                self.assertEqual(code, 0, error)
                self.assertEqual(value["seedvr2_vae"], {"encoder":"buffered","history":"compact","share_upscale_history":False,"spatial":"full","encoder_spatial":"full", "spatial_rows":64,"spatial_disk_budget_bytes":8<<30,**options})
                v2 = dict(schema_version=2, model=r["model"], operation=r["operation"], inputs=r["inputs"],
                    outputs=[dict(kind="video", path=r["output"], width=r["width"], height=r["height"], frames=r["frames"], fps=24, audio=False)],
                    sampling=dict(seed=42, steps=1), execution=dict(policy="gpu", seedvr2_vae=options))
                code, value, error = plan(v2)
                self.assertEqual(code, 0, error)
                self.assertEqual(value["seedvr2_vae"], {"encoder":"buffered","history":"compact","share_upscale_history":False,"spatial":"full","encoder_spatial":"full", "spatial_rows":64,"spatial_disk_budget_bytes":8<<30,**options})
        self.assertEqual(plan({**request(True), "seedvr2_vae": {}})[1]["seedvr2_vae"]["decoder"], "buffered")

    def test_invalid_vae_controls_never_silently_fall_back(self):
        for options in (None, [], True, {"decoder": "auto"}, {"encoder":"auto"}, {"encoder":True}, {"convolution": "spatial_tiling"},
                        {"convolution": "taps2d"}, {"tile_rows": 0}, {"tile_rows": 129},
                        {"tile_rows": True}, {"tile_rows": 65537}, {"allocator_cache_bytes": -1},
                        {"allocator_cache_bytes": (16 << 30) + 1}, {"ane": True}, {"history":"concat"}, {"history":True},
                        {"spatial":"independent_tiles"},{"encoder_spatial":"independent_tiles"},{"encoder_spatial":True},{"spatial_rows":True},{"spatial_rows":7},{"spatial_rows":129},
                        {"spatial_disk_budget_bytes":0},{"spatial_disk_budget_bytes":(8<<30)+1}):
            with self.subTest(options=options):
                self.assertNotEqual(plan({**request(True), "seedvr2_vae": options})[0], 0)
        self.assertNotEqual(plan({**request(), "seedvr2_vae": {"decoder": "streaming"}})[0], 0)
        self.assertNotEqual(plan({**request(True), "seedvr2_vae": {"decoder": "streaming"}, "dump_tensors": "dump"})[0], 0)
        self.assertNotEqual(plan({**request(), "seedvr2_vae": {"encoder":"streaming"}})[0],0)
        self.assertNotEqual(plan({**request(True), "seedvr2_vae": {"encoder":"streaming"}, "dump_tensors":"dump"})[0],0)
        other = dict(model="flux2-klein-4b", prompt="a fox", output="result.png", execution="gpu", seedvr2_vae={})
        self.assertNotEqual(plan(other)[0], 0)

    def test_spatial_rows_are_explicit_for_image_video_and_do_not_silently_mix_recipes(self):
        for video in (False,True):
            for spatial in ("full","file_rows"):
                options=dict(spatial=spatial,spatial_rows=32,spatial_disk_budget_bytes=8<<30)
                r={**request(video),"seedvr2_vae":options}
                code,value,error=plan(r);self.assertEqual(code,0,error)
                self.assertEqual(value["seedvr2_vae"]["spatial"],spatial)
                self.assertEqual(value["seedvr2_vae"]["spatial_rows"],32)
                v2=dict(schema_version=2,model=r["model"],operation=r["operation"],inputs=r["inputs"],
                    outputs=[dict(kind="video" if video else "image",path=r["output"],width=r["width"],height=r["height"],frames=r["frames"],fps=24,audio=False)],
                    sampling=dict(seed=42,steps=1),execution=dict(policy="gpu",seedvr2_vae=options))
                code,value,error=plan(v2);self.assertEqual(code,0,error);self.assertEqual(value["seedvr2_vae"]["spatial_rows"],32)
        for convolution in ("mixed3d","tiled3d"):
            self.assertNotEqual(plan({**request(True),"seedvr2_vae":dict(spatial="file_rows",convolution=convolution)})[0],0)
        self.assertNotEqual(plan({**request(True),"width":2048,"seedvr2_vae":dict(spatial="file_rows")})[0],0)
        self.assertEqual(plan({**request(True),"seedvr2_vae":{}})[1]["seedvr2_vae"]["spatial"],"full")

    def test_encoder_spatial_control_is_separate_and_preserves_schema_defaults(self):
        for video in (False,True):
            for spatial in ("full","file_rows"):
                options=dict(encoder_spatial="file_rows",spatial=spatial,encoder="streaming" if video else "buffered")
                code,value,error=plan({**request(video),"seedvr2_vae":options})
                self.assertEqual(code,0,error);self.assertEqual(value["seedvr2_vae"]["encoder_spatial"],"file_rows")
                r=request(video);v2=dict(schema_version=2,model=r["model"],operation=r["operation"],inputs=r["inputs"],
                    outputs=[dict(kind="video" if video else "image",path=r["output"],width=r["width"],height=r["height"],frames=r["frames"],fps=24,audio=False)],
                    sampling=dict(seed=42,steps=1),execution=dict(policy="gpu",seedvr2_vae=options))
                self.assertEqual(plan(v2)[0],0,plan(v2)[2])
        for conv in ("mixed3d","tiled3d"):
            self.assertNotEqual(plan({**request(True),"seedvr2_vae":dict(encoder_spatial="file_rows",convolution=conv)})[0],0)
        self.assertNotEqual(plan({**request(True),"width":2048,"seedvr2_vae":dict(encoder_spatial="file_rows")})[0],0)
        self.assertEqual(plan({**request(True),"seedvr2_vae":{}})[1]["seedvr2_vae"]["encoder_spatial"],"full")

    def test_streaming_encoder_is_explicit_request_local_and_default_stays_buffered(self):
        for encoder in ("buffered","streaming"):
            for residency in ("resident","component_staged"):
                r={**request(True),"residency":residency,"seedvr2_vae":{"encoder":encoder}}
                code,value,error=plan(r)
                self.assertEqual(code,0,error);self.assertEqual(value["seedvr2_vae"]["encoder"],encoder)
                v2=dict(schema_version=2,model=r["model"],operation=r["operation"],inputs=r["inputs"],
                    outputs=[dict(kind="video",path=r["output"],width=r["width"],height=r["height"],frames=r["frames"],fps=24,audio=False)],
                    sampling=dict(seed=42,steps=1),execution=dict(policy="gpu",residency=residency,seedvr2_vae=r["seedvr2_vae"]))
                code,value,error=plan(v2)
                self.assertEqual(code,0,error);self.assertEqual(value["seedvr2_vae"]["encoder"],encoder)
        self.assertEqual(plan({**request(True),"seedvr2_vae":{}})[1]["seedvr2_vae"]["encoder"],"buffered")

    def test_frame_history_is_request_local_and_legacy_default_remains_compact(self):
        for history in ("compact","frames"):
            for convolution in ("singleton2d","conv3d","mixed3d","tiled3d"):
                options = dict(history=history,convolution=convolution)
                code,value,error = plan({**request(True),"seedvr2_vae":options})
                self.assertEqual(code,0,error)
                self.assertEqual(value["seedvr2_vae"]["history"],history)
                self.assertEqual(value["seedvr2_vae"]["convolution"],convolution)
        self.assertEqual(plan({**request(True),"seedvr2_vae":{}})[1]["seedvr2_vae"]["history"],"compact")

    def test_upscale_history_sharing_is_explicit_typed_and_not_a_budget_change(self):
        options=dict(history="frames",share_upscale_history=True)
        for video in (False,True):
            r={**request(video),"seedvr2_vae":options};code,value,error=plan(r);self.assertEqual(code,0,error)
            self.assertTrue(value["seedvr2_vae"]["share_upscale_history"])
            control=plan({**r,"seedvr2_vae":dict(history="frames")})[1]
            self.assertEqual(value["memory_estimate_bytes"],control["memory_estimate_bytes"])
            v2=dict(schema_version=2,model=r["model"],operation=r["operation"],inputs=r["inputs"],
                outputs=[dict(kind="video" if video else "image",path=r["output"],width=r["width"],height=r["height"],frames=r["frames"],fps=24,audio=False)],
                sampling=dict(seed=42,steps=1),execution=dict(policy="gpu",seedvr2_vae=options))
            self.assertTrue(plan(v2)[1]["seedvr2_vae"]["share_upscale_history"])
        self.assertFalse(plan({**request(True),"seedvr2_vae":{}})[1]["seedvr2_vae"]["share_upscale_history"])
        for change in ({"share_upscale_history":1},{"share_upscale_history":"true"},{"history":"compact"},
                       {"convolution":"mixed3d"},{"spatial":"file_rows"}):
            with self.subTest(change=change):self.assertNotEqual(plan({**request(True),"seedvr2_vae":{**options,**change}})[0],0)

    def test_qualified_hybrid_bank_precision_and_residency_are_explicit(self):
        manifest = ROOT / "models/coreml/seedvr2-original-fp16-prefix2048-rows512-v1/manifest.json"
        if not manifest.exists():
            self.skipTest("qualified compiled bank fixture unavailable")
        for video in (False, True):
            for residency in ("resident", "component_staged"):
                r = {**request(video), "execution": "gpu_ane", "ane_manifest": str(manifest),
                     "allow_approximation": True, "residency": residency}
                code, value, error = plan(r)
                self.assertEqual(code, 0, error)
                self.assertEqual(value["execution"], "gpu_ane_experimental")
                self.assertEqual(value["precision"],
                    "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_coreml_ffn_prefix+fp32_ffn_join+fp16_vae+fp32_euler")
                self.assertEqual(value["residency"], residency)
                for invalid in ({"allow_approximation": False}, {"execution": "gpu"}, {"execution": "auto"}):
                    self.assertNotEqual(plan({**r, **invalid})[0], 0)
                self.assertIn("seedvr2_original_fp16_video_ffn_prefix_fp16_partial_down_fp32_join",value["algorithm_approximations"])
                self.assertNotIn("single_block_mlp_coreml_approximation",value["algorithm_approximations"])
        with tempfile.TemporaryDirectory(prefix="seedvr2-missing-bank-") as directory:
            orphan = Path(directory) / "manifest.json"
            orphan.write_bytes(manifest.read_bytes())
            self.assertNotEqual(plan({**request(), "execution": "gpu_ane", "allow_approximation": True,
                                     "ane_manifest": str(orphan)})[0], 0)

    def test_rejects_unsupported_inputs_and_silent_option_changes(self):
        invalid = (
            dict(operation="image.generate"), dict(prompt="A fox"), dict(inputs=[]),
            dict(inputs=[dict(kind="image", role="reference", path="source.png")]),
            dict(inputs=[dict(kind="video", role="source", path="source.mp4")]),
            dict(steps=2), dict(frames=2), dict(width=513), dict(audio=True), dict(compile_gpu=True),
            dict(residency="streamed"), dict(ane_manifest="unqualified.json"),
            dict(execution="gpu_ane", ane_manifest="unqualified.json", allow_approximation=True),
            dict(loras=[dict(path="adapter.safetensors", strength=1, role="transformer")]),
            dict(vsa=True), dict(quantized_cache="cache"), dict(warmup_iterations=1),
            dict(hybrid_mlp_mode="runtime"), dict(hybrid_mlp_mode="runtime_qkv"),
        )
        for override in invalid:
            with self.subTest(override=override):
                code, _, error = plan({**request(), **override})
                self.assertNotEqual(code, 0, error)
        for frames in (1, 122):
            self.assertNotEqual(plan({**request(True), "frames": frames})[0], 0)


if __name__ == "__main__":
    unittest.main()
