"""Metadata/request tests for the independent executor; no model imports."""
import ctypes as C
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
PRIVATE_ANE = os.environ.get("TURBOCIDER_TEST_PRIVATE_ANE", "0") == "1"
VIGGLE_R128 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
LIB = C.CDLL(str(NATIVE / "libturbocider.dylib"))
LIB.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
LIB.tc_string_free.argtypes = [C.c_void_p]


def plan(request):
    output, error = C.c_void_p(), C.c_void_p()
    status = LIB.tc_plan_json(json.dumps(request).encode(), C.byref(output), C.byref(error))
    def consume(pointer):
        if not pointer.value:
            return None
        value = C.string_at(pointer).decode()
        LIB.tc_string_free(pointer)
        return value
    value, message = consume(output), consume(error)
    return status, json.loads(value) if value else None, message


class NativeRouteTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        for key in tuple(os.environ):
            if key.startswith(("TURBOCIDER_QWEN21_", "TURBOCIDER_ANE_CALIBRATION_")):
                os.environ.pop(key)
        self.base = dict(model="qwen-image-2.1", operation="image.generate", prompt="A blue teapot",
                         width=512, height=512, steps=20, frames=1, audio=False,
                         execution="gpu", residency="component_staged", allow_approximation=True,
                         qwen21_denoiser_backend="native-metal-graph", qwen21_dit_cache="off")
        # The metadata API must not read/create this directory or load a model.
        self.w1 = {**self.base, "execution": "gpu_ane", "qwen21_w8a8": True,
                   "qwen21_native_ane_cache": "/tmp/turbocider-native-w1-plan-only"}
        self.viggle = {**self.base, "steps": 6, "lora_strategy": "inference_time",
                       "loras": [dict(path="/tmp/turbocider-native-lora-plan-only/" + VIGGLE_R128,
                                      strength=1.0, role="transformer")]}

    @staticmethod
    def candidate_v2(request):
        """Use actual v2 asset placement: LoRA assets/strategy stay top-level."""
        execution_keys = ("residency", "allow_approximation", "qwen21_denoiser_backend",
                          "qwen21_dit_cache", "ane_manifest", "encoder_ane_manifest", "hybrid_mlp_mode",
                          "qwen21_w8a8", "qwen21_gpu_w8a16", "qwen21_gpu_full_ffn_blocks",
                          "qwen21_native_ane_cache", "qwen21_native_w8_staging",
                          "qwen21_native_scale_cache", "qwen21_native_lora_kernel", "quantized_cache")
        parameters = {key: request[key] for key in ("compile_gpu", "prompt_enhance",
                      "prompt_enhancer_path", "vsa") if key in request}
        value = dict(schema_version=2, model=request["model"], operation=request["operation"],
                     inputs=[dict(kind="text", role="prompt", text=request["prompt"])] + request.get("inputs", []),
                     outputs=[dict(kind="image", path="result.png", width=request["width"],
                                   height=request["height"], frames=request["frames"], audio=False)],
                     sampling=dict(seed=42, steps=request["steps"]),
                     execution={"policy": request["execution"],
                                **{key: request[key] for key in execution_keys if key in request}},
                     loras=request["loras"])
        if "lora_strategy" in request:
            value["lora_strategy"] = request["lora_strategy"]
        if parameters:
            value["parameters"] = parameters
        return value

    def tearDown(self):
        self.environment.stop()

    def test_explicit_executor_v1_v2_and_dense_estimate(self):
        v2 = dict(schema_version=2, model=self.base["model"], operation=self.base["operation"],
                  inputs=[dict(kind="text", role="prompt", text=self.base["prompt"])],
                  outputs=[dict(kind="image", path="result.png", width=512, height=512,
                                frames=1, audio=False)], sampling=dict(seed=42, steps=20),
                  execution=dict(policy="gpu", residency="component_staged", allow_approximation=True,
                                 qwen21_denoiser_backend="native-metal-graph", qwen21_dit_cache="off"))
        for request in (self.base, v2):
            with self.subTest(schema=request.get("schema_version", 1)):
                status, result, error = plan(request)
                self.assertEqual(status, 0, error)
                self.assertEqual(result["qwen21_denoiser_backend"], "native-metal-graph")
                self.assertEqual(result["qwen21_denoiser_qualification"], "experimental-unqualified")
                self.assertEqual(result["backend"], "native_metal_graph+mlx_encoder_vae")
                self.assertEqual(result["gpu_graph"], "native_full_dit_euler")
                self.assertEqual(result["memory_estimate_bytes"], (22 << 30) + 512 * 512 * 10240)
        self.assertNotEqual(plan({**v2, "parameters": {"qwen21_denoiser_backend": "native-metal-graph"}})[0], 0)
        self.assertNotEqual(plan({**v2, "qwen21_denoiser_backend": "native-metal-graph"})[0], 0)

    def test_default_remains_mlx_and_does_not_inherit_candidate_restrictions(self):
        ordinary = {key: value for key, value in self.base.items()
                    if key != "qwen21_denoiser_backend"}
        ordinary.update(steps=40, allow_approximation=False)
        status, result, error = plan(ordinary)
        self.assertEqual(status, 0, error)
        self.assertEqual(result["qwen21_denoiser_backend"], "mlx")
        self.assertNotIn("qwen21_denoiser_qualification", result)

    def test_viggle_r128_explicit_v1_v2_is_metadata_only_and_unqualified(self):
        # This deliberately nonexistent named artifact must not be opened by
        # planning. Session still verifies the actual pin from its held fd.
        candidates = [self.viggle, {**self.viggle, "lora_strategy": "auto"},
                      {key: value for key, value in self.viggle.items() if key != "lora_strategy"}]
        for request in (request for candidate in candidates for request in (candidate, self.candidate_v2(candidate))):
            with self.subTest(schema=request.get("schema_version", 1)):
                status, result, error = plan(request)
                self.assertEqual(status, 0, error)
                self.assertEqual(result["qwen21_denoiser_backend"], "native-metal-graph")
                self.assertEqual(result["qwen21_denoiser_qualification"], "experimental-unqualified")
                self.assertEqual(result["backend"], "native_metal_graph+mlx_encoder_vae")
                self.assertEqual(result["gpu_graph"], "native_full_dit_euler_runtime_lora")
                self.assertEqual(result["lora_strategy"], "inference_time")
                self.assertEqual(result["qwen21_native_variant"], "viggle-r128-6step")
                self.assertEqual(result["qwen21_native_lora_kernel"], "scalar")
                self.assertEqual(result["memory_estimate_bytes"], (23 << 30) + 512 * 512 * 10240)

    def test_native_lora_kernel_candidate_is_explicit_and_confined(self):
        for kernel in ("scalar", "sgmma"):
            candidate = {**self.viggle, "qwen21_native_lora_kernel": kernel}
            for request in (candidate, self.candidate_v2(candidate)):
                with self.subTest(kernel=kernel, schema=request.get("schema_version", 1)):
                    status, result, error = plan(request)
                    self.assertEqual(status, 0, error)
                    self.assertEqual(result["qwen21_native_lora_kernel"], kernel)
                    self.assertEqual(result["qwen21_denoiser_qualification"], "experimental-unqualified")
            for route in (self.base, self.w1, {**self.viggle, "qwen21_denoiser_backend": "mlx"},
                          {**self.base, "model": "z-image-turbo", "steps": 4}):
                self.assertNotEqual(plan({**route, "qwen21_native_lora_kernel": kernel})[0], 0)
        for value in (None, True, 0, 1, [], {}, "", "auto", "SGMMA", "half"):
            candidate = {**self.viggle, "qwen21_native_lora_kernel": value}
            for request in (candidate, self.candidate_v2(candidate)):
                self.assertNotEqual(plan(request)[0], 0)
        v2 = self.candidate_v2(self.viggle)
        for wrong in ({**v2, "qwen21_native_lora_kernel": "sgmma"},
                      {**v2, "parameters": {"qwen21_native_lora_kernel": "sgmma"}}):
            self.assertNotEqual(plan(wrong)[0], 0)

    def test_viggle_asset_shape_strength_role_rank_and_strategy_are_exact(self):
        original = self.viggle["loras"][0]
        invalid_assets = [
            [{**original, "strength": value}] for value in (0, -1, 0.75, 2, True, "1", None)
        ] + [
            [{**original, "role": role}] for role in ("text_encoder", "refiner", "", "unknown")
        ] + [
            [{**original, "path": path}] for path in (
                original["path"].replace("r128", "r256"), "other-adapter.safetensors",
                original["path"] + ".backup", "https://example.invalid/" + VIGGLE_R128,
                original["path"] + "/", "\x00" + original["path"])
        ] + [[original, original], [], [{**original, "rank": 128}],
             [{**original, "inference_time": True}], [{**original, "kind": "lora"}]]
        for assets in invalid_assets:
            candidate = {**self.viggle, "loras": assets}
            for request in (candidate, self.candidate_v2(candidate)):
                with self.subTest(schema=request.get("schema_version", 1), assets=assets):
                    self.assertNotEqual(plan(request)[0], 0)
        for strategy in ("in_memory_merge", "disk_premerge", "", None):
            candidate = {**self.viggle, "lora_strategy": strategy}
            for request in (candidate, self.candidate_v2(candidate)):
                with self.subTest(schema=request.get("schema_version", 1), strategy=strategy):
                    self.assertNotEqual(plan(request)[0], 0)
        v2 = self.candidate_v2(self.viggle)
        # Assets are neither execution parameters nor arbitrary input records.
        for container in ("execution", "parameters"):
            wrong = {**v2, "loras": [], container: {**v2.get(container, {}), "loras": self.viggle["loras"]}}
            self.assertNotEqual(plan(wrong)[0], 0)

    def test_viggle_remains_gpu_only_six_step_generation_without_other_recipes(self):
        changes = (
            dict(steps=20), dict(steps=5), dict(steps=7), dict(width=768), dict(height=768),
            dict(frames=2), dict(operation="image.edit"),
            dict(inputs=[dict(kind="image", role="reference", path="local.png")]),
            dict(execution="gpu_ane", qwen21_w8a8=True,
                 qwen21_native_ane_cache=self.w1["qwen21_native_ane_cache"]),
            dict(execution="auto"), dict(allow_approximation=False),
            dict(ane_manifest="legacy.json"), dict(encoder_ane_manifest="legacy.json"),
            dict(hybrid_mlp_mode="runtime"), dict(hybrid_mlp_mode="lora_fused"),
            dict(qwen21_w8a8=True), dict(qwen21_gpu_w8a16=True), dict(qwen21_gpu_full_ffn_blocks=[0]),
            dict(qwen21_native_ane_cache=self.w1["qwen21_native_ane_cache"]),
            dict(qwen21_native_w8_staging="generic"), dict(qwen21_native_scale_cache=False),
            dict(qwen21_dit_cache="balanced"), dict(compile_gpu=True), dict(vsa=True),
            dict(quantized_cache="local"), dict(prompt_enhance=True, prompt_enhancer_path="local"),
        )
        for change in changes:
            candidate = {**self.viggle, **change}
            for request in (candidate, self.candidate_v2(candidate)):
                with self.subTest(schema=request.get("schema_version", 1), change=change):
                    status, result, error = plan(request)
                    self.assertNotEqual(status, 0)
                    self.assertIsNone(result)
                    self.assertTrue(error)
        with patch.dict(os.environ, {"TURBOCIDER_QWEN21_VIGGLE_LORA_FP16": "1"}):
            self.assertNotEqual(plan(self.viggle)[0], 0)

    def test_w1_private_build_gate_and_v1_v2_report(self):
        v2 = dict(schema_version=2, model=self.base["model"], operation=self.base["operation"],
                  inputs=[dict(kind="text", role="prompt", text=self.base["prompt"])],
                  outputs=[dict(kind="image", path="result.png", width=512, height=512,
                                frames=1, audio=False)], sampling=dict(seed=42, steps=20),
                  execution=dict(policy="gpu_ane", residency="component_staged", allow_approximation=True,
                                 qwen21_denoiser_backend="native-metal-graph", qwen21_dit_cache="off",
                                 qwen21_w8a8=True, qwen21_native_ane_cache=self.w1["qwen21_native_ane_cache"]))
        for request in (self.w1, v2):
            with self.subTest(schema=request.get("schema_version", 1)):
                status, result, error = plan(request)
                if not PRIVATE_ANE:
                    self.assertNotEqual(status, 0)
                    self.assertIsNone(result)
                    self.assertIn("private-ANE-enabled build", error)
                    continue
                self.assertEqual(status, 0, error)
                self.assertEqual(result["backend"], "native_metal_graph+private_ane_w1+mlx_encoder_vae")
                self.assertEqual(result["gpu_graph"], "native_full_dit_euler_w1")
                self.assertEqual(result["qwen21_denoiser_qualification"], "experimental-unqualified")
                self.assertEqual(result["qwen21_native_ane_cache"], self.w1["qwen21_native_ane_cache"])
                self.assertIs(result["qwen21_native_scale_cache"], False)
                self.assertEqual(result["planned_native_w1_geometry"],
                                 dict(hidden=4096, intermediate=12288, ane_channels=6144, target_rows=1024, slots=1))
                self.assertEqual(result["native_ane_hardware_arithmetic"], "unverified")
                self.assertIn("qwen21_native_w1_sylvester_w8a8_channel_split", result["algorithm_approximations"])
                self.assertNotIn("qwen21_decode_mlp_w8a8_per_tensor", result["algorithm_approximations"])
                self.assertEqual(result["memory_estimate_bytes"], (24 << 30) + 512 * 512 * 10240)
        # Cache placement is as strict as executor placement in schema v2.
        for wrong in ({**v2, "parameters": {"qwen21_native_ane_cache": self.w1["qwen21_native_ane_cache"]}},
                      {**v2, "qwen21_native_ane_cache": self.w1["qwen21_native_ane_cache"]}):
            self.assertNotEqual(plan(wrong)[0], 0)

    def test_w1_requires_exact_recipe_and_explicit_cache(self):
        for change in (
            dict(qwen21_w8a8=False), dict(qwen21_native_ane_cache=""),
            dict(qwen21_native_ane_cache="relative/path"), dict(qwen21_native_ane_cache="/"),
            dict(qwen21_native_ane_cache="/tmp/../cache"), dict(qwen21_native_ane_cache="/tmp/./cache"),
            dict(qwen21_native_ane_cache="/tmp/\x00cache"), dict(qwen21_native_ane_cache="/" + "a" * 4096),
            dict(qwen21_denoiser_backend="mlx"), dict(execution="gpu"),
            dict(allow_approximation=False), dict(ane_manifest="legacy.json"),
            dict(hybrid_mlp_mode="runtime"), dict(qwen21_gpu_w8a16=True),
            dict(loras=[dict(path="adapter.safetensors")]), dict(width=768), dict(steps=6),
            dict(operation="image.edit", inputs=[dict(kind="image", role="reference", path="local.png")]),
        ):
            with self.subTest(change=change):
                status, result, error = plan({**self.w1, **change})
                self.assertNotEqual(status, 0)
                self.assertIsNone(result)
                self.assertTrue(error)
        missing = {key: value for key, value in self.w1.items() if key != "qwen21_native_ane_cache"}
        self.assertNotEqual(plan(missing)[0], 0)
        self.assertNotEqual(plan({**self.base, "qwen21_native_ane_cache": self.w1["qwen21_native_ane_cache"]})[0], 0)
        # Existing MLX GPU+ANE cannot inherit the native manifest exemption.
        legacy = {**missing, "qwen21_denoiser_backend": "mlx"}
        self.assertNotEqual(plan(legacy)[0], 0)

    def test_w1_staging_variant_is_explicit_and_confined_to_native_private_route(self):
        for stage in ("generic", "register"):
            with self.subTest(stage=stage):
                status, result, error = plan({**self.w1, "qwen21_native_w8_staging": stage})
                if PRIVATE_ANE:
                    self.assertEqual(status, 0, error)
                    self.assertEqual(result["qwen21_native_w8_staging"], stage)
                else:
                    self.assertNotEqual(status, 0)
                for request in (self.base, {**self.base, "qwen21_denoiser_backend": "mlx"},
                                {**self.base, "model": "z-image-turbo", "steps": 4}):
                    self.assertNotEqual(plan({**request, "qwen21_native_w8_staging": stage})[0], 0)
        for stage in (None, True, 1, [], {}, "", "Register", "auto"):
            with self.subTest(invalid=stage):
                self.assertNotEqual(plan({**self.w1, "qwen21_native_w8_staging": stage})[0], 0)
        v2 = dict(schema_version=2, model=self.base["model"], operation=self.base["operation"],
                  inputs=[dict(kind="text", role="prompt", text=self.base["prompt"])],
                  outputs=[dict(kind="image", path="result.png", width=512, height=512,
                                frames=1, audio=False)], sampling=dict(seed=42, steps=20),
                  execution=dict(policy="gpu_ane", residency="component_staged", allow_approximation=True,
                                 qwen21_denoiser_backend="native-metal-graph", qwen21_dit_cache="off",
                                 qwen21_w8a8=True, qwen21_native_ane_cache=self.w1["qwen21_native_ane_cache"],
                                 qwen21_native_w8_staging="register"))
        status, result, error = plan(v2)
        if PRIVATE_ANE:
            self.assertEqual(status, 0, error)
            self.assertEqual(result["qwen21_native_w8_staging"], "register")
        else:
            self.assertNotEqual(status, 0)
        del v2["execution"]["qwen21_native_w8_staging"]
        for misplaced in ({**v2, "parameters": dict(qwen21_native_w8_staging="register")},
                          {**v2, "qwen21_native_w8_staging": "register"}):
            self.assertNotEqual(plan(misplaced)[0], 0)

    def test_w1_scale_cache_strict_boolean_presence_and_route(self):
        for enabled in (False, True):
            for staging in ("generic", "register"):
                request = {**self.w1, "qwen21_native_scale_cache": enabled,
                           "qwen21_native_w8_staging": staging}
                with self.subTest(enabled=enabled, staging=staging):
                    status, result, error = plan(request)
                    if PRIVATE_ANE:
                        self.assertEqual(status, 0, error)
                        self.assertIs(result["qwen21_native_scale_cache"], enabled)
                        self.assertEqual(result["qwen21_native_w8_staging"], staging)
                    else:
                        self.assertNotEqual(status, 0)
            # Explicit false is still a native-W1-only request option.
            for request in (self.base, {**self.base, "qwen21_denoiser_backend": "mlx"},
                            {**self.base, "model": "z-image-turbo", "steps": 4},
                            {**self.w1, "loras": [dict(path="adapter.safetensors")]}):
                self.assertNotEqual(plan({**request, "qwen21_native_scale_cache": enabled})[0], 0)
        for value in (None, 0, 1, 0.0, "true", "false", [], {}):
            with self.subTest(invalid=value):
                self.assertNotEqual(plan({**self.w1, "qwen21_native_scale_cache": value})[0], 0)
        v2 = dict(schema_version=2, model=self.base["model"], operation=self.base["operation"],
                  inputs=[dict(kind="text", role="prompt", text=self.base["prompt"])],
                  outputs=[dict(kind="image", path="result.png", width=512, height=512,
                                frames=1, audio=False)], sampling=dict(seed=42, steps=20),
                  execution=dict(policy="gpu_ane", residency="component_staged", allow_approximation=True,
                                 qwen21_denoiser_backend="native-metal-graph", qwen21_dit_cache="off",
                                 qwen21_w8a8=True, qwen21_native_ane_cache=self.w1["qwen21_native_ane_cache"]))
        for enabled in (False, True):
            valid = {**v2, "execution": {**v2["execution"], "qwen21_native_scale_cache": enabled}}
            status, result, error = plan(valid)
            if PRIVATE_ANE:
                self.assertEqual(status, 0, error)
                self.assertIs(result["qwen21_native_scale_cache"], enabled)
            else:
                self.assertNotEqual(status, 0)
            for misplaced in ({**v2, "parameters": dict(qwen21_native_scale_cache=enabled)},
                              {**v2, "qwen21_native_scale_cache": enabled}):
                self.assertNotEqual(plan(misplaced)[0], 0)
        for invalid in (None, 0, 1, "true", [], {}):
            self.assertNotEqual(plan({**v2, "execution": {
                **v2["execution"], "qwen21_native_scale_cache": invalid}})[0], 0)

    def test_unsupported_recipes_fail_without_silent_fallback(self):
        for change in (
            dict(qwen21_denoiser_backend="unknown"), dict(operation="image.edit"),
            dict(inputs=[dict(kind="image", role="reference", path="local.png")]),
            dict(width=768), dict(steps=6), dict(execution="auto"), dict(execution="gpu_ane"),
            dict(allow_approximation=False), dict(loras=[dict(path="adapter.safetensors")]),
            dict(ane_manifest="local.json"), dict(encoder_ane_manifest="local.json"),
            dict(hybrid_mlp_mode="runtime"), dict(qwen21_w8a8=True),
            dict(qwen21_gpu_full_ffn_blocks=[0]), dict(compile_gpu=True),
            dict(qwen21_dit_cache="balanced"), dict(quantized_cache="local"),
            dict(prompt_enhance=True, prompt_enhancer_path="local"),
        ):
            with self.subTest(change=change):
                status, result, error = plan({**self.base, **change})
                self.assertNotEqual(status, 0)
                self.assertIsNone(result)
                self.assertTrue(error)
        other = {**self.base, "model": "z-image-turbo", "steps": 4, "residency": "resident"}
        self.assertNotEqual(plan(other)[0], 0)
        self.assertEqual(plan({**other, "qwen21_denoiser_backend": "mlx"})[0], 0)

    def test_strict_type_and_nonempty_strings(self):
        for value in (None, True, 1, [], {}, "", "native_metal_graph"):
            with self.subTest(value=value):
                self.assertNotEqual(plan({**self.base, "qwen21_denoiser_backend": value})[0], 0)
        for value in (None, True, 1, [], {}, ""):
            with self.subTest(cache=value):
                self.assertNotEqual(plan({**self.w1, "qwen21_native_ane_cache": value})[0], 0)
        for wrong_key in ("qwen21_native_ane_caches", "native_ane_cache"):
            with self.subTest(key=wrong_key):
                self.assertNotEqual(plan({**self.base, wrong_key: "/tmp/cache"})[0], 0)

    def test_bf16_streaming_does_not_override_native_dit_before_loading_weights(self):
        # Header-only constructor; the conflict must fail before parsing any
        # checkpoint, allocating model tensors, or publishing output.
        LIB.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        LIB.tc_engine_prepare.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        LIB.tc_engine_free.argtypes = [C.c_void_p]
        with tempfile.TemporaryDirectory(prefix="tc-native-stream-conflict-") as raw:
            root=Path(raw)
            for relative in ("text_encoders/qwen3vl_8b_bf16.safetensors",
                             "diffusion_models/qwen_image_2.1_bf16.safetensors",
                             "vae/qwen_image_2.1_vae_bf16.safetensors", "processor/tokenizer.json"):
                file=root/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b"not a checkpoint")
            engine,error=C.c_void_p(),C.c_void_p()
            status=LIB.tc_engine_create_model(b"qwen-image-2.1",str(root).encode(),C.byref(engine),C.byref(error))
            message=C.string_at(error).decode() if error.value else ""
            if error.value:LIB.tc_string_free(error)
            self.assertEqual(status,0,message)
            try:
                request={**self.base,"output":str(root/"must-not-exist.png")}
                result,error=C.c_void_p(),C.c_void_p()
                with patch.dict(os.environ,{"TURBOCIDER_QWEN21_BF16_STREAMING":"1"}):
                    status=LIB.tc_engine_prepare(engine,json.dumps(request).encode(),0,None,None,C.byref(result),C.byref(error))
                message=C.string_at(error).decode() if error.value else ""
                if result.value:LIB.tc_string_free(result)
                if error.value:LIB.tc_string_free(error)
                self.assertNotEqual(status,0,message)
                self.assertIn("BF16 streaming cannot silently override",message)
                self.assertFalse((root/"must-not-exist.png").exists())
            finally:LIB.tc_engine_free(engine)

    def test_foreign_executor_diagnostics_cannot_change_recipe(self):
        for key in ("TURBOCIDER_QWEN21_METAL_QK_ROPE", "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS",
                    "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN", "TURBOCIDER_ANE_CALIBRATION_DIR",
                    "TURBOCIDER_QWEN21_LORA_BF16_AB", "TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS",
                    "TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE", "TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
                    "TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS", "TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS"):
            with self.subTest(key=key), patch.dict(os.environ, {key: "1"}):
                self.assertNotEqual(plan(self.base)[0], 0)
        with patch.dict(os.environ, {"TURBOCIDER_QWEN21_METAL_QK_ROPE": "0"}):
            self.assertEqual(plan(self.base)[0], 0)


if __name__ == "__main__":
    unittest.main()
