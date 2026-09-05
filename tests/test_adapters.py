from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from turbocider.adapters import adapter_for, register_adapter
from turbocider.adapters.base import CommandSpec, EngineAdapter
from turbocider.adapters.flux2 import Flux2Adapter
from turbocider.adapters.fastmetal import FastMetalAdapter
from turbocider.adapters.h3 import H3Adapter
from turbocider.adapters.ltx import LTXAdapter
from turbocider.errors import EngineUnavailableError, ValidationError
from turbocider.models import ExecutionPlan, GenerationRequest, ModelDescriptor


ROOT = Path(__file__).resolve().parents[2]


class AdapterTests(unittest.TestCase):
    def test_developer_adapter_can_be_registered_without_changing_runtime(self):
        class TestAdapter(EngineAdapter):
            name = "test-developer-adapter"

            def build_command(self, model, request, plan, output_path):
                return CommandSpec(argv=["/usr/bin/true"], cwd=Path("/tmp"))

        register_adapter(TestAdapter.name, TestAdapter)
        self.assertIsInstance(adapter_for(TestAdapter.name), TestAdapter)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_adapter(TestAdapter.name, TestAdapter)

    def test_registered_adapter_name_must_match_engine_id(self):
        class MismatchedAdapter(EngineAdapter):
            name = "different-name"

        register_adapter("test-mismatched-adapter", MismatchedAdapter)
        with self.assertRaisesRegex(ValueError, "returned adapter named"):
            adapter_for("test-mismatched-adapter")

    def test_fastmetal_hybrid_maps_fixed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            script = engine / "mlx_wan_prompt_to_video.py"
            model_path = root / "model"
            manifest = root / "coreml" / "manifest.json"
            bridge = root / "bridge"
            engine.mkdir()
            model_path.mkdir()
            manifest.parent.mkdir()
            manifest.touch()
            bridge.mkdir()
            script.touch()
            model = ModelDescriptor(
                id="fastmetal", name="FastMetal", engine="fastmetal", version="1",
                capabilities={},
                config={
                    "python_path": "/usr/bin/python3",
                    "engine_root": str(engine),
                    "script_path": str(script),
                    "model_path": str(model_path),
                    "ane_manifest": str(manifest),
                    "bridge_dir": str(bridge),
                },
                plans=[],
            )
            request = GenerationRequest.from_dict({
                "model": "fastmetal", "prompt": "hello",
                "output": {
                    "type": "video", "width": 832, "height": 480,
                    "frames": 81, "fps": 16, "audio": False,
                },
                "sampling": {"steps": 3},
            })
            plan = ExecutionPlan.from_dict({
                "id": "hybrid", "execution": "gpu_ane", "quality": "validated"
            })

            spec = FastMetalAdapter().build_command(
                model, request, plan, root / "out.mp4"
            )

            self.assertIn("--ane-manifest", spec.argv)
            self.assertIn(str(manifest), spec.argv)
            self.assertIn("--ane-bridge-dir", spec.argv)
            self.assertIn("--mlx-compile", spec.argv)
            self.assertEqual(spec.output_paths[-1], root / "out.engine.json")

    def test_fastmetal_rejects_non_native_dmd_step_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            model_path = root / "model"
            engine.mkdir()
            model_path.mkdir()
            script = engine / "run.py"
            script.touch()
            model = ModelDescriptor(
                id="fastmetal", name="FastMetal", engine="fastmetal", version="1",
                capabilities={},
                config={
                    "python_path": "/usr/bin/python3", "engine_root": str(engine),
                    "script_path": str(script), "model_path": str(model_path),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact"
            })
            request = GenerationRequest.from_dict({
                "model": "fastmetal", "prompt": "hello",
                "output": {"width": 832, "height": 480, "frames": 81},
                "sampling": {"steps": 4},
            })
            with self.assertRaisesRegex(ValidationError, "three-step DMD"):
                FastMetalAdapter().build_command(model, request, plan, root / "out.mp4")

    def test_h3_maps_reference_inputs_and_clears_ane_environment(self):
        model = ModelDescriptor(
            id="h3", name="h3", engine="h3", version="1", capabilities={},
            config={
                "executable_path": str(ROOT / "h3.c" / "h3"),
                "model_path": str(ROOT / "h3.c" / "models" / "MiniMax-H3-LightX2V-Turbo"),
            },
            plans=[],
        )
        request = GenerationRequest.from_dict({
            "model": "h3",
            "prompt": "hello",
            "inputs": [
                {"type": "image", "role": "first_frame", "path": "/tmp/first.png"},
                {"type": "video", "role": "reference", "path": "/tmp/ref.mp4", "include_embedded_audio": False},
            ],
        })
        plan = ExecutionPlan.from_dict({"id": "gpu", "execution": "gpu", "quality": "exact"})
        spec = H3Adapter().build_command(model, request, plan, Path("/tmp/out.mp4"))
        self.assertIn("--first-frame", spec.argv)
        self.assertIn("--ref-silent-video", spec.argv)
        self.assertIn("H3_PRIVATE_ANE_", spec.unset_environment_prefixes)

    def test_h3_hybrid_rejects_environment_without_an_ane_route(self):
        model = ModelDescriptor(
            id="h3", name="h3", engine="h3", version="1", capabilities={},
            config={
                "executable_path": str(ROOT / "h3.c" / "h3"),
                "model_path": str(ROOT / "h3.c" / "models" / "MiniMax-H3-LightX2V-Turbo"),
            },
            plans=[],
        )
        request = GenerationRequest.from_dict({
            "model": "h3", "prompt": "hello",
            "engine_options": {"h3": {"env": {"UNRELATED": "1"}}},
        })
        plan = ExecutionPlan.from_dict({
            "id": "hybrid", "execution": "gpu_ane", "quality": "validated"
        })
        with self.assertRaisesRegex(EngineUnavailableError, "requires a Core ML"):
            H3Adapter().build_command(model, request, plan, Path("/tmp/out.mp4"))

    def test_h3_hybrid_accepts_checkpoint_backed_ane_projection(self):
        model = ModelDescriptor(
            id="h3", name="h3", engine="h3", version="1", capabilities={},
            config={
                "executable_path": str(ROOT / "h3.c" / "h3"),
                "model_path": str(ROOT / "h3.c" / "models" / "MiniMax-H3-LightX2V-Turbo"),
            },
            plans=[],
        )
        request = GenerationRequest.from_dict({
            "model": "h3", "prompt": "hello",
            "engine_options": {
                "h3": {"env": {"H3_PRIVATE_ANE_QKV_CHECKPOINT": "1"}}
            },
        })
        plan = ExecutionPlan.from_dict({
            "id": "hybrid", "execution": "gpu_ane", "quality": "validated"
        })
        spec = H3Adapter().build_command(model, request, plan, Path("/tmp/out.mp4"))
        self.assertEqual(
            spec.metadata["ane_configuration"],
            ["H3_PRIVATE_ANE_QKV_CHECKPOINT"],
        )

    def test_flux_hybrid_maps_manifests(self):
        model = ModelDescriptor(
            id="flux", name="flux", engine="flux2", version="1", capabilities={},
            config={
                "python_path": "/usr/bin/python3",
                "engine_root": str(ROOT / "gpu_ane" / "flux2-engine"),
                "model_path": "/tmp/model",
                "mflux_root": "/tmp/mflux",
                "ane_manifests": ["/tmp/manifest.json"],
            },
            plans=[],
        )
        for path in (Path("/tmp/model"), Path("/tmp/mflux")):
            path.mkdir(exist_ok=True)
        Path("/tmp/manifest.json").touch()
        request = GenerationRequest.from_dict({
            "model": "flux", "prompt": "hello", "task": "image",
            "output": {"type": "image"},
            "policy": {"persistent": True},
        })
        plan = ExecutionPlan.from_dict({"id": "hybrid", "execution": "gpu_ane", "quality": "validated"})
        spec = Flux2Adapter().build_command(model, request, plan, Path("/tmp/out.png"))
        self.assertIn("mlx-ane", spec.argv)
        self.assertIn("--ane-manifest", spec.argv)
        self.assertIn("--model-variant", spec.argv)

    def test_flux_persistent_worker_preserves_advanced_runtime_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            model_path = root / "model"
            mflux = root / "mflux"
            manifest = root / "manifest.json"
            bridge = root / "bridge"
            for path in (engine, model_path, mflux, bridge):
                path.mkdir()
            manifest.touch()
            model = ModelDescriptor(
                id="flux", name="flux", engine="flux2", version="1",
                capabilities={},
                config={
                    "python_path": "/usr/bin/python3",
                    "engine_root": str(engine),
                    "model_path": str(model_path),
                    "mflux_root": str(mflux),
                    "bridge_dir": str(bridge),
                    "ane_manifests": [str(manifest)],
                    "clear_mlx_cache_between_requests": False,
                },
                plans=[],
            )
            request = GenerationRequest.from_dict({
                "model": "flux", "prompt": "hello", "task": "image",
                "output": {"type": "image"},
                "policy": {"persistent": True},
                "engine_options": {"flux2": {
                    "attention": "mlx",
                    "dynamic_text_length": False,
                    "reuse_ane_outputs": False,
                    "compile_quantized_gpu_attention": False,
                }},
            })
            plan = ExecutionPlan.from_dict({
                "id": "hybrid", "execution": "gpu_ane", "quality": "validated"
            })
            spec = Flux2Adapter().build_command(
                model, request, plan, root / "out.png"
            )
            worker = spec.metadata["persistent_worker"]
            self.assertIn("--attention", worker["argv"])
            self.assertIn("mlx", worker["argv"])
            self.assertIn("--no-reuse-ane-outputs", worker["argv"])
            self.assertIn("--no-compile-quantized-gpu-attention", worker["argv"])
            self.assertIn("--no-clear-mlx-cache-between-requests", worker["argv"])
            self.assertFalse(worker["request"]["dynamic_text_length"])

            override = GenerationRequest.from_dict({
                "model": "flux", "prompt": "hello", "task": "image",
                "output": {"type": "image"},
                "policy": {"persistent": True},
                "engine_options": {"flux2": {
                    "clear_mlx_cache_between_requests": True,
                }},
            })
            override_spec = Flux2Adapter().build_command(
                model, override, plan, root / "override.png"
            )
            override_argv = override_spec.metadata["persistent_worker"]["argv"]
            self.assertIn("--clear-mlx-cache-between-requests", override_argv)
            self.assertNotIn("--no-clear-mlx-cache-between-requests", override_argv)

    def test_flux_single_image_and_multi_reference_inputs_reach_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            model_path = root / "model"
            mflux = root / "mflux"
            for path in (engine, model_path, mflux):
                path.mkdir()
            model = ModelDescriptor(
                id="flux", name="flux", engine="flux2", version="1",
                capabilities={},
                config={
                    "python_path": "/usr/bin/python3",
                    "engine_root": str(engine),
                    "model_path": str(model_path),
                    "mflux_root": str(mflux),
                },
                plans=[],
            )
            gpu = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact",
            })
            init = root / "init.png"
            first = root / "one.png"
            second = root / "two.png"
            for path in (init, first, second):
                path.touch()

            img2img = GenerationRequest.from_dict({
                "model": "flux", "task": "image", "prompt": "restyle",
                "mode": "image_to_image",
                "inputs": [{
                    "type": "image", "role": "init_image", "path": str(init),
                    "strength": 0.4,
                }],
                "output": {"type": "image"},
            })
            img2img_spec = Flux2Adapter().build_command(
                model, img2img, gpu, root / "img2img.png"
            )
            self.assertIn("--pipeline", img2img_spec.argv)
            self.assertEqual(img2img_spec.argv[img2img_spec.argv.index("--pipeline") + 1], "standard")
            self.assertEqual(
                img2img_spec.argv[img2img_spec.argv.index("--image-path") + 1],
                str(init.resolve()),
            )
            self.assertEqual(img2img_spec.argv[img2img_spec.argv.index("--image-strength") + 1], "0.4")

            edit = GenerationRequest.from_dict({
                "model": "flux", "task": "image", "prompt": "combine",
                "mode": "image_edit",
                "inputs": [
                    {"type": "image", "role": "reference", "path": str(first)},
                    {"type": "image", "role": "reference", "path": str(second)},
                ],
                "output": {"type": "image"},
                "policy": {"persistent": True},
            })
            edit_spec = Flux2Adapter().build_command(
                model, edit, gpu, root / "edit.png"
            )
            worker = edit_spec.metadata["persistent_worker"]
            self.assertEqual(worker["argv"][worker["argv"].index("--pipeline") + 1], "edit")
            self.assertEqual(
                worker["request"]["image_paths"],
                [str(first.resolve()), str(second.resolve())],
            )
            self.assertIsNone(worker["request"]["image_path"])

    def test_ltx_rejects_silently_ignored_sampling_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "build" / "bench_block_mlx"
            engine.parent.mkdir()
            engine.touch()
            files = {}
            for name in ("transformer", "upsampler", "video_vae"):
                files[name] = root / name
                files[name].touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "gpu_executable_path": str(engine),
                    "transformer_path": str(files["transformer"]),
                    "upsampler_path": str(files["upsampler"]),
                    "video_vae_path": str(files["video_vae"]),
                    "conditioning_directory": str(conditioning),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact"
            })
            request = GenerationRequest.from_dict({
                "model": "ltx", "prompt": "hello",
                "output": {
                    "width": 704, "height": 480, "frames": 97,
                    "fps": 24, "audio": False,
                },
                "sampling": {"steps": 4},
            })
            with self.assertRaisesRegex(ValidationError, "fixed 8\+3 schedule"):
                LTXAdapter().build_command(model, request, plan, root / "out.mp4")

    def test_ltx_hybrid_uses_resolved_model_pack_ane_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "build" / "bench_block_mlx_ane"
            engine.parent.mkdir()
            engine.touch()
            files = {}
            for name in ("transformer", "upsampler", "video_vae"):
                files[name] = root / name
                files[name].touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            ane = {}
            for name in ("stage1", "stage2", "kv"):
                ane[name] = root / name
                ane[name].mkdir()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "hybrid_executable_path": str(engine),
                    "transformer_path": str(files["transformer"]),
                    "upsampler_path": str(files["upsampler"]),
                    "video_vae_path": str(files["video_vae"]),
                    "conditioning_directory": str(conditioning),
                    "ane_mlp_stage1_directory": str(ane["stage1"]),
                    "ane_mlp_stage2_directory": str(ane["stage2"]),
                    "ane_kv_directory": str(ane["kv"]),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "hybrid", "execution": "gpu_ane",
                "quality": "validated",
                "environment": {"LTX_ANE_MLP_STAGE1_DIR": "/stale"},
            })
            request = GenerationRequest.from_dict({
                "model": "ltx", "prompt": "hello",
                "output": {
                    "width": 704, "height": 480, "frames": 97,
                    "fps": 24, "audio": False,
                },
            })
            spec = LTXAdapter().build_command(model, request, plan, root / "out.mp4")
            self.assertEqual(spec.environment["LTX_ANE_MLP_STAGE1_DIR"], str(ane["stage1"]))
            self.assertEqual(spec.environment["LTX_ANE_MLP_STAGE2_DIR"], str(ane["stage2"]))
            self.assertEqual(spec.environment["LTX_ANE_KV_DIR"], str(ane["kv"]))

    def test_ltx_fast_hybrid_uses_r256_kv_and_fast_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "build" / "bench_block_mlx_ane"
            engine.parent.mkdir()
            engine.touch()
            for name in ("transformer", "upsampler", "video_vae"):
                (root / name).touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            ane = {}
            for name in ("stage1", "stage2", "kv", "kv_r256"):
                ane[name] = root / name
                ane[name].mkdir()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "hybrid_executable_path": str(engine),
                    "transformer_path": str(root / "transformer"),
                    "upsampler_path": str(root / "upsampler"),
                    "video_vae_path": str(root / "video_vae"),
                    "conditioning_directory": str(conditioning),
                    "ane_mlp_stage1_directory": str(ane["stage1"]),
                    "ane_mlp_stage2_directory": str(ane["stage2"]),
                    "ane_kv_directory": str(ane["kv"]),
                    "ane_kv_r256_directory": str(ane["kv_r256"]),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "fast", "execution": "gpu_ane",
                "quality": "experimental", "profile": "preview",
                "environment": {
                    "LTX_ANE_KV_DIR": "/stale-r256",
                    "LTX_ANE_KV_STAGE1": "1",
                    "LTX_ANE_KV_STAGE2": "1",
                    "LTX_TEXT_ROWS_LIMIT": "256",
                    "LTX_SOL_VIDEO_SELF_STAGE2": "1",
                },
                "metadata": {
                    "ltx_ane_paths": {
                        "LTX_ANE_KV_DIR": "ane_kv_r256_directory",
                    },
                },
            })
            request = GenerationRequest.from_dict({
                "model": "ltx", "prompt": "hello",
                "output": {
                    "width": 704, "height": 480, "frames": 97,
                    "fps": 24, "audio": False,
                },
            })

            spec = LTXAdapter().build_command(
                model, request, plan, root / "out.mp4"
            )

            self.assertEqual(
                spec.environment["LTX_ANE_KV_DIR"], str(ane["kv_r256"])
            )
            self.assertEqual(spec.environment["LTX_TEXT_ROWS_LIMIT"], "256")
            self.assertEqual(spec.environment["LTX_ANE_KV_STAGE1"], "1")
            self.assertEqual(spec.environment["LTX_ANE_KV_STAGE2"], "1")
            self.assertEqual(spec.environment["LTX_SOL_VIDEO_SELF_STAGE2"], "1")

    def test_ltx_audio_request_requires_audio_vae(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "build" / "bench_block_mlx"
            engine.parent.mkdir()
            engine.touch()
            for name in ("transformer", "upsampler", "video_vae"):
                (root / name).touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "gpu_executable_path": str(engine),
                    "transformer_path": str(root / "transformer"),
                    "upsampler_path": str(root / "upsampler"),
                    "video_vae_path": str(root / "video_vae"),
                    "conditioning_directory": str(conditioning),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact"
            })
            request = GenerationRequest.from_dict({
                "model": "ltx", "prompt": "hello",
                "output": {
                    "width": 704, "height": 480, "frames": 97,
                    "fps": 24, "audio": True,
                },
                "sampling": {"steps": 11},
            })
            with self.assertRaisesRegex(
                EngineUnavailableError, "audio_vae=<unset>"
            ):
                LTXAdapter().build_command(model, request, plan, root / "out.mp4")

    def test_ltx_image_to_video_maps_first_frame_for_gpu_and_hybrid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            build.mkdir()
            gpu_engine = build / "bench_block_mlx"
            hybrid_engine = build / "bench_block_mlx_ane"
            helper = build / "bench_mlx_video_vae"
            for path in (gpu_engine, hybrid_engine, helper):
                path.touch()
            for name in ("transformer", "upsampler", "video_vae"):
                (root / name).touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            first_frame = root / "first.png"
            first_frame.touch()
            ane = {}
            for name in ("stage1", "stage2", "kv"):
                ane[name] = root / name
                ane[name].mkdir()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "gpu_executable_path": str(gpu_engine),
                    "hybrid_executable_path": str(hybrid_engine),
                    "video_vae_helper_path": str(helper),
                    "transformer_path": str(root / "transformer"),
                    "upsampler_path": str(root / "upsampler"),
                    "video_vae_path": str(root / "video_vae"),
                    "conditioning_directory": str(conditioning),
                    "ane_mlp_stage1_directory": str(ane["stage1"]),
                    "ane_mlp_stage2_directory": str(ane["stage2"]),
                    "ane_kv_directory": str(ane["kv"]),
                },
                plans=[],
            )
            request = GenerationRequest.from_dict({
                "model": "ltx", "task": "video", "prompt": "animate",
                "mode": "image_to_video",
                "inputs": [{
                    "type": "image", "role": "first_frame",
                    "path": str(first_frame), "strength": 0.65,
                    "frame_index": 0,
                }],
                "output": {
                    "type": "video", "width": 704, "height": 480,
                    "frames": 97, "fps": 24, "audio": False,
                },
                "engine_options": {"ltx": {"image_crf": 31}},
            })
            gpu = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact",
            })
            hybrid = ExecutionPlan.from_dict({
                "id": "hybrid", "execution": "gpu_ane",
                "quality": "validated",
            })

            gpu_spec = LTXAdapter().build_command(
                model, request, gpu, root / "gpu.mp4"
            )
            hybrid_spec = LTXAdapter().build_command(
                model, request, hybrid, root / "hybrid.mp4"
            )

            for spec in (gpu_spec, hybrid_spec):
                self.assertEqual(
                    spec.argv[spec.argv.index("--first-frame") + 1],
                    str(first_frame.resolve()),
                )
                self.assertEqual(
                    spec.argv[spec.argv.index("--image-strength") + 1],
                    "0.65",
                )
                self.assertEqual(
                    spec.argv[spec.argv.index("--image-crf") + 1], "31"
                )
                self.assertEqual(
                    spec.argv[spec.argv.index("--video-vae-helper") + 1],
                    str(helper),
                )
                self.assertEqual(spec.metadata["conditioning_mode"], "image_to_video")
            self.assertIn("LTX_ANE_", gpu_spec.unset_environment_prefixes)
            self.assertEqual(
                hybrid_spec.environment["LTX_ANE_MLP_STAGE1_DIR"],
                str(ane["stage1"]),
            )
            self.assertEqual(
                hybrid_spec.environment["LTX_ANE_MLP_STAGE2_DIR"],
                str(ane["stage2"]),
            )

    def test_ltx_image_modes_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            build.mkdir()
            engine = build / "bench_block_mlx"
            engine.touch()
            for name in ("transformer", "upsampler", "video_vae"):
                (root / name).touch()
            conditioning = root / "conditioning"
            conditioning.mkdir()
            image = root / "image.png"
            image.touch()
            model = ModelDescriptor(
                id="ltx", name="ltx", engine="ltx25", version="1",
                capabilities={},
                config={
                    "gpu_executable_path": str(engine),
                    "transformer_path": str(root / "transformer"),
                    "upsampler_path": str(root / "upsampler"),
                    "video_vae_path": str(root / "video_vae"),
                    "conditioning_directory": str(conditioning),
                },
                plans=[],
            )
            plan = ExecutionPlan.from_dict({
                "id": "gpu", "execution": "gpu", "quality": "exact",
            })

            def request(mode, inputs, options=None):
                return GenerationRequest.from_dict({
                    "model": "ltx", "task": "video", "prompt": "animate",
                    "mode": mode, "inputs": inputs,
                    "output": {
                        "type": "video", "width": 704, "height": 480,
                        "frames": 97, "fps": 24, "audio": False,
                    },
                    "engine_options": {"ltx": options or {}},
                })

            with self.assertRaisesRegex(ValidationError, "does not accept image"):
                LTXAdapter().build_command(
                    model,
                    request("text_to_video", [{
                        "type": "image", "role": "first_frame",
                        "path": str(image),
                    }]),
                    plan, root / "text.mp4",
                )
            with self.assertRaisesRegex(ValidationError, "exactly one image"):
                LTXAdapter().build_command(
                    model,
                    request("image_to_video", [{
                        "type": "image", "role": "reference",
                        "path": str(image),
                    }]),
                    plan, root / "wrong-role.mp4",
                )
            with self.assertRaisesRegex(ValidationError, "unsupported LTX-2.5 mode"):
                LTXAdapter().build_command(
                    model,
                    request("keyframe_interpolation", [{
                        "type": "image", "role": "first_frame",
                        "path": str(image),
                    }]),
                    plan, root / "keyframes.mp4",
                )
            with self.assertRaisesRegex(ValidationError, "image strength"):
                LTXAdapter().build_command(
                    model,
                    request("image_to_video", [{
                        "type": "image", "role": "first_frame",
                        "path": str(image),
                    }], {"image_strength": 1.1}),
                    plan, root / "strength.mp4",
                )
            with self.assertRaisesRegex(ValidationError, "image_crf"):
                LTXAdapter().build_command(
                    model,
                    request("image_to_video", [{
                        "type": "image", "role": "first_frame",
                        "path": str(image),
                    }], {"image_crf": 52}),
                    plan, root / "crf.mp4",
                )
            with self.assertRaisesRegex(EngineUnavailableError, "encoder helper"):
                LTXAdapter().build_command(
                    model,
                    request("image_to_video", [{
                        "type": "image", "role": "first_frame",
                        "path": str(image),
                    }]),
                    plan, root / "helper.mp4",
                )


if __name__ == "__main__":
    unittest.main()
