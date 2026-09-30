"""Native CLI ANE manifest override contract; no model weights needed."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CLI = Path(os.environ.get("TURBOCIDER_TEST_CLI", ROOT / "build/native/turbocider"))


def clean_acceleration_environment():
    """Keep the caller's experimental switches out of route contract tests."""
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("TURBOCIDER_QWEN21_", "TURBOCIDER_Z_",
                                   "TURBOCIDER_RUNTIME_ANE_"))}


@unittest.skipUnless(CLI.is_file(), "build the native CLI before this test")
class CLIAneOverrideTests(unittest.TestCase):
    def test_qwen_runtime_ref512_switch_accepts_base_without_relaxing_other_routes(self):
        env = clean_acceleration_environment()
        env["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"] = "1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "request.json"
            base = {"model": "qwen-image-2.1", "operation": "image.edit", "prompt": "A fox",
                    "output": str(root / "out.png"), "steps": 6, "width": 512, "height": 512,
                    "execution": "gpu_ane", "audio": False, "residency": "resident",
                    "allow_approximation": True, "hybrid_mlp_mode": "runtime",
                    "ane_manifest": "checked-at-load.json", "qwen21_reference_size": 512,
                    "inputs": [{"kind": "image", "role": "reference", "path": "ref.png"}]}
            adapter = {"path": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
                       "role": "transformer", "strength": 1.0}

            def plan(data):
                request.write_text(json.dumps(data))
                return subprocess.run([str(CLI), "plan", str(request)], cwd=ROOT,
                                      env=env, capture_output=True, text=True)

            for count in (1, 2, 3):
                for lora in (False, True):
                    data = {**base, "inputs": base["inputs"] * count}
                    if lora:
                        data.update(lora_strategy="inference_time", loras=[adapter])
                    with self.subTest(count=count, lora=lora):
                        result = plan(data)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(json.loads(result.stdout)["backend"],
                                         "mlx_cpp_metal+coreml_runtime_weight")
            for change in ({"execution": "gpu", "hybrid_mlp_mode": "auto"},
                           {"hybrid_mlp_mode": "auto"},
                           {"hybrid_mlp_mode": "lora_fused", "qwen21_w8a8": True},
                           {"allow_approximation": False}, {"steps": 8}, {"width": 1024},
                           {"inputs": []}, {"inputs": base["inputs"] * 4},
                           {"qwen21_reference_size": 1024},
                           {"lora_strategy": "inference_time", "loras": [adapter, adapter]}):
                with self.subTest(change=change):
                    self.assertNotEqual(plan({**base, **change}).returncode, 0)

    def test_runtime_tuning_does_not_opt_default_requests_into_runtime(self):
        env = clean_acceleration_environment()
        env.update(TURBOCIDER_RUNTIME_ANE_CHUNKS="2", TURBOCIDER_RUNTIME_ANE_PROFILE="1")
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary) / "request.json"
            for model in ("z-image-turbo", "qwen-image-2.1", "z-image-turbo-gguf"):
                for execution in ("auto", "gpu"):
                    with self.subTest(model=model, execution=execution):
                        request.write_text(json.dumps({
                            "model": model, "operation": "image.generate", "prompt": "A fox",
                            "output": str(Path(temporary) / "out.png"), "steps": 8,
                            "width": 512, "height": 512, "execution": execution,
                            "residency": "resident", "audio": False}))
                        result = subprocess.run([str(CLI), "plan", str(request)], cwd=ROOT,
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        data = json.loads(result.stdout)
                        self.assertEqual(data["hybrid_mlp_mode"], "auto")
                        self.assertNotIn("runtime_weight", data["backend"])
                        self.assertNotEqual(data["gpu_graph"], "runtime_weight_token_row_ffn")

    def test_runtime_weight_base_route_and_rejections(self):
        env = clean_acceleration_environment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            request = root / "request.json"
            for model in ("z-image-turbo", "qwen-image-2.1"):
                base = {"model": model, "operation": "image.generate", "prompt": "A fox",
                        "output": str(root / "out.png"), "steps": 8, "width": 512,
                        "height": 512, "execution": "gpu", "audio": False,
                        "residency": "resident", "allow_approximation": True}
                request.write_text(json.dumps(base))
                command = [str(CLI), "plan", str(request), "--hybrid-mode", "runtime",
                           "--ane-manifest", str(manifest)]
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["hybrid_mlp_mode"], "runtime")
                self.assertEqual(data["backend"], "mlx_cpp_metal+coreml_runtime_weight")
                self.assertEqual(data["gpu_graph"], "runtime_weight_token_row_ffn")
                self.assertIn("runtime_weight_fp16_token_row_ffn", data["algorithm_approximations"])
                for change in (
                    {"loras": [{"path": "adapter.safetensors", "role": "transformer", "strength": 1.0}]},
                    {"qwen21_w8a8": True}, {"residency": "streamed"},
                ):
                    request.write_text(json.dumps({**base, **change}))
                    rejected = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
                    self.assertNotEqual(rejected.returncode, 0, change)
                # The existing --ane-manifest shortcut explicitly opts into
                # approximation. Check the JSON-only accuracy gate without it.
                request.write_text(json.dumps({**base, "execution": "gpu_ane",
                                               "hybrid_mlp_mode": "runtime",
                                               "ane_manifest": str(manifest),
                                               "allow_approximation": False}))
                rejected = subprocess.run([str(CLI), "plan", str(request)], cwd=ROOT,
                                          env=env, capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0, model)

    def test_gguf_runtime_explicit_route_and_rejections(self):
        env = clean_acceleration_environment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "request.json"
            base = {"model": "z-image-turbo-gguf", "operation": "image.generate",
                    "prompt": "A fox", "output": str(root / "out.png"), "steps": 8,
                    "width": 512, "height": 512, "execution": "gpu_ane", "audio": False,
                    "residency": "resident", "allow_approximation": True,
                    "hybrid_mlp_mode": "runtime", "ane_manifest": "checked-at-load.json"}
            request.write_text(json.dumps(base))
            command = [str(CLI), "plan", str(request)]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["backend"], "mlx_cpp_metal_gguf+coreml_runtime_weight")
            self.assertEqual(data["gpu_graph"], "runtime_weight_token_row_ffn")
            self.assertIn("runtime_weight_fp16_token_row_ffn", data["algorithm_approximations"])
            for change in ({"execution": "gpu"}, {"execution": "auto"},
                           {"allow_approximation": False}, {"ane_manifest": ""},
                           {"residency": "streamed"}, {"encoder_ane_manifest": "encoder.json"},
                           {"loras": [{"path": "adapter.safetensors", "role": "transformer"}]},
                           {"hybrid_mlp_mode": "lora_fused"}, {"hybrid_mlp_mode": "lora_suffix"}):
                request.write_text(json.dumps({**base, **change}))
                rejected = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
                with self.subTest(change=change):
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
            example = ROOT / "examples/requests/z-image-gguf-base-512.json"
            original = example.read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            for options, backend in (((), "mlx_cpp_metal_gguf"),
                (("--hybrid-mode", "runtime", "--ane-manifest", str(manifest)),
                 "mlx_cpp_metal_gguf+coreml_runtime_weight")):
                result = subprocess.run([str(CLI), "plan", str(example), *options],
                    cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["backend"], backend)
                if options:
                    self.assertEqual(plan["precision"], "gguf_native_gpu+runtime_fp16_ffn")
                    self.assertIn("checkpoint_defined_gguf_weight_quantization",
                                  plan["algorithm_approximations"])
            self.assertEqual(example.read_bytes(), original)
            self.assertFalse(Path(json.loads(original)["output"]).is_absolute())

    def test_portable_runtime_lora_examples(self):
        # Planning needs no real weights. Artifact identity is checked at load,
        # so this empty manifest checks only the public CLI/request contract.
        env = clean_acceleration_environment()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "compiled.json"
            manifest.write_text("{}")
            cases = (
                ("z-image-runtime-lora-512.json", (), "gpu"),
                ("qwen21-viggle-runtime-lora-512.json", (), "gpu"),
                ("z-image-runtime-lora-512.json",
                 ("--ane-manifest", str(manifest), "--hybrid-mode", "lora_fused"), "gpu_ane"),
                ("qwen21-viggle-runtime-lora-hybrid-512.json",
                 ("--ane-manifest", str(manifest)), "gpu_ane"),
                ("z-image-runtime-lora-512.json",
                 ("--ane-manifest", str(manifest), "--hybrid-mode", "runtime"), "gpu_ane"),
                ("qwen21-viggle-runtime-lora-512.json",
                 ("--ane-manifest", str(manifest), "--hybrid-mode", "runtime"), "gpu_ane"),
            )
            for name, options, execution in cases:
                with self.subTest(name=name, execution=execution):
                    request = ROOT / "examples/requests" / name
                    original = request.read_bytes()
                    result = subprocess.run([str(CLI), "plan", str(request), *options],
                                            capture_output=True, text=True, cwd=ROOT, env=env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan["requested_execution"], execution)
                    if execution == "gpu_ane":
                        self.assertEqual(plan["hybrid_mlp_mode"], "runtime" if "runtime" in options else "lora_fused")
                    self.assertEqual(request.read_bytes(), original)
                    data = json.loads(original)
                    self.assertFalse(Path(data["output"]).is_absolute())
                    self.assertTrue(all(not Path(lora["path"]).is_absolute()
                                        for lora in data["loras"]))
                    self.assertNotIn("ane_manifest", data)

    def plan(self, request, manifest):
        return subprocess.run([str(CLI), "plan", str(request), "--ane-manifest", str(manifest)],
                              capture_output=True, text=True, cwd=ROOT)

    def test_schema2_z_image_and_other_ane_model_remain_manifest_driven(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            for model, steps in (("z-image-turbo", 8), ("flux2-klein-4b", 4)):
                request = root / f"{model}.json"
                original = json.dumps({"schema_version": 2, "model": model,
                    "operation": "image.generate", "inputs": [
                        {"kind": "text", "role": "prompt", "text": "A red fox."}],
                    "outputs": [{"kind": "image", "path": str(root / "fox.png"),
                                 "width": 512, "height": 512}],
                    "sampling": {"steps": steps, "seed": 42},
                    "execution": {"policy": "gpu", "residency": "resident"}})
                request.write_text(original)
                result = self.plan(request, manifest)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["model"], model)
                self.assertEqual(data["requested_execution"], "gpu_ane")
                self.assertEqual(request.read_text(), original)

    def test_schema1_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            request = root / "request.json"
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A red fox.",
                "output": str(root / "out.png"), "width": 512, "height": 512,
                "steps": 8, "audio": False, "execution": "gpu"}))
            result = self.plan(request, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["requested_execution"], "gpu_ane")

    def test_rejects_missing_or_conflicting_manifests_without_touching_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = root / "chosen.json"
            chosen.write_text("{}")
            request = root / "request.json"
            original = json.dumps({"model": "z-image-turbo", "operation": "image.generate",
                "prompt": "A red fox.", "output": str(root / "out.png"),
                "width": 512, "height": 512, "steps": 8, "audio": False,
                "execution": "gpu", "ane_manifest": str(root / "different.json")})
            request.write_text(original)
            missing = self.plan(request, root / "missing.json")
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("existing manifest JSON file", missing.stderr)
            conflict = self.plan(request, chosen)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("conflicts with request ane_manifest", conflict.stderr)
            self.assertEqual(request.read_text(), original)

    def test_generate_and_batch_validate_the_override_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            request = root / "request.json"
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A red fox.",
                "output": str(root / "out.png"), "width": 512, "height": 512,
                "steps": 8, "audio": False, "execution": "gpu",
                "ane_manifest": str(root / "different.json")}))
            for command in (("generate", "model", str(request)),
                            ("batch", "model", str(request), str(request))):
                result = subprocess.run(
                    [str(CLI), *command, "--ane-manifest", str(manifest)],
                    capture_output=True, text=True, cwd=ROOT)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflicts with request ane_manifest", result.stderr)
                self.assertFalse((root / "out.png").exists())

    def test_hybrid_mode_override_preserves_accuracy_policy_and_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "request.json"
            original = json.dumps({"schema_version": 2, "model": "qwen-image-2.1",
                "operation": "image.generate", "inputs": [
                    {"kind": "text", "role": "prompt", "text": "A fox"}],
                "outputs": [{"kind": "image", "path": str(root / "out.png"),
                             "width": 512, "height": 512}],
                "sampling": {"steps": 6}, "execution": {"policy": "gpu", "residency": "resident"}})
            request.write_text(original)
            result = subprocess.run([str(CLI), "plan", str(request),
                                     "--hybrid-mode", "auto"],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["hybrid_mlp_mode"], "auto")
            self.assertEqual(request.read_text(), original)
            protected = json.loads(original)
            protected["sampling"]["steps"] = 5
            protected["execution"] = {"policy": "gpu_ane", "qwen21_w8a8": True,
                                      "ane_manifest": "checked-at-execution.json"}
            request.write_text(json.dumps(protected))
            denied = subprocess.run([str(CLI), "plan", str(request),
                                     "--hybrid-mode", "base_fused"],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("allow_approximation", denied.stderr)

    def test_hybrid_mode_manifest_order_and_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "compiled.json"
            manifest.write_text("{}")
            request = root / "request.json"
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A fox", "steps": 8,
                "width": 512, "height": 512, "execution": "gpu",
                "output": str(root / "out.png")}))
            for tail in (("--hybrid-mode", "base_fused", "--ane-manifest", str(manifest)),
                         ("--ane-manifest", str(manifest), "--hybrid-mode", "base_fused")):
                result = subprocess.run([str(CLI), "plan", str(request), *tail],
                                        capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["hybrid_mlp_mode"], "base_fused")
            request.write_text(json.dumps({"model": "z-image-turbo",
                "operation": "image.generate", "prompt": "A fox", "steps": 8,
                "width": 512, "height": 512, "execution": "gpu",
                "hybrid_mlp_mode": "lora_suffix", "output": str(root / "out.png")}))
            conflict = subprocess.run([str(CLI), "plan", str(request),
                                       "--hybrid-mode", "base_fused"],
                                      capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("conflicts with request hybrid_mlp_mode", conflict.stderr)

    def test_base_request_can_explicitly_select_reusable_lora_fused_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "shared-base.json"
            manifest.write_text("{}")
            for model, extra in (
                ("z-image-turbo", {"steps": 8, "residency": "resident"}),
                ("qwen-image-2.1", {"steps": 6, "qwen21_w8a8": True}),
            ):
                with self.subTest(model=model):
                    request = root / f"{model}.json"
                    original = json.dumps({"model": model, "operation": "image.generate",
                        "prompt": "A fox", "output": str(root / "out.png"),
                        "width": 512, "height": 512, "execution": "gpu",
                        "allow_approximation": True, **extra})
                    request.write_text(original)
                    result = subprocess.run([str(CLI), "plan", str(request),
                                             "--ane-manifest", str(manifest),
                                             "--hybrid-mode", "lora_fused"],
                                            capture_output=True, text=True, cwd=ROOT)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["hybrid_mlp_mode"], "lora_fused")
                    self.assertEqual(request.read_text(), original)


if __name__ == "__main__":
    unittest.main()
