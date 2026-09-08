#!/usr/bin/env python3
"""Persistent PyTorch/MPS worker for LLaDA-Image-Turbo.

The worker keeps the official Diffusers pipeline resident.  TurboCider's
native session owns the process lifetime and communicates with it using one
JSON object per line, which avoids reloading roughly 46 GB of weights for
every request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--ane-manifest", type=Path)
    parser.add_argument("--native-library", type=Path)
    return parser.parse_args()


class CoreMLFFNBridge:
    """ctypes wrapper for one checkpoint-bound 32-block FFN artifact set."""

    def __init__(self, library_path: Path, manifest: Path, checkpoint: Path, rows: int):
        import ctypes

        self.ctypes = ctypes
        self.library = ctypes.CDLL(str(library_path.resolve()))
        pointer = ctypes.POINTER(ctypes.c_void_p)
        fp16_pointer = ctypes.POINTER(ctypes.c_uint16)
        self.library.tc_coreml_ffn_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, pointer, pointer
        ]
        self.library.tc_coreml_ffn_create.restype = ctypes.c_int
        self.library.tc_coreml_ffn_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_int, fp16_pointer, ctypes.c_int, fp16_pointer, pointer
        ]
        self.library.tc_coreml_ffn_predict.restype = ctypes.c_int
        self.library.tc_coreml_ffn_metrics_json.argtypes = [ctypes.c_void_p, pointer, pointer]
        self.library.tc_coreml_ffn_metrics_json.restype = ctypes.c_int
        self.library.tc_coreml_ffn_free.argtypes = [ctypes.c_void_p]
        self.library.tc_string_free.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        error = ctypes.c_void_p()
        status = self.library.tc_coreml_ffn_create(
            str(manifest.resolve()).encode(), str(checkpoint.resolve()).encode(),
            int(rows), 0, ctypes.byref(self.handle), ctypes.byref(error)
        )
        if status:
            raise RuntimeError(self._take(error) or f"Core ML FFN create failed ({status})")
        self.configuration = self.metrics()
        self.prefix_width = int(self.configuration["ane_mlp_range"][1])
        self.output_scale = float(self.configuration["output_scale"])
        self.validation = []

    def _take(self, pointer) -> str:
        if not pointer or not pointer.value:
            return ""
        try:
            return self.ctypes.cast(pointer, self.ctypes.c_char_p).value.decode()
        finally:
            self.library.tc_string_free(pointer)

    def stage_input(self, value):
        import numpy as np

        # Core ML's C ABI accepts host FP16 bit patterns.  The explicit CPU
        # round-trip is intentional for this first integration: it makes the
        # device boundary measurable and keeps the default MPS path untouched.
        return np.ascontiguousarray(
            value.detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    def predict_host(self, block: int, host, device, dtype):
        import numpy as np

        output = np.empty_like(host)
        error = self.ctypes.c_void_p()
        status = self.library.tc_coreml_ffn_predict(
            self.handle, int(block),
            host.view(np.uint16).ctypes.data_as(self.ctypes.POINTER(self.ctypes.c_uint16)),
            int(host.shape[1]),
            output.view(np.uint16).ctypes.data_as(self.ctypes.POINTER(self.ctypes.c_uint16)),
            self.ctypes.byref(error),
        )
        if status:
            raise RuntimeError(self._take(error) or f"Core ML FFN predict failed ({status})")
        return torch.from_numpy(output).to(device=device, dtype=dtype) * self.output_scale

    def metrics(self) -> dict:
        result = self.ctypes.c_void_p()
        error = self.ctypes.c_void_p()
        status = self.library.tc_coreml_ffn_metrics_json(
            self.handle, self.ctypes.byref(result), self.ctypes.byref(error)
        )
        if status:
            raise RuntimeError(self._take(error) or f"Core ML FFN metrics failed ({status})")
        try:
            return json.loads(self.ctypes.cast(result, self.ctypes.c_char_p).value.decode())
        finally:
            self.library.tc_string_free(result)

    def close(self) -> None:
        if self.handle.value:
            self.library.tc_coreml_ffn_free(self.handle)
            self.handle = self.ctypes.c_void_p()


class HybridFeedForward(torch.nn.Module):
    """Run an ANE/Core ML prefix and the complementary MPS suffix."""

    def __init__(self, original, bridge: CoreMLFFNBridge, block: int):
        super().__init__()
        self.original = original
        self.bridge = bridge
        self.block = block
        self.prefix_width = bridge.prefix_width
        self.validate = os.getenv("TURBOCIDER_LLADA_HYBRID_VALIDATE") == "1"
        self.validation_calls = 0

    def forward(self, hidden_states):
        import torch.nn.functional as functional

        # Materialize the common FFN input on the host before submitting the
        # suffix.  Once staged, the MPS suffix can run asynchronously while the
        # synchronous Core ML prefix is in flight.
        host = self.bridge.stage_input(hidden_states)
        start = self.prefix_width
        w1 = functional.linear(hidden_states, self.original.w1.weight[start:, :], None)
        w3 = functional.linear(hidden_states, self.original.w3.weight[start:, :], None)
        suffix = functional.linear(
            functional.silu(w1) * w3, self.original.w2.weight[:, start:], self.original.w2.bias
        )
        prefix = self.bridge.predict_host(
            self.block, host, hidden_states.device, hidden_states.dtype
        )
        if self.validate and self.validation_calls < 2:
            reference_w1 = functional.linear(
                hidden_states, self.original.w1.weight[:start, :], None
            )
            reference_w3 = functional.linear(
                hidden_states, self.original.w3.weight[:start, :], None
            )
            reference = functional.linear(
                functional.silu(reference_w1) * reference_w3,
                self.original.w2.weight[:, :start],
                None,
            )
            difference = (prefix - reference).float()
            left = prefix.float().reshape(-1)
            right = reference.float().reshape(-1)
            left_centered = left - left.mean()
            right_centered = right - right.mean()
            cosine = torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
            correlation = torch.dot(left_centered, right_centered) / (
                torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
            )
            self.bridge.validation.append({
                "block": self.block,
                "rows": int(hidden_states.shape[1]),
                "mae": float(difference.abs().mean().cpu()),
                "max_abs": float(difference.abs().max().cpu()),
                "cosine": float(cosine.cpu()),
                "correlation": float(correlation.cpu()),
                "prefix_max": float(prefix.float().abs().max().cpu()),
                "reference_max": float(reference.float().abs().max().cpu()),
            })
            self.validation_calls += 1
        return prefix + suffix


def attach_coreml_ffn(pipeline, bridge: CoreMLFFNBridge) -> list:
    """Patch only the 2 noise-refiner + 30 main FFNs covered by the artifact."""
    attached = []
    for block, layer in enumerate(pipeline.transformer.noise_refiner):
        layer.feed_forward = HybridFeedForward(layer.feed_forward, bridge, block)
        attached.append(f"noise_refiner.{block}")
    offset = len(pipeline.transformer.noise_refiner)
    for index, layer in enumerate(pipeline.transformer.layers):
        layer.feed_forward = HybridFeedForward(layer.feed_forward, bridge, offset + index)
        attached.append(f"layers.{index}")
    return attached


class ConditioningCache:
    """Resident prompt cache with an optional content-addressed disk handoff.

    The disk form is primarily an acceptance aid: paired GPU and hybrid
    processes can consume the exact same BF16 conditioning instead of comparing
    two independently scheduled MPS MoE executions.
    """

    def __init__(self, pipeline, model_root: Path):
        self.pipeline = pipeline
        self.model_root = model_root.resolve()
        self.memory = {}
        configured = os.getenv("TURBOCIDER_LLADA_CONDITIONING_CACHE_DIR")
        self.directory = Path(configured).resolve() if configured else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _identity(self, prompt: str, guidance_scale: float) -> str:
        value = json.dumps({
            "schema_version": 1,
            "model": str(self.model_root),
            "prompt": prompt,
            "guidance_scale": guidance_scale,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(value).hexdigest()

    def _to_device(self, values):
        prompt, mask, negative, negative_mask = values
        device = self.pipeline.transformer.device
        dtype = self.pipeline.transformer.dtype
        return (
            prompt.to(device=device, dtype=dtype),
            mask.to(device=device, dtype=torch.bool),
            None if negative is None else negative.to(device=device, dtype=dtype),
            None if negative_mask is None else negative_mask.to(device=device, dtype=torch.bool),
        )

    def get(self, prompt: str, guidance_scale: float):
        identity = self._identity(prompt, guidance_scale)
        if identity in self.memory:
            return self.memory[identity], True
        path = self.directory / f"{identity}.pt" if self.directory else None
        if path and path.is_file():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            values = self._to_device((
                stored["prompt_embeds"], stored["prompt_attention_mask"],
                stored.get("negative_prompt_embeds"), stored.get("negative_prompt_attention_mask"),
            ))
            self.memory[identity] = values
            return values, True
        values = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=guidance_scale > 1.0,
            num_images_per_prompt=1,
            max_sequence_length=2048,
            device=self.pipeline.transformer.device,
        )
        values = tuple(None if value is None else value.detach() for value in values)
        self.memory[identity] = values
        if path:
            stored = {
                "prompt_embeds": values[0].to("cpu"),
                "prompt_attention_mask": values[1].to("cpu"),
            }
            if values[2] is not None:
                stored["negative_prompt_embeds"] = values[2].to("cpu")
                stored["negative_prompt_attention_mask"] = values[3].to("cpu")
            temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
            torch.save(stored, temporary)
            os.replace(temporary, path)
        return values, False


class LayerProfiler:
    """Synchronization-heavy, opt-in MPS phase profiler.

    The hooks are intentionally disabled during normal benchmarks because a
    device synchronization around every block changes scheduling.  They are
    used to discover the real sequence buckets and the fraction of request
    time that can be moved to an ANE FFN branch.
    """

    def __init__(self, pipeline, torch) -> None:
        self.enabled = os.getenv("TURBOCIDER_LLADA_PROFILE") == "1"
        self.torch = torch
        self.handles = []
        self.started = {}
        self.seconds = {}
        self.calls = {}
        self.rows = {}
        if not self.enabled:
            return

        def register(module, name: str) -> None:
            def before(_module, inputs):
                torch.mps.synchronize()
                self.started[name] = time.perf_counter()
                if inputs and hasattr(inputs[0], "shape") and len(inputs[0].shape) >= 2:
                    self.rows.setdefault(name, set()).add(int(inputs[0].shape[1]))

            def after(_module, _inputs, _output):
                torch.mps.synchronize()
                elapsed = time.perf_counter() - self.started.pop(name)
                self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
                self.calls[name] = self.calls.get(name, 0) + 1

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))

        for name in ("text_encoder", "queryformer", "text_projection", "sigvq", "transformer", "vae"):
            module = getattr(pipeline, name, None)
            if module is not None:
                register(module, name)
        for index, layer in enumerate(pipeline.transformer.noise_refiner):
            register(layer, f"noise_refiner.{index}")
        for index, layer in enumerate(pipeline.transformer.context_refiner):
            register(layer, f"context_refiner.{index}")
        for index, layer in enumerate(pipeline.transformer.sigvq_refiner):
            register(layer, f"sigvq_refiner.{index}")
        for index, layer in enumerate(pipeline.transformer.layers):
            register(layer, f"layers.{index}")

    def start_request(self) -> None:
        if not self.enabled:
            return
        self.started.clear()
        self.seconds.clear()
        self.calls.clear()
        self.rows.clear()

    def report(self) -> dict | None:
        if not self.enabled:
            return None
        blocks = [name for name in self.seconds if name.startswith(("noise_refiner.", "context_refiner.",
                                                                     "sigvq_refiner.", "layers."))]
        return {
            "method": "MPS synchronized forward hooks; profiling only",
            "components_seconds": {name: self.seconds[name] for name in sorted(self.seconds)
                                   if name not in blocks},
            "block_seconds_total": sum(self.seconds[name] for name in blocks),
            "block_calls_total": sum(self.calls[name] for name in blocks),
            "block_rows": sorted({row for name in blocks for row in self.rows.get(name, set())}),
            "block_groups_seconds": {
                "noise_refiner": sum(value for name, value in self.seconds.items()
                                     if name.startswith("noise_refiner.")),
                "context_refiner": sum(value for name, value in self.seconds.items()
                                       if name.startswith("context_refiner.")),
                "sigvq_refiner": sum(value for name, value in self.seconds.items()
                                     if name.startswith("sigvq_refiner.")),
                "main": sum(value for name, value in self.seconds.items() if name.startswith("layers.")),
            },
        }


def main() -> int:
    args = arguments()
    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch
    from PIL import Image
    from src import LLaDAImagePipeline

    load_start = time.perf_counter()
    pipeline = LLaDAImagePipeline.from_pretrained(
        args.model_root.resolve(), torch_dtype=torch.bfloat16, device=args.device
    )
    hybrid_bridge = None
    hybrid_blocks = []
    if args.ane_manifest:
        if not args.native_library:
            raise ValueError("--native-library is required with --ane-manifest")
        manifest = json.loads(args.ane_manifest.read_text())
        shape = manifest.get("shape", {})
        buckets = shape.get("buckets", [])
        if not isinstance(buckets, list) or len(buckets) != 1:
            raise ValueError("LLaDA Core ML manifest must contain one fixed row bucket")
        if int(shape.get("K", 0)) != 3840 or int(shape.get("N", 0)) != 3840:
            raise ValueError("LLaDA Core ML manifest hidden dimension must be 3840")
        if int(shape.get("mlp_width", 0)) != 10240 or int(shape.get("ane_mlp_end", 0)) != 4096:
            raise ValueError("LLaDA Core ML manifest must use the validated [0,4096) prefix")
        checkpoint = args.model_root / "transformer/diffusion_pytorch_model.safetensors.index.json"
        hybrid_bridge = CoreMLFFNBridge(
            args.native_library, args.ane_manifest, checkpoint, int(buckets[0])
        )
        hybrid_blocks = attach_coreml_ffn(pipeline, hybrid_bridge)
    conditioning_cache = ConditioningCache(pipeline, args.model_root)
    profiler = LayerProfiler(pipeline, torch)
    load_seconds = time.perf_counter() - load_start
    emit({
        "type": "ready",
        "model": "llada-image-turbo",
        "device": str(pipeline.transformer.device),
        "dtype": "bfloat16",
        "execution": "gpu_ane" if hybrid_bridge else "gpu",
        "hybrid_blocks": hybrid_blocks,
        "hybrid_manifest": str(args.ane_manifest.resolve()) if args.ane_manifest else None,
        "load_seconds": load_seconds,
    })
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if request.get("action") != "generate":
                    raise ValueError("unsupported worker action")
                mode = request.get("generation_mode", "text")
                if mode not in {"text", "editing"}:
                    raise ValueError("generation_mode must be text or editing")
                if mode == "editing" and not request.get("input_image"):
                    raise ValueError("editing requires input_image")

                output = Path(request["output"]).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                seed = int(request.get("seed", 42))
                latent_shape = (
                    1,
                    pipeline.transformer.config.in_channels,
                    int(request["height"]) // pipeline.latent_scale_factor,
                    int(request["width"]) // pipeline.latent_scale_factor,
                )
                cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
                latents = torch.randn(latent_shape, generator=cpu_generator, dtype=torch.float32)
                latents = latents.to(pipeline.transformer.device).to(pipeline.transformer.dtype).float()
                started = time.perf_counter()
                profiler.start_request()
                guidance_scale = float(request.get("guidance_scale", 1.0))

                def progress(_pipe, step, _timestep, _kwargs):
                    emit({
                        "type": "progress",
                        "phase": "denoise",
                        "completed": int(step) + 1,
                        "total": int(request.get("steps", 4)),
                    })
                    return {}

                # The official pipeline is inference-only.  Disabling autograd is
                # safe for both the reference path and the persistent worker and
                # avoids retaining backward metadata across four denoise steps.
                with torch.inference_mode():
                    conditioning, prompt_cache_hit = conditioning_cache.get(
                        request["prompt"], guidance_scale
                    )
                    kwargs = {
                        "prompt": None,
                        "prompt_embeds": conditioning[0],
                        "prompt_attention_mask": conditioning[1],
                        "negative_prompt_embeds": conditioning[2],
                        "negative_prompt_attention_mask": conditioning[3],
                        "generation_mode": mode,
                        "height": int(request["height"]),
                        "width": int(request["width"]),
                        "num_inference_steps": int(request.get("steps", 4)),
                        "guidance_scale": guidance_scale,
                        "latents": latents,
                    }
                    if mode == "editing":
                        kwargs["image"] = Image.open(request["input_image"]).convert("RGB")
                    result = pipeline(callback_on_step_end=progress, **kwargs)
                result.images[0].save(output)
                seconds = time.perf_counter() - started
                worker_result = {
                    "type": "result",
                    "result": {
                        "schema_version": 1,
                        "model": "llada-image-turbo",
                        "output": str(output),
                        "width": int(request["width"]),
                        "height": int(request["height"]),
                        "seed": seed,
                        "steps": int(request.get("steps", 4)),
                        "generation_mode": mode,
                        "runtime_backend": "torch_mps_coreml_ffn_hybrid" if hybrid_bridge else "torch_mps_persistent",
                        "runtime_precision": "bf16",
                        "persistent_model": True,
                        "prompt_cache_hit": prompt_cache_hit,
                        "initial_noise": "cpu-seeded-fp32-bf16-rounded",
                        "timings_seconds": {
                            "request_wall": seconds,
                            "model_load": load_seconds,
                        },
                    },
                }
                profile = profiler.report()
                if profile is not None:
                    worker_result["result"]["profile"] = profile
                if hybrid_bridge:
                    worker_result["result"]["hybrid"] = hybrid_bridge.metrics()
                    if hybrid_bridge.validation:
                        worker_result["result"]["hybrid_validation"] = hybrid_bridge.validation
                emit(worker_result)
            except Exception as error:  # noqa: BLE001 - worker must report and stay alive
                emit({"type": "error", "error": f"{type(error).__name__}: {error}"})
    finally:
        if hybrid_bridge:
            hybrid_bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
