#!/usr/bin/env python3
"""Legacy Python oracle for Wan 1.3B development comparisons; never packaged.

The protocol is newline-delimited JSON on stdin/stdout. Human-readable output
from imported FastVideo modules may also appear on stdout; the native parent
ignores non-protocol lines and keeps their tail for diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def emit(kind: str, **values: Any) -> None:
    payload = {"type": kind, **values}
    sys.__stdout__.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()


def process_peak_rss_bytes() -> int:
    """Return this worker's peak resident set size in bytes when available.

    macOS reports ``ru_maxrss`` in bytes while Linux reports KiB.  FastMetal
    is an Apple Silicon runtime, but handling both units keeps the diagnostic
    metric useful in CPU-only CI and local protocol tests.
    """
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return 0
    if sys.platform == "darwin":
        return value
    return value * 1024


def process_rss_bytes() -> int | None:
    """Return current RSS without making psutil a required dependency."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError, ValueError):
        return None


def load_entrypoint(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_turbocider_fastmetal_entrypoint", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import FastMetal entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastMetalWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_root = args.model_root.resolve()
        self.mlx_checkpoint = args.mlx_checkpoint.resolve()
        if not (self.mlx_checkpoint / "mlx_dit.json").is_file() or not (
            self.mlx_checkpoint / "mlx_dit.safetensors"
        ).is_file():
            raise ValueError(
                "FastMetal MLX checkpoint must contain mlx_dit.json and "
                f"mlx_dit.safetensors: {self.mlx_checkpoint}"
            )
        self.entrypoint = load_entrypoint(args.entrypoint.resolve())

        import mlx.core as mx
        import torch
        from fastvideo.mlx_runtime.checkpoint import load_mlx_dit_checkpoint
        from fastvideo.mlx_runtime.sampling import MLXDMDSchedule

        self.mx = mx
        self.torch = torch
        self.mx_dtype = {
            "fp16": mx.float16,
            "bf16": mx.bfloat16,
            "fp32": mx.float32,
        }[args.mlx_dtype]
        worker_started = time.perf_counter()
        mx.clear_cache()
        mx.reset_peak_memory()
        dit_started = time.perf_counter()
        self.dit = load_mlx_dit_checkpoint(
            self.mlx_checkpoint,
            compile=args.mlx_compile and args.ane_manifest is None,
        )
        self.dit_load_s = time.perf_counter() - dit_started
        self.compile_requested = bool(args.mlx_compile and args.ane_manifest is None)
        self.compile_enabled = bool(getattr(self.dit, "_enable_compile", False))
        self.config = self.dit.config
        self.patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))
        self.ane_session = None
        # Defer Core ML activation until after prompt encoding.  Keeping 30 ANE
        # bindings alive while UMT5 or TAEHV owns unified memory caused large
        # RSS spikes and a 4+s decode regression; they are cheap enough to load
        # at the denoise boundary and release before decode.
        self.ane_load_s = 0.0
        # FastMetal QAD uses the fixed FlowMatch shift=8 training schedule.
        # Construct precisely the float32 table used by the upstream scheduler
        # without importing its optional SciPy-only schedule variants.
        training_timesteps = np.linspace(
            1, 1000, 1000, dtype=np.float32
        )[::-1].copy()
        sigmas = training_timesteps / np.float32(1000.0)
        sigmas = np.float32(8.0) * sigmas / (
            np.float32(1.0) + np.float32(7.0) * sigmas
        )
        self.schedule = MLXDMDSchedule(
            sigmas=sigmas.astype(np.float64),
            timesteps=(sigmas * np.float32(1000.0)).astype(np.float64),
        )
        self.timesteps = (1000.0, 757.0, 522.0)
        self.model_load_s = time.perf_counter() - worker_started
        self.model_load_peak_bytes = int(mx.get_peak_memory())
        self.generation_count = 0
        self.peak_rss_bytes = process_peak_rss_bytes()
        self.rotary_cache: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self.taehv_decoder = None
        self.taehv_checkpoint: Path | None = None
        self.decoder_load_s = 0.0
        if args.decode_backend == "taehv":
            from fastvideo.mlx_runtime.wan_vae import (
                MLXTAEHVDecoder,
                ensure_taehv_checkpoint,
            )

            decoder_started = time.perf_counter()
            self.taehv_checkpoint = ensure_taehv_checkpoint(
                z_dim=int(self.config["in_channels"]),
                checkpoint_path=args.taehv_checkpoint,
            )
            self.taehv_decoder = MLXTAEHVDecoder(
                self.taehv_checkpoint,
                z_dim=int(self.config["in_channels"]),
            )
            self.decoder_load_s = time.perf_counter() - decoder_started
        self.worker_init_s = time.perf_counter() - worker_started

    def activate_ane(self) -> float:
        if self.args.ane_manifest is None or self.ane_session is not None:
            return 0.0
        from fastvideo.mlx_runtime.fastmetal_ane import activate_fastmetal_ane

        started = time.perf_counter()
        self.ane_session = activate_fastmetal_ane(
            self.dit,
            manifest_path=self.args.ane_manifest,
            bridge_dir=self.args.ane_bridge_dir,
        )
        return time.perf_counter() - started

    def prompt_embeds(self, prompt: str) -> tuple[Any, bool, float]:
        from fastvideo.mlx_runtime.prompt_cache import load_prompt_cache

        cache_path = None
        fingerprint = None
        if self.args.prompt_cache:
            fingerprint = self.entrypoint._prompt_cache_fingerprint(
                model_root=self.model_root,
                prompt=prompt,
                max_sequence_length=512,
                dtype_arg=self.args.text_encoder_dtype,
            )
            cache_path = self.entrypoint._default_prompt_cache_path(
                model_root=self.model_root,
                prompt=prompt,
                max_sequence_length=512,
                dtype_arg=self.args.text_encoder_dtype,
            )
            cached = load_prompt_cache(cache_path, fingerprint)
            if cached is not None:
                return self.torch.from_numpy(cached).contiguous(), True, 0.0

        started = time.perf_counter()
        embeds = self.entrypoint.get_prompt_embeds(
            model_root=self.model_root,
            prompt=prompt,
            max_sequence_length=512,
            device_arg="auto",
            dtype_arg=self.args.text_encoder_dtype,
            encode_mode="inline",
            cache_path=cache_path,
        )
        return embeds, False, time.perf_counter() - started

    def rotary_embeddings(
        self, *, latent_frames: int, latent_height: int, latent_width: int
    ) -> tuple[Any, Any]:
        key = (latent_frames, latent_height, latent_width)
        cached = self.rotary_cache.get(key)
        if cached is not None:
            return cached
        num_heads = int(self.config["num_attention_heads"])
        head_dim = int(self.config["attention_head_dim"])
        patch_t, patch_h, patch_w = self.patch_size
        sizes = (
            latent_frames // patch_t,
            latent_height // patch_h,
            latent_width // patch_w,
        )
        sixth = head_dim // 6
        dimensions = (head_dim - 4 * sixth, 2 * sixth, 2 * sixth)
        axes = [
            self.torch.linspace(0, size, size + 1, dtype=self.torch.float32)[:size]
            for size in sizes
        ]
        grid = self.torch.stack(
            self.torch.meshgrid(*axes, indexing="ij"), dim=0
        )
        cosines = []
        sines = []
        for axis, dimension in enumerate(dimensions):
            positions = grid[axis].reshape(-1)
            frequencies = 1.0 / (
                10000.0
                ** (
                    self.torch.arange(0, dimension, 2, dtype=self.torch.float32)
                    / dimension
                )
            )
            angles = self.torch.outer(positions, frequencies)
            cosines.append(angles.cos().repeat_interleave(2, dim=-1))
            sines.append(angles.sin().repeat_interleave(2, dim=-1))
        result = (
            self.mx.array(self.torch.cat(cosines, dim=1).numpy()).astype(
                self.mx.float32
            ),
            self.mx.array(self.torch.cat(sines, dim=1).numpy()).astype(
                self.mx.float32
            ),
        )
        self.rotary_cache[key] = result
        return result

    def decode_taehv(self, latents: np.ndarray, output: Path, fps: int) -> dict[str, float]:
        from fastvideo.mlx_runtime.memory import cleanup_mlx

        assert self.taehv_decoder is not None
        started = time.perf_counter()
        prepare_started = started
        value = self.mx.array(
            latents.transpose(0, 2, 1, 3, 4).astype(np.float32)
        )
        prepare_s = time.perf_counter() - prepare_started
        decode_started = time.perf_counter()
        pixels = self.taehv_decoder.decode_ntchw(value)
        self.mx.eval(pixels)
        decode_s = time.perf_counter() - decode_started
        numpy_started = time.perf_counter()
        frames = np.array(pixels).transpose(0, 1, 3, 4, 2)[0]
        frames = np.clip(frames, 0.0, 1.0)
        numpy_s = time.perf_counter() - numpy_started
        export_started = time.perf_counter()
        self.export_mp4(frames, output, fps)
        export_s = time.perf_counter() - export_started
        del value, pixels, frames
        cleanup_mlx(self.mx)
        return {
            "decode_s": time.perf_counter() - started,
            "decode_prepare_s": prepare_s,
            "decode_mlx_s": decode_s,
            "decode_numpy_s": numpy_s,
            "decode_export_s": export_s,
        }

    @staticmethod
    def export_mp4(frames: np.ndarray, output: Path, fps: int) -> None:
        """Export RGB float frames without importing Diffusers.

        The upstream helper only uses ``export_to_video`` for this final step;
        OpenCV's native AVFoundation/FFmpeg writer is sufficient for the
        TurboCider contract and keeps the persistent worker's dependency set
        limited to MLX, Torch and Transformers.
        """
        import cv2

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected [T,H,W,3] RGB frames, got {frames.shape}")
        output.parent.mkdir(parents=True, exist_ok=True)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"avc1"), float(fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer for {output}")
        try:
            for frame in frames:
                bgr = np.asarray(np.clip(frame, 0.0, 1.0) * 255.0, dtype=np.uint8)
                writer.write(cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def decode_wan_vae(self, latents: np.ndarray, output: Path, fps: int) -> float:
        started = time.perf_counter()
        self.entrypoint.decode_latents_to_video(
            model_root=self.model_root,
            latents_np=latents,
            output_path=output,
            fps=fps,
            device_arg="auto",
            dtype_arg=self.args.vae_decode_dtype,
            backend="wan-vae",
            taehv_source_path=None,
            taehv_checkpoint_path=None,
            taehv_parallel=False,
        )
        return time.perf_counter() - started

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request["prompt"]
        width = int(request["width"])
        height = int(request["height"])
        frames = int(request["frames"])
        fps = int(request["fps"])
        seed = int(request["seed"])
        output = Path(request["output"])
        dump = request.get("dump")
        if not prompt:
            raise ValueError("FastMetal prompt is required")
        if width % 16 or height % 16 or frames % 4 != 1:
            raise ValueError("FastMetal shape is not VAE/patch aligned")

        latent_frames = (frames - 1) // 4 + 1
        latent_height = height // 8
        latent_width = width // 8
        pt, ph, pw = self.patch_size
        if latent_frames % pt or latent_height % ph or latent_width % pw:
            raise ValueError("FastMetal latent grid is not patch aligned")
        rows = latent_frames * (latent_height // ph) * (latent_width // pw)
        total_started = time.perf_counter()
        prompt_embeds, prompt_cache_hit, prompt_s = self.prompt_embeds(prompt)
        emit("progress", phase="text_encode", completed=1, total=1)

        ane_reactivation_s = self.activate_ane()
        if self.generation_count == 0:
            self.ane_load_s = ane_reactivation_s
        if self.ane_session is not None and rows != self.ane_session.manifest.rows:
            raise ValueError(
                "FastMetal ANE manifest rows do not match the request: "
                f"manifest={self.ane_session.manifest.rows}, request={rows}"
            )

        self.mx.random.seed(seed)
        self.torch.manual_seed(seed)
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        noise = self.torch.randn(
            (
                1,
                int(self.config["in_channels"]),
                latent_frames,
                latent_height,
                latent_width,
            ),
            generator=generator,
            dtype=self.torch.float32,
        )
        latents = self.mx.array(noise.numpy()).astype(self.mx_dtype)
        conditioning = self.mx.array(prompt_embeds.numpy()).astype(self.mx_dtype)
        frequencies = self.rotary_embeddings(
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        del noise, prompt_embeds

        ane_offsets = (
            self.ane_session.call_offsets() if self.ane_session is not None else None
        )
        denoise_started = time.perf_counter()
        denoise_steps: list[float] = []
        self.mx.reset_peak_memory()
        for index, timestep in enumerate(self.timesteps):
            step_started = time.perf_counter()
            original = latents
            timestep_array = self.mx.array([timestep]).astype(self.mx.float32)
            prediction = self.dit(
                latents.astype(self.mx_dtype),
                conditioning,
                timestep_array,
                frequencies,
            )
            original_f32 = original.astype(self.mx.float32)
            prediction_f32 = prediction.astype(self.mx.float32)
            if index + 1 < len(self.timesteps):
                next_timestep: float | None = self.timesteps[index + 1]
                renoise = self.mx.random.normal(original_f32.shape).astype(
                    self.mx.float32
                )
            else:
                next_timestep = None
                renoise = None
            from fastvideo.mlx_runtime.sampling import dmd_step

            latents = dmd_step(
                latents=original_f32,
                noise_input_latent=original_f32,
                pred_noise=prediction_f32,
                schedule=self.schedule,
                timestep=timestep,
                next_timestep=next_timestep,
                noise=renoise,
            ).astype(self.mx_dtype)
            self.mx.eval(latents)
            emit(
                "progress",
                phase="dit",
                completed=index + 1,
                total=len(self.timesteps),
                elapsed_s=time.perf_counter() - step_started,
            )
            denoise_steps.append(time.perf_counter() - step_started)
            del original, timestep_array, prediction, original_f32, prediction_f32
            if renoise is not None:
                del renoise

        denoise_s = time.perf_counter() - denoise_started
        denoise_peak = int(self.mx.get_peak_memory())
        active_after_denoise = int(self.mx.get_active_memory())
        compile_enabled = bool(
            getattr(self.dit, "_enable_compile", self.compile_enabled)
        )
        compiled_graph_cached = getattr(self.dit, "_compiled_forward", None) is not None
        latents_np = np.array(latents.astype(self.mx.float32))
        if dump:
            dump_path = Path(dump)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dump_path, latents_np)
        del latents, conditioning, frequencies

        ane_metrics = (
            self.ane_session.metrics(calls_after=ane_offsets)
            if self.ane_session is not None
            else None
        )
        ane_release_started = time.perf_counter()
        ane_released = self.ane_session is not None
        if self.ane_session is not None:
            self.ane_session.close(self.dit)
            self.ane_session = None
            self.mx.clear_cache()
        ane_release_s = time.perf_counter() - ane_release_started

        emit("progress", phase="video_vae", completed=0, total=1)
        if self.args.decode_backend == "taehv":
            decode_metrics = self.decode_taehv(latents_np, output, fps)
        else:
            decode_s = self.decode_wan_vae(latents_np, output, fps)
            decode_metrics = {
                "decode_s": decode_s,
                "decode_prepare_s": 0.0,
                "decode_mlx_s": decode_s,
                "decode_numpy_s": 0.0,
                "decode_export_s": 0.0,
            }
        emit("progress", phase="video_vae", completed=1, total=1)
        self.generation_count += 1
        self.peak_rss_bytes = max(self.peak_rss_bytes, process_peak_rss_bytes())
        result = {
            "schema_version": 1,
            "model": "wan2.1-1.3b-qad",
            "output": str(output),
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "seed": seed,
            "prompt_encode_s": prompt_s,
            "prompt_cache_hit": prompt_cache_hit,
            "mlx_dit_load_s": self.model_load_s if self.generation_count == 1 else 0.0,
            "mlx_checkpoint_load_s": self.dit_load_s if self.generation_count == 1 else 0.0,
            "ane_activation_s": ane_reactivation_s,
            "ane_reactivation_s": ane_reactivation_s,
            "ane_release_s": ane_release_s,
            "ane_released_before_decode": ane_released,
            "decoder_load_s": self.decoder_load_s if self.generation_count == 1 else 0.0,
            "worker_init_s": self.worker_init_s if self.generation_count == 1 else 0.0,
            "model_cache_hit": self.generation_count > 1,
            "mlx_denoise_s": denoise_s,
            "decode_export_s": decode_metrics["decode_s"],
            "decode_prepare_s": decode_metrics["decode_prepare_s"],
            "decode_mlx_s": decode_metrics["decode_mlx_s"],
            "decode_numpy_s": decode_metrics["decode_numpy_s"],
            "decode_encode_s": decode_metrics["decode_export_s"],
            "denoise_step_s": denoise_steps,
            "denoise_step_count": len(denoise_steps),
            "compile_requested": self.compile_requested,
            "compile_enabled": compile_enabled,
            "compiled_graph_cached": compiled_graph_cached,
            "mlx_compile_requested": self.compile_requested,
            "mlx_compile": compile_enabled,
            "worker_pid": os.getpid(),
            "worker_rss_bytes": process_rss_bytes(),
            "worker_peak_rss_bytes": self.peak_rss_bytes,
            "total_s": time.perf_counter() - total_started,
            "mlx_load_peak_bytes": self.model_load_peak_bytes,
            "mlx_denoise_peak_bytes": denoise_peak,
            "mlx_active_after_denoise_bytes": active_after_denoise,
            "persistent_model": True,
            "worker_generation": self.generation_count,
            "decode_backend": self.args.decode_backend,
            "fastmetal_ane": ane_metrics,
            "latents_path": str(dump) if dump else None,
        }
        del latents_np
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--mlx-checkpoint", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True, type=Path)
    parser.add_argument("--ane-manifest", type=Path)
    parser.add_argument("--ane-bridge-dir", type=Path)
    parser.add_argument("--decode-backend", choices=("taehv", "wan-vae"), default="taehv")
    parser.add_argument("--text-encoder-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--vae-decode-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--mlx-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--mlx-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--taehv-parallel", action="store_true")
    parser.add_argument("--taehv-checkpoint", type=Path)
    return parser.parse_args()


def dispatch_protocol_request(worker: Any, request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one decoded JSONL request and return a result when applicable.

    Keeping this small dispatcher separate from stdin/stdout makes the worker
    protocol testable without importing MLX or loading a multi-gigabyte model.
    Progress is still emitted by ``generate`` through the existing callback
    path in production; the persistent native parent remains the owner of
    cancellation and process lifetime.
    """
    action = request.get("action")
    if action == "shutdown":
        return {"type": "stopped"}
    if action == "ping":
        return {"type": "pong"}
    if action != "generate":
        raise ValueError(f"unknown action: {action!r}")
    return {"type": "result", "result": worker.generate(request["request"])}


def main() -> None:
    args = parse_args()
    if args.ane_manifest is not None and args.ane_bridge_dir is None:
        raise SystemExit("--ane-manifest requires --ane-bridge-dir")
    try:
        worker = FastMetalWorker(args)
    except BaseException as error:
        emit(
            "fatal",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise SystemExit(1) from error
    emit(
        "ready",
        model="wan2.1-1.3b-qad",
        mlx_checkpoint=str(worker.mlx_checkpoint),
        model_load_s=worker.model_load_s,
        mlx_checkpoint_load_s=worker.dit_load_s,
        ane_activation_s=worker.ane_load_s,
        decoder_load_s=worker.decoder_load_s,
        worker_init_s=worker.worker_init_s,
        ane=args.ane_manifest is not None,
        ane_activation_deferred=args.ane_manifest is not None,
        compile_requested=worker.compile_requested,
        compile_enabled=worker.compile_enabled,
        worker_pid=os.getpid(),
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = dispatch_protocol_request(worker, request)
            if response is None:
                continue
            response_type = response.pop("type")
            emit(response_type, **response)
            if response_type == "stopped":
                return
        except BaseException as error:
            emit(
                "error",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )


if __name__ == "__main__":
    main()
