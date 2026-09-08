"""Benchmark ultra-fast-image-gen's full-precision Z-Image path on MPS.

The harness keeps one Diffusers pipeline resident, uses the same Comfy BF16
transformer accepted by TurboCider, and records synchronized stage timings.
It intentionally mirrors references/ultra-fast-image-gen/loaders.py rather
than changing the reference repository.
"""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
import types
from pathlib import Path

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel


TRANSFORMER_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2560,
    "siglip_feat_dim": None,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1024, 512, 512],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--vae-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--e2e-only",
        action="store_true",
        help="disable per-stage MPS synchronization for pure wall-clock E2E runs",
    )
    return parser.parse_args()


def sync():
    torch.mps.synchronize()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(runs, key):
    values = [run[key] for run in runs]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")

    model_root = Path(args.model_root).resolve()
    transformer_path = Path(args.transformer).resolve()
    vae_path = Path(args.vae_file).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="z-image-diffusers-config-") as config_dir:
        transformer_dir = Path(config_dir) / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG))
        transformer = ZImageTransformer2DModel.from_single_file(
            str(transformer_path),
            config=config_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        # The local VAE is in the original/Comfy key layout, so it must also
        # go through Diffusers' single-file conversion (loading it with
        # from_pretrained would silently leave most layers randomly initialized).
        vae = AutoencoderKL.from_single_file(
            str(vae_path),
            config=str(model_root),
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    pipe = ZImagePipeline.from_pretrained(
        str(model_root),
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )
    pipe.to("mps")
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    sync()
    load_seconds = time.perf_counter() - load_start

    active = None
    original_encode_prompt = pipe.encode_prompt
    original_transformer_forward = pipe.transformer.forward
    original_vae_decode = pipe.vae.decode

    def timed_encode_prompt(*call_args, **call_kwargs):
        sync()
        start = time.perf_counter()
        result = original_encode_prompt(*call_args, **call_kwargs)
        sync()
        active["text_encode_seconds"] += time.perf_counter() - start
        return result

    def timed_transformer_forward(_self, *call_args, **call_kwargs):
        sync()
        start = time.perf_counter()
        result = original_transformer_forward(*call_args, **call_kwargs)
        sync()
        active["denoise_seconds"] += time.perf_counter() - start
        active["transformer_forwards"] += 1
        return result

    def timed_vae_decode(_self, *call_args, **call_kwargs):
        sync()
        start = time.perf_counter()
        result = original_vae_decode(*call_args, **call_kwargs)
        sync()
        active["vae_decode_seconds"] += time.perf_counter() - start
        return result

    if not args.e2e_only:
        pipe.encode_prompt = timed_encode_prompt
        pipe.transformer.forward = types.MethodType(timed_transformer_forward, pipe.transformer)
        pipe.vae.decode = types.MethodType(timed_vae_decode, pipe.vae)

    runs = []
    for run_index in range(args.runs):
        active = {
            "text_encode_seconds": 0.0,
            "denoise_seconds": 0.0,
            "vae_decode_seconds": 0.0,
            "transformer_forwards": 0,
        }
        generator = torch.Generator("mps").manual_seed(args.seed)
        sync()
        request_start = time.perf_counter()
        result = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        )
        sync()
        pipeline_seconds = time.perf_counter() - request_start

        image_path = output / f"ultra-fast-image-gen-{run_index}.png"
        export_start = time.perf_counter()
        result.images[0].save(image_path)
        export_seconds = time.perf_counter() - export_start
        external_e2e_seconds = pipeline_seconds + export_seconds
        stage_sum = (
            active["text_encode_seconds"]
            + active["denoise_seconds"]
            + active["vae_decode_seconds"]
        )
        runs.append(
            {
                "run": run_index,
                "cold": run_index == 0,
                "external_e2e_seconds": external_e2e_seconds,
                "pipeline_seconds": pipeline_seconds,
                **active,
                "pipeline_other_seconds": None if args.e2e_only else pipeline_seconds - stage_sum,
                "export_seconds": export_seconds,
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                "output": str(image_path),
                "output_sha256": sha256(image_path),
            }
        )
        report = {
            "schema_version": 1,
            "engine": "ultra-fast-image-gen full precision (Diffusers/PyTorch MPS)",
            "reference_commit": "72ccd5f735ed02fec0e7d12b517cf04b9d219cda",
            "method": "resident pipeline; synchronized stage timers; same prompt and seed repeated",
            "measurement_mode": "e2e_only" if args.e2e_only else "synchronized_stage_profile",
            "hardware": platform.platform(),
            "torch_version": torch.__version__,
            "model_root": str(model_root),
            "transformer": str(transformer_path),
            "vae": str(vae_path),
            "precision": "bf16",
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "max_sequence_length": args.max_sequence_length,
            "load_seconds": load_seconds,
            "settings": {
                "device": "mps",
                "attention_slicing": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "use_beta_sigmas": True,
            },
            "runs": runs,
        }
        if len(runs) > 1:
            warm = runs[1:]
            summary_keys = ["external_e2e_seconds", "pipeline_seconds", "export_seconds"]
            if not args.e2e_only:
                summary_keys.extend(
                    [
                        "text_encode_seconds",
                        "denoise_seconds",
                        "vae_decode_seconds",
                        "pipeline_other_seconds",
                    ]
                )
            report["warm_summary"] = {key: summarize(warm, key) for key in summary_keys}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(runs[-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
