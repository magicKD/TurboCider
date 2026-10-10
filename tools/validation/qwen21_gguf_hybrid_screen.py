#!/usr/bin/env python3
"""Serial matched mixed-K GPU/Private FFN screens; no performance promotion."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "A red fox sitting in a snowy forest, natural winter light, detailed fur.",
    "A red fox sitting in a snowy forest, soft winter light, detailed fur.",
    "A red fox sitting in a snowy forest, warm winter light, detailed fur.",
)


def counter(record, name):
    value = record.get(name)
    if type(value) is not int or value < 0:
        raise ValueError("missing or invalid actual counter: " + name)
    return value


def validate_receipts(rows, channels, steps=40, phase="all"):
    if len(rows) != 3 or phase not in ("all", "prefill", "decode"):
        raise ValueError("three original fresh-condition GPU/base receipts required")
    expected_blocks = (steps if phase == "all" else 1 if phase == "prefill" else steps - 1) * 32
    for index, row in enumerate(rows):
        if (row.get("model") != "qwen-image-2.1" or row.get("operation") != "image.generate" or
            row.get("width") != 512 or row.get("height") != 512 or row.get("seed") != 29 or
            row.get("steps") != steps or row.get("actual_denoise_steps") != steps or
            row.get("reference_tokens") != 0 or row.get("prompt_cache_hit") is not False or
            row.get("encoder_execution") != "gpu" or row.get("encoder_hybrid") or
            row.get("encoder_runtime_precision") != "q4_k_m_affine_fp16_io"):
            raise ValueError("wrong model/source/geometry/cache/encoder execution")
        source = row.get("encoder_weight_residency") or {}
        if (source.get("enabled") is not True or source.get("weights_retained") is not True or
            source.get("weights_reused") is not (index > 0) or counter(source,"loads_session_total") != 1):
            raise ValueError("same real retained original encoder lifecycle required")
        for field in ("request_wall", "text_encode", "denoise"):
            value = (row.get("timings_seconds") or {}).get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("missing finite positive native request timing")
        phases = row.get("qwen_ffn_phases") or {}
        if phases.get("policy") != (phase if channels else "gpu"):
            raise ValueError("actual phase receipt mismatched")
        for name, count in (("prefill", 1), ("decode", steps - 1)):
            p = phases.get(name) or {}
            blocks = 32 * count if channels and phase in (name, "all") else 0
            if (counter(p,"steps_this_request") != count or counter(p,"completed_channel_blocks_this_request") != blocks or
                counter(p,"runtime_calls_this_request") != blocks):
                raise ValueError("actual single-bucket phase work differs")
        hybrid = row.get("hybrid")
        if not channels:
            if hybrid or row.get("runtime_backend") != "mlx_cpp_metal_qwen21_gguf":
                raise ValueError("complete packed GPU control changed")
            continue
        w = (hybrid or {}).get("runtime_weight") or {}
        if (row.get("runtime_backend") != "mlx_cpp_metal_gguf+private_ane_runtime_weight_experimental" or
            hybrid.get("runtime_failed") is not False or counter(hybrid,"runtime_failures_session_total") != 0 or
            counter(hybrid,"runtime_calls_session_total") != expected_blocks * (index + 1) or
            w.get("executor_backend") != "private_ane" or w.get("data_path") != "w8a8_hadamard" or
            w.get("partition_axis") != "intermediate_channels" or counter(w,"ane_channels") != channels or
            counter(w,"channel_blocks_session_total") != expected_blocks * (index + 1) or
            counter(w,"async_hybrid_blocks_session_total") != expected_blocks * (index + 1) or
            counter(w,"fallback_blocks_session_total") != 0 or counter(w,"overflow_retries_session_total") != 0 or
            w.get("fp32_channel_join_enabled") is not False or w.get("io_path") != "gpu_iosurface" or
            w.get("headroom_scale") != 1):
            raise ValueError("native hybrid failed, retried, fell back or performed different work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=[4096, 5120])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--shared-down", action="store_true", help="same explicit typed shared-word down kernel on all arms")
    parser.add_argument("--order", help="gpu,a4096,a5120 or exact reverse/subset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not 2 <= args.steps <= 40 or not args.channels or len(set(args.channels)) != len(args.channels) or
        any(not 0 < v < 12288 or v % 512 for v in args.channels)):
        parser.error("bounded steps and distinct fixed aligned partial channel shares required")
    allowed = ["gpu", *(f"a{v}" for v in args.channels)]
    order = args.order.split(",") if args.order else allowed
    if len(order) < 2 or set(order) != set(allowed) or len(order) != len(allowed):
        parser.error("exact matched GPU/channel arm order required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh evidence directory required")
    cli = args.cli.resolve(strict=True)
    model = args.model.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    inputs = [cli, cli.parent / "libturbocider.dylib", manifest,
              model / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
              model / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
              model / "vae/qwen_image_2.1_vae_bf16.safetensors"]
    identities = {str(p): sha256_file(p) for p in inputs}
    args.output.mkdir(parents=True)
    summary = dict(schema="tc-qwen21-gguf-hybrid-screen-v1", status="running", qualification_passed=False,
                   source_identities=identities, steps=args.steps, phase=args.phase, shared_down=args.shared_down, order=order, trials=[])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for mode in order:
        channels = 0 if mode == "gpu" else int(mode[1:])
        env = benchmark_environment()
        env.update(TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS="1", TURBOCIDER_QWEN21_PROFILE_STEPS="1")
        env["TURBOCIDER_QWEN21_GGUF_SHARED_DOWN"] = "1" if args.shared_down else "0"
        if channels:
            env.update(TURBOCIDER_ANE_BACKEND="private", TURBOCIDER_ALLOW_PRIVATE_ANE="1",
                       TURBOCIDER_PRIVATE_ANE_CHANNELS=str(channels), TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
                       TURBOCIDER_PRIVATE_ANE_GPU_IO="1", TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",
                       TURBOCIDER_RUNTIME_ANE_CHUNKS="1", TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",
                       TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1", TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="0",
                       TURBOCIDER_PRIVATE_ANE_PREFETCH="0", TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",
                       TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE=args.phase)
        requests = []
        for index, prompt in enumerate(PROMPTS):
            request = dict(model="qwen-image-2.1", operation="image.generate", prompt=prompt, width=512,
                           height=512, steps=args.steps, seed=29, audio=False, residency="resident",
                           execution="gpu_ane" if channels else "gpu", allow_approximation=True,
                           output=str((args.output / f"{mode}-{index}.png").resolve()))
            if channels:
                request.update(hybrid_mlp_mode="runtime", ane_manifest=str(manifest))
            path = args.output / f"{mode}-{index}.json"
            path.write_text(json.dumps(request, indent=2) + "\n")
            requests.append(str(path.resolve()))
        print(json.dumps({"starting": mode}), flush=True)
        with (args.output / f"{mode}.stdout.jsonl").open("x") as stdout, (args.output / f"{mode}.stderr.txt").open("x") as stderr:
            memory = run_sampled([str(cli), "batch", str(model), *requests], repo=ROOT, output=args.output,
                                 stem=mode, env=env, stdout=stdout, stderr=stderr,
                                 timeout=900, interval_ms=100, max_gap_ms=500)
        rows = [json.loads(line) for line in (args.output / f"{mode}.stdout.jsonl").read_text().splitlines()]
        validate_receipts(rows, channels, args.steps, args.phase)
        if any(sha256_file(p) != identities[str(p)] for p in inputs):
            raise ValueError("model/runtime source changed during screen")
        trial = dict(mode=mode, channels=channels, timings=[r["timings_seconds"] for r in rows], memory=memory,
                     warm_medians={k: statistics.median(r["timings_seconds"][k] for r in rows[1:])
                                   for k in ("request_wall", "text_encode", "denoise")},
                     png_sha256=[sha256_file(args.output / f"{mode}-{i}.png") for i in range(3)])
        summary["trials"].append(trial)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
