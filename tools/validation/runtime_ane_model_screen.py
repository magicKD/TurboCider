#!/usr/bin/env python3
"""Explicit resident GPU/frozen/runtime-weight screen; keep PNGs and raw results.

This is not exclusive device isolation or automatic visual quality approval.
The cold request is excluded. Use reversed order in a new output directory
before drawing performance conclusions. No checkpoint or adapter is modified.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

# Re-export shared contracts for existing local analysis scripts. New tools
# should import runtime_ane_common directly, not this runner entry point.
from runtime_ane_common import (
    benchmark_environment, check_load, chunk_policy, session_counter, sha256_file,
    system_memory, validate_edit_results, validate_results, wait_for_idle,
    qwen_qk_environment, validate_qwen_qk_receipts, validate_lora_channel_range, validate_fixed_async,
    qwen_lora_1024_environment, validate_qwen_lora_1024_receipts,
    validate_deferred_channel_join,
    gpu_layer_policy, z_gpu_layer_environment, validate_requested_gpu_layers,
)
from runtime_ane_memory import run_sampled, run_owned
from runtime_ane_load import LoadObservation
from runtime_ane_calibration import channel_policy


ROOT = Path(__file__).resolve().parents[2]


def edit_workload(model_id, references, reference_size, output_size, routes, has_lora):
    """Only expose existing native edit routes; never silently resize/drop inputs."""
    if not references:
        if reference_size != 1024:
            raise ValueError("reference-size requires reference images")
        return {}
    if model_id != "qwen-image-2.1" or not 1 <= len(references) <= 3:
        raise ValueError("editing screen requires Qwen21 and 1...3 ordered references")
    if reference_size not in (256, 512, 1024):
        raise ValueError("reference-size must be 256, 512 or 1024")
    if any(not path.is_file() for path in references):
        raise ValueError("reference image is missing")
    if has_lora and (reference_size == 256 or output_size != 512):
        raise ValueError("Qwen LoRA editing requires 512px output and 512/1024px references")
    if "frozen" in routes and output_size != 512:
        raise ValueError("frozen Qwen editing is limited to 512px output")
    return {"operation": "image.edit", "qwen21_reference_size": reference_size,
            "inputs": [{"kind": "image", "role": "reference", "path": str(path.resolve())}
                       for path in references]}


def reference_receipts(edit):
    """Record ordered source bytes; pre/post hashes are not an immutable lease."""
    receipts = []
    for item in edit.get("inputs", []):
        path = Path(item["path"])
        receipts.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return receipts


def check_references(edit, expected):
    if reference_receipts(edit) != expected:
        raise ValueError("reference images changed; do not compare these trials")


def reference_environment(route, edit, has_lora):
    flags = {}
    if edit:
        if route == "frozen" and edit["qwen21_reference_size"] == 1024:
            flags["TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC"] = "1"
        if has_lora and edit["qwen21_reference_size"] == 512:
            flags["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"] = "1"
    return flags


def validate_qwen_lora_precision(rows, fp16):
    """Check the native opt-in receipt, not the unchanged BF16 base dtype.

    This reports rank-matmul selection, not numerical/visual qualification.
    Apply the same setting to GPU and hybrid when comparing their speed.
    """
    marker = "experimental FP16 low-rank LoRA matmuls"
    for row in rows:
        selection = row.get("acceleration_selection")
        if not isinstance(selection, str) or (marker in selection) != fp16:
            raise ValueError("Qwen LoRA rank precision does not match the requested experiment")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", type=Path,
                   help="optional isolated native build; requires adjacent libturbocider.dylib")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", choices=("z-image-turbo", "qwen-image-2.1", "z-image-turbo-gguf"), required=True)
    p.add_argument("--runtime-manifest", type=Path)
    p.add_argument("--runtime-backend", choices=("public", "private", "auto"), default="public",
                   help="private/auto explicitly authorize experimental private API in a private-enabled build")
    p.add_argument("--private-gpu-io", action="store_true", help="explicit private IOSurface GPU transfer experiment; verifies actual I/O receipt")
    p.add_argument("--private-data-path", choices=("fp16", "w8a8"), default="fp16",
                   help="private runtime representation; W8A8 requires private backend and GPU I/O")
    p.add_argument("--private-channels", type=channel_policy, default=0,
                   help="0: rows; positive aligned width: fixed channels; auto: native calibrated candidate with raw evidence")
    p.add_argument("--private-prefetch", choices=("0","1"), default="0",
                   help="private W8 future-bank staging ablation; source-matched activation/reuse fences remain checked")
    p.add_argument("--private-scale-cache",choices=("0","1"),default="1",
                   help="private W8 compact immutable-generation row-scale cache ablation")
    p.add_argument("--private-launch-fence",choices=("0","1"),default="0",
                   help="wait for first ANE request/ready-producer submission before the GPU branch, not completed ANE output")
    p.add_argument("--private-a8-lookahead",choices=("0","1"),default="0",
                   help="bounded two-slot A8 staging: prepare next row chunk while current ANE request runs")
    p.add_argument("--private-stage-specialize",choices=("0","1"),default="0",
                   help="format/dtype/H-block Metal function-constant specialization; same weights/recipe")
    p.add_argument("--private-lora-channel-range",choices=("0","1"),default=None,
                   help="explicit full-vs-ANE-only gate/up LoRA correction ablation; requires private channel LoRA")
    p.add_argument("--fixed-async",choices=("0","1"),default=None,
                   help="explicit fixed-partition untimed/async head ablation; requires positive fixed chunks and no profile")
    p.add_argument("--defer-channel-join",choices=("0","1"),default=None,
                   help="private channel fixed-async owned lazy-join ablation; request_wall remains the timing scope")
    p.add_argument("--z-runtime-gpu-blocks",type=gpu_layer_policy,
                   help="explicit complete GPU blocks by FFN ordinal; Z BF16 runtime route only, never applied to GPU/frozen")
    p.add_argument("--qkv-manifest", type=Path,
                   help="Qwen base-only Q/K/V MatMul runtime graph; separate from FFN runtime")
    p.add_argument("--frozen-manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--routes", default="gpu,runtime,frozen")
    p.add_argument("--chunks", type=chunk_policy, default="auto",
                   help="auto (default), 0 (GPU split boundary), or fixed chunk count 1...128")
    p.add_argument("--qkv-chunks", type=chunk_policy, default="1",
                   help="fixed QKV tail chunks 1...128 (default 1), or auto full-block on/off with one chunk")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--prompt", default="A curious red fox sitting in falling snow beside pine trees, detailed fur, natural winter light, photorealistic.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm-repeats", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--sample-memory", action="store_true",
                   help="independent per-trial process-tree sampling on every route")
    p.add_argument("--observe-load", action="store_true",
                   help="continuous competing-process CPU checks on every route; incomplete/contaminated runs rejected")
    p.add_argument("--memory-interval-ms", type=int, default=100)
    p.add_argument("--memory-max-gap-ms", type=int, default=500)
    p.add_argument("--lora", type=Path, help="optional inference-time adapter; never merged into weights")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--qwen-lora-fp16", action="store_true",
                   help="explicit approximate FP16 LoRA rank matmuls on EVERY route; Qwen 512px/6-step LoRA only")
    p.add_argument("--qwen-lora-1024", action="store_true",
                   help="explicit 1024px six-step LoRA generation on EVERY GPU/runtime route; original FP32 rank only")
    p.add_argument("--qwen-qk-norm-rope", action="store_true",
                   help="explicit fused GPU Q/K norm-RoPE on EVERY route; Qwen 512px or 1024px base generation")
    p.add_argument("--reference", type=Path, action="append", default=[],
                   help="Qwen edit image, repeat in reference order (1...3)")
    p.add_argument("--reference-size", type=int, choices=(256, 512, 1024), default=1024,
                   help="native reference resize policy; separate from output size")
    args = p.parse_args()
    try:
        qk_environment = qwen_qk_environment(args.model_id, args.size, args.qwen_qk_norm_rope,
                                           base_generation=not args.reference and args.lora is None)
    except ValueError as error:
        p.error(str(error))
    cli = (args.cli or ROOT / "build/native/turbocider").resolve()
    library = cli.parent / "libturbocider.dylib"
    if args.cli and (not cli.is_file() or not os.access(cli, os.X_OK) or not library.is_file()):
        p.error("--cli requires an existing executable and adjacent libturbocider.dylib")
    routes = args.routes.split(",")
    if not routes or any(route not in ("gpu", "runtime", "qkv", "frozen") for route in routes):
        p.error("routes must be comma-separated gpu,runtime,qkv,frozen")
    try:
        z_gpu_layer_environment(args.model_id,"runtime",args.z_runtime_gpu_blocks,routes)
    except ValueError as error:
        p.error(str(error))
    try:
        lora_1024_environment = qwen_lora_1024_environment(
            args.model_id, args.size, args.steps, args.qwen_lora_1024,
            has_lora=args.lora is not None, references=bool(args.reference),
            fp16=args.qwen_lora_fp16, routes=routes)
    except ValueError as error:
        p.error(str(error))
    if args.runtime_backend != "public" and "runtime" not in routes:
        p.error("runtime backend selection requires runtime route")
    if args.private_gpu_io and (args.runtime_backend == "public" or "runtime" not in routes):
        p.error("private GPU I/O requires an explicitly private/auto runtime route")
    if args.private_data_path == "w8a8" and (args.runtime_backend != "private" or not args.private_gpu_io):
        p.error("W8A8 requires --runtime-backend private --private-gpu-io")
    full_width = 12288 if args.model_id == "qwen-image-2.1" else 10240
    native_auto = args.private_channels == "auto"
    if (native_auto and (args.runtime_backend != "private" or args.private_data_path != "w8a8" or
                        args.chunks not in ("auto","1"))) or (not native_auto and (args.private_channels < 0 or (args.private_channels and
            (args.private_channels % 512 or args.private_channels >= full_width or args.private_data_path != "w8a8" or
             args.chunks not in ("auto","0","1"))))):
        p.error("private channels require W8A8, a positive 512 multiple below FFN width and chunks=auto,0,1")
    if args.private_lora_channel_range is not None and (args.runtime_backend!="private" or
            not args.private_channels or not args.lora or args.chunks=="0"):
        p.error("LoRA channel range ablation requires private W8A8 channels, an adapter and nonzero chunks")
    if args.fixed_async is not None and ("runtime" not in routes or args.chunks in ("auto","0") or args.profile):
        p.error("fixed async ablation requires runtime, positive fixed chunks and profiling disabled")
    if args.defer_channel_join is not None and (args.runtime_backend!="private" or
            not args.private_channels or args.fixed_async!="1" or args.profile or "runtime" not in routes):
        p.error("deferred channel join requires private channels, fixed-async=1 and no profile")
    if "qkv" in routes and (args.model_id != "qwen-image-2.1" or args.lora or args.reference or
                             args.qkv_chunks == "0"):
        p.error("qkv screen requires Qwen base generation and positive or auto QKV chunks")
    if args.warm_repeats < 1 or not 1 <= args.steps <= 50:
        p.error("need warm repeats >= 1 and steps in 1...50")
    if args.timeout <= 0:
        p.error("timeout must be positive (seconds per route, including cold and warm requests)")
    if args.memory_interval_ms <= 0 or args.memory_max_gap_ms < args.memory_interval_ms:
        p.error("memory interval must be positive and max gap must be >= interval")
    if not args.sample_memory and (args.memory_interval_ms != 100 or args.memory_max_gap_ms != 500):
        p.error("memory timing overrides require --sample-memory")
    if not args.prompt.strip() or not 0 <= args.seed <= 2147483647:
        p.error("need a nonempty prompt and seed in 0...2147483647")
    if not math.isfinite(args.lora_strength) or not -8 <= args.lora_strength <= 8:
        p.error("LoRA strength must be finite and in [-8,8]")
    if args.lora and (not args.lora.is_file() or args.model_id == "z-image-turbo-gguf"):
        p.error("runtime LoRA requires an existing adapter and a supported BF16 model")
    if args.qwen_lora_fp16 and (args.model_id != "qwen-image-2.1" or not args.lora or
                               args.size != 512 or args.steps != 6):
        p.error("--qwen-lora-fp16 requires Qwen LoRA, 512px output and 6 steps")
    for route, manifest in (("runtime", args.runtime_manifest), ("qkv", args.qkv_manifest),
                            ("frozen", args.frozen_manifest)):
        if route in routes and (not manifest or not manifest.is_file()):
            p.error(f"{route} requires an existing manifest")
    try:
        edit = edit_workload(args.model_id, args.reference, args.reference_size,
                             args.size, routes, bool(args.lora))
        references = reference_receipts(edit)
    except (OSError, ValueError) as error:
        p.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "incomplete", "model": args.model_id, "size": args.size, "steps": args.steps,
               "prompt": args.prompt, "seed": args.seed,
               "routes": routes, "chunks": args.chunks, "qkv_chunks": args.qkv_chunks,
               "runtime_backend_preference": args.runtime_backend,
               "private_gpu_io": args.private_gpu_io,
               "private_data_path": args.private_data_path,
               "private_channels": args.private_channels,
               "private_prefetch": args.private_prefetch,
               "private_scale_cache": args.private_scale_cache,
               "private_launch_fence": args.private_launch_fence,
               "private_a8_lookahead": args.private_a8_lookahead,
               "private_stage_specialize": args.private_stage_specialize,
               "private_lora_channel_range": args.private_lora_channel_range,
               "fixed_async": args.fixed_async,
               "defer_channel_join": args.defer_channel_join,
               "z_runtime_gpu_blocks": args.z_runtime_gpu_blocks,
               "placement": "unknown",
               "profile": args.profile, "warm_repeats": args.warm_repeats,
               "continuous_load_observation": args.observe_load,
               "timing_scope": "native request_wall; includes VAE/PNG; excludes cold request",
               "library_sha256": sha256_file(library),
               "trials": []}
    if args.cli:
        summary["cli"] = {"path": str(cli), "sha256": sha256_file(cli)}
    summary["operation"] = edit.get("operation", "image.generate")
    summary["memory_sampling"] = {"enabled": args.sample_memory,
                                  "interval_ms": args.memory_interval_ms,
                                  "max_gap_ms": args.memory_max_gap_ms}
    if edit:
        summary.update(references=references, reference_size=args.reference_size)
    if args.lora:
        summary["lora"] = {"sha256": sha256_file(args.lora), "strength": args.lora_strength,
                           "strategy": "inference_time"}
        if args.model_id == "qwen-image-2.1":
            summary["lora"]["rank_matmul_dtype"] = "fp16" if args.qwen_lora_fp16 else "fp32"
    env = benchmark_environment()
    env.update(qk_environment)
    env.update(lora_1024_environment)
    if args.qwen_lora_1024:
        summary["qwen_lora_1024_diagnostic"] = True
    if args.qwen_qk_norm_rope:
        summary["qwen_qk_norm_rope"] = True
    if args.qwen_lora_fp16:
        env["TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"] = "1"
    # A failed/lost later route must not make an earlier partial receipt look
    # like a completed comparison. Preserve raw evidence on every failure.
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for trial, route in enumerate(routes):
        snapshot = wait_for_idle()
        check_references(edit, references)
        (args.output / f"{trial}-{route}-processes.txt").write_text(snapshot)
        requests = []
        for index in range(args.warm_repeats + 1):
            request = {"model": args.model_id, "operation": "image.generate",
                       "prompt": args.prompt,
                       "width": args.size, "height": args.size, "steps": args.steps, "seed": args.seed,
                       "frames": 1, "audio": False, "residency": "resident",
                       "execution": "gpu" if route == "gpu" else "gpu_ane",
                       "allow_approximation": True,
                       "output": str((args.output / f"{trial}-{route}-{index}.png").resolve())}
            request.update(edit)
            if args.lora:
                request.update(lora_strategy="inference_time", loras=[{
                    "path": str(args.lora.resolve()), "role": "transformer", "strength": args.lora_strength}])
            if route == "runtime":
                request.update(hybrid_mlp_mode="runtime", ane_manifest=str(args.runtime_manifest.resolve()))
            if route == "qkv":
                request.update(hybrid_mlp_mode="runtime_qkv", ane_manifest=str(args.qkv_manifest.resolve()))
            if route == "frozen":
                request["ane_manifest"] = str(args.frozen_manifest.resolve())
                if args.model_id == "qwen-image-2.1":
                    request["qwen21_w8a8"] = True
                if args.lora:
                    request["hybrid_mlp_mode"] = "lora_fused"
            path = args.output / f"{trial}-{route}-{index}.json"
            path.write_text(json.dumps(request, indent=2) + "\n")
            requests.append(path.resolve())
        route_env = dict(env)
        route_env.update(reference_environment(route, edit, bool(args.lora)))
        route_env.update(z_gpu_layer_environment(args.model_id,route,args.z_runtime_gpu_blocks,routes))
        if route == "runtime":
            route_env["TURBOCIDER_RUNTIME_ANE_CHUNKS"] = args.chunks
            if args.fixed_async is not None:
                route_env["TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC"] = args.fixed_async
            if args.defer_channel_join is not None:
                route_env["TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN"] = args.defer_channel_join
            route_env["TURBOCIDER_ANE_BACKEND"] = args.runtime_backend
            if args.runtime_backend != "public":
                route_env["TURBOCIDER_ALLOW_PRIVATE_ANE"] = "1"
                route_env["TURBOCIDER_PRIVATE_ANE_DATA_PATH"] = args.private_data_path
                route_env["TURBOCIDER_PRIVATE_ANE_CHANNELS"] = str(args.private_channels)
                route_env["TURBOCIDER_PRIVATE_ANE_PREFETCH"] = args.private_prefetch
                route_env["TURBOCIDER_PRIVATE_ANE_SCALE_CACHE"] = args.private_scale_cache
                route_env["TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE"] = args.private_launch_fence
                route_env["TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD"] = args.private_a8_lookahead
                route_env["TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE"] = args.private_stage_specialize
                if args.private_lora_channel_range is not None:
                    route_env["TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE"] = args.private_lora_channel_range
            if args.private_gpu_io:
                route_env["TURBOCIDER_PRIVATE_ANE_GPU_IO"] = "1"
            if args.profile:
                route_env["TURBOCIDER_RUNTIME_ANE_PROFILE"] = "1"
        if route == "qkv":
            route_env["TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS"] = args.qkv_chunks
        # Full-size Qwen frozen graphs remain a separately gated diagnostic.
        if route == "frozen" and args.model_id == "qwen-image-2.1" and args.size == 1024:
            route_env["TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC"] = "1"
        print(json.dumps({"starting": route, "trial": trial}), flush=True)
        memory_before = system_memory()
        sampled_memory = None
        observed_load = LoadObservation(args.output / f"{trial}-{route}-load.jsonl") if args.observe_load else None
        command = [str(cli), "batch", str(args.model.resolve()), *(str(path) for path in requests)]
        with (args.output / f"{trial}-{route}.stdout.jsonl").open("w") as stdout, \
             (args.output / f"{trial}-{route}.stderr.txt").open("w") as stderr:
            if args.sample_memory:
                sampled_memory = run_sampled(
                    command, repo=ROOT, output=args.output, stem=f"{trial}-{route}",
                    env=route_env, stdout=stdout, stderr=stderr, timeout=args.timeout,
                    interval_ms=args.memory_interval_ms, max_gap_ms=args.memory_max_gap_ms, observer=observed_load)
            else:
                result = run_owned(command, cwd=ROOT, env=route_env, stdout=stdout,
                                   stderr=stderr, timeout=args.timeout, observer=observed_load)
                if result.returncode:
                    raise RuntimeError(f"{route} failed ({result.returncode}); inspect saved stderr")
        check_references(edit, references)
        load_receipt = observed_load.verify() if observed_load else None
        rows = [json.loads(line) for line in (args.output / f"{trial}-{route}.stdout.jsonl").read_text().splitlines()]
        validate_results(rows, route, len(requests), args.model_id, bool(args.lora), args.runtime_backend, args.private_gpu_io,
                         "w8a8_hadamard" if args.private_data_path == "w8a8" else None,
                         channel_auto=native_auto and route=="runtime")
        active_runtime_rows = [row for row in rows if not native_auto or
            ((row.get("hybrid") or {}).get("runtime_weight") or {}).get("executor_backend")=="private_ane"]
        if route=="runtime":validate_requested_gpu_layers(rows,args.z_runtime_gpu_blocks,args.steps)
        if route=="runtime" and args.private_lora_channel_range is not None:
            validate_lora_channel_range(rows,args.private_lora_channel_range=="1")
        if route=="runtime" and args.fixed_async is not None and active_runtime_rows:
            validate_fixed_async(active_runtime_rows,args.fixed_async=="1")
        if route=="runtime" and args.defer_channel_join is not None and active_runtime_rows:
            validate_deferred_channel_join(active_runtime_rows,args.defer_channel_join=="1")
        if route == "runtime" and args.private_channels and not native_auto:
            for row in rows:
                receipt = row["hybrid"]["runtime_weight"]
                if (receipt.get("partition_axis") != "intermediate_channels" or
                        receipt.get("ane_channels") != args.private_channels or
                        receipt.get("gpu_channels") != full_width-args.private_channels):
                    raise ValueError("requested physical channel partition was not reported")
        if route == "runtime" and args.private_data_path=="w8a8":
            for row in active_runtime_rows:
                receipt=row["hybrid"]["runtime_weight"]
                if receipt.get("prefetch_enabled") is not (args.private_prefetch=="1"):
                    raise ValueError("requested W8 future-bank prefetch policy was not reported")
                if receipt.get("scale_cache_enabled") is not (args.private_scale_cache=="1"):
                    raise ValueError("requested W8 scale-cache policy was not reported")
                if receipt.get("launch_fence_enabled") is not (args.private_launch_fence=="1"):
                    raise ValueError("requested first-submission policy was not reported")
                if receipt.get("a8_lookahead_enabled") is not (args.private_a8_lookahead=="1"):
                    raise ValueError("requested bounded A8 lookahead policy was not reported")
                if receipt.get("stage_specialized") is not (args.private_stage_specialize=="1"):
                    raise ValueError("requested W8 pipeline specialization was not reported")
        if args.model_id == "qwen-image-2.1":
            validate_qwen_qk_receipts(rows, args.qwen_qk_norm_rope)
        if args.model_id == "qwen-image-2.1" and args.lora:
            validate_qwen_lora_precision(rows, args.qwen_lora_fp16)
            validate_qwen_lora_1024_receipts(rows, args.qwen_lora_1024)
        reference_tokens = validate_edit_results(rows, edit)
        if edit:
            if summary.get("reference_tokens", reference_tokens) != reference_tokens:
                raise ValueError("reference token geometry differs across routes")
            summary["reference_tokens"] = reference_tokens
        wall = [row["timings_seconds"]["request_wall"] for row in rows]
        denoise = [row["timings_seconds"]["denoise"] for row in rows]
        record = {"trial": trial, "route": route, "wall_seconds": wall,
                  "denoise_seconds": denoise, "warm_median": statistics.median(wall[1:]),
                  "hybrid": rows[-1].get("hybrid"), "qkv": rows[-1].get("qkv"),
                  "memory": rows[-1].get("memory"),
                  "system_memory_before": memory_before, "system_memory_after": system_memory()}
        if route == "runtime":
            # Auto may validly decline every warm block after its cold probe.
            # Make that visible: a private session label with zero new calls
            # must not be mistaken for steady W8A8 acceleration evidence.
            cumulative = [row["hybrid"]["runtime_calls_session_total"] for row in rows]
            record["runtime_calls_per_request"] = [current - prior for current, prior in
                                                   zip(cumulative, [0, *cumulative[:-1]])]
            if native_auto:
                record["native_channel_calibration"] = rows[-1]["hybrid"]["runtime_weight"]["channel_calibration"]
                record["native_channel_auto_declined"] = record["native_channel_calibration"]["selected_channels"] == 0
        if sampled_memory is not None:
            record["sampled_memory"] = sampled_memory
        if load_receipt is not None:
            record["load_observation"] = load_receipt
        summary["trials"].append(record)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    summary["status"] = "complete"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
