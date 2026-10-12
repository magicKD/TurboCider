#!/usr/bin/env python3
"""Same-library local Z base-format GPU/Public/Private serial diagnostics."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import generation, check_generations
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "models/Comfy-Org-z_image_turbo"
CHECKPOINTS = {
    "bf16": COMMON / "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "nvfp4": COMMON / "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
    "convrot": COMMON / "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
    "q4": ROOT / "models/z-image-runtime-gguf-q4-main/z_image_turbo-Q4_0.gguf",
    "q8": ROOT / "models/z-image-runtime-gguf-q8/z_image_turbo-Q8_0.gguf",
}
PRIVATE = ROOT / "outputs/runtime-ane/private-channel-z-c1152-k1024-n512-v2-20261009/manifest.json"
PUBLIC = ROOT / "outputs/runtime-ane/public-w8-z-c352-h3840-f10240-v1-20261007/manifest.json"
PUBLIC_CONVROT = ROOT / "outputs/runtime-ane/public-w8-convrot-c352-h3840-f10240-v1-20261007/manifest.json"
FROZEN = ROOT / "models/coreml/z_image_w8a8_imageonly_hidden_c4_a5120_b1024_diverse8_compiled/manifest-0718a2ff4c18e5c283b91f45ec21d867cd6532e899e6b48520e8396efea20e72.json"
PROMPTS = (
    "A red fox sitting in a snowy forest, natural winter light, detailed fur.",
    "A red fox sitting in a snowy forest, soft winter light, detailed fur.",
    "A red fox sitting in a snowy forest, warm winter light, detailed fur.",
    "A red fox sitting in a snowy forest, warm winter light, detailed fur.",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cases():
    value = [("bf16-gpu", "bf16", "gpu"), ("bf16-frozen", "bf16", "frozen"),
             ("bf16-private", "bf16", "private"), ("nvfp4-gpu", "nvfp4", "gpu"),
             ("convrot-gpu", "convrot", "gpu"), ("convrot-fast", "convrot", "fast"),
             ("convrot-private", "convrot", "private"), ("convrot-public", "convrot", "public")]
    for source in ("q4", "q8"):
        value += [(f"{source}-{mode}", source, mode) for mode in ("gpu", "fast", "private", "public")]
    return {name: dict(format=source, mode=mode) for name, source, mode in value}


def environment(case):
    env = benchmark_environment()
    source, mode = case["format"], case["mode"]
    if source in ("bf16", "nvfp4", "convrot"):
        env["TURBOCIDER_Z_IMAGE_TRANSFORMER"] = str(CHECKPOINTS[source])
    if source == "convrot" and mode == "fast":
        env.update(TURBOCIDER_Z_CONVROT_GPU_RECIPE="compiled_dense", TURBOCIDER_Z_CONVROT_CACHE_BYTES=str(3 << 30),
                   TURBOCIDER_Z_CONVROT_CACHE_RETAIN="1")
    if source in ("q4", "q8") and mode != "gpu":
        env.update(TURBOCIDER_Z_GGUF_IMPORT="cpu_direct", TURBOCIDER_Z_GGUF_AFFINE_PACK="fused",
                   TURBOCIDER_Z_GGUF_RETAIN_PACKED="1", TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES=str(1 << 30))
        if mode == "fast":
            env.update(TURBOCIDER_Z_GGUF_COMPILE_PACKED="1", TURBOCIDER_Z_GGUF_COMPUTE="qmm_f16_down64")
    if mode in ("public", "private"):
        env.update(TURBOCIDER_ANE_BACKEND=mode, TURBOCIDER_PRIVATE_ANE_GPU_IO="1",
                   TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8", TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",
                   TURBOCIDER_RUNTIME_ANE_CHUNKS="1", TURBOCIDER_PRIVATE_ANE_SCALE_CACHE="1",
                   TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1", TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
                   TURBOCIDER_PRIVATE_ANE_PREFETCH="0", TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0")
        if mode == "private":
            env.update(TURBOCIDER_ALLOW_PRIVATE_ANE="1", TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",
                       TURBOCIDER_Z_RUNTIME_MATCH_ROWS="1")
            if source in ("bf16", "convrot"):
                env["TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN"] = "1"
        if source == "convrot":
            env["TURBOCIDER_Z_RUNTIME_CONVROT"] = "1"
            if mode == "private":
                env.update(TURBOCIDER_PRIVATE_ANE_DATA_PATH="convrot_w8a8", TURBOCIDER_Z_CONVROT_BF16_PARTIAL="1")
    return env


def create_gguf_view(output, source):
    root = output / f"model-{source}"
    (root / "split_files").mkdir(parents=True)
    (root / "tokenizer").symlink_to(COMMON / "tokenizer", target_is_directory=True)
    for name in ("text_encoders", "vae"):
        (root / "split_files" / name).symlink_to(COMMON / "split_files" / name, target_is_directory=True)
    (root / CHECKPOINTS[source].name).symlink_to(CHECKPOINTS[source])
    return root


def request(case, prompt, output, steps):
    source, mode = case["format"], case["mode"]
    result = dict(model="z-image-turbo-gguf" if source in ("q4", "q8") else "z-image-turbo",
                  operation="image.generate", prompt=prompt, width=512, height=512, steps=steps,
                  seed=42, audio=False, residency="resident", dynamic_text=True,
                  execution="gpu" if mode in ("gpu", "fast") else "gpu_ane",
                  allow_approximation=True, output=str(output))
    if mode in ("private", "public"):
        manifest = PRIVATE if mode == "private" else PUBLIC_CONVROT if source == "convrot" else PUBLIC
        result.update(hybrid_mlp_mode="runtime", ane_manifest=str(manifest))
    elif mode == "frozen":
        result["ane_manifest"] = str(FROZEN)
    if source == "convrot" and mode == "fast":
        result["compile_gpu"] = True
    return result


def validate_rows(rows, case, steps):
    if len(rows) != 4:
        raise ValueError("cold/two-fresh/repeat actual request sequence required")
    source, mode = case["format"], case["mode"]
    previous_calls = 0
    for index, row in enumerate(rows):
        if (row.get("model") != ("z-image-turbo-gguf" if source in ("q4", "q8") else "z-image-turbo") or
                row.get("operation") != "image.generate" or row.get("width") != 512 or row.get("height") != 512 or
                row.get("seed") != 42 or row.get("steps") != steps or row.get("prompt_cache_hit") is not (index == 3) or
                row.get("lora_applied_projections", 0) != 0 or row.get("lora_identity") or
                any(type(row["timings_seconds"].get(key)) not in (float, int) or
                    not math.isfinite(row["timings_seconds"][key]) or row["timings_seconds"][key] <= 0
                    for key in ("request_wall", "denoise", "vae_decode"))):
            raise ValueError("wrong actual base GPU/source/task/cache or finite timing")
        backend, precision = row.get("runtime_backend", ""), row.get("runtime_precision", "")
        if source == "nvfp4" and ("nvfp4" not in backend or precision != "nvfp4_weights_bf16_activations"):
            raise ValueError("NVFP4 did not execute its actual weight-only path")
        if source == "convrot" and "convrot" not in backend:
            raise ValueError("ConvRot source not actually selected")
        if source in ("q4", "q8") and "gguf" not in backend:
            raise ValueError("GGUF source not actually selected")
        hybrid = row.get("hybrid")
        if mode in ("gpu", "fast"):
            if hybrid:
                raise ValueError("pure GPU control executed hybrid work")
            continue
        if not hybrid or hybrid.get("runtime_failed") is not False or hybrid.get("runtime_failures_session_total") != 0:
            raise ValueError("actual successful hybrid work required, not fallback")
        calls = hybrid.get("runtime_calls_session_total")
        if type(calls) is not int or calls <= previous_calls:
            raise ValueError("actual successful per-request calls required")
        if mode in ("private", "public"):
            runtime = hybrid.get("runtime_weight") or {}
            if (runtime.get("executor_backend") != ("private_ane" if mode == "private" else "public_coreml") or
                    runtime.get("fallback_blocks_session_total") != 0):
                raise ValueError("wrong actual runtime backend or error fallback")
        previous_calls = calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 8), default=8)
    parser.add_argument("--cases", help="explicit serial subset from declared cases")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    all_cases = cases()
    names = args.cases.split(",") if args.cases else list(all_cases)
    if not names or len(set(names)) != len(names) or any(name not in all_cases for name in names) or args.output.exists():
        parser.error("distinct declared cases and fresh output required")
    if args.reverse:
        names.reverse()
    cli = args.cli.resolve(strict=True)
    library = cli.parent / "libturbocider.dylib"
    args.output = args.output.absolute()
    args.output.mkdir(parents=True)
    views = {source: create_gguf_view(args.output, source) for source in ("q4", "q8") if any(all_cases[name]["format"] == source for name in names)}
    sources = sorted(set([*(CHECKPOINTS[all_cases[name]["format"]] for name in names),
                          COMMON / "split_files/text_encoders/qwen_3_4b.safetensors",
                          COMMON / "split_files/vae/ae.safetensors", COMMON / "tokenizer/tokenizer.json",
                          cli, library, PRIVATE, PUBLIC, PUBLIC_CONVROT, FROZEN, Path(__file__).resolve()]))
    identities = {str(path): generation(path) for path in sources}
    hashes = {str(path): sha256_file(path) for path in sources}
    summary = dict(schema="tc-z-base-formats-screen-v1", status="running", qualification_passed=False,
                   date="2026-10-11", steps=args.steps, names=names, source_sha256=hashes, source_generations=identities, trials=[],
                   scope="same library/components/512/seed42; cold/two fresh warm/repeat; source representations and opt-in arithmetic explicitly differ",
                   source_policy="initial whole-window SHA once; generation checks between native processes; no per-inference payload hash")
    path = args.output / "summary.json"
    write_json(path, summary)
    try:
        for name in names:
            check_generations(identities)
            case = all_cases[name]
            directory = args.output / name
            directory.mkdir()
            requests = []
            for index, prompt in enumerate(PROMPTS):
                location = directory / f"request-{index}.json"
                write_json(location, request(case, prompt, directory / f"{index}.png", args.steps))
                requests.append(str(location))
            model = views.get(case["format"], COMMON)
            print(json.dumps(dict(starting=name)), flush=True)
            env = environment(case)
            with (directory / "stdout.jsonl").open("x") as out, (directory / "stderr.txt").open("x") as err:
                memory = run_sampled([str(cli), "batch", str(model), *requests], repo=ROOT, output=directory,
                    stem="run", env=env, stdout=out, stderr=err, timeout=900, interval_ms=100, max_gap_ms=500)
            check_generations(identities)
            rows = [json.loads(line) for line in (directory / "stdout.jsonl").read_text().splitlines()]
            validate_rows(rows, case, args.steps)
            from PIL import Image
            pngs = []
            for index in range(4):
                image_path = directory / f"{index}.png"
                with Image.open(image_path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete original-size output required")
                pngs.append(sha256_file(image_path))
            summary["trials"].append(dict(name=name, case=case, rows=rows, memory=memory,
                controls={key: value for key, value in env.items() if key.startswith("TURBOCIDER_")}, png_sha256=pngs,
                fresh_warm_median=statistics.median(row["timings_seconds"]["request_wall"] for row in rows[1:3]),
                repeated_wall=rows[3]["timings_seconds"]["request_wall"]))
            write_json(path, summary)
        summary["status"] = "complete_diagnostic"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        changes = []
        for name, before in identities.items():
            try:
                after = generation(Path(name))
                if after != before:
                    changes.append(dict(path=name, before=before, after=after))
            except ValueError as failure:
                changes.append(dict(path=name, before=before, error=str(failure)))
        summary["source_generation_changes"] = changes
        raise
    finally:
        write_json(path, summary)


if __name__ == "__main__":
    main()
