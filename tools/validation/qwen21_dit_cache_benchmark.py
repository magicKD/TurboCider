"""Run paired local Qwen 2.1 DiT-cache requests; never fetch models.

The first request per condition warms encoding. Measured requests share model,
ordered references, prompt, seed, schedule, precision and residency. Prefix
snapshots are disabled in this comparison to isolate within-request DiT reuse.
Image metrics require visual review and do not establish semantic equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

MODES = ("off", "conservative", "balanced", "fast", "sglang")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:4]))
    parser.add_argument("--operations", nargs="+", choices=["generate", "edit"], default=["generate", "edit"])
    parser.add_argument("--generation-prompt", default="A ceramic blue teapot on a wooden table, warm sunlight, detailed product photography.")
    parser.add_argument("--edit-prompt", default="Change the blue ceramic teapot to red. Preserve the wooden table, lighting and composition.")
    parser.add_argument("--repeat-off", action="store_true", help="Append a second uncached control to check timing drift")
    parser.add_argument("--prefix-snapshot", choices=["0", "1"], default="0",
        help="Use 1 for the App's repeated-edit comparison; DiT cache itself bypasses snapshots")
    args = parser.parse_args()
    if args.prefix_snapshot == "1" and args.repeat_off:
        parser.error("repeat-off after a DiT request has a cold prefix; use separate warmed-prefix comparisons")
    if "sglang" in args.modes and (args.lora or set(args.modes) - {"off", "sglang"}):
        parser.error("sglang is a Base diagnostic comparison; use --modes off sglang without a LoRA")
    if any(step < 20 or step > 40 for step in args.steps):
        parser.error("this qualification matrix supports 20...40 steps")
    for path in [args.cli, args.reference, *([args.lora] if args.lora else [])]:
        if not path.is_file():
            parser.error(f"local file missing: {path}")
    if not args.model.is_dir():
        parser.error("model must be an existing local directory")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    requests = []
    for operation in args.operations:
        # Warm encoding at the actual step count; exclude this request from timing comparisons.
        for steps in args.steps:
            modes = ["warmup", *args.modes, *(["off-repeat"] if args.repeat_off else [])]
            for label in modes:
                mode = "off" if label in ("warmup", "off-repeat") else label
                name = f"{operation}-{steps}-{label}"
                request = dict(schema_version=1, model="qwen-image-2.1",
                    operation=f"image.{operation}",
                    prompt=args.generation_prompt if operation == "generate" else args.edit_prompt,
                    output=str(out / f"{name}.png"), width=512, height=512,
                    steps=steps, seed=args.seed, frames=1, audio=False,
                    execution="gpu", residency="component_staged", dynamic_text=True,
                    prompt_enhance=False, allow_approximation=bool(args.lora) or mode != "off",
                    qwen21_dit_cache=mode)
                if mode == "sglang":
                    # Omission follows diagnostic defaults; explicit off overrides them.
                    del request["qwen21_dit_cache"]
                if operation == "edit":
                    request["inputs"] = [dict(kind="image", role="reference", path=str(args.reference.resolve()))]
                if args.lora:
                    request.update(lora_strategy="inference_time",
                        loras=[dict(path=str(args.lora.resolve()), role="transformer", strength=1.0)])
                path = out / f"{name}.json"
                path.write_text(json.dumps(request, indent=2) + "\n")
                requests.append(path)
    metadata = dict(cli=str(args.cli.resolve()), cli_sha256=sha256(args.cli),
        runtime_sha256=sha256(args.cli.parent / "libturbocider.dylib"),
        model=str(args.model.resolve()), reference_sha256=sha256(args.reference),
        lora_sha256=sha256(args.lora) if args.lora else None,
        timing_scope="request wall and denoise; warmup excluded; encoding warmed; staged weights reload",
        prefix_snapshot=args.prefix_snapshot,
        requests=[str(path) for path in requests])
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TURBOCIDER_QWEN21_"):
            del env[key]
    env.pop("MLX_DISABLE_COMPILE", None)
    env.pop("TURBOCIDER_DISABLE_FUSED_RMSNORM", None)
    env.update(TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=args.prefix_snapshot, TURBOCIDER_QWEN21_PROFILE_STEPS="1")
    if "sglang" in args.modes:
        env.update(TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC="1",
            TURBOCIDER_QWEN21_DBCACHE_THRESHOLD="0.24",
            TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE="3",
            TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS="1",
            TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS="4")
    with (out / "results.jsonl").open("w") as results, (out / "events.jsonl").open("w") as events:
        subprocess.run([str(args.cli.resolve()), "batch", str(args.model.resolve()),
            *map(str, requests)], env=env, stdout=results, stderr=events, check=True)
    if sha256(args.reference) != metadata["reference_sha256"] or (
            args.lora and sha256(args.lora) != metadata["lora_sha256"]):
        raise RuntimeError("reference or adapter contents changed during the comparison")
    if sha256(args.cli) != metadata["cli_sha256"] or sha256(args.cli.parent / "libturbocider.dylib") != metadata["runtime_sha256"]:
        raise RuntimeError("executable or native library changed during the comparison")
    records = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != len(requests):
        raise RuntimeError("request/result count mismatch")
    profiles = []
    for line in (out / "events.jsonl").read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "qwen21_step" not in item:
            continue
        if item["qwen21_step"] == 0:
            profiles.append([])
        profiles[-1].append(item)
    if len(profiles) != len(requests):
        raise RuntimeError("missing per-step profiles")
    summaries = []
    for path, result, profile in zip(requests, records, profiles):
        request = json.loads(path.read_text())
        for key in ("model", "operation", "output", "seed", "steps", "width", "height"):
            if result.get(key) != request[key]:
                raise RuntimeError(f"{path.name}: mismatched {key}")
        if result.get("actual_denoise_steps") != request["steps"]:
            raise RuntimeError(f"{path.name}: incorrect sampling schedule length")
        plan = result["plan"]
        if (plan.get("execution"), plan.get("backend"), plan.get("precision"), plan.get("residency")) != (
                "gpu", "mlx_cpp_metal", "bf16", "component_staged"):
            raise RuntimeError(f"{path.name}: did not execute the matched BF16 GPU route")
        cache = result.get("qwen21_dbcache")
        mode = request.get("qwen21_dit_cache", "diagnostic")
        enabled = mode != "off"
        if bool(cache) != enabled:
            raise RuntimeError(f"{path.name}: requested cache mode was not honored")
        if enabled:
            if cache["saved_middle_blocks"] != cache["cached_steps"] * (32 - cache["front_blocks"] - cache["back_blocks"]):
                raise RuntimeError(f"{path.name}: invalid saved-block accounting")
            if cache.get("mode") != mode:
                raise RuntimeError(f"{path.name}: cache receipt mode mismatch")
            expected = {"conservative": (8, 8, .15, 1), "balanced": (8, 8, .25, 2),
                "fast": (8, 8, .25, 4), "diagnostic": (1, 4, .24, 3)}[mode]
            if (cache["front_blocks"], cache["warmup_steps"], cache["max_consecutive"]) != (expected[0], expected[1], expected[3]) or abs(cache["threshold"] - expected[2]) > 1e-6:
                raise RuntimeError(f"{path.name}: cache parameters were not honored")
        if [item["qwen21_step"] for item in profile] != list(range(request["steps"])):
            raise RuntimeError(f"{path.name}: missing or reordered sampling steps")
        if args.prefix_snapshot == "0" and any(item["prefix_cache_hit"] for item in profile):
            raise RuntimeError(f"{path.name}: unintended cross-request prefix reuse")
        skipped = [item["qwen21_step"] for item in profile if item["dit_cache_skipped"]]
        if len(skipped) != (cache or {}).get("cached_steps", 0):
            raise RuntimeError(f"{path.name}: profile/receipt skip counts differ")
        consecutive = 0
        for item in profile:
            if item["dit_cache_skipped"]:
                consecutive += 1
                if item["qwen21_step"] < cache["warmup_steps"] or item["qwen21_step"] == request["steps"] - 1 or consecutive > cache["max_consecutive"]:
                    raise RuntimeError(f"{path.name}: warmup/final/consecutive guard violated")
            else:
                consecutive = 0
        if args.lora and result.get("lora_applied_projections") != 224:
            raise RuntimeError(f"{path.name}: expected all 224 projections of the qualification adapter")
        summaries.append(dict(name=path.stem, warmup=path.stem.endswith("warmup"),
            prompt_cache_hit=result.get("prompt_cache_hit"),
            timings_seconds=result["timings_seconds"], cache=cache,
            lora_applied_projections=result.get("lora_applied_projections", 0),
            skipped_step_indices=skipped, first_step_seconds=profile[0]["seconds"],
            prefix_cache_hit=profile[0]["prefix_cache_hit"],
            image_sha256=sha256(Path(request["output"])), memory=result.get("memory")))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"status": "passed", "requests": len(records), "output": str(out)}))


if __name__ == "__main__":
    main()
