#!/usr/bin/env python3
"""Serial BF16 resident/streaming and mixed GGUF GPU/ANE full-request screens."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
from qwen21_gguf_hybrid_screen import PROMPTS, counter, validate_receipts

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("bf16-stream", "gguf-gpu", "gguf-ane", "bf16-resident")


def validate_bf16_rows(rows, steps, streamed, prefixes, budget, hashes):
    if len(rows) != 3:
        raise ValueError("three actual fresh-condition requests required")
    for index, row in enumerate(rows):
        if (row.get("model") != "qwen-image-2.1" or row.get("operation") != "image.generate" or
            row.get("width") != 512 or row.get("height") != 512 or row.get("steps") != steps or
            row.get("actual_denoise_steps") != steps or row.get("seed") != 29 or
            row.get("reference_tokens") != 0 or row.get("prompt_cache_hit") is not False or row.get("hybrid")):
            raise ValueError("wrong BF16 workload, cache or model execution")
        for name in ("request_wall", "text_encode", "denoise", "vae_decode"):
            value = (row.get("timings_seconds") or {}).get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid native whole-phase timing")
        metrics = row.get("qwen_bf16_streaming")
        if not streamed:
            if metrics or row.get("runtime_backend") != "mlx_cpp_metal":
                raise ValueError("resident BF16 control changed")
            retention = row.get("encoder_weight_residency") or {}
            if (retention.get("enabled") is not True or retention.get("weights_reused") is not (index > 0) or
                retention.get("weights_retained") is not True or counter(retention, "loads_session_total") != 1):
                raise ValueError("real original resident encoder lifecycle differs")
            continue
        if (row.get("runtime_backend") != "mlx_cpp_metal_qwen21_bf16_streaming_experimental" or
            (metrics or {}).get("weight_precision") != "original_bf16" or
            metrics.get("dynamic_layer_weight_arguments") is not True or
            metrics.get("components_released_before_vae") is not True or
            metrics.get("public_memory_qualification") is not False or metrics.get("physical_overlap_proved") is not False):
            raise ValueError("missing real original-BF16 dynamic/retired component evidence")
        for name, layers, passes, prefix in (("encoder", 36, 1, prefixes[0]), ("denoiser", 32, steps, prefixes[1])):
            stage = metrics.get(name) or {}
            fills = (layers - prefix) * passes
            for field, expected in (("layers", layers), ("prefix", prefix), ("slots", 2), ("completed_passes", passes),
                ("completed_layers", layers * passes), ("completed_prefix_layers", prefix * passes),
                ("completed_streamed_layers", fills), ("fills", fills), ("reader_fences", fills),
                ("completed_reader_fences", fills), ("managed_weight_budget_bytes", budget)):
                if counter(stage, field) != expected:
                    raise ValueError("incomplete original-BF16 source/pass/last-reader coverage: " + field)
            if (stage.get("drained") is not True or stage.get("source_sha256") != hashes[name] or
                not 0 < counter(stage, "managed_weight_capacity_bytes") <= budget or
                counter(stage, "resident_source_bytes") == 0 or counter(stage, "streamed_source_bytes") == 0):
                raise ValueError("missing source proof, managed weight admission, real reads or drain")
            if index == 0:
                if counter(stage, "verification_bytes") != counter(stage, "source_file_bytes"):
                    raise ValueError("cold original source full-SHA proof missing")
            elif counter(stage, "verification_bytes") != 0 or counter(stage, "verification_cache_hits") != 1:
                raise ValueError("warm native unchanged-generation content proof differs")
        phases = row.get("qwen_ffn_phases") or {}
        if (phases.get("policy") != "gpu" or counter(phases.get("prefill") or {}, "steps_this_request") != 1 or
            counter(phases.get("decode") or {}, "steps_this_request") != steps - 1):
            raise ValueError("original prefix KV/complete denoise phase coverage differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(4, 40), default=40)
    parser.add_argument("--order", default=",".join(ARMS))
    parser.add_argument("--encoder-prefix", type=int, default=24)
    parser.add_argument("--dit-prefix", type=int, default=24)
    parser.add_argument("--weight-gib", type=int, default=11)
    parser.add_argument("--dump-latents", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    order = args.order.split(",")
    if (len(order) != len(set(order)) or len(order) < 2 or set(order) - set(ARMS) or
        "bf16-stream" not in order or not 0 <= args.encoder_prefix <= 34 or not 0 <= args.dit_prefix <= 30 or
        not 1 <= args.weight_gib <= 20 or args.output.exists() or args.output.is_symlink()):
        parser.error("fresh evidence and actual bounded source/control arms required")
    cli, bf16, gguf, manifest = (p.resolve(strict=True) for p in (args.cli, args.bf16, args.gguf, args.manifest))
    sources = [cli, cli.parent / "libturbocider.dylib", manifest,
        bf16 / "text_encoders/qwen3vl_8b_bf16.safetensors", bf16 / "diffusion_models/qwen_image_2.1_bf16.safetensors",
        bf16 / "vae/qwen_image_2.1_vae_bf16.safetensors", gguf / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        gguf / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf"]
    hashes = {str(p): sha256_file(p) for p in sources}
    args.output.mkdir(parents=True)
    summary = dict(schema="tc-qwen21-bf16-stream-comparison-v1", status="running", qualification_passed=False,
                   source_identities=hashes, order=order, steps=args.steps, encoder_prefix=args.encoder_prefix,
                   dit_prefix=args.dit_prefix, managed_weight_budget_bytes=args.weight_gib << 30, trials=[])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for mode in order:
        print(json.dumps({"starting": mode}), flush=True)
        env = benchmark_environment()
        env.update(TURBOCIDER_QWEN21_PROFILE_STEPS="1", TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS="8")
        streamed = mode == "bf16-stream"
        env["TURBOCIDER_QWEN21_BF16_STREAMING"] = "1" if streamed else "0"
        env["TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS"] = "0" if streamed else "1"
        env.update(TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX=str(args.encoder_prefix),
                   TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX=str(args.dit_prefix),
                   TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB=str(args.weight_gib))
        if mode == "gguf-ane":
            env.update(TURBOCIDER_ANE_BACKEND="private", TURBOCIDER_ALLOW_PRIVATE_ANE="1",
                TURBOCIDER_PRIVATE_ANE_CHANNELS="5120", TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
                TURBOCIDER_PRIVATE_ANE_GPU_IO="1", TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1", TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1", TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1",
                TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="0", TURBOCIDER_PRIVATE_ANE_PREFETCH="0",
                TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0", TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="all")
        requests = []
        for index, prompt in enumerate(PROMPTS):
            request = dict(model="qwen-image-2.1", operation="image.generate", prompt=prompt, width=512, height=512,
                steps=args.steps, seed=29, audio=False, execution="gpu_ane" if mode == "gguf-ane" else "gpu",
                residency="component_staged" if streamed else "resident", allow_approximation=True,
                output=str((args.output / f"{mode}-{index}.png").resolve()))
            if mode == "gguf-ane":
                request.update(hybrid_mlp_mode="runtime", ane_manifest=str(manifest))
            if args.dump_latents:
                request["dump_tensors"] = str((args.output / f"{mode}-{index}-dump").resolve())
            path = args.output / f"{mode}-{index}.json"
            path.write_text(json.dumps(request, indent=2) + "\n")
            requests.append(str(path.resolve()))
        model = gguf if mode.startswith("gguf") else bf16
        with (args.output / f"{mode}.stdout.jsonl").open("x") as stdout, (args.output / f"{mode}.stderr.txt").open("x") as stderr:
            memory = run_sampled([str(cli), "batch", str(model), *requests], repo=ROOT, output=args.output, stem=mode,
                env=env, stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500)
        rows = [json.loads(line) for line in (args.output / f"{mode}.stdout.jsonl").read_text().splitlines()]
        if mode.startswith("gguf"):
            validate_receipts(rows, 5120 if mode == "gguf-ane" else 0, args.steps, "all")
        else:
            validate_bf16_rows(rows, args.steps, streamed, (args.encoder_prefix, args.dit_prefix), args.weight_gib << 30,
                dict(encoder=hashes[str(sources[3])], denoiser=hashes[str(sources[4])]))
        trial = dict(mode=mode, timings=[r["timings_seconds"] for r in rows], memory=memory,
            warm_medians={name: statistics.median(r["timings_seconds"][name] for r in rows[1:])
                          for name in ("request_wall", "text_encode", "denoise", "vae_decode")},
            actual_streaming=[r.get("qwen_bf16_streaming") for r in rows],
            actual_phases=[r.get("qwen_ffn_phases") for r in rows],
            png_sha256=[sha256_file(args.output / f"{mode}-{i}.png") for i in range(3)])
        summary["trials"].append(trial)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha256_file(p) != hashes[str(p)] for p in sources):
        raise ValueError("original model/runtime source changed during window")
    arms = {t["mode"]: t for t in summary["trials"]}
    if "gguf-ane" in arms:
        ratio = arms["bf16-stream"]["memory"]["verified"]["tree_peak_phys_footprint_bytes"] / arms["gguf-ane"]["memory"]["verified"]["tree_peak_phys_footprint_bytes"]
        summary["stream_to_gguf_ane_tree_peak_ratio"] = ratio
        summary["observed_memory_within_ten_percent"] = .9 <= ratio <= 1.1
    if "bf16-resident" in arms:
        summary["stream_vs_resident_png_equal"] = [a == b for a, b in zip(arms["bf16-stream"]["png_sha256"], arms["bf16-resident"]["png_sha256"])]
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
