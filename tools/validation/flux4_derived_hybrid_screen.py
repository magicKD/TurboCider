#!/usr/bin/env python3
"""One source-bound FLUX4 candidate: derived dense-single GPU vs W8/F16 hybrid.

Never retimes the original packed-Q4 or external frameworks. Four owned
processes in forward/reverse order; full PNG requests and independent memory.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import statistics
import sys
import time

from flux4_hybrid_source_prepare import prefixes
from flux9_bf16_framework_screen import PROMPTS, check_generations, generation, registered_hashes, write_json
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("gpu", "hybrid", "hybrid", "gpu")


def requests(output, route, manifest):
    if route not in ORDER:
        raise ValueError("only derivedGPU/hybrid candidate routes supported")
    return [dict(model="flux2-klein-4b", operation="image.generate", prompt=prompt,
                 width=512, height=512, steps=4, seed=42, audio=False,
                 execution="gpu_ane" if route == "hybrid" else "gpu",
                 residency="resident", dynamic_text=False, allow_approximation=True,
                 compile_gpu=route == "gpu", warmup_iterations=0,
                 **({"ane_manifest": str(manifest)} if route == "hybrid" else {}),
                 output=str(output / f"{index}.png")) for index, prompt in enumerate(PROMPTS)]


def validate_runs(rows, route):
    if route not in ORDER or len(rows) != 4:
        raise ValueError("complete derived cold/twofresh/repeat sequence required")
    for index, row in enumerate(rows):
        actual = row["metrics"]
        wall = row.get("request_wall_including_export")
        counts = row.get("native_event_counts", {})
        if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0 or
                actual.get("model") != "flux2-klein-4b" or actual.get("width") != 512 or
                actual.get("height") != 512 or actual.get("steps") != 4 or
                actual.get("actual_denoise_steps") != 4 or actual.get("seed") != 42 or
                actual.get("text_tokens") != 512 or actual.get("lora_strategy") != "none" or
                actual.get("prompt_cache_hit") is not (index == 3) or
                counts.get("transformer_block") != 100 or
                counts.get("acceleration_gpu_fallback", 0) != 0 or
                counts.get("compiled_single_gpu_block", 0) != (80 if route == "gpu" else 0)):
            raise ValueError("actual source-bound derived4B geometry/branches/base/cache differ")
        hybrid = actual.get("hybrid")
        if route == "hybrid":
            if (not isinstance(hybrid, dict) or hybrid.get("bucket") != 1536 or
                    hybrid.get("hidden") != 3072 or hybrid.get("output_channels") != 3072 or
                    hybrid.get("block_count") != 20 or hybrid.get("mlp_width") != 9216 or
                    hybrid.get("ane_mlp_range") != [0, 6144] or
                    hybrid.get("checkpoint_sha256_verified") is not True or
                    hybrid.get("runtime_failed") is not False or
                    hybrid.get("runtime_failures_session_total") != 0 or
                    hybrid.get("runtime_calls_session_total") != 80 * (index + 1) or
                    hybrid.get("calls_session_total") != 80 * (index + 1) or
                    hybrid.get("warmup_calls_session_total") != 0 or
                    hybrid.get("weight_variant") != "int8_pc" or
                    hybrid.get("compute_units") != "cpuAndNeuralEngine"):
                raise ValueError("actual complete source-bound80-call W8/F16 hybrid required")
        elif hybrid:
            raise ValueError("derivedGPU control must not execute hybrid work")


def child(args):
    lib = C.CDLL(str(args.library))
    pointer = C.c_void_p
    lib.tc_string_free.argtypes = [pointer]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(pointer), C.POINTER(pointer)]
    lib.tc_engine_generate.argtypes = [pointer, C.c_char_p, pointer, pointer, C.POINTER(pointer), C.POINTER(pointer)]
    lib.tc_engine_free.argtypes = [pointer]

    def consume(value):
        result = C.string_at(value).decode() if value.value else ""
        if value.value:
            lib.tc_string_free(value)
        return result

    engine, failure = pointer(), pointer()
    started = time.perf_counter()
    if lib.tc_engine_create_model(b"flux2-klein-4b", str(args.model).encode(), C.byref(engine), C.byref(failure)):
        raise RuntimeError(consume(failure))
    report = dict(status="running", route=args.child, constructor_seconds=time.perf_counter() - started,
                  instrumentation="host native branch-event counter, no additional GPU fences", runs=[])
    counts, errors = {}, []
    callback_type = C.CFUNCTYPE(None, C.c_char_p, pointer)

    def observed(payload, _):
        try:
            phase = json.loads(payload.decode())["phase"]
            counts[phase] = counts.get(phase, 0) + 1
        except Exception as error:
            errors.append(str(error))

    callback = callback_type(observed)
    try:
        for index, request in enumerate(requests(args.output, args.child, args.ane_manifest)):
            counts.clear()
            errors.clear()
            write_json(args.output / f"request-{index}.json", request)
            result, failure = pointer(), pointer()
            started = time.perf_counter()
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), C.cast(callback, pointer), None,
                                            C.byref(result), C.byref(failure))
            wall = time.perf_counter() - started
            text, error = consume(result), consume(failure)
            if status or errors:
                raise RuntimeError(error or str(errors))
            report["runs"].append(dict(index=index, request_wall_including_export=wall,
                                       native_event_counts=counts.copy(), metrics=json.loads(text)))
            write_json(args.output / "report.json", report)
        validate_runs(report["runs"], args.child)
        report["status"] = "complete_diagnostic"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        lib.tc_engine_free(engine)
        write_json(args.output / "report.json", report)


def parent(args):
    if args.output.exists():
        raise ValueError("fresh dedicated screen output required")
    source = json.loads((args.model / "derived-receipt.json").read_text())
    if (source.get("status") != "complete_derived_source_only" or source.get("decoded_projections") != prefixes() or
            source.get("all_saved_tensor_bitpatterns_verified") is not True or
            source.get("unchanged_tensor_bitpatterns_verified") is not True):
        raise ValueError("complete single40-derived source proof required")
    manifest = json.loads(args.ane_manifest.read_text())
    sys.path.insert(0, str(ROOT / "tools/coreml"))
    from compile_flux_manifest import validate_manifest
    validate_manifest(manifest, "flux2-klein-4b")
    shape = manifest["shape"]
    if (shape.get("buckets"), shape.get("ane_mlp_start"), shape.get("ane_mlp_end")) != ([1536], 0, 6144):
        raise ValueError("source-bound6144-prefix1536-row manifest required")
    checkpoint = args.model / "transformer/diffusion_pytorch_model.safetensors"
    if (not checkpoint.samefile(manifest["source"]["checkpoint"]) or
            manifest["source"]["checkpoint_sha256"] != source["checkpoint_sha256"]):
        raise ValueError("compiled artifacts must bind the new derived checkpoint")
    paths = [checkpoint, args.model / "transformer/config.json", args.model / "derived-receipt.json",
             args.ane_manifest, args.library, Path(__file__).resolve()]
    for component in ("text_encoder", "vae", "tokenizer"):
        paths.extend(sorted(path for path in (args.model / component).resolve().rglob("*") if path.is_file()))
    for block in manifest["artifacts"].values():
        artifact = (args.ane_manifest.parent / block["int8_pc"]).resolve(strict=True)
        if not artifact.is_dir() or artifact.suffix != ".mlmodelc":
            raise ValueError("complete managed compiled partitions required")
        paths.extend(sorted(path for path in artifact.rglob("*") if path.is_file()))
    identities = {str(path): generation(path) for path in paths}
    hashes = registered_hashes(paths, ROOT / "outputs/benchmark-source-registrations/flux4-derived-hybrid.json")
    if hashes[str(checkpoint)] != source["checkpoint_sha256"]:
        raise ValueError("derived source changed after conversion")
    args.output.mkdir(parents=True)
    summary = dict(schema="tc-flux4-derived-hybrid-screen-v1", status="running", qualification_passed=False,
        goal_complete=False, source_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in paths},
        task=dict(model="flux2-klein-4b", width=512, height=512, steps=4, seed=42, text_tokens=512, loras=[]),
        source_recipe=source["recipe"], order=list(ORDER), trials=[],
        original_packed_native_retested=False, external_retested=False,
        timing_scope="complete PNG; cold/twofresh/repeat perprocess; derivedGPU vs same-derived-source hybrid; not originalpacked speed/quality parity",
        precision_scope="BF16 single projections derived from Q4/g64; CoreML INT8 weights/FP16 IO+math, no activationW8A8 claim",
        source_policy="first generation SHA registered, unchanged metadata at arm boundaries; no perrequest model scans/cache purge")
    target = args.output / "summary.json"
    write_json(target, summary)
    try:
        from PIL import Image
        for index, route in enumerate(ORDER):
            check_generations(identities)
            destination = args.output / f"{index}-{route}"
            destination.mkdir()
            env = benchmark_environment()
            for key in tuple(env):
                if key.startswith(("TURBOCIDER_FLUX_", "TURBOCIDER_FLUX9_", "TURBOCIDER_QWEN3_")):
                    del env[key]
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", route,
                       "--model", str(args.model), "--library", str(args.library),
                       "--ane-manifest", str(args.ane_manifest), "--output", str(destination)]
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (destination / "stdout.txt").open("x") as out, (destination / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=destination, stem="run", env=env,
                    stdout=out, stderr=err, timeout=600, interval_ms=100, max_gap_ms=500, async_evidence=True)
            report_path = destination / "report.json"
            report = json.loads(report_path.read_text())
            validate_runs(report["runs"], route)
            media = []
            for request_index in range(4):
                image_path = destination / f"{request_index}.png"
                with Image.open(image_path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete full-size PNG required")
                media.append(dict(path=str(image_path.relative_to(ROOT)), sha256=sha256_file(image_path)))
            check_generations(identities)
            summary["trials"].append(dict(index=index, route=route, report=report,
                report_sha256=sha256_file(report_path), memory=memory, pngs=media))
            write_json(target, summary)
        summary["routes"] = []
        for route in ("gpu", "hybrid"):
            trials = [trial for trial in summary["trials"] if trial["route"] == route]
            samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
            summary["routes"].append(dict(route=route, fresh_warm_seconds=samples,
                fresh_warm_median=statistics.median(samples),
                cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
                physical_peaks=[trial["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
        gpu, hybrid = [row["fresh_warm_median"] for row in summary["routes"]]
        summary.update(status="complete_diagnostic", derived_native_comparison_valid=True,
                       comparisons=dict(gpu_over_hybrid=gpu / hybrid, hybrid_reduction_percent=100 * (1 - hybrid / gpu)))
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, summary)
    print(json.dumps(dict(status=summary["status"], routes=summary["routes"], comparisons=summary["comparisons"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "ane-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--child", choices=("gpu", "hybrid"))
    args = parser.parse_args()
    for name in ("model", "library", "ane_manifest"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    child(args) if args.child else parent(args)


if __name__ == "__main__":
    main()
