#!/usr/bin/env python3
"""Actual native4B saved-Q4 complete PNG sequence; not an external speed ranking."""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import statistics
import sys
import time

from flux9_bf16_framework_screen import PROMPTS, generation, check_generations, registered_hashes, write_json
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]


def validate_runs(rows):
    if len(rows) != 4:
        raise ValueError("complete cold/two fresh/repeat sequence required")
    for index, row in enumerate(rows):
        wall = row.get("request_wall_including_export")
        if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive complete PNG wall required")
        metrics = row["metrics"]
        if (metrics.get("model") != "flux2-klein-4b" or metrics.get("width") != 512 or
                metrics.get("height") != 512 or metrics.get("steps") != 4 or
                metrics.get("actual_denoise_steps") != 4 or metrics.get("lora_strategy") != "none" or
                metrics.get("text_tokens") != 512 or metrics.get("seed") != 42 or
                metrics.get("prompt_cache_hit") is not (index == 3) or metrics.get("hybrid") or
                row["native_event_counts"].get("transformer_block") != 100 or
                row["native_event_counts"].get("compiled_single_gpu_block", 0) != 0):
            raise ValueError("actual complete packed4B GPU/100 blocks/base/cache sequence required")


def child(args):
    library = C.CDLL(str(args.library))
    library.tc_string_free.argtypes = [C.c_void_p]
    library.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_free.argtypes = [C.c_void_p]
    def consume(pointer):
        text = C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:
            library.tc_string_free(pointer)
        return text
    engine, failure = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    if library.tc_engine_create_model(b"flux2-klein-4b", str(args.model).encode(), C.byref(engine), C.byref(failure)):
        raise RuntimeError(consume(failure))
    report = dict(status="running", model="flux2-klein-4b", constructor_seconds=time.perf_counter()-started,
        performance_verified=False, runtime_verified=False, instrumentation="host native event counter only; no extra GPU fences", runs=[])
    counts, errors = {}, []
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    def observed(payload, _):
        try:
            phase = json.loads(payload.decode())["phase"]
            counts[phase] = counts.get(phase, 0) + 1
        except Exception as error:
            errors.append(str(error))
    callback = callback_type(observed)
    try:
        for index, prompt in enumerate(PROMPTS):
            counts.clear()
            errors.clear()
            request = dict(model="flux2-klein-4b", operation="image.generate", prompt=prompt,
                width=512, height=512, steps=4, seed=42, audio=False, execution="gpu", residency="resident",
                dynamic_text=False, allow_approximation=True, output=str(args.output / f"{index}.png"))
            write_json(args.output / f"request-{index}.json", request)
            result, failure = C.c_void_p(), C.c_void_p()
            started = time.perf_counter()
            status = library.tc_engine_generate(engine, json.dumps(request).encode(), C.cast(callback, C.c_void_p), None,
                C.byref(result), C.byref(failure))
            wall = time.perf_counter() - started
            text, error = consume(result), consume(failure)
            if status or errors:
                raise RuntimeError(error or str(errors))
            report["runs"].append(dict(index=index, request_wall_including_export=wall,
                native_event_counts=counts.copy(), metrics=json.loads(text)))
            write_json(args.output / "report.json", report)
        validate_runs(report["runs"])
        report.update(status="complete_precheck_only", runtime_verified=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        library.tc_engine_free(engine)
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    args.output = args.output.resolve()
    if args.child:
        child(args)
        return
    if args.output.exists():
        parser.error("fresh native precheck output required")
    layout = json.loads((args.model / "layout-receipt.json").read_text())
    if layout.get("status") != "complete_layout_only" or (layout.get("bits"),layout.get("group_size")) != (4,64):
        raise ValueError("complete bitpattern-verified native4B Q4/g64 view required")
    paths = sorted(args.model.rglob("*.safetensors"))
    hashes = registered_hashes(paths, ROOT / "outputs/benchmark-source-registrations/flux4-native-q4.json")
    identities = {str(p): generation(p) for p in paths}
    for name, records in layout["components"].items():
        for record in records:
            if hashes[str(args.model / name / record["file"])] != record["sha256"]:
                raise ValueError("native view differs from full payload/bitpattern proof")
    args.output.mkdir(parents=True)
    data = dict(status="running", qualification_passed=False, latency_comparison_valid=False,
        scope="one native packed4B/Q4 process: cold/two fresh/repeat; no external paired median or GPU/ANE claim",
        source_layout_sha256=sha256_file(args.model / "layout-receipt.json"), library_sha256=sha256_file(args.library))
    target = args.output / "summary.json"
    write_json(target,data)
    try:
        command = [sys.executable,"-B",str(Path(__file__).resolve()),"--model",str(args.model),"--library",str(args.library),"--output",str(args.output),"--child"]
        with (args.output / "stdout.txt").open("x") as out, (args.output / "stderr.txt").open("x") as err:
            memory = run_sampled(command,repo=ROOT,output=args.output,stem="run",env=benchmark_environment(),
                stdout=out,stderr=err,timeout=600,interval_ms=100,max_gap_ms=500,async_evidence=True)
        check_generations(identities)
        report = json.loads((args.output / "report.json").read_text())
        validate_runs(report["runs"])
        from PIL import Image
        images = []
        for index in range(4):
            path = args.output / f"{index}.png"
            with Image.open(path) as image:
                image.load()
                if image.size != (512,512):
                    raise ValueError("full-size PNG required")
            images.append(sha256_file(path))
        data.update(status="complete_precheck_only", runtime_verified=True, performance_verified=False,
            report=report,memory=memory,png_sha256=images,
            fresh_median_seconds=statistics.median(row["request_wall_including_export"] for row in report["runs"][1:3]))
    except BaseException as error:
        data.update(status="failed",error=str(error))
        raise
    finally:
        write_json(target,data)
    print(json.dumps(dict(status=data["status"],fresh_median_seconds=data["fresh_median_seconds"])))


if __name__ == "__main__":
    main()
