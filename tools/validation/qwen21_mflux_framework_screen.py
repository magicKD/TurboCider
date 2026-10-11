#!/usr/bin/env python3
"""Serial original-source Qwen BF16/affine mflux and native full-image screen."""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from flux9_bf16_framework_screen import PROMPTS, generation, check_generations, write_json
from mflux_original_layout import validate_layout
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native-gpu", "native-frozen", "mflux-bf16", "mflux-q4", "mflux-q8")


def requests(output, steps, count, route, manifest=None):
    values = []
    for index, prompt in enumerate(PROMPTS[:count]):
        request = dict(model="qwen-image-2.1", operation="image.generate", prompt=prompt, width=512,
                       height=512, steps=steps, seed=29, audio=False, execution="gpu", residency="resident",
                       allow_approximation=route == "native-frozen", output=str(output / f"{index}.png"))
        if route == "native-frozen":
            if not manifest:
                raise ValueError("existing source-matched frozen manifest required")
            request.update(execution="gpu_ane", qwen21_w8a8=True, ane_manifest=str(manifest))
        values.append(request)
    return values


def validate_runs(report, route, steps, count):
    if len(report["runs"]) != count:
        raise ValueError("complete actual request sequence required")
    for index, row in enumerate(report["runs"]):
        wall = row["request_wall_including_export"]
        if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive complete image latency required")
        if route.startswith("mflux"):
            bits = {"mflux-bf16": None, "mflux-q4": 4, "mflux-q8": 8}[route]
            if row["actual_steps"] != list(range(steps)) or row["actual_quantization"] != bits or row["loras"]:
                raise ValueError("actual mflux steps/precision/base task differs")
            if row["prompt_cache_hit"] is not (index == 3):
                raise ValueError("actual mflux fresh/repeat cache lifecycle differs")
        else:
            metrics = row["metrics"]
            if (metrics["model"] != "qwen-image-2.1" or metrics["operation"] != "image.generate" or
                    (metrics["width"], metrics["height"], metrics["steps"], metrics["seed"]) != (512, 512, steps, 29) or
                    metrics["actual_denoise_steps"] != steps or metrics["prompt_cache_hit"] is not (index == 3)):
                raise ValueError("actual native geometry/steps/cache task differs")
            hybrid = metrics.get("hybrid")
            if route == "native-gpu" and hybrid:
                raise ValueError("GPU control cannot claim hybrid work")
            if route == "native-frozen" and (not hybrid or hybrid.get("runtime_failed") is not False or
                    hybrid.get("checkpoint_sha256_verified") is not True or hybrid.get("runtime_weight") is not None or
                    hybrid.get("runtime_calls_session_total") != (index + 1) * (steps - 1) * 32):
                raise ValueError("actual successful frozen work required, not fallback")


def native_child(args):
    lib = C.CDLL(str(args.library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(pointer):
        value = C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:
            lib.tc_string_free(pointer)
        return value

    engine, error = C.c_void_p(), C.c_void_p()
    start = time.perf_counter()
    if lib.tc_engine_create_model(b"qwen-image-2.1", str(args.model).encode(), C.byref(engine), C.byref(error)):
        raise RuntimeError(consume(error))
    report = dict(engine=args.child, constructor_seconds=time.perf_counter() - start, runs=[])
    try:
        for index, request in enumerate(requests(args.output, args.steps, args.count, args.child, args.frozen_manifest)):
            write_json(args.output / f"request-{index}.json", request)
            result, error = C.c_void_p(), C.c_void_p()
            start = time.perf_counter()
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), None, None, C.byref(result), C.byref(error))
            wall = time.perf_counter() - start
            text, failure = consume(result), consume(error)
            if status:
                raise RuntimeError(failure)
            report["runs"].append(dict(index=index, request_wall_including_export=wall, metrics=json.loads(text)))
            write_json(args.output / "report.json", report)
    finally:
        lib.tc_engine_free(engine)
    validate_runs(report, args.child, args.steps, args.count)


def mflux_child(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
    from mflux_weight_coverage import inspect_reference_coverage
    import mlx.core as mx
    bits = {"mflux-bf16": None, "mflux-q4": 4, "mflux-q8": 8}[args.child]
    coverage = inspect_reference_coverage(args.reference_model, Qwen21WeightDefinition)
    start = time.perf_counter()
    model = QwenImage21(model_path=str(args.reference_model), quantize=bits)
    report = dict(engine=args.child, constructor_seconds=time.perf_counter() - start,
                  precision=str(ModelConfig.precision), quantization=model.bits,
                  text_embedding_dtype=str(model.text_encoder.embed_tokens.weight.dtype),
                  transformer_input_weight_dtype=str(model.transformer.img_in.weight.dtype),
                  required_weight_coverage=coverage,
                  instrumentation="host step counter only; stock model, cache and synchronization",
                  packages={name: importlib.metadata.version(name) for name in ("mlx", "torch", "transformers", "numpy", "Pillow", "safetensors")}, runs=[])
    if str(ModelConfig.precision) != "mlx.core.bfloat16" or model.bits != bits:
        raise ValueError("declared reference precision differs")

    class Counter:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)

    counter = Counter()
    model.callbacks.register(counter)
    for index, request in enumerate(requests(args.output, args.steps, args.count, args.child)):
        counter.steps = []
        write_json(args.output / f"request-{index}.json", request)
        before_cache = request["prompt"] in model.prompt_cache
        start = time.perf_counter()
        image = model.generate_image(seed=29, prompt=request["prompt"], num_inference_steps=args.steps,
                                     height=512, width=512, guidance=1.0)
        image.save(request["output"])
        row = dict(index=index, request_wall_including_export=time.perf_counter() - start,
                   actual_steps=counter.steps, actual_quantization=image.quantization, loras=image.lora_paths,
                   prompt_cache_hit=before_cache, mlx_peak_bytes=mx.get_peak_memory())
        report["runs"].append(row)
        write_json(args.output / "report.json", report)
    validate_runs(report, args.child, args.steps, args.count)
    # Validate exact token input only after all timed image generations.
    import numpy as np
    from tokenizers import Tokenizer
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
    original = Tokenizer.from_file(str(args.model / "processor/tokenizer.json"))
    token_rows = []
    for prompt in PROMPTS[:args.count]:
        tokens = model.tokenizers["qwen21"].tokenize(prompt)
        mx.eval(tokens.input_ids, tokens.attention_mask)
        ids = np.array(tokens.input_ids).reshape(-1).tolist()
        expected = original.encode(Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I.format(prompt), add_special_tokens=False).ids
        if ids != expected:
            raise ValueError("original tokenizer and template inputs differ")
        token_rows.append(dict(ids=ids, valid=len(ids), prefix_drop=Qwen21PromptEncoder._system_prefix_length(model.tokenizers["qwen21"])))
    write_json(args.output / "tokenizer-evidence.json", token_rows)


def parent(args):
    layout = validate_layout(args.reference_model)
    if layout["family"] != "qwen21":
        raise ValueError("original Qwen layout required")
    if args.frozen_manifest:
        args.frozen_manifest = args.frozen_manifest.resolve(strict=True)
    args.output.mkdir(parents=True)
    # Layout conversion already verified complete source bytes. Only the DiT
    # symlink and tokenizer need first registration; all later sweeps reuse it.
    cache_path = args.reference_model / "benchmark-source-registrations.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    files = [*(Path(record["source"]) for record in layout["files"]), *(Path(link["source"]) for link in layout["links"])]
    hashes = {}
    for path in files:
        current = generation(path)
        record = cache.get(str(path))
        converted = next((item for item in layout["files"] if item["source"] == str(path) and item["source_generation"] == current), None)
        digest = (converted["source_sha256"] if converted else record["sha256"] if record and record["generation"] == current else sha256_file(path))
        cache[str(path)] = dict(generation=current, sha256=digest)
        hashes[str(path.relative_to(ROOT))] = digest
    write_json(cache_path, cache)
    reference_code = sorted((args.mflux / "src").rglob("*.py"))
    code = reference_code + [args.library, Path(__file__).resolve(), ROOT / "tools/validation/mflux_original_layout.py",
                             ROOT / "tools/validation/mflux_weight_coverage.py"]
    if args.frozen_manifest:
        files.append(args.frozen_manifest.resolve(strict=True))
    metadata_files = sorted(args.reference_model.rglob("*.json"))
    identities = {str(path): generation(path) for path in (*files, *code, *metadata_files)}
    summary = dict(schema="tc-qwen21-mflux-framework-screen-v1", status="running", qualification_passed=False,
                   steps=args.steps, count=args.count, order=args.order.split(","), trials=[], model_sha256=hashes,
                   library_sha256=sha256_file(args.library), layout_receipt_sha256=sha256_file(args.reference_model / "layout-receipt.json"),
                   mflux_commit=subprocess.check_output(["git", "-C", str(args.mflux), "rev-parse", "HEAD"], text=True).strip(),
                   mflux_source_snapshot=hashlib.sha256(json.dumps([(str(path.relative_to(args.mflux)), sha256_file(path)) for path in reference_code]).encode()).hexdigest(),
                   source_policy="first verified source generation only; persistent metadata-bound reuse; no per-inference model SHA",
                   timing_scope="constructor separately; full PNG request wall; cold/two fresh/repeat, no added phase eval or model instrumentation",
                   representation_scope="same original BF16 values through lossless layout; Q4/Q8 are explicitly separate mflux affine recipes, not original GGUF equality")
    target = args.output / "summary.json"
    write_json(target, summary)
    try:
        for index, route in enumerate(summary["order"]):
            check_generations(identities)
            validate_layout(args.reference_model)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = benchmark_environment()
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=str(args.extra), MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            command = [str(args.python), "-B", str(Path(__file__).resolve()), "--child", route,
                       "--model", str(args.model), "--reference-model", str(args.reference_model), "--library", str(args.library),
                       "--mflux", str(args.mflux), "--output", str(output), "--steps", str(args.steps), "--count", str(args.count)]
            if args.frozen_manifest:
                command += ["--frozen-manifest", str(args.frozen_manifest)]
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                                     timeout=1800, interval_ms=100, max_gap_ms=500,
                                     async_evidence=True,
                                     child_roles=(("python_helper", str(args.python.resolve(strict=True))),))
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            validate_runs(report, route, args.steps, args.count)
            from PIL import Image
            pngs = []
            for request_index in range(args.count):
                path = output / f"{request_index}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete full-resolution PNG required")
                    rgb = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                pngs.append(dict(file_sha256=sha256_file(path), rgb_sha256=rgb))
            summary["trials"].append(dict(route=route, index=index, report=report, memory=memory, pngs=pngs))
            write_json(target, summary)
        if args.count >= 3:
            summary["medians"] = {}
            for route in sorted(set(summary["order"])):
                trials = [trial for trial in summary["trials"] if trial["route"] == route]
                samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
                summary["medians"][route] = dict(fresh_warm_seconds=statistics.median(samples), samples=samples,
                    cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials])
        summary["status"] = "complete_diagnostic" if args.steps == 40 and args.count == 4 else "complete_precheck_only"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "reference-model", "library", "mflux", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--steps", type=int, choices=(4, 40), default=40)
    parser.add_argument("--count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--order", default="native-gpu,mflux-bf16,mflux-bf16,native-gpu")
    parser.add_argument("--child", choices=ROUTES)
    args = parser.parse_args()
    for name in ("model", "reference_model", "library", "mflux"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.absolute()
    if args.child:
        (native_child if args.child.startswith("native") else mflux_child)(args)
    else:
        if args.output.exists() or not args.python or not args.extra or any(route not in ROUTES for route in args.order.split(",")) or (
                "native-frozen" in args.order.split(",") and (not args.frozen_manifest or args.steps != 40)):
            parser.error("fresh output, reference environment and known serial routes required")
        args.python = args.python.absolute()
        args.extra = args.extra.resolve(strict=True)
        parent(args)


if __name__ == "__main__":
    main()
