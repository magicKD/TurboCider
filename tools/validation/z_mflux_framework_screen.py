#!/usr/bin/env python3
"""Same original Z BF16 and explicitly separate affine reference routes."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from flux9_bf16_framework_screen import generation, check_generations, write_json
from image_framework_native import generate_requests
from mflux_original_layout import validate_layout
from runtime_ane_common import benchmark_environment, sha256_file, validate_fp32_channel_join
from runtime_ane_memory import run_sampled
from z_base_formats_screen import COMMON, FROZEN, PRIVATE, PUBLIC, PROMPTS, environment, request, validate_rows

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native-gpu", "native-frozen", "native-private", "native-public", "mflux-bf16", "mflux-q4", "mflux-q8")
PRIVATE_SHARES = (2048, 3072, 4096, 5120, 6144, 7168, 8192)


def native_case(route):
    if route not in ROUTES or not route.startswith("native-"):
        raise ValueError("declared native Z route required")
    return dict(format="bf16", mode=route.removeprefix("native-"))


def requests(output, route, steps, count):
    case = native_case(route) if route.startswith("native-") else dict(format="bf16", mode="gpu")
    return [request(case, prompt, output / f"{index}.png", steps) for index, prompt in enumerate(PROMPTS[:count])]


def route_environment(route, private_channels=4096, private_join="f32"):
    if private_channels not in PRIVATE_SHARES or private_join not in ("f16", "f32"):
        raise ValueError("known aligned partial channel split and explicit join recipe required")
    env = environment(native_case(route)) if route.startswith("native-") else benchmark_environment()
    if route == "native-private":
        env["TURBOCIDER_PRIVATE_ANE_CHANNELS"] = str(private_channels)
        env["TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN"] = "1" if private_join == "f32" else "0"
    return env


def validate_runs(report, route, steps, count):
    if len(report["runs"]) != count:
        raise ValueError("complete actual Z request sequence required")
    for index, row in enumerate(report["runs"]):
        wall = row["request_wall_including_export"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive full PNG wall required")
        if route.startswith("mflux-"):
            bits = {"mflux-bf16": None, "mflux-q4": 4, "mflux-q8": 8}[route]
            if row["actual_steps"] != list(range(steps)) or row["actual_quantization"] != bits or row["loras"]:
                raise ValueError("actual Z reference steps/precision/base differs")
            if row.get("prompt_cache_hit") is not False:
                raise ValueError("stock Z reference re-encodes prompts, no fabricated cache hits")
        else:
            metrics = row["metrics"]
            if (metrics["model"], metrics["operation"], metrics["width"], metrics["height"], metrics["steps"], metrics["seed"]) != (
                    "z-image-turbo", "image.generate", 512, 512, steps, 42) or metrics["prompt_cache_hit"] is not (index == 3):
                raise ValueError("actual native Z geometry/cache task differs")
            hybrid = metrics.get("hybrid")
            if route == "native-gpu" and hybrid:
                raise ValueError("GPU control cannot be hybrid")
            if route != "native-gpu" and (not hybrid or hybrid.get("runtime_failed") is not False or
                    hybrid.get("runtime_failures_session_total") != 0 or hybrid.get("runtime_calls_session_total", 0) <= 0):
                raise ValueError("actual successful hybrid work required")
    if route.startswith("native-") and count == 4:
        validate_rows([row["metrics"] for row in report["runs"]], native_case(route), steps)


def mflux_child(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
    from mflux_weight_coverage import inspect_reference_coverage
    import mlx.core as mx
    bits = {"mflux-bf16": None, "mflux-q4": 4, "mflux-q8": 8}[args.child]
    start = time.perf_counter()
    model = ZImage(model_path=str(args.reference_model), quantize=bits)
    constructor_seconds = time.perf_counter() - start
    coverage = inspect_reference_coverage(args.reference_model, ZImageWeightDefinition,
        dict(vae=model.vae, transformer=model.transformer, text_encoder=model.text_encoder))
    report = dict(engine=args.child, constructor_seconds=constructor_seconds,
                  precision=str(ModelConfig.precision), quantization=model.bits,
                  text_embedding_dtype=str(model.text_encoder.embed_tokens.weight.dtype),
                  required_weight_coverage=coverage,
                  instrumentation="host step counter only; unchanged stock compile, tokenizer and synchronization",
                  packages={name: importlib.metadata.version(name) for name in ("mlx", "torch", "transformers", "numpy", "Pillow", "safetensors")}, runs=[])
    if str(ModelConfig.precision) != "mlx.core.bfloat16" or model.bits != bits:
        raise ValueError("declared Z reference precision differs")

    class Counter:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)

    counter = Counter()
    model.callbacks.register(counter)
    for index, actual in enumerate(requests(args.output, args.child, args.steps, args.count)):
        counter.steps = []
        write_json(args.output / f"request-{index}.json", actual)
        start = time.perf_counter()
        image = model.generate_image(seed=42, prompt=actual["prompt"], num_inference_steps=args.steps,
                                     height=512, width=512, guidance=0.0)
        image.save(actual["output"])
        report["runs"].append(dict(index=index, request_wall_including_export=time.perf_counter() - start,
            actual_steps=counter.steps, actual_quantization=image.quantization, loras=image.lora_paths,
            prompt_cache_hit=False, mlx_peak_bytes=mx.get_peak_memory()))
        write_json(args.output / "report.json", report)
    validate_runs(report, args.child, args.steps, args.count)
    # No extra encoder execution or GPU fences during timed image requests.
    import numpy as np
    token_rows = []
    for prompt in PROMPTS[:args.count]:
        tokens = model.tokenizers["z_image"].tokenize(prompt)
        mx.eval(tokens.input_ids, tokens.attention_mask)
        ids, mask = np.array(tokens.input_ids).reshape(-1).tolist(), np.array(tokens.attention_mask).reshape(-1).tolist()
        expected = model.tokenizers["z_image"].tokenizer.encode(
            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        valid = sum(mask)
        if ids[:valid] != expected or any(value not in (0, 1) for value in mask):
            raise ValueError("original Z tokenizer chat-template inputs differ")
        token_rows.append(dict(ids=ids, valid=valid, padded=len(ids)))
    write_json(args.output / "tokenizer-evidence.json", token_rows)


def parent(args):
    layout = validate_layout(args.reference_model)
    if layout["family"] != "z":
        raise ValueError("original Z lossless layout required")
    selected = args.order.split(",")
    required_manifests = {"native-frozen": FROZEN, "native-private": PRIVATE, "native-public": PUBLIC}
    manifests = [required_manifests[route].resolve(strict=True) for route in set(selected) if route in required_manifests]
    args.output.mkdir(parents=True)
    cache_path = args.reference_model / "benchmark-source-registrations.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    files = [*(Path(item["source"]) for item in layout["files"]), *(Path(item["source"]) for item in layout["links"]),
             *(COMMON / "tokenizer").glob("*.json"), *(COMMON / "tokenizer").glob("*.jinja"), COMMON / "tokenizer/merges.txt"]
    files = sorted(set(files))
    hashes = {}
    for path in files:
        current, previous = generation(path), cache.get(str(path))
        converted = next((item for item in layout["files"] if item["source"] == str(path) and item["source_generation"] == current), None)
        digest = (converted["source_sha256"] if converted else previous["sha256"] if previous and previous["generation"] == current else sha256_file(path))
        cache[str(path)] = dict(generation=current, sha256=digest)
        hashes[str(path.relative_to(ROOT))] = digest
    write_json(cache_path, cache)
    reference_code = sorted((args.mflux / "src").rglob("*.py"))
    own_code = [args.library, Path(__file__).resolve(), ROOT / "tools/validation/image_framework_native.py",
                ROOT / "tools/validation/mflux_original_layout.py", ROOT / "tools/validation/z_base_formats_screen.py",
                ROOT / "tools/validation/mflux_weight_coverage.py"]
    identities = {str(path): generation(path) for path in (*files, *manifests, *reference_code, *own_code, *args.reference_model.rglob("*.json"))}
    summary = dict(schema="tc-z-mflux-framework-screen-v1", status="running", qualification_passed=False,
        steps=args.steps, count=args.count, order=selected, trials=[], model_sha256=hashes,
        private_channels=args.private_channels, private_join=args.private_join,
        library_sha256=sha256_file(args.library), layout_receipt_sha256=sha256_file(args.reference_model / "layout-receipt.json"),
        mflux_commit=subprocess.check_output(["git", "-C", str(args.mflux), "rev-parse", "HEAD"], text=True).strip(),
        mflux_source_snapshot=hashlib.sha256(json.dumps([(str(path.relative_to(args.mflux)), sha256_file(path)) for path in reference_code]).encode()).hexdigest(),
        source_policy="first source generation verified; durable metadata-bound reuse; no per-inference model SHA",
        timing_scope="constructor separately; native/API and stock mflux full PNG wall; cold/two fresh/repeat; no extra phase fences",
        representation_scope="same original BF16 values through lossless layout; reference affine4/8 quantizes eligible text/DiT/VAE modules, not original GGUF equality")
    target = args.output / "summary.json"
    write_json(target, summary)
    try:
        for index, route in enumerate(selected):
            check_generations(identities)
            validate_layout(args.reference_model)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = route_environment(route, args.private_channels, args.private_join)
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=str(args.extra), MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            command = [str(args.python), "-B", str(Path(__file__).resolve()), "--child", route,
                       "--reference-model", str(args.reference_model), "--library", str(args.library), "--mflux", str(args.mflux),
                       "--output", str(output), "--steps", str(args.steps), "--count", str(args.count)]
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=900, interval_ms=100, max_gap_ms=500, child_roles=(("python_helper", str(args.python.resolve(strict=True))),))
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            validate_runs(report, route, args.steps, args.count)
            if route == "native-private" and any(row["metrics"]["hybrid"]["runtime_weight"].get("ane_channels") != args.private_channels
                                                 for row in report["runs"]):
                raise ValueError("actual native channel split differs from requested candidate")
            if route == "native-private":
                validate_fp32_channel_join([row["metrics"] for row in report["runs"]], args.private_join == "f32")
            from PIL import Image
            pngs = []
            for request_index in range(args.count):
                path = output / f"{request_index}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete full-resolution PNG required")
                    pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                pngs.append(dict(file_sha256=sha256_file(path), rgb_sha256=pixels))
            summary["trials"].append(dict(route=route, index=index, report=report, memory=memory, pngs=pngs,
                controls={key: value for key, value in env.items() if key.startswith("TURBOCIDER_")}))
            write_json(target, summary)
        if args.count == 4:
            summary["medians"] = {}
            for route in sorted(set(selected)):
                trials = [trial for trial in summary["trials"] if trial["route"] == route]
                samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
                summary["medians"][route] = dict(fresh_warm_seconds=statistics.median(samples), samples=samples,
                    cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                    repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials])
        summary["status"] = "complete_diagnostic" if args.count == 4 and args.steps == 8 else "complete_precheck_only"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-model", "library", "mflux", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--steps", type=int, choices=(2, 8), default=8)
    parser.add_argument("--count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--private-channels", type=int, choices=PRIVATE_SHARES, default=4096)
    parser.add_argument("--private-join", choices=("f16", "f32"), default="f32")
    parser.add_argument("--order", default="native-gpu,native-frozen,mflux-bf16,mflux-bf16,native-frozen,native-gpu")
    parser.add_argument("--child", choices=ROUTES)
    args = parser.parse_args()
    for name in ("reference_model", "library", "mflux"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.absolute()
    if args.child:
        if args.child.startswith("native-"):
            report = generate_requests(args.library, "z-image-turbo", COMMON,
                requests(args.output, args.child, args.steps, args.count), args.output, args.child)
            validate_runs(report, args.child, args.steps, args.count)
        else:
            mflux_child(args)
    else:
        if args.output.exists() or not args.python or not args.extra or any(route not in ROUTES for route in args.order.split(",")):
            parser.error("fresh output, reference environment and declared serial routes required")
        args.python = args.python.absolute()
        args.extra = args.extra.resolve(strict=True)
        parent(args)


if __name__ == "__main__":
    main()
