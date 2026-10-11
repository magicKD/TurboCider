#!/usr/bin/env python3
"""Serial same-source FLUX9B BF16 native/mflux cold, fresh and repeat screen."""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "A red fox sitting in a snowy forest, natural winter light, detailed fur.",
    "A red fox sitting in a snowy forest, soft winter light, detailed fur.",
    "A red fox sitting in a snowy forest, warm winter light, detailed fur.",
    "A red fox sitting in a snowy forest, warm winter light, detailed fur.",
)


def generation(path):
    """Bind code too: empty package __init__.py is a valid source input."""
    try:
        resolved, status = path.resolve(strict=True), path.stat()
    except OSError as error:
        raise ValueError("source unavailable") from error
    if not path.is_file() or (status.st_size == 0 and path.suffix != ".py"):
        raise ValueError("missing regular original source")
    return dict(target=str(resolved), device=status.st_dev, inode=status.st_ino, bytes=status.st_size,
                mtime_ns=status.st_mtime_ns, ctime_ns=status.st_ctime_ns)


def check_generations(identities):
    if any(generation(Path(name)) != value for name, value in identities.items()):
        raise ValueError("source generation changed during screen")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def registered_hashes(paths, registration):
    """Benchmark-only generation-bound source reuse; never write model dirs."""
    cache = json.loads(registration.read_text()) if registration.is_file() else {}
    result = {}
    for path in paths:
        identity = generation(path)
        previous = cache.get(str(path))
        digest = previous.get("sha256") if isinstance(previous, dict) and previous.get("generation") == identity else None
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            digest = sha256_file(path)
            if generation(path) != identity:
                raise ValueError("source changed during initial verification")
        result[str(path)] = digest
        cache[str(path)] = dict(generation=identity, sha256=digest)
    registration.parent.mkdir(parents=True, exist_ok=True)
    write_json(registration, cache)
    return result


def requests(output):
    return [dict(model="flux2-klein-9b", operation="image.generate", prompt=prompt, width=512,
                 height=512, steps=4, seed=42, audio=False, execution="gpu", residency="resident",
                 dynamic_text=False, output=str(output / f"{index}.png"))
            for index, prompt in enumerate(PROMPTS)]


def validate_native(rows):
    if len(rows) != 4:
        raise ValueError("complete cold/two-fresh/repeat sequence required")
    for index, row in enumerate(rows):
        metrics = row["metrics"]
        if (metrics["model"] != "flux2-klein-9b" or metrics["operation"] != "image.generate" or
                metrics["width"] != 512 or metrics["height"] != 512 or metrics["steps"] != 4 or
                metrics["seed"] != 42 or metrics["text_tokens"] != 512 or metrics.get("hybrid") or
                metrics["prompt_cache_hit"] is not (index == 3)):
            raise ValueError("wrong actual BF16 GPU/prompt/cache/geometry")
        if type(row["request_wall_including_export"]) not in (float, int) or not math.isfinite(row["request_wall_including_export"]) or row["request_wall_including_export"] <= 0:
            raise ValueError("missing actual complete request timing")


def native_child(args, output):
    lib = C.CDLL(str(args.library))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
                                     C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(pointer):
        value = C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:
            lib.tc_string_free(pointer)
        return value

    engine, failure = C.c_void_p(), C.c_void_p()
    start = time.perf_counter()
    status = lib.tc_engine_create_model(b"flux2-klein-9b", str(args.model).encode(), C.byref(engine), C.byref(failure))
    if status:
        raise RuntimeError(consume(failure))
    report = dict(engine="native", constructor_seconds=time.perf_counter() - start, runs=[])
    try:
        for index, request in enumerate(requests(output)):
            write_json(output / f"request-{index}.json", request)
            result, failure = C.c_void_p(), C.c_void_p()
            start = time.perf_counter()
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), None, None,
                                            C.byref(result), C.byref(failure))
            wall = time.perf_counter() - start
            text, error = consume(result), consume(failure)
            if status:
                raise RuntimeError(error)
            report["runs"].append(dict(index=index, prompt=request["prompt"], request_wall_including_export=wall,
                                       metrics=json.loads(text)))
            write_json(output / "report.json", report)
    finally:
        lib.tc_engine_free(engine)
    validate_native(report["runs"])
    return report


def mflux_child(args, output):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux / "src"))
    import mlx.core as mx
    import numpy as np
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux_weight_coverage import inspect_reference_coverage

    previous_cache_limit = mx.set_cache_limit(args.reference_cache_mib << 20) if args.reference_cache_mib is not None else None
    started = time.perf_counter()
    model = Flux2Klein(model_path=str(args.model), model_config=ModelConfig.flux2_klein_9b(), quantize=None)
    report = dict(engine="mflux", constructor_seconds=time.perf_counter() - started,
                  precision=str(ModelConfig.precision), quantization=model.bits,
                  reference_cache_limit_bytes=(args.reference_cache_mib << 20) if args.reference_cache_mib is not None else None,
                  previous_reference_cache_limit_bytes=previous_cache_limit,
                  instrumentation="stage-eval-observer" if args.stage_probe else "host-step-counter-only; no extra GPU sync or encoder wrapper",
                  packages={name: importlib.metadata.version(name) for name in
                            ("mlx", "torch", "transformers", "numpy", "Pillow", "safetensors")}, runs=[])
    report["required_weight_coverage"] = inspect_reference_coverage(args.model, Flux2KleinWeightDefinition,
        dict(vae=model.vae, transformer=model.transformer, text_encoder=model.text_encoder))
    if model.bits is not None or str(ModelConfig.precision) != "mlx.core.bfloat16":
        raise ValueError("original BF16, no quantization, required")

    class Steps:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)

    class Stages(Steps):
        def call_before_loop(self, latents, **kwargs):
            mx.eval(latents, self.prompt_embeds)
            self.before = time.perf_counter()

        def call_after_loop(self, latents, **kwargs):
            mx.eval(latents)
            self.after = time.perf_counter()

    stages = Stages() if args.stage_probe else Steps()
    model.callbacks.register(stages)
    # Observe the existing method's returned arrays; do not replace its math,
    # truncate layers, alter tokenizer padding or enable another compile path.
    original_encode = model._encode_prompt_pair

    def observed_encode(*a, **kw):
        values = original_encode(*a, **kw)
        stages.prompt_embeds = values[0]
        return values

    if args.stage_probe:
        model._encode_prompt_pair = observed_encode
    token_rows = []
    for index, request in enumerate(requests(output)):
        stages.steps = []
        write_json(output / f"request-{index}.json", request)
        start = time.perf_counter()
        image = model.generate_image(seed=42, prompt=request["prompt"], num_inference_steps=4,
                                     height=512, width=512, guidance=1.0)
        image.save(request["output"])
        end = time.perf_counter()
        if stages.steps != [0, 1, 2, 3] or image.lora_paths or image.quantization is not None:
            raise ValueError("original actual base four-step execution differs")
        row = dict(index=index, prompt=request["prompt"], request_wall_including_export=end - start,
                   actual_steps=stages.steps, text_rows=512, actual_quantization=image.quantization,
                   loras=image.lora_paths, mlx_peak_bytes=mx.get_peak_memory())
        if args.stage_probe:
            row["timings_seconds"] = dict(encoder_and_noise=stages.before - start,
                denoise=stages.after - stages.before, vae_and_export=end - stages.after)
        report["runs"].append(row)
        write_json(output / "report.json", report)
    # Only after ALL timed generations. Verification must not warm the first
    # GPU/tokenizer call or retain an extra conditioning array during timing.
    for index, request in enumerate(requests(output)):
        token = model.tokenizers["qwen3"].tokenize(request["prompt"], max_length=512)
        mx.eval(token.input_ids, token.attention_mask)
        ids = np.array(token.input_ids).reshape(-1).tolist()
        mask = np.array(token.attention_mask).reshape(-1).tolist()
        expected = model.tokenizers["qwen3"].tokenizer.encode(
            "<|im_start|>user\n" + request["prompt"] +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        expected += [151643] * (512 - len(expected))
        if ids != expected or len(ids) != 512 or any(value not in (0, 1) for value in mask):
            raise ValueError("original tokenizer differs from native template")
        token_rows.append(dict(index=index, ids=ids, valid=sum(mask)))
    write_json(output / "tokenizer-evidence.json", token_rows)
    return report


def parent(args):
    from PIL import Image
    args.output.mkdir(parents=True)
    model_sources = sorted([*args.model.rglob("*.safetensors"), *args.model.rglob("*.json"),
                            *args.model.rglob("*.jinja"), *args.model.rglob("merges.txt")])
    code_sources = sorted((args.mflux / "src").rglob("*.py")) + [args.library, Path(__file__).resolve(),
        ROOT / "tools/validation/mflux_weight_coverage.py"]
    identities = {str(path): generation(path) for path in (*model_sources, *code_sources)}
    # Initial source generation verified once across campaigns; code is small
    # and independently bound each time. No inference-time model payload scan.
    hashes = registered_hashes(model_sources, ROOT / "outputs/benchmark-source-registrations/flux9-bf16.json")
    hashes.update({str(path): sha256_file(path) for path in code_sources})
    reference_sources = code_sources[:-3]
    summary = dict(schema="tc-flux9-bf16-framework-screen-v1", status="running", qualification_passed=False,
                   stage_probe=args.stage_probe,
                   task=dict(model="flux2-klein-9b", size=512, steps=4, seed=42, dynamic_text=False, loras=[]),
                   model_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in model_sources},
                   source_sha256={"libturbocider.dylib": hashes[str(args.library)],
                                  "flux9_bf16_framework_screen.py": hashes[str(Path(__file__).resolve())],
                                  "mflux_weight_coverage.py": hashes[str(ROOT / "tools/validation/mflux_weight_coverage.py")]},
                   mflux_source_files=len(reference_sources),
                   mflux_source_snapshot=hashlib.sha256(json.dumps(
                       [(str(path.relative_to(args.mflux)), hashes[str(path)]) for path in reference_sources],
                       separators=(",", ":")).encode()).hexdigest(),
                   mflux_commit=subprocess.check_output(["git", "-C", str(args.mflux), "rev-parse", "HEAD"], text=True).strip(),
                   order=args.order.split(","), trials=[],
                   reference_cache_mib=args.reference_cache_mib,
                   timing_scope="first process request; two fresh warm prompts; one immediate repeated-prompt request; complete PNG export wall",
                   source_policy="first source generation SHA then durable benchmark metadata reuse; unchanged canonical/stat guard at arm boundaries; no per-inference model hash or OS cache purge",
                   observer_policy="bounded-async-record-fsync-v1" if args.async_evidence else "synchronous-record-fsync-v1")
    write_json(args.output / "summary.json", summary)
    try:
        for index, engine in enumerate(summary["order"]):
            check_generations(identities)
            destination = args.output / f"{index}-{engine}"
            destination.mkdir()
            env = benchmark_environment()
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=str(args.extra), MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            command = [str(args.python), "-B", str(Path(__file__).resolve()), "--child", engine,
                       "--model", str(args.model), "--library", str(args.library), "--mflux", str(args.mflux),
                       "--output", str(destination)]
            if args.stage_probe:
                command.append("--stage-probe")
            if args.reference_cache_mib is not None:
                command += ["--reference-cache-mib", str(args.reference_cache_mib)]
            print(json.dumps(dict(starting=index, engine=engine)), flush=True)
            with (destination / "stdout.txt").open("x") as out, (destination / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=destination, stem="run", env=env, stdout=out,
                                     stderr=err, timeout=900, interval_ms=100, max_gap_ms=500,
                                     async_evidence=args.async_evidence,
                                     child_roles=(("python_helper", str(args.python.resolve(strict=True))),)
                                                 if engine == "mflux" else ())
            check_generations(identities)
            report = json.loads((destination / "report.json").read_text())
            pngs = []
            for request_index in range(4):
                path = destination / f"{request_index}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete output image has wrong dimensions")
                    pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                pngs.append(dict(file_sha256=sha256_file(path), rgb_sha256=pixels))
            summary["trials"].append(dict(engine=engine, index=index, report=report, memory=memory, pngs=pngs))
            write_json(args.output / "summary.json", summary)
        check_generations(identities)
        summary["medians"] = {}
        for engine in sorted(set(summary["order"])):
            trials = [trial for trial in summary["trials"] if trial["engine"] == engine]
            samples = [run["request_wall_including_export"] for trial in trials for run in trial["report"]["runs"][1:3]]
            summary["medians"][engine] = dict(fresh_warm_seconds=statistics.median(samples), samples=samples,
                repeated_prompt_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
                cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials])
        summary["status"] = "complete_diagnostic"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--order", default="native,mflux,mflux,native")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("native", "mflux"))
    parser.add_argument("--stage-probe", action="store_true", help="separate optional observer with eval phase fences; not default whole-request ranking")
    parser.add_argument("--async-evidence", action="store_true", help="same bounded async record writer on both arms; gap/verification gate unchanged")
    parser.add_argument("--reference-cache-mib", type=int, choices=(0, 1024, 4096),
                        help="explicit process-local reference free-cache candidate; unset is stock")
    args = parser.parse_args()
    for field in ("model", "library", "mflux"):
        setattr(args, field, getattr(args, field).resolve(strict=True))
    args.output = args.output.absolute()
    if args.child:
        (native_child if args.child == "native" else mflux_child)(args, args.output)
    else:
        if args.output.exists() or not args.python or not args.extra or args.order.split(",") not in (
                ["native", "mflux", "mflux", "native"], ["native", "mflux"], ["mflux", "native"]):
            parser.error("fresh output, reference Python/extra dependencies and exact serial arm order required")
        args.python = args.python.absolute()  # preserve venv launcher path
        args.extra = args.extra.resolve(strict=True)
        parent(args)


if __name__ == "__main__":
    main()
