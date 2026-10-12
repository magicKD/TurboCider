#!/usr/bin/env python3
"""Current mflux BF16/partial-FP16/Q4 versus retained native GPU/hybrid recipes."""
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tomllib

from flux9_bf16_framework_screen import (
    check_generations, generation, native_child, registered_hashes, requests, validate_native, write_json)
from flux9_matched_affine_screen import validate_pack
from mflux_weight_coverage import inspect_reference_coverage, inspect_saved_full_coverage
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native-bf16", "native-hybrid", "native-q4", "native-predecoded",
          "mflux-bf16", "mflux-fp16", "mflux-q4")
ORDER = (*ROUTES, *reversed(ROUTES))
QUANTIZED = ("native-q4", "native-predecoded", "mflux-q4")


def validate_report(report, route, count=4):
    if route not in ROUTES or report.get("engine") != route or len(report.get("runs", [])) != count:
        raise ValueError("complete declared current-framework request sequence required")
    if route.startswith("native-"):
        if count != 4:
            raise ValueError("native controls require complete four-request sequence")
        validate_native(report["runs"], route == "native-hybrid")
        expected = 96 if route in ("native-bf16", "native-predecoded") else 0
        for index, row in enumerate(report["runs"]):
            if row.get("actual_compiled_single_blocks") != expected or row["native_event_counts"].get("transformer_block") != 128:
                raise ValueError("actual dense compiled/packed/hybrid full9B blocks required")
            if route == "native-predecoded" and (
                    row["native_event_counts"].get("affine_predecoded_gpu_route") != 1 or
                    row["native_event_counts"].get("affine_predecode_projection", 0) != (153 if index == 0 else 0)):
                raise ValueError("actual one-time153 projection predecode required")
        return
    bits = 4 if route == "mflux-q4" else None
    if report.get("precision") != "mlx.core.bfloat16" or report.get("quantization") != bits:
        raise ValueError("actual source/quantization precision differs")
    coverage = report.get("required_weight_coverage", {})
    if set(coverage) != {"transformer", "text_encoder", "vae"} or any(
            type(row.get("required_targets")) is not int or row["required_targets"] <= 0 or
            row.get("covered_targets") != row["required_targets"] or row.get("missing") != []
            for row in coverage.values()):
        raise ValueError("complete actual reference source parameters required")
    if report.get("reference_cache_limit_bytes") != 0:
        raise ValueError("current FLUX reference requires actual retained cache0 candidate")
    reduced = route == "mflux-fp16"
    if report.get("compute_precision_name") != ("float16" if reduced else None):
        raise ValueError("actual upstream compute precision differs from declared route")
    if reduced and (report.get("transformer_floating_parameter_dtypes", {}).get("mlx.core.float16", 0) <= 0 or
                    report.get("transformer_floating_parameter_dtypes", {}).get("mlx.core.bfloat16", 0) <= 0 or
                    report.get("text_embedding_dtype") != "mlx.core.bfloat16"):
        raise ValueError("partial FP16 needs actual cast parameters and BF16 stream/encoder")
    if route == "mflux-q4" and any(
            (coverage[name].get("quantized_linears"), coverage[name].get("bits"), coverage[name].get("group_size")) != (linears, 4, 32)
            for name, linears in (("transformer", 153), ("text_encoder", 252))):
        raise ValueError("complete matched encoder/DiT Q4/g32 parameters required")
    for row in report["runs"]:
        wall = row.get("request_wall_including_export")
        if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite positive full PNG wall required")
        if row.get("actual_steps") != [0, 1, 2, 3] or row.get("actual_quantization") != bits or row.get("loras"):
            raise ValueError("actual full four-step base generation required")
        if row.get("prompt_cache_hit") is not False or row.get("prompt_cache_entries") != 0:
            raise ValueError("stock reference re-encodes prompts, no invented repeat-cache hit")
        if row.get("generation_parameters", {}).get("compute_precision") != ("float16" if reduced else None):
            raise ValueError("actual output compute metadata differs")


def reference_child(args, root):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux / "src"))
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    previous = mx.set_cache_limit(0)
    options = dict(model_path=str(root), model_config=ModelConfig.flux2_klein_9b(), quantize=None)
    if args.child == "mflux-fp16":
        options["compute_precision"] = mx.float16
    started = time.perf_counter()
    model = Flux2Klein(**options)
    constructor = time.perf_counter() - started
    provider = inspect_saved_full_coverage if args.child == "mflux-q4" else inspect_reference_coverage
    coverage = provider(root, Flux2KleinWeightDefinition,
                        dict(transformer=model.transformer, text_encoder=model.text_encoder, vae=model.vae))
    dtypes = {}
    for _, value in tree_flatten(model.transformer.parameters()):
        if mx.issubdtype(value.dtype, mx.floating):
            name = str(value.dtype)
            dtypes[name] = dtypes.get(name, 0) + 1
    report = dict(engine=args.child, constructor_seconds=constructor, precision=str(ModelConfig.precision),
        quantization=model.bits, required_weight_coverage=coverage,
        compute_precision_name=model.compute_precision.name, transformer_floating_parameter_dtypes=dtypes,
        text_embedding_dtype=str(model.text_encoder.embed_tokens.weight.dtype),
        reference_cache_limit_bytes=0, previous_reference_cache_limit_bytes=previous,
        packages={name: importlib.metadata.version(name) for name in
                  ("mflux", "mlx", "torch", "transformers", "numpy", "Pillow", "safetensors")},
        instrumentation="host step counter only; unchanged stock graph, tokenizer and synchronization", runs=[])

    class Counter:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)

    counter = Counter()
    model.callbacks.register(counter)
    tasks = requests(args.output)[:args.count]
    for index, request in enumerate(tasks):
        counter.steps = []
        write_json(args.output / f"request-{index}.json", request)
        before_cache = request["prompt"] in model.prompt_cache
        started = time.perf_counter()
        image = model.generate_image(seed=42, prompt=request["prompt"], num_inference_steps=4,
                                     height=512, width=512, guidance=1.0)
        image.save(request["output"])
        report["runs"].append(dict(index=index, request_wall_including_export=time.perf_counter() - started,
            actual_steps=counter.steps, actual_quantization=image.quantization, loras=image.lora_paths,
            prompt_cache_hit=before_cache, prompt_cache_entries=len(model.prompt_cache),
            generation_parameters=image.generation_parameters, mlx_peak_bytes=mx.get_peak_memory()))
        write_json(args.output / "report.json", report)
    validate_report(report, args.child, args.count)
    token_rows = []
    for request in tasks:
        tokens = model.tokenizers["qwen3"].tokenize(request["prompt"], max_length=512)
        mx.eval(tokens.input_ids, tokens.attention_mask)
        ids, mask = np.array(tokens.input_ids).reshape(-1).tolist(), np.array(tokens.attention_mask).reshape(-1).tolist()
        expected = model.tokenizers["qwen3"].tokenizer.encode(
            "<|im_start|>user\n" + request["prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        expected += [151643] * (512 - len(expected))
        if ids != expected or len(ids) != 512 or any(value not in (0, 1) for value in mask):
            raise ValueError("current stock tokenizer differs from bound original/native input")
        token_rows.append(dict(ids=ids, valid=sum(mask)))
    write_json(args.output / "tokenizer-evidence.json", token_rows)
    return report


def child(args):
    receipt = validate_pack(args.pack, args.model)
    if receipt["converted_components"] != ["transformer", "text_encoder"]:
        raise ValueError("retained full-linear Q4/g32 source required")
    root = args.pack / ("native" if args.child.startswith("native-") else "mflux") if args.child in QUANTIZED else args.model
    if args.child.startswith("native-"):
        call = copy.copy(args)
        call.model = root
        call.native_encoder_eval_interval = 0
        call.native_encoder_fused_attention = True
        call.ane_manifest = args.ane_manifest if args.child == "native-hybrid" else None
        call.child = "native-compiled" if args.child in ("native-bf16", "native-predecoded") else "native-hybrid" if args.child == "native-hybrid" else "native"

        def factory(output, compiled, manifest):
            tasks = requests(output, compiled, manifest)
            if args.child in QUANTIZED:
                for request in tasks:
                    request["allow_approximation"] = True
            return tasks

        report = native_child(call, args.output, request_factory=factory)
    else:
        report = reference_child(args, root)
    report.update(engine=args.child, model_root=str(root),
                  source_representation="encoder+DiT eligible linears Q4/g32, dense BF16 embedding/VAE" if args.child in QUANTIZED else "original BF16 source",
                  quantized_components=receipt["converted_components"] if args.child in QUANTIZED else [])
    validate_report(report, args.child, args.count)
    write_json(args.output / "report.json", report)


def parent(args):
    from PIL import Image
    receipt = validate_pack(args.pack, args.model)
    if receipt["converted_components"] != ["transformer", "text_encoder"]:
        raise ValueError("complete full-linear stored source required")
    order = args.order.split(",")
    if args.count == 4 and tuple(order) != ORDER:
        raise ValueError("full current-framework experiment requires all seven matched routes in exact palindrome")
    if args.count == 1 and any(route.startswith("native-") for route in order):
        raise ValueError("short precheck is reference-only, never a complete native ranking")
    if "native-hybrid" in order and not args.ane_manifest:
        raise ValueError("retained complete compiled hybrid manifest required")
    sources = sorted([*args.model.rglob("*.safetensors"), *args.model.rglob("*.json"), *args.model.rglob("*.jinja"),
        *(args.pack / layout / component / "0.safetensors" for layout in ("native", "mflux") for component in receipt["converted_components"]),
        args.pack / "conversion.json"])
    reference = sorted((args.mflux / "src").rglob("*.py")) + [args.mflux / "pyproject.toml"]
    local = [args.library, Path(__file__).resolve(), ROOT / "tools/validation/flux9_bf16_framework_screen.py",
        ROOT / "tools/validation/flux9_matched_affine_screen.py", ROOT / "tools/validation/mflux_weight_coverage.py",
        ROOT / "tools/validation/mflux_original_layout.py"]
    ane = []
    if "native-hybrid" in order:
        manifest = json.loads(args.ane_manifest.read_text())
        sys.path.insert(0, str(ROOT / "tools/coreml"))
        from compile_flux_manifest import validate_manifest
        validate_manifest(manifest, "flux2-klein-9b")
        shape = manifest["shape"]
        if (shape.get("buckets"), shape.get("ane_mlp_start"), shape.get("ane_mlp_end")) != ([1536], 0, 4096):
            raise ValueError("retained exact4096-prefix/1536-row9B hybrid required")
        ane = [args.ane_manifest]
        for block in manifest["artifacts"].values():
            path = (args.ane_manifest.parent / block["int8_pc"]).resolve(strict=True)
            if path.suffix != ".mlmodelc" or not path.is_dir():
                raise ValueError("already-compiled retained CoreML partitions required")
            ane.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    identities = {str(path): generation(path) for path in (*sources, *reference, *local, *ane)}
    hashes = registered_hashes(sources, ROOT / "outputs/benchmark-source-registrations/flux9-bf16.json")
    hashes.update(registered_hashes(ane, ROOT / "outputs/benchmark-source-registrations/flux9-hybrid.json"))
    hashes.update({str(path): sha256_file(path) for path in (*reference, *local)})
    for layout in ("native", "mflux"):
        for component in receipt["converted_components"]:
            if hashes[str(args.pack / layout / component / "0.safetensors")] != receipt["components"][component][layout + "_sha256"]:
                raise ValueError("actual complete matched compressed source differs")
    if any(hashes.get(name) != value for name, value in receipt["source_sha256"].items()):
        raise ValueError("original source differs from retained conversion")
    version = tomllib.loads((args.mflux / "pyproject.toml").read_text())["project"]["version"]
    if version != "0.22.0":
        raise ValueError("current experiment requires verified0.22.0 checkout")
    args.output.mkdir(parents=True)
    data = dict(schema="tc-flux9-current-framework-screen-v1", status="running", goal_complete=False,
        qualification_passed=False, count=args.count, order=order, trials=[],
        task=dict(model="flux2-klein-9b", width=512, height=512, steps=4, seed=42, dynamic_text=False, loras=[]),
        model_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in sources},
        local_source_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in local},
        mflux_source_sha256={str(path.relative_to(args.mflux)): hashes[str(path)] for path in reference},
        ane_source_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in ane},
        ane_manifest=str(args.ane_manifest) if args.ane_manifest else None,
        mflux_package_version=version,
        mflux_commit=subprocess.check_output(["git", "-C", str(args.mflux), "rev-parse", "HEAD"], text=True).strip(),
        conversion=receipt, reference_cache_mib=0,
        timing_scope="complete PNG request wall; constructor separate; cold/two fresh/repeat, no added phase fences",
        source_policy="first source-generation SHA then durable registration reuse; stat guards between arms, no per-inference model SHA or OS cache purge",
        representation_scope="same base checkpoint; BF16 source versus matched full-linear Q4/g32; partial FP16, BF16 predecode rounding and INT8-weight/FP16-I/O hybrid explicitly separate arithmetic")
    target = args.output / "summary.json"
    write_json(target, data)
    try:
        for index, route in enumerate(order):
            check_generations(identities)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = benchmark_environment()
            for key in tuple(env):
                if key.startswith(("TURBOCIDER_FLUX_", "TURBOCIDER_FLUX9_", "TURBOCIDER_QWEN3_")):
                    del env[key]
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=str(args.extra), MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            if route.startswith("native-"):
                env.update(TURBOCIDER_QWEN3_EVAL_INTERVAL="0", TURBOCIDER_QWEN3_FUSED_SDPA="1")
            if route == "native-predecoded":
                env["TURBOCIDER_FLUX9_PREDECODE_AFFINE"] = "1"
            if route == "native-hybrid":
                env["TURBOCIDER_FLUX9_EXPERIMENTAL_HYBRID"] = "1"
            command = [str(args.python), "-B", str(Path(__file__).resolve()), "--model", str(args.model),
                "--pack", str(args.pack), "--library", str(args.library), "--mflux", str(args.mflux),
                "--output", str(output), "--child", route, "--count", str(args.count)]
            if route == "native-hybrid":
                command.extend(["--ane-manifest", str(args.ane_manifest)])
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=900, interval_ms=100, max_gap_ms=500, async_evidence=True,
                    child_roles=(("python_helper", str(args.python.resolve(strict=True))),) if route.startswith("mflux-") else ())
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            validate_report(report, route, args.count)
            pngs = []
            for request_index in range(args.count):
                path = output / f"{request_index}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG" or image.size != (512, 512):
                        raise ValueError("complete full-resolution PNG required")
                pngs.append(sha256_file(path))
            data["trials"].append(dict(index=index, route=route, report=report, memory=memory, png_sha256=pngs))
            write_json(target, data)
        if args.count == 4:
            data["medians"] = {}
            for route in dict.fromkeys(order):
                trials = [trial for trial in data["trials"] if trial["route"] == route]
                samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
                data["medians"][route] = dict(samples=samples, fresh_warm_seconds=statistics.median(samples),
                    cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                    repeated_prompt_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
                    constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials])
        data["status"] = "complete_diagnostic" if args.count == 4 else "complete_precheck_only"
    except BaseException as error:
        data.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "pack", "library", "mflux", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--ane-manifest", type=Path)
    parser.add_argument("--count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--order", default=",".join(ORDER))
    parser.add_argument("--child", choices=ROUTES)
    args = parser.parse_args()
    for name in ("model", "pack", "library", "mflux"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.ane_manifest:
        args.ane_manifest = args.ane_manifest.resolve(strict=True)
    args.output = args.output.absolute()
    if args.child:
        child(args)
    else:
        if args.output.exists() or not args.python or not args.extra or any(route not in ROUTES for route in args.order.split(",")):
            parser.error("fresh output, declared routes and isolated reference environment required")
        args.python = args.python.absolute()
        args.extra = args.extra.resolve(strict=True)
        parent(args)


if __name__ == "__main__":
    main()
