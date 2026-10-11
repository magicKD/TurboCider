#!/usr/bin/env python3
"""Stock mflux FLUX9B BF16/Q4/Q8 full-image reference sweeps, not native ranks."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from flux9_bf16_framework_screen import PROMPTS, generation, check_generations, write_json
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
BITS = {"bf16": None, "q4": 4, "q8": 8}


def validate_report(report, mode):
    if mode not in BITS or report["quantization"] != BITS[mode] or len(report["runs"]) != 4:
        raise ValueError("actual full stock precision/request sequence differs")
    for row in report["runs"]:
        if row["actual_steps"] != [0, 1, 2, 3] or row["actual_quantization"] != BITS[mode] or row["loras"]:
            raise ValueError("actual base four-step work required")


def child(args):
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux_weight_coverage import inspect_reference_coverage
    import mlx.core as mx
    start = time.perf_counter()
    model = Flux2Klein(model_path=str(args.model), model_config=ModelConfig.flux2_klein_9b(), quantize=BITS[args.child])
    constructor = time.perf_counter() - start
    coverage = inspect_reference_coverage(args.model, Flux2KleinWeightDefinition,
        dict(vae=model.vae, transformer=model.transformer, text_encoder=model.text_encoder))
    report = dict(quantization=model.bits, constructor_seconds=constructor, precision=str(ModelConfig.precision),
        required_weight_coverage=coverage,
        packages={name: importlib.metadata.version(name) for name in ("mlx", "torch", "transformers", "numpy", "Pillow", "safetensors")},
        source_recipe="stock mflux quantize argument; includes eligible text/DiT modules; not equal to native BF16 or GGUF recipes", runs=[])
    class Counter:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)
    counter = Counter()
    model.callbacks.register(counter)
    for index, prompt in enumerate(PROMPTS):
        counter.steps = []
        request = dict(model="flux2-klein-9b", operation="image.generate", prompt=prompt,
            width=512, height=512, steps=4, seed=42, guidance=1.0, quantize=BITS[args.child], output=str(args.output / f"{index}.png"))
        write_json(args.output / f"request-{index}.json", request)
        start = time.perf_counter()
        image = model.generate_image(seed=42, prompt=prompt, num_inference_steps=4, height=512, width=512, guidance=1.0)
        image.save(request["output"])
        report["runs"].append(dict(index=index, request_wall_including_export=time.perf_counter() - start,
            actual_steps=counter.steps, actual_quantization=image.quantization, loras=image.lora_paths, mlx_peak_bytes=mx.get_peak_memory()))
        write_json(args.output / "report.json", report)
    validate_report(report, args.child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "mflux", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--order", default="bf16,q4,q8,q8,q4,bf16")
    parser.add_argument("--child", choices=tuple(BITS))
    args = parser.parse_args()
    args.model, args.mflux, args.output = args.model.resolve(strict=True), args.mflux.resolve(strict=True), args.output.absolute()
    if args.child:
        child(args)
        return
    order = args.order.split(",")
    if args.output.exists() or not args.python or not args.extra or any(name not in BITS for name in order):
        parser.error("fresh output and known serial stock reference precision routes required")
    args.python, args.extra = args.python.absolute(), args.extra.resolve(strict=True)
    args.output.mkdir(parents=True)
    sources = sorted([*args.model.rglob("*.safetensors"), *args.model.rglob("*.json"), *args.model.rglob("*.jinja"),
                      *args.mflux.joinpath("src").rglob("*.py"), Path(__file__).resolve(), ROOT / "tools/validation/mflux_weight_coverage.py"])
    identities = {str(path): generation(path) for path in sources}
    hashes = {str(path): sha256_file(path) for path in sources}
    result = dict(schema="tc-flux9-mflux-stock-quant-screen-v1", status="running", qualification_passed=False,
        order=order, source_sha256=hashes, trials=[], scope="stock same-original local BF16 source; reference precision sweep; no native equal-precision ranking claim")
    target = args.output / "summary.json"
    write_json(target, result)
    try:
        for index, mode in enumerate(order):
            check_generations(identities)
            output = args.output / f"{index}-{mode}"
            output.mkdir()
            env = benchmark_environment()
            env.update(PYTHONPATH=str(args.extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            command = [str(args.python), "-B", str(Path(__file__).resolve()), "--model", str(args.model),
                       "--mflux", str(args.mflux), "--output", str(output), "--child", mode]
            print(json.dumps(dict(starting=index, mode=mode)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=900, interval_ms=100, max_gap_ms=500, async_evidence=True,
                    child_roles=(("python_helper", str(args.python.resolve(strict=True))),))
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            validate_report(report, mode)
            from PIL import Image
            images = []
            for request in range(4):
                path = output / f"{request}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete full-size PNG required")
                images.append(sha256_file(path))
            result["trials"].append(dict(index=index, mode=mode, report=report, memory=memory, png_sha256=images))
            write_json(target, result)
        result["routes"] = []
        for mode in sorted(set(order)):
            trials = [trial for trial in result["trials"] if trial["mode"] == mode]
            warm = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
            result["routes"].append(dict(mode=mode, fresh_warm_seconds=warm, fresh_warm_median=statistics.median(warm)))
        result["status"] = "complete_diagnostic"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, result)


if __name__ == "__main__":
    main()
