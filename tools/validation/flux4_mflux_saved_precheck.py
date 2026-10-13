#!/usr/bin/env python3
"""Actual FLUX4B upstream saved-Q4 full-image precheck, not a speed ranking."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from flux9_bf16_framework_screen import PROMPTS, write_json
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file


def saved_coverage(headers, descriptions):
    expected = dict(descriptions)
    if "rotary_emb.inv_freq" not in headers:
        expected.pop("rotary_emb.inv_freq", None)
    if set(expected) != set(headers):
        raise ValueError("all actual saved parameters must be loaded exactly")
    for key, value in expected.items():
        if headers[key]["shape"] != value["shape"] or headers[key]["dtype"] != value["dtype"]:
            raise ValueError("saved parameter shape/dtype differs: " + key)
        if value["dtype"] not in ("BF16", "U32") and not (key == "rotary_emb.inv_freq" and value["dtype"] == "F32"):
            raise ValueError("unsupported learned parameter dtype")
    return dict(required_targets=len(expected), covered_targets=len(expected), missing=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "mflux", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-count", type=int, choices=(1, 4), default=1,
                        help="four requests collect cold/two fresh/repeat reference-only timing")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh precheck output required")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux.resolve() / "src"))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    args.output.mkdir(parents=True)
    mx.set_cache_limit(0)
    started = time.perf_counter()
    model = Flux2Klein(model_path=str(args.model.resolve()), model_config=ModelConfig.flux2_klein_4b(), quantize=None)
    constructor = time.perf_counter() - started
    if model.bits != 4 or str(ModelConfig.precision) != "mlx.core.bfloat16":
        raise ValueError("actual saved Q4/BF164B recipe required")
    coverage = {}
    dtype_names = {str(mx.bfloat16): "BF16", str(mx.uint32): "U32"}
    for name in ("transformer", "text_encoder", "vae"):
        headers = {}
        for path in sorted((args.model / name).glob("*.safetensors")):
            header, _ = read_header(path)
            for key, value in header.items():
                if key != "__metadata__":
                    if key in headers:
                        raise ValueError("duplicate saved parameter")
                    headers[key] = value
        params = dict(tree_flatten(getattr(model, name).parameters()))
        # Upstream saved packs may retain this F32 architecture constant;
        # older derived packs omit it. Never discard a stored constant from
        # source coverage just because another recipe generated it locally.
        if "rotary_emb.inv_freq" not in headers:
            params.pop("rotary_emb.inv_freq", None)
        if set(params) != set(headers):
            write_json(args.output / "report.json", dict(status="failed_required_parameter_coverage", runtime_verified=False,
                performance_verified=False, component=name, actual_parameter_count=len(params), saved_parameter_count=len(headers),
                missing_saved=sorted(set(params)-set(headers)), unexpected_saved=sorted(set(headers)-set(params))))
            raise ValueError("all actual saved parameters must be loaded exactly: " + name)
        descriptions = {key: dict(shape=list(value.shape),
            dtype="F32" if key == "rotary_emb.inv_freq" and value.dtype == mx.float32 else dtype_names.get(str(value.dtype)))
            for key, value in params.items()}
        coverage[name] = saved_coverage(headers, descriptions)

    class Counter:
        def call_in_loop(self, t, **kwargs):
            self.steps.append(t)

    counter = Counter()
    model.callbacks.register(counter)
    result = dict(status="running", scope="one cold complete4B/Q4/512/4 PNG only, not fresh median or ranking" if args.request_count == 1 else
        "four complete reference-only PNGs: cold/two fresh/repeated prompt new seed; native best reused, no new paired ranking",
        model="flux2-klein-4b", constructor_seconds=constructor, required_weight_coverage=coverage,
        packages={name: importlib.metadata.version(name) for name in ("mflux", "mlx", "torch", "transformers")},
        runtime_verified=False, performance_verified=False, runs=[])
    write_json(args.output / "report.json", result)
    try:
        for index, prompt in enumerate(PROMPTS[:args.request_count]):
            counter.steps = []
            seed = 43 if index == 3 else 42
            started = time.perf_counter()
            image = model.generate_image(seed=seed, prompt=prompt, num_inference_steps=4, height=512, width=512, guidance=1.)
            image.save(str(args.output / f"{index}.png"))
            wall = time.perf_counter() - started
            if counter.steps != [0, 1, 2, 3] or image.quantization != 4 or image.lora_paths:
                raise ValueError("actual complete four-step base request differs")
            row = dict(index=index, prompt=prompt, seed=seed, request_wall_including_export=wall,
                actual_steps=counter.steps, actual_quantization=image.quantization, loras=image.lora_paths,
                png_sha256=sha256_file(args.output / f"{index}.png"), mlx_peak_bytes=mx.get_peak_memory())
            result["runs"].append(row)
            write_json(args.output / "report.json", result)
        result.update(status="complete_precheck_only" if args.request_count == 1 else "complete_reference_only", runtime_verified=True)
        if args.request_count == 1:
            result.update({k:v for k,v in result["runs"][0].items() if k not in ("index", "prompt", "seed")})
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
