#!/usr/bin/env python3
"""Run repeatable end-to-end benchmarks through one persistent engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flux2_engine import (
    EngineConfig,
    ExecutionMode,
    Flux2Engine,
    GenerationRequest,
    ModelVariant,
)


def _defaults() -> tuple[Path, Path, Path, tuple[Path, ...]]:
    engine_root = Path(__file__).resolve().parents[1]
    image_root = engine_root.parent
    project_root = image_root.parent
    manifests = tuple(sorted((project_root / "mac_local_ai/models/coreml").glob("flux2_stack_runtime*/manifest.json")))
    return (
        image_root / "models/FLUX.2-klein-4B",
        image_root / "mflux",
        image_root / "outputs/flux2-engine/benchmark",
        manifests,
    )


def main() -> None:
    default_model, default_mflux, default_output, default_manifests = _defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--mode", type=ExecutionMode, choices=list(ExecutionMode), default=ExecutionMode.MLX_ANE)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument(
        "--model-variant",
        type=ModelVariant,
        choices=list(ModelVariant),
        default=ModelVariant.AUTO,
    )
    parser.add_argument("--mflux-root", type=Path, default=default_mflux)
    parser.add_argument("--ane-manifest", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ane-block", type=int, action="append")
    parser.add_argument(
        "--reuse-ane-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-quantized-gpu-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clear-mlx-cache-between-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamic-text-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")

    manifests = tuple(args.ane_manifest or default_manifests)
    config = EngineConfig(
        model_path=args.model,
        mflux_root=args.mflux_root,
        model_variant=args.model_variant,
        mode=args.mode,
        ane_manifests=manifests,
        persistent=True,
        ane_blocks=tuple(args.ane_block) if args.ane_block else None,
        reuse_ane_outputs=args.reuse_ane_outputs,
        compile_quantized_gpu_attention=args.compile_quantized_gpu_attention,
        clear_mlx_cache_between_requests=args.clear_mlx_cache_between_requests,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with Flux2Engine(config) as engine:
        for index in range(1, args.repeats + 1):
            output = args.output_dir / f"{args.mode}-{args.width}x{args.height}-{index}.png"
            result = engine.generate(
                GenerationRequest(
                    prompt=args.prompt,
                    output=output,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    seed=args.seed,
                    dynamic_text_length=args.dynamic_text_length,
                )
            )
            report = result.metrics.as_dict()
            report["index"] = index
            report["output"] = str(result.output)
            reports.append(report)
            # Do not keep a 1024+ decoded image alive while the next request runs.
            del result

    rendered = json.dumps(reports, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
