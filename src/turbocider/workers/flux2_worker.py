#!/usr/bin/env python3
"""Persistent JSON-lines worker for FLUX.2 MLX and MLX+ANE generation."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from flux2_engine import (
    AttentionMode,
    EngineConfig,
    ExecutionMode,
    Flux2Engine,
    GenerationRequest,
    ModelVariant,
)
from flux2_engine.config import GenerationProfile


def emit_progress(phase: str, completed: int, total: int = 1) -> None:
    print(
        "phase=%s %d/%d" % (phase, completed, total),
        file=sys.stderr,
        flush=True,
    )


class DenoiseProgress:
    def call_before_loop(self, config, **kwargs) -> None:
        emit_progress("denoise", 0, int(config.num_inference_steps))

    def call_in_loop(self, t, config, **kwargs) -> None:
        emit_progress("denoise", int(t) + 1, int(config.num_inference_steps))

    def call_after_loop(self, config, **kwargs) -> None:
        emit_progress("decode", 0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--model", type=Path, required=True)
    root.add_argument(
        "--model-variant",
        type=ModelVariant,
        choices=list(ModelVariant),
        default=ModelVariant.AUTO,
    )
    root.add_argument("--mflux-root", type=Path, required=True)
    root.add_argument("--mode", type=ExecutionMode, choices=list(ExecutionMode), required=True)
    root.add_argument(
        "--attention",
        type=AttentionMode,
        choices=list(AttentionMode),
        default=AttentionMode.AUTO,
    )
    root.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    root.add_argument("--ane-manifest", type=Path, action="append", default=[])
    root.add_argument("--bridge-dir", type=Path)
    root.add_argument("--mlx-eager", action="store_true")
    root.add_argument("--ane-variant", choices=("int8_pc", "fp16"), default="int8_pc")
    root.add_argument("--ane-block", type=int, action="append")
    root.add_argument(
        "--reuse-ane-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    root.add_argument(
        "--compile-quantized-gpu-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    root.add_argument(
        "--clear-mlx-cache-between-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return root


def main() -> None:
    args = parser().parse_args()
    config = EngineConfig(
        model_path=args.model,
        mflux_root=args.mflux_root,
        model_variant=args.model_variant,
        mode=args.mode,
        attention=args.attention,
        precision=args.precision,
        ane_manifests=tuple(args.ane_manifest),
        bridge_dir=args.bridge_dir,
        persistent=True,
        mlx_eager=args.mlx_eager,
        ane_variant=args.ane_variant,
        ane_blocks=tuple(args.ane_block) if args.ane_block else None,
        reuse_ane_outputs=args.reuse_ane_outputs,
        compile_quantized_gpu_attention=args.compile_quantized_gpu_attention,
        clear_mlx_cache_between_requests=args.clear_mlx_cache_between_requests,
    )
    request_index = 0
    emit_progress("model_load", 0)
    with Flux2Engine(config) as engine:
        emit_progress("model_load", 1)
        model = getattr(engine.backend, "model", None)
        if model is not None and hasattr(model, "callbacks"):
            model.callbacks.register(DenoiseProgress())
        for raw in sys.stdin:
            if not raw.strip():
                continue
            request_index += 1
            try:
                payload = json.loads(raw)
                output = Path(payload["output"]).expanduser().resolve()
                result = engine.generate(GenerationRequest(
                    prompt=str(payload["prompt"]),
                    output=output,
                    width=int(payload["width"]),
                    height=int(payload["height"]),
                    steps=int(payload["steps"]) if payload.get("steps") is not None else None,
                    seed=int(payload["seed"]),
                    guidance=float(payload.get("guidance", 1.0)),
                    profile=GenerationProfile(str(payload.get("profile", "quality"))),
                    dynamic_text_length=bool(payload.get("dynamic_text_length", True)),
                ))
                report = result.metrics.as_dict()
                report["output"] = str(result.output) if result.output else None
                report["policy_reason"] = engine.decision.reason
                sidecar = output.with_suffix(".engine.json")
                sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                emit_progress("decode", 1)
                response = {
                    "ok": True,
                    "worker_request_index": request_index,
                    "metrics": report,
                }
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {
                    "ok": False,
                    "worker_request_index": request_index,
                    "error": "%s: %s" % (type(error).__name__, error),
                }
            print(
                "turbocider_worker_result=" + json.dumps(response),
                flush=True,
            )


if __name__ == "__main__":
    main()
