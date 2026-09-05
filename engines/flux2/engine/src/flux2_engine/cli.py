"""Command-line interface for one-shot and persistent-style FLUX.2 generation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from flux2_engine.attention import analyze_attention_candidates
from flux2_engine.config import (
    AttentionMode,
    EngineConfig,
    ExecutionMode,
    GenerationProfile,
    GenerationRequest,
    ModelVariant,
    PipelineMode,
)
from flux2_engine.engine import Flux2Engine
from flux2_engine.errors import Flux2EngineError


def _emit_progress(phase: str, completed: int, total: int = 1) -> None:
    print("phase=%s %d/%d" % (phase, completed, total), flush=True)


class _DenoiseProgress:
    def call_before_loop(self, config, **kwargs) -> None:
        _emit_progress("denoise", 0, int(config.num_inference_steps))

    def call_in_loop(self, t, config, **kwargs) -> None:
        _emit_progress("denoise", int(t) + 1, int(config.num_inference_steps))

    def call_after_loop(self, config, **kwargs) -> None:
        _emit_progress("decode", 0)


def _defaults() -> tuple[Path, Path, Path]:
    engine_root = Path(__file__).resolve().parents[2]
    image_root = engine_root.parent
    return image_root / "models" / "FLUX.2-klein-4B", image_root / "mflux", image_root / "outputs"


def _discover_manifests(mflux_root: Path) -> tuple[Path, ...]:
    project_root = mflux_root.resolve().parents[1]
    coreml_root = project_root / "mac_local_ai" / "models" / "coreml"
    if not coreml_root.is_dir():
        return ()
    return tuple(sorted(coreml_root.glob("flux2_stack_runtime*/manifest.json")))


def build_parser() -> argparse.ArgumentParser:
    default_model, default_mflux, default_outputs = _defaults()
    parser = argparse.ArgumentParser(prog="flux2-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate an image")
    generate.add_argument("prompt")
    generate.add_argument("--model", type=Path, default=default_model)
    generate.add_argument(
        "--model-variant",
        type=ModelVariant,
        choices=list(ModelVariant),
        default=ModelVariant.AUTO,
    )
    generate.add_argument("--mflux-root", type=Path, default=default_mflux)
    generate.add_argument(
        "--pipeline", type=PipelineMode, choices=list(PipelineMode),
        default=PipelineMode.STANDARD,
    )
    generate.add_argument("--image-path", type=Path)
    generate.add_argument("--image-paths", type=Path, nargs="+")
    generate.add_argument("--image-strength", type=float, default=0.75)
    generate.add_argument("--output", type=Path, default=default_outputs / "flux2-engine.png")
    generate.add_argument("--mode", type=ExecutionMode, choices=list(ExecutionMode), default=ExecutionMode.AUTO)
    generate.add_argument(
        "--attention",
        type=AttentionMode,
        choices=list(AttentionMode),
        default=AttentionMode.AUTO,
    )
    generate.add_argument(
        "--profile",
        type=GenerationProfile,
        choices=list(GenerationProfile),
        default=GenerationProfile.QUALITY,
    )
    generate.add_argument("--width", type=int, default=512)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--steps", type=int)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--guidance", type=float, default=1.0)
    generate.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    generate.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=False)
    generate.add_argument("--mlx-eager", action=argparse.BooleanOptionalAction, default=False)
    generate.add_argument(
        "--dynamic-text-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate.add_argument(
        "--reuse-ane-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate.add_argument(
        "--compile-quantized-gpu-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate.add_argument(
        "--clear-mlx-cache-between-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate.add_argument("--ane-manifest", type=Path, action="append", default=[])
    generate.add_argument("--bridge-dir", type=Path)
    generate.add_argument("--ane-block", type=int, action="append")
    generate.add_argument("--ane-variant", choices=("int8_pc", "fp16"), default="int8_pc")
    generate.add_argument("--json", type=Path, help="Optional metrics sidecar path")

    doctor = subparsers.add_parser("doctor", help="Inspect local runtime and artifact availability")
    doctor.add_argument("--model", type=Path, default=default_model)
    doctor.add_argument(
        "--model-variant",
        type=ModelVariant,
        choices=list(ModelVariant),
        default=ModelVariant.AUTO,
    )
    doctor.add_argument("--mflux-root", type=Path, default=default_mflux)
    doctor.add_argument("--bridge-dir", type=Path)
    doctor.add_argument("--ane-manifest", type=Path, action="append", default=[])

    attention_plan = subparsers.add_parser(
        "attention-plan",
        help="Report GPU/ANE attention placement traffic and readiness gates",
    )
    attention_plan.add_argument("--width", type=int, default=512)
    attention_plan.add_argument("--height", type=int, default=512)
    attention_plan.add_argument("--text-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor":
        _doctor(args)
        return
    if args.command == "attention-plan":
        candidates = analyze_attention_candidates(
            width=args.width,
            height=args.height,
            text_tokens=args.text_tokens,
        )
        print(json.dumps([candidate.as_dict() for candidate in candidates], indent=2))
        return
    manifests = tuple(args.ane_manifest) or _discover_manifests(args.mflux_root)
    config = EngineConfig(
        model_path=args.model,
        mflux_root=args.mflux_root,
        model_variant=args.model_variant,
        pipeline=args.pipeline,
        mode=args.mode,
        attention=args.attention,
        precision=args.precision,
        mlx_eager=args.mlx_eager,
        ane_manifests=manifests,
        bridge_dir=args.bridge_dir,
        persistent=args.persistent,
        ane_variant=args.ane_variant,
        ane_blocks=tuple(args.ane_block) if args.ane_block else None,
        reuse_ane_outputs=args.reuse_ane_outputs,
        compile_quantized_gpu_attention=args.compile_quantized_gpu_attention,
        clear_mlx_cache_between_requests=args.clear_mlx_cache_between_requests,
    )
    request = GenerationRequest(
        prompt=args.prompt,
        output=args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        profile=args.profile,
        dynamic_text_length=args.dynamic_text_length,
        image_path=args.image_path,
        image_paths=tuple(args.image_paths or ()),
        image_strength=args.image_strength,
    )
    try:
        _emit_progress("model_load", 0)
        with Flux2Engine(config) as engine:
            _emit_progress("model_load", 1)
            model = getattr(engine.backend, "model", None)
            if model is not None and hasattr(model, "callbacks"):
                model.callbacks.register(_DenoiseProgress())
            result = engine.generate(request)
            _emit_progress("decode", 1)
            report = result.metrics.as_dict()
            report["output"] = str(result.output) if result.output else None
            report["policy_reason"] = engine.decision.reason
    except (Flux2EngineError, ValueError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(report, indent=2) + "\n"
    sidecar = args.json or args.output.with_suffix(".engine.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(rendered)
    print(rendered, end="")


def _doctor(args: argparse.Namespace) -> None:
    bridge_dirs = [
        args.bridge_dir,
        Path(__file__).resolve().parents[2] / "build",
        args.mflux_root.resolve().parents[1] / "mac_local_ai" / "build",
    ]
    bridge_dirs = [path for path in bridge_dirs if path is not None]
    manifests = tuple(args.ane_manifest) or _discover_manifests(args.mflux_root)
    config = EngineConfig(
        model_path=args.model,
        mflux_root=args.mflux_root,
        model_variant=args.model_variant,
    )
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "model": str(args.model.resolve()),
        "model_exists": args.model.is_dir(),
        "model_variant": config.resolved_model_variant.value,
        "mflux_root": str(args.mflux_root.resolve()),
        "mflux_source_exists": (args.mflux_root / "src" / "mflux").is_dir(),
        "native_bridge": next(
            (str(path) for path in bridge_dirs if path.is_dir() and any(path.glob("_flux2_ane_bridge*.so"))),
            None,
        ),
        "ane_manifests": [str(path) for path in manifests],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
