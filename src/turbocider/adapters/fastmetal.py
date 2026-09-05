"""Adapter for FastMetal-QAD's Apple Silicon MLX/Core ML runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from turbocider.adapters.base import (
    CommandSpec,
    EngineAdapter,
    append_extra_args,
    extra_environment,
)
from turbocider.errors import EngineUnavailableError, ValidationError
from turbocider.models import ExecutionMode, ExecutionPlan, GenerationRequest, ModelDescriptor


class FastMetalAdapter(EngineAdapter):
    name = "fastmetal"

    def build_command(
        self,
        model: ModelDescriptor,
        request: GenerationRequest,
        plan: ExecutionPlan,
        output_path: Path,
    ) -> CommandSpec:
        options = request.engine_options.get("fastmetal", {})
        if not isinstance(options, dict):
            raise ValidationError("engine_options.fastmetal must be an object")

        python = Path(str(options.get("python_path") or model.config.get("python_path", "")))
        engine_root = Path(str(model.config.get("engine_root", "")))
        script = Path(str(options.get("script_path") or model.config.get("script_path", "")))
        model_path = Path(str(options.get("model_path") or model.config.get("model_path", "")))
        required = {
            "python": python,
            "engine_root": engine_root,
            "script": script,
            "model": model_path,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise EngineUnavailableError(
                "FastMetal configuration is incomplete: %s" % ", ".join(missing)
            )
        if request.output.frames % 4 != 1:
            raise ValidationError("FastMetal frames must satisfy frames = 4n + 1")
        if request.output.width % 16 or request.output.height % 16:
            raise ValidationError("FastMetal width and height must be divisible by 16")
        if request.sampling.steps is not None and request.sampling.steps != 3:
            raise ValidationError(
                "FastMetal-1.3B-QAD uses its fixed three-step DMD schedule; "
                "sampling.steps must be omitted or set to 3"
            )
        if request.sampling.guidance is not None:
            raise ValidationError("FastMetal-1.3B-QAD does not expose guidance")

        decode_backend = str(options.get("decode_backend", "taehv"))
        if decode_backend not in {"taehv", "wan-vae"}:
            raise ValidationError(
                "engine_options.fastmetal.decode_backend must be taehv or wan-vae"
            )
        text_dtype = str(options.get("text_encoder_dtype", "bf16"))
        if text_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValidationError(
                "engine_options.fastmetal.text_encoder_dtype must be bf16, fp16, or fp32"
            )

        metrics_path = output_path.with_suffix(".engine.json")
        worker = Path(__file__).resolve().parents[1] / "workers" / "fastmetal_worker.py"
        engine_argv: List[str] = [
            str(script),
            "--model-root", str(model_path),
            "--mlx-checkpoint", str(model_path),
            "--prompt", request.prompt,
            "--output-path", str(output_path),
            "--metrics-json", str(metrics_path),
            "--height", str(request.output.height),
            "--width", str(request.output.width),
            "--num-frames", str(request.output.frames),
            "--fps", str(request.output.fps),
            "--seed", str(request.sampling.seed),
            "--denoising-mode", "dmd",
            "--dmd-denoising-steps", "1000,757,522",
            "--mlx-dtype", "fp16",
            "--mlx-quantization", "int8",
            "--mlx-compile",
            "--decode-backend", decode_backend,
            "--text-encoder-dtype", text_dtype,
        ]
        if decode_backend == "wan-vae":
            vae_dtype = str(options.get("vae_decode_dtype", "bf16"))
            if vae_dtype not in {"bf16", "fp16", "fp32"}:
                raise ValidationError(
                    "engine_options.fastmetal.vae_decode_dtype must be bf16, fp16, or fp32"
                )
            engine_argv.extend(["--vae-decode-dtype", vae_dtype])
        if bool(options.get("taehv_parallel", False)):
            engine_argv.append("--taehv-parallel")
        prompt_cache = options.get("prompt_embeds_cache")
        if prompt_cache:
            engine_argv.extend(["--prompt-embeds-cache", str(prompt_cache)])

        if plan.execution is ExecutionMode.GPU_ANE:
            manifest = Path(str(options.get("ane_manifest") or model.config.get("ane_manifest", "")))
            bridge_dir = Path(str(options.get("bridge_dir") or model.config.get("bridge_dir", "")))
            if not manifest.is_file() or not bridge_dir.is_dir():
                raise EngineUnavailableError(
                    "FastMetal GPU+ANE requires a compiled manifest and native bridge"
                )
            engine_argv.extend([
                "--ane-manifest", str(manifest),
                "--ane-bridge-dir", str(bridge_dir),
            ])
        append_extra_args(engine_argv, options)

        environment = dict(plan.environment)
        environment.update(extra_environment(options))
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(engine_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        return CommandSpec(
            argv=[str(python), str(worker), *engine_argv],
            cwd=engine_root,
            environment=environment,
            output_paths=[output_path, metrics_path],
            metadata={
                "engine": self.name,
                "plan": plan.id,
                "expected_seconds": float(plan.metadata.get("expected_seconds", 73.0)),
                "progress_phases": [
                    {"name": "denoise step", "weight": 0.94},
                    {"name": "decode", "weight": 0.06},
                ],
            },
        )

    def doctor(self, model: ModelDescriptor) -> Dict[str, Any]:
        python = Path(str(model.config.get("python_path", "")))
        engine_root = Path(str(model.config.get("engine_root", "")))
        script = Path(str(model.config.get("script_path", "")))
        model_path = Path(str(model.config.get("model_path", "")))
        manifest = Path(str(model.config.get("ane_manifest", "")))
        bridge_dir = Path(str(model.config.get("bridge_dir", "")))
        report: Dict[str, Any] = {
            "engine": self.name,
            "paths": {
                "python": {"path": str(python), "exists": python.is_file()},
                "engine_root": {"path": str(engine_root), "exists": engine_root.is_dir()},
                "script": {"path": str(script), "exists": script.is_file()},
                "model": {"path": str(model_path), "exists": model_path.is_dir()},
                "ane_manifest": {"path": str(manifest), "exists": manifest.is_file()},
                "bridge_dir": {"path": str(bridge_dir), "exists": bridge_dir.is_dir()},
            },
        }
        if python.is_file() and engine_root.is_dir():
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(engine_root)
            probe = subprocess.run(
                [
                    str(python), "-c",
                    "import mlx.core, torch; import fastvideo.mlx_runtime.fastwan",
                ],
                cwd=str(engine_root),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                check=False,
            )
            report["probe_exit_code"] = probe.returncode
            report["probe"] = probe.stdout.strip()
        report["gpu_available"] = bool(
            python.is_file() and engine_root.is_dir() and script.is_file()
            and model_path.is_dir()
        )
        report["gpu_ane_available"] = bool(
            report["gpu_available"] and manifest.is_file() and bridge_dir.is_dir()
        )
        report["available"] = report["gpu_available"]
        return report
