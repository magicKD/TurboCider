"""Adapter for the native LTX-2.5 two-stage generation pipeline."""

from __future__ import annotations

import os
import sys
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


_ENVIRONMENT_KEYS = {
    "transformer_path": "LTX_TRANSFORMER",
    "upsampler_path": "LTX_UPSAMPLER",
    "video_vae_path": "LTX_VIDEO_VAE",
    "conditioning_directory": "LTX_CONDITIONING_DIR",
    "audio_vae_path": "LTX_AUDIO_VAE",
}

_DEFAULT_ANE_PATHS = {
    "LTX_ANE_MLP_STAGE1_DIR": "ane_mlp_stage1_directory",
    "LTX_ANE_MLP_STAGE2_DIR": "ane_mlp_stage2_directory",
    "LTX_ANE_KV_DIR": "ane_kv_directory",
}


def _configured(model: ModelDescriptor, options: Dict[str, Any], key: str) -> str:
    value = options.get(key) or model.config.get(key) or os.environ.get(_ENVIRONMENT_KEYS[key])
    return str(value or "")


class LTXAdapter(EngineAdapter):
    name = "ltx25"

    def build_command(
        self,
        model: ModelDescriptor,
        request: GenerationRequest,
        plan: ExecutionPlan,
        output_path: Path,
    ) -> CommandSpec:
        options = request.engine_options.get("ltx", {})
        if not isinstance(options, dict):
            raise ValidationError("engine_options.ltx must be an object")

        executable_key = "hybrid_executable_path" if plan.execution is ExecutionMode.GPU_ANE else "gpu_executable_path"
        executable = Path(str(model.config.get(executable_key, "")))
        transformer = _configured(model, options, "transformer_path")
        upsampler = _configured(model, options, "upsampler_path")
        video_vae = _configured(model, options, "video_vae_path")
        conditioning = _configured(model, options, "conditioning_directory")
        audio_vae_value = _configured(model, options, "audio_vae_path")
        required = {
            "executable": str(executable),
            "transformer": transformer,
            "upsampler": upsampler,
            "video_vae": video_vae,
            "conditioning": conditioning,
        }
        if request.output.audio:
            required["audio_vae"] = audio_vae_value
        missing = [name for name, value in required.items() if not value or not Path(value).exists()]
        if missing:
            raise EngineUnavailableError(
                "LTX generation configuration is incomplete: %s"
                % ", ".join("%s=%s" % (name, required[name] or "<unset>") for name in missing)
            )

        if request.output.frames != 97 or request.output.fps != 24:
            raise ValidationError(
                "the current native LTX pipeline requires frames=97 and fps=24"
            )
        if request.sampling.steps is not None and request.sampling.steps != 11:
            raise ValidationError(
                "the current distilled LTX pipeline uses a fixed 8+3 schedule; "
                "sampling.steps must be omitted or set to 11"
            )
        if request.sampling.guidance is not None:
            raise ValidationError(
                "the current distilled LTX pipeline does not expose guidance"
            )
        output_dir = output_path.parent / (output_path.stem + ".ltx-artifacts")
        ltx_root = executable.parent.parent
        worker = Path(__file__).resolve().parents[1] / "workers" / "ltx_pipeline.py"
        comfy_python = Path(str(options.get("comfy_python_path") or model.config.get("comfy_python_path", sys.executable)))
        mlx_python = Path(str(options.get("mlx_python_path") or model.config.get("mlx_python_path", sys.executable)))
        comfy_root = Path(str(options.get("comfy_root") or model.config.get("comfy_root", "")))
        text_encoder = Path(str(options.get("text_encoder_path") or model.config.get("text_encoder_path", "")))
        audio_vae = Path(audio_vae_value) if audio_vae_value else Path(
            "/__turbocider_missing_ltx_audio_vae__"
        )
        dynamic_conditioning = bool(options.get("dynamic_conditioning", True))
        argv: List[str] = [
            sys.executable,
            str(worker),
            "--engine", str(executable),
            "--ltx-root", str(ltx_root),
            "--comfy-python", str(comfy_python),
            "--mlx-python", str(mlx_python),
            "--comfy-root", str(comfy_root),
            "--text-encoder", str(text_encoder),
            "--transformer", transformer,
            "--upsampler", upsampler,
            "--video-vae", video_vae,
            "--conditioning", conditioning,
            "--artifact-dir", str(output_dir),
            "--output", str(output_path),
            "--prompt", request.prompt,
            "--width", str(request.output.width),
            "--height", str(request.output.height),
            "--frames", str(request.output.frames),
            "--fps", str(request.output.fps),
            "--seed", str(request.sampling.seed),
        ]
        if dynamic_conditioning:
            argv.append("--dynamic-conditioning")
        if request.output.audio and audio_vae.is_file():
            argv.extend(["--audio-vae", str(audio_vae)])
        extra = options.get("args", [])
        if not isinstance(extra, list):
            raise ValidationError("engine_options.ltx.args must be a list")
        if extra:
            raise ValidationError(
                "the native LTX generation command has no trailing argument "
                "surface; use engine_options.ltx.env for supported LTX_* controls"
            )

        environment = {
            "LTX_OUTPUT_WIDTH": str(request.output.width),
            "LTX_OUTPUT_HEIGHT": str(request.output.height),
            "LTX_MEDIA_BACKEND": str(options.get("media_backend", "mlx")),
        }
        environment.update(plan.environment)
        if plan.execution is ExecutionMode.GPU_ANE:
            configured_paths = dict(_DEFAULT_ANE_PATHS)
            plan_paths = plan.metadata.get("ltx_ane_paths", {})
            if isinstance(plan_paths, dict):
                configured_paths.update(
                    (str(environment_key), str(config_key))
                    for environment_key, config_key in plan_paths.items()
                )
            for environment_key, config_key in configured_paths.items():
                value = model.config.get(config_key)
                if value:
                    environment[environment_key] = str(value)
        environment.update(extra_environment(options))
        unset = ()
        if plan.execution is ExecutionMode.GPU:
            unset = ("LTX_ANE_",)
        return CommandSpec(
            argv=argv,
            cwd=ltx_root,
            environment=environment,
            unset_environment_prefixes=unset,
            output_paths=[output_path, output_dir],
            metadata={
                "engine": "ltx25",
                "plan": plan.id,
                "requested_frames": request.output.frames,
                "native_frames": 97,
                "requested_fps": request.output.fps,
                "dynamic_conditioning": dynamic_conditioning,
                "expected_seconds": float(plan.metadata.get("expected_seconds", 113.0)),
                "progress_phases": [
                    {"name": "text_conditioning", "weight": 0.17},
                    {"name": "connector", "weight": 0.03},
                    {"name": "native_generation", "weight": 0.76},
                    {"name": "video_encode", "weight": 0.02},
                    {"name": "audio_decode", "weight": 0.01},
                    {"name": "mux", "weight": 0.01},
                ],
            },
        )

    def doctor(self, model: ModelDescriptor) -> Dict[str, Any]:
        paths = {
            "gpu_executable": Path(str(model.config.get("gpu_executable_path", ""))),
            "hybrid_executable": Path(str(model.config.get("hybrid_executable_path", ""))),
            "transformer": Path(str(model.config.get("transformer_path", ""))),
            "upsampler": Path(str(model.config.get("upsampler_path", ""))),
            "video_vae": Path(str(model.config.get("video_vae_path", ""))),
            "conditioning": Path(str(model.config.get("conditioning_directory", ""))),
            "text_encoder": Path(str(model.config.get("text_encoder_path", ""))),
            "comfy_root": Path(str(model.config.get("comfy_root", ""))),
            "comfy_python": Path(str(model.config.get("comfy_python_path", ""))),
            "mlx_python": Path(str(model.config.get("mlx_python_path", ""))),
        }
        report: Dict[str, Any] = {
            "engine": self.name,
            "paths": {key: {"path": str(value), "exists": value.exists()} for key, value in paths.items()},
        }
        executable = paths["gpu_executable"]
        if executable.is_file():
            probe = subprocess.run(
                [str(model.config.get("info_executable_path", executable)), "--self-test"],
                cwd=str(executable.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                check=False,
            )
            report["self_test_exit_code"] = probe.returncode
            report["self_test"] = probe.stdout.strip()
        report["available"] = all(
            paths[key].exists()
            for key in (
                "gpu_executable", "transformer", "upsampler", "video_vae",
                "conditioning", "text_encoder", "comfy_root", "comfy_python", "mlx_python",
            )
        )
        return report
