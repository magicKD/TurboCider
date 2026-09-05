"""Adapter for the native h3.c executable."""

from __future__ import annotations

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


class H3Adapter(EngineAdapter):
    name = "h3"

    _INTEGER_OPTIONS = {
        "render_width": "--render-width",
        "render_height": "--render-height",
        "reuse": "--reuse",
        "layers": "--layers",
        "core_reuse": "--core-reuse",
        "zoom": "--zoom",
    }
    _FLOAT_OPTIONS = {
        "video_flow_shift": "--video-flow-shift",
        "audio_flow_shift": "--audio-flow-shift",
    }
    _BOOLEAN_OPTIONS = {
        "token_reduction": "--token-reduction",
        "ssd_streaming": "--ssd-streaming",
        "use_int8_row_fc2": "--use-int8-row-fc2",
        "use_reference_rope": "--use-reference-rope",
        "show": "--show",
        "profile_timing": "--profile",
        "super": "--super",
    }

    _ANE_DIRECTORY_KEYS = (
        "H3_COREML_ANE_DIR",
        "H3_PRIVATE_ANE_MLP_DIR",
        "H3_PRIVATE_ANE_QKV_DIR",
    )
    _ANE_PATH_KEYS = (
        "H3_COREML_ANE_MODEL",
    )
    _ANE_CHECKPOINT_KEYS = (
        "H3_PRIVATE_ANE_QKV_CHECKPOINT",
        "H3_PRIVATE_ANE_ATTENTION_OUT_CHECKPOINT",
    )

    @classmethod
    def _validate_ane_environment(cls, environment: Dict[str, str]) -> List[str]:
        active: List[str] = []
        for key in cls._ANE_DIRECTORY_KEYS:
            value = environment.get(key)
            if not value:
                continue
            if not Path(value).expanduser().is_dir():
                raise EngineUnavailableError(
                    "H3 GPU+ANE directory does not exist: %s=%s" % (key, value)
                )
            active.append(key)
        for key in cls._ANE_PATH_KEYS:
            value = environment.get(key)
            if not value:
                continue
            if not Path(value).expanduser().exists():
                raise EngineUnavailableError(
                    "H3 GPU+ANE artifact does not exist: %s=%s" % (key, value)
                )
            active.append(key)
        for key in cls._ANE_CHECKPOINT_KEYS:
            value = environment.get(key)
            if not value:
                continue
            if value != "1" and not Path(value).expanduser().exists():
                raise EngineUnavailableError(
                    "H3 GPU+ANE checkpoint does not exist: %s=%s" % (key, value)
                )
            active.append(key)
        if not active:
            raise EngineUnavailableError(
                "H3 GPU+ANE requires a Core ML/private-ANE model directory, "
                "model artifact, or checkpoint-backed ANE projection"
            )
        return active

    def build_command(
        self,
        model: ModelDescriptor,
        request: GenerationRequest,
        plan: ExecutionPlan,
        output_path: Path,
    ) -> CommandSpec:
        executable = Path(str(model.config["executable_path"]))
        model_path = Path(str(model.config["model_path"]))
        if not executable.is_file():
            raise EngineUnavailableError("H3 executable does not exist: %s" % executable)
        if not model_path.is_dir():
            raise EngineUnavailableError("H3 model directory does not exist: %s" % model_path)
        if request.output.fps != 24:
            raise ValidationError("the native H3 engine currently outputs 24 fps")

        argv: List[str] = [
            str(executable),
            "-d",
            str(model_path),
            "-p",
            request.prompt,
            "-o",
            str(output_path),
            "--width",
            str(request.output.width),
            "--height",
            str(request.output.height),
            "--frames",
            str(request.output.frames),
            "--seed",
            str(request.sampling.seed),
        ]
        if request.sampling.steps is not None:
            argv.extend(["--steps", str(request.sampling.steps)])

        for item in request.inputs:
            if item.type == "text":
                continue
            if not item.path:
                raise ValidationError("H3 %s input requires path" % item.type)
            if item.type == "image" and item.role == "first_frame":
                argv.extend(["--first-frame", item.path])
            elif item.type == "image" and item.role == "last_frame":
                argv.extend(["--last-frame", item.path])
            elif item.type == "image":
                argv.extend(["--ref-image", item.path])
            elif item.type == "audio":
                argv.extend(["--ref-audio", item.path])
            elif item.type == "video" and item.audio_path:
                argv.extend(["--ref-video-audio", item.path, item.audio_path])
            elif item.type == "video" and not item.include_embedded_audio:
                argv.extend(["--ref-silent-video", item.path])
            elif item.type == "video":
                argv.extend(["--ref-video", item.path])
            else:
                raise ValidationError("unsupported H3 input type: %s" % item.type)

        options = request.engine_options.get("h3", {})
        if not isinstance(options, dict):
            raise ValidationError("engine_options.h3 must be an object")
        for name, flag in self._INTEGER_OPTIONS.items():
            if options.get(name) is not None:
                argv.extend([flag, str(int(options[name]))])
        for name, flag in self._FLOAT_OPTIONS.items():
            if options.get(name) is not None:
                argv.extend([flag, str(float(options[name]))])
        for name, flag in self._BOOLEAN_OPTIONS.items():
            if options.get(name):
                argv.append(flag)

        defaults = plan.metadata.get("h3_options", {})
        for name, value in defaults.items():
            if name in options:
                continue
            if name in self._INTEGER_OPTIONS:
                argv.extend([self._INTEGER_OPTIONS[name], str(int(value))])
            elif name in self._BOOLEAN_OPTIONS and value:
                argv.append(self._BOOLEAN_OPTIONS[name])

        reference_size = options.get("reference_image_size")
        if reference_size:
            argv.extend(["--ref-image-size", str(reference_size)])
        append_extra_args(argv, options)

        environment = dict(plan.environment)
        environment.update(extra_environment(options))
        unset = ()
        if plan.execution is ExecutionMode.GPU:
            unset = ("H3_PRIVATE_ANE_", "H3_COREML_ANE_")
            active_ane = []
        else:
            active_ane = self._validate_ane_environment(environment)
        return CommandSpec(
            argv=argv,
            cwd=executable.parent,
            environment=environment,
            unset_environment_prefixes=unset,
            output_paths=[output_path],
            metadata={
                "engine": "h3",
                "plan": plan.id,
                "ane_configuration": active_ane,
                "expected_seconds": float(plan.metadata.get("expected_seconds", 30.0)),
                "progress_phases": [
                    {"name": "tokenizer", "weight": 0.01},
                    {"name": "text encoder", "weight": 0.12},
                    {"name": "refine text", "weight": 0.01},
                    {"name": "precompute AdaLN", "weight": 0.05},
                    {"name": "load transformer core", "weight": 0.20},
                    {"name": "denoise", "weight": 0.42},
                    {"name": "audio VAE", "weight": 0.05},
                    {"name": "video VAE load", "weight": 0.10},
                    {"name": "FFmpeg", "weight": 0.04},
                ],
            },
        )

    def doctor(self, model: ModelDescriptor) -> Dict[str, Any]:
        executable = Path(str(model.config.get("executable_path", "")))
        model_path = Path(str(model.config.get("model_path", "")))
        report: Dict[str, Any] = {
            "engine": self.name,
            "executable": str(executable),
            "executable_exists": executable.is_file(),
            "model_path": str(model_path),
            "model_exists": model_path.is_dir(),
        }
        if executable.is_file() and model_path.is_dir():
            probe = subprocess.run(
                [str(executable), "--info", "-d", str(model_path)],
                cwd=str(executable.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
            report["probe_exit_code"] = probe.returncode
            report["probe"] = probe.stdout.strip()
        report["available"] = bool(
            report["executable_exists"]
            and report["model_exists"]
            and report.get("probe_exit_code", 1) == 0
        )
        return report
