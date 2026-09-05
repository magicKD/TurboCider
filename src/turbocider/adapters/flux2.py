"""Adapter for the persistent FLUX.2 MLX/Core ML engine CLI."""

from __future__ import annotations

import os
import subprocess
import hashlib
import json
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


class Flux2Adapter(EngineAdapter):
    name = "flux2"

    @staticmethod
    def _boolean_flag(argv: List[str], name: str, value: bool) -> None:
        argv.append("--%s%s" % ("" if value else "no-", name))

    def build_command(
        self,
        model: ModelDescriptor,
        request: GenerationRequest,
        plan: ExecutionPlan,
        output_path: Path,
    ) -> CommandSpec:
        options = request.engine_options.get("flux2", {})
        if not isinstance(options, dict):
            raise ValidationError("engine_options.flux2 must be an object")
        image_assets = [item for item in request.inputs if item.type == "image"]
        request_mode = request.resolved_mode
        image_path = None
        image_paths: List[str] = []
        image_strength = float(options.get("image_strength", 0.75))
        if request_mode == "text_to_image":
            if image_assets:
                raise ValidationError("text_to_image does not accept image inputs")
            pipeline = "standard"
        elif request_mode == "image_to_image":
            if len(image_assets) != 1 or image_assets[0].role != "init_image":
                raise ValidationError(
                    "image_to_image requires exactly one image with role=init_image"
                )
            pipeline = "standard"
            image_path = image_assets[0].path
            if image_assets[0].strength is not None:
                image_strength = image_assets[0].strength
        elif request_mode == "image_edit":
            if not image_assets or any(item.role != "reference" for item in image_assets):
                raise ValidationError(
                    "image_edit requires one or more images with role=reference"
                )
            pipeline = "edit"
            image_paths = [str(item.path) for item in image_assets]
        else:
            raise ValidationError("unsupported FLUX.2 mode: %s" % request_mode)
        if not 0.0 <= image_strength <= 1.0:
            raise ValidationError("FLUX.2 image strength must be between 0 and 1")
        python_value = options.get("python_path") or model.config.get("python_path") or os.environ.get("MFLUX_PY", "")
        model_value = options.get("model_path") or model.config.get("model_path") or os.environ.get("FLUX2_MODEL_PATH", "")
        mflux_value = options.get("mflux_root") or model.config.get("mflux_root") or os.environ.get("MFLUX_ROOT", "")
        python = Path(str(python_value)) if python_value else Path("/__turbocider_missing_python__")
        engine_root = Path(str(model.config.get("engine_root", "")))
        model_path = Path(str(model_value)) if model_value else Path("/__turbocider_missing_flux_model__")
        mflux_root = Path(str(mflux_value)) if mflux_value else Path("/__turbocider_missing_mflux__")
        bridge_dir = Path(str(
            options.get("bridge_dir")
            or model.config.get("bridge_dir")
            or engine_root / "build"
        ))
        missing = [
            name
            for name, path in (
                ("python", python),
                ("engine_root", engine_root),
                ("model", model_path),
                ("mflux_root", mflux_root),
            )
            if not path.exists()
        ]
        if plan.execution is ExecutionMode.GPU_ANE and not bridge_dir.is_dir():
            missing.append("native_bridge")
        if missing:
            raise EngineUnavailableError(
                "FLUX.2 configuration is incomplete: %s" % ", ".join(missing)
            )

        backend_mode = "mlx-ane" if plan.execution is ExecutionMode.GPU_ANE else "mlx"
        precision = str(options.get("precision", "bf16"))
        if precision not in ("bf16", "fp16"):
            raise ValidationError("engine_options.flux2.precision must be bf16 or fp16")
        model_variant = str(
            options.get("model_variant")
            or model.config.get("model_variant")
            or "auto"
        )
        if model_variant not in {
            "auto", "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-9b-kv"
        }:
            raise ValidationError(
                "engine_options.flux2.model_variant must be auto, "
                "flux2-klein-4b, flux2-klein-9b, or flux2-klein-9b-kv"
            )
        attention = str(options.get("attention", "auto"))
        if attention not in {
            "auto", "mlx", "ane-experimental", "head-split-experimental"
        }:
            raise ValidationError(
                "engine_options.flux2.attention must be auto, mlx, "
                "ane-experimental, or head-split-experimental"
            )
        dynamic_text_length = bool(options.get("dynamic_text_length", True))
        reuse_ane_outputs = bool(options.get("reuse_ane_outputs", True))
        compile_gpu_attention = bool(
            options.get("compile_quantized_gpu_attention", True)
        )
        clear_mlx_cache = bool(
            options.get(
                "clear_mlx_cache_between_requests",
                model.config.get("clear_mlx_cache_between_requests", True),
            )
        )
        ane_variant = str(options.get("ane_variant", "int8_pc"))
        if ane_variant not in ("int8_pc", "fp16"):
            raise ValidationError(
                "engine_options.flux2.ane_variant must be int8_pc or fp16"
            )
        ane_blocks = options.get("ane_blocks", [])
        if not isinstance(ane_blocks, list) or not all(
            isinstance(block, int) and block >= 0 for block in ane_blocks
        ):
            raise ValidationError(
                "engine_options.flux2.ane_blocks must be a list of non-negative integers"
            )
        argv: List[str] = [
            str(python),
            "-m",
            "flux2_engine.cli",
            "generate",
            request.prompt,
            "--model",
            str(model_path),
            "--model-variant",
            model_variant,
            "--pipeline",
            pipeline,
            "--mflux-root",
            str(mflux_root),
            "--output",
            str(output_path),
            "--mode",
            backend_mode,
            "--attention",
            attention,
            "--profile",
            request.policy.profile.value,
            "--width",
            str(request.output.width),
            "--height",
            str(request.output.height),
            "--seed",
            str(request.sampling.seed),
            "--precision",
            precision,
        ]
        if image_path:
            argv.extend([
                "--image-path", str(image_path),
                "--image-strength", str(image_strength),
            ])
        if image_paths:
            argv.append("--image-paths")
            argv.extend(image_paths)
        if request.sampling.steps is not None:
            argv.extend(["--steps", str(request.sampling.steps)])
        if request.sampling.guidance is not None:
            argv.extend(["--guidance", str(request.sampling.guidance)])
        if request.policy.persistent:
            argv.append("--persistent")
        if bool(options.get("mlx_eager", False)):
            argv.append("--mlx-eager")
        if bridge_dir.is_dir():
            argv.extend(["--bridge-dir", str(bridge_dir)])
        argv.extend(["--ane-variant", ane_variant])
        self._boolean_flag(argv, "dynamic-text-length", dynamic_text_length)
        self._boolean_flag(argv, "reuse-ane-outputs", reuse_ane_outputs)
        self._boolean_flag(
            argv,
            "compile-quantized-gpu-attention",
            compile_gpu_attention,
        )
        self._boolean_flag(
            argv,
            "clear-mlx-cache-between-requests",
            clear_mlx_cache,
        )
        for block in ane_blocks:
            argv.extend(["--ane-block", str(block)])

        manifests = options.get("ane_manifests") or model.config.get("ane_manifests", [])
        if isinstance(manifests, str):
            manifests = [item for item in manifests.split(os.pathsep) if item]
        if plan.execution is ExecutionMode.GPU_ANE:
            if not manifests:
                raise EngineUnavailableError("FLUX.2 GPU+ANE requires at least one ANE manifest")
            for manifest in manifests:
                argv.extend(["--ane-manifest", str(manifest)])
        append_extra_args(argv, options)

        environment = dict(plan.environment)
        environment.update(extra_environment(options))
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        source_path = str(engine_root / "src")
        environment["PYTHONPATH"] = source_path + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        metadata: Dict[str, Any] = {
            "engine": "flux2",
            "plan": plan.id,
            "mode": request_mode,
            "expected_seconds": float(
                plan.metadata.get(
                    "expected_seconds_warm" if request.policy.persistent else "expected_seconds",
                    4.0 if request.policy.persistent else 6.0,
                )
            ),
            "expected_seconds_cold": float(
                plan.metadata.get("expected_seconds_cold", 16.0)
            ),
            "progress_phases": [
                {"name": "model_load", "weight": 0.30},
                {"name": "denoise", "weight": 0.60},
                {"name": "decode", "weight": 0.10},
            ],
        }
        if request.policy.persistent:
            worker = Path(__file__).resolve().parents[1] / "workers" / "flux2_worker.py"
            worker_argv: List[str] = [
                str(python), str(worker),
                "--model", str(model_path),
                "--model-variant", model_variant,
                "--pipeline", pipeline,
                "--mflux-root", str(mflux_root),
                "--mode", backend_mode,
                "--attention", attention,
                "--precision", precision,
                "--bridge-dir", str(bridge_dir),
                "--ane-variant", ane_variant,
            ]
            self._boolean_flag(worker_argv, "reuse-ane-outputs", reuse_ane_outputs)
            self._boolean_flag(
                worker_argv,
                "compile-quantized-gpu-attention",
                compile_gpu_attention,
            )
            self._boolean_flag(
                worker_argv,
                "clear-mlx-cache-between-requests",
                clear_mlx_cache,
            )
            if bool(options.get("mlx_eager", False)):
                worker_argv.append("--mlx-eager")
            for block in ane_blocks:
                worker_argv.extend(["--ane-block", str(block)])
            for manifest in manifests:
                worker_argv.extend(["--ane-manifest", str(manifest)])
            key_material = json.dumps(worker_argv, separators=(",", ":"))
            metadata["persistent_worker"] = {
                "key": hashlib.sha256(key_material.encode()).hexdigest(),
                "argv": worker_argv,
                "cwd": str(engine_root),
                "request": {
                    "prompt": request.prompt,
                    "output": str(output_path),
                    "width": request.output.width,
                    "height": request.output.height,
                    "steps": request.sampling.steps,
                    "seed": request.sampling.seed,
                    "guidance": request.sampling.guidance if request.sampling.guidance is not None else 1.0,
                    "profile": request.policy.profile.value,
                    "dynamic_text_length": dynamic_text_length,
                    "image_path": str(image_path) if image_path else None,
                    "image_paths": image_paths,
                    "image_strength": image_strength,
                },
            }
            argv = worker_argv
        return CommandSpec(
            argv=argv,
            cwd=engine_root,
            environment=environment,
            output_paths=[output_path, output_path.with_suffix(".engine.json")],
            metadata=metadata,
        )

    def doctor(self, model: ModelDescriptor) -> Dict[str, Any]:
        python_value = model.config.get("python_path") or os.environ.get("MFLUX_PY", "")
        model_value = model.config.get("model_path") or os.environ.get("FLUX2_MODEL_PATH", "")
        mflux_value = model.config.get("mflux_root") or os.environ.get("MFLUX_ROOT", "")
        python = Path(str(python_value)) if python_value else Path("/__turbocider_missing_python__")
        engine_root = Path(str(model.config.get("engine_root", "")))
        model_path = Path(str(model_value)) if model_value else Path("/__turbocider_missing_flux_model__")
        mflux_root = Path(str(mflux_value)) if mflux_value else Path("/__turbocider_missing_mflux__")
        bridge_dir = Path(str(
            model.config.get("bridge_dir") or engine_root / "build"
        ))
        manifests = model.config.get("ane_manifests", [])
        if isinstance(manifests, str):
            manifests = [item for item in manifests.split(os.pathsep) if item]
        manifest_paths = [Path(str(item)) for item in manifests]
        report: Dict[str, Any] = {
            "engine": self.name,
            "python": {"path": str(python), "exists": python.exists()},
            "engine_root": {"path": str(engine_root), "exists": engine_root.is_dir()},
            "model": {"path": str(model_path), "exists": model_path.is_dir()},
            "mflux_root": {"path": str(mflux_root), "exists": mflux_root.is_dir()},
            "native_bridge": {
                "path": str(bridge_dir),
                "exists": bridge_dir.is_dir(),
            },
            "ane_manifests": [
                {"path": str(path), "exists": path.is_file()}
                for path in manifest_paths
            ],
        }
        if python.is_file() and engine_root.is_dir():
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(engine_root / "src")
            command = [
                str(python), "-m", "flux2_engine.cli", "doctor",
                "--model", str(model_path),
                "--model-variant", str(model.config.get("model_variant", "auto")),
                "--mflux-root", str(mflux_root),
                "--bridge-dir", str(bridge_dir),
            ]
            for manifest in manifest_paths:
                command.extend(["--ane-manifest", str(manifest)])
            probe = subprocess.run(
                command,
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
            python.is_file() and engine_root.is_dir() and model_path.is_dir() and mflux_root.is_dir()
        )
        report["gpu_ane_available"] = bool(
            report["gpu_available"]
            and bridge_dir.is_dir()
            and manifest_paths
            and all(path.is_file() for path in manifest_paths)
        )
        report["available"] = report["gpu_available"]
        return report
