"""Top-level model registry, policy, adapter, and job orchestration."""

from __future__ import annotations

import platform
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from turbocider.adapters import adapter_for
from turbocider.adapters.base import CommandSpec
from turbocider.jobs import JobManager
from turbocider.models import ExecutionPlan, GenerationRequest, JobRecord, ModelDescriptor
from turbocider.device import DeviceProfileCatalog, current_device
from turbocider.paths import output_root, state_root
from turbocider.policy import available_plans, choose_plan, device_compatibility
from turbocider.registry import ModelRegistry
from turbocider.errors import ValidationError


class PreparedGeneration:
    def __init__(
        self,
        model: ModelDescriptor,
        plan: ExecutionPlan,
        request: GenerationRequest,
        spec: CommandSpec,
    ):
        self.model = model
        self.plan = plan
        self.request = request
        self.spec = spec

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.as_dict(),
            "plan": self.plan.as_dict(),
            "request": self.request.as_dict(),
            "command": self.spec.as_dict(),
        }


class TurboCiderRuntime:
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        state_directory: Optional[Path] = None,
        output_directory: Optional[Path] = None,
        max_workers: int = 1,
    ):
        self.registry = registry or ModelRegistry()
        self.state_directory = state_directory or state_root()
        self.output_directory = output_directory or output_root()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.jobs = JobManager(self.state_directory, max_workers=max_workers)

    def prepare(self, request: GenerationRequest) -> PreparedGeneration:
        model = self.registry.get(request.model)
        tasks = model.capabilities.get("tasks", [])
        if tasks and request.task not in tasks:
            raise ValidationError(
                "model %s does not support task %s; supported: %s"
                % (model.id, request.task, ", ".join(tasks))
            )
        input_types = model.capabilities.get("inputs", [])
        unsupported = sorted({item.type for item in request.inputs if item.type not in input_types})
        if input_types and unsupported:
            raise ValidationError(
                "model %s does not support inputs: %s"
                % (model.id, ", ".join(unsupported))
            )
        if model.capabilities.get("audio_output") is False and request.output.audio:
            raise ValidationError(
                "model %s does not support audio output" % model.id
            )
        if (
            model.capabilities.get("audio_required") is True
            and request.task == "video"
            and not request.output.audio
        ):
            raise ValidationError(
                "model %s requires audio output on its current native path"
                % model.id
            )
        plan = choose_plan(model, request)
        extension = {
            "image": ".png",
            "video": ".mp4",
            "audio": ".wav",
        }.get(request.output.type, ".bin")
        if request.output.path:
            destination = Path(request.output.path).expanduser().resolve()
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            destination = self.output_directory / ("%s-%s%s" % (model.id, stamp, extension))
        destination.parent.mkdir(parents=True, exist_ok=True)
        adapter = adapter_for(model.engine)
        spec = adapter.build_command(model, request, plan, destination)
        compatibility = device_compatibility(plan)
        metadata = dict(spec.metadata)
        metadata["device_fingerprint"] = current_device().fingerprint
        if plan.execution.value == "gpu_ane":
            metadata["ane_device_profile"] = compatibility.get("matched_profile")
            metadata["ane_device_matched"] = bool(compatibility.get("matched"))
            metadata["ane_device_forced"] = bool(
                request.policy.execution.value == "gpu_ane"
                and not compatibility.get("matched")
            )
        spec = replace(spec, metadata=metadata)
        return PreparedGeneration(model, plan, request, spec)

    def submit(self, request: GenerationRequest) -> JobRecord:
        prepared = self.prepare(request)
        return self.jobs.submit(request, prepared.plan.id, prepared.spec)

    def doctor(self) -> Dict[str, Any]:
        model_reports = []
        for model in self.registry.all():
            try:
                report = adapter_for(model.engine).doctor(model)
            except (ValueError, TypeError) as error:
                report = {
                    "engine": model.engine,
                    "available": False,
                    "error": str(error),
                }
            report["id"] = model.id
            report["plans"] = [plan.as_dict() for plan in model.plans]
            model_reports.append(report)
        return {
            "framework": "TurboCider",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "device": current_device().as_dict(),
            "device_profiles": sorted(DeviceProfileCatalog().profiles),
            "models": model_reports,
        }

    def public_system(self) -> Dict[str, Any]:
        """Return health/capability status without exposing local filesystem paths."""
        doctor = self.doctor()
        return {
            "framework": doctor["framework"],
            "platform": doctor["platform"],
            "machine": doctor["machine"],
            "python": doctor["python"],
            "device": doctor["device"],
            "device_profiles": doctor["device_profiles"],
            "models": [
                {
                    "id": report["id"],
                    "engine": report["engine"],
                    "available": bool(report.get("available", False)),
                }
                for report in doctor["models"]
            ],
        }

    def plans(self, request: GenerationRequest):
        model = self.registry.get(request.model)
        return [
            {"plan": plan.as_dict(), "failures": failures, "available": not failures}
            for plan, failures in available_plans(model, request)
        ]

    def close(self) -> None:
        self.jobs.close()
