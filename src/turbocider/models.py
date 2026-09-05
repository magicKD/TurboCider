"""Stable request, model, plan, and job data models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from turbocider.errors import ValidationError


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ExecutionMode(StringEnum):
    AUTO = "auto"
    GPU = "gpu"
    GPU_ANE = "gpu_ane"


class GenerationProfile(StringEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    PREVIEW = "preview"


class ApproximationMode(StringEnum):
    EXACT = "exact"
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"


class JobState(StringEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InputAsset:
    type: str
    role: str
    path: Optional[str] = None
    text: Optional[str] = None
    audio_path: Optional[str] = None
    include_embedded_audio: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "InputAsset":
        return cls(
            type=str(raw.get("type", "")),
            role=str(raw.get("role", "reference")),
            path=raw.get("path"),
            text=raw.get("text"),
            audio_path=raw.get("audio_path"),
            include_embedded_audio=bool(raw.get("include_embedded_audio", True)),
        )


@dataclass(frozen=True)
class OutputSpec:
    type: str = "video"
    path: Optional[str] = None
    width: int = 512
    height: int = 512
    frames: int = 22
    fps: int = 24
    audio: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OutputSpec":
        output_type = str(raw.get("type", "video"))
        return cls(
            type=output_type,
            path=raw.get("path"),
            width=int(raw.get("width", 512)),
            height=int(raw.get("height", 512)),
            frames=int(raw.get("frames", 1 if output_type == "image" else 22)),
            fps=int(raw.get("fps", 24)),
            audio=bool(raw.get("audio", output_type == "video")),
        )


@dataclass(frozen=True)
class SamplingSpec:
    seed: int = 42
    steps: Optional[int] = None
    guidance: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SamplingSpec":
        steps = raw.get("steps")
        guidance = raw.get("guidance")
        return cls(
            seed=int(raw.get("seed", 42)),
            steps=int(steps) if steps is not None else None,
            guidance=float(guidance) if guidance is not None else None,
        )


@dataclass(frozen=True)
class PolicySpec:
    execution: ExecutionMode = ExecutionMode.AUTO
    profile: GenerationProfile = GenerationProfile.QUALITY
    approximation: ApproximationMode = ApproximationMode.VALIDATED
    persistent: bool = False
    allow_fallback: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PolicySpec":
        return cls(
            execution=ExecutionMode(str(raw.get("execution", "auto"))),
            profile=GenerationProfile(str(raw.get("profile", "quality"))),
            approximation=ApproximationMode(str(raw.get("approximation", "validated"))),
            persistent=bool(raw.get("persistent", False)),
            allow_fallback=bool(raw.get("allow_fallback", True)),
        )


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    task: str
    prompt: str
    inputs: List[InputAsset] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    policy: PolicySpec = field(default_factory=PolicySpec)
    engine_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GenerationRequest":
        prompt = str(raw.get("prompt", ""))
        inputs = [InputAsset.from_dict(item) for item in raw.get("inputs", [])]
        if not prompt:
            for item in inputs:
                if item.type == "text" and item.role == "prompt" and item.text:
                    prompt = item.text
                    break
        task = str(raw.get("task", "video"))
        output_raw = dict(raw.get("output", {}))
        output_raw.setdefault("type", task)
        request = cls(
            model=str(raw.get("model", "")),
            task=task,
            prompt=prompt,
            inputs=inputs,
            output=OutputSpec.from_dict(output_raw),
            sampling=SamplingSpec.from_dict(dict(raw.get("sampling", {}))),
            policy=PolicySpec.from_dict(dict(raw.get("policy", {}))),
            engine_options=dict(raw.get("engine_options", {})),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if not self.model:
            raise ValidationError("model must not be empty")
        if not self.prompt.strip():
            raise ValidationError("prompt must not be empty")
        if self.task not in {"image", "video", "audio"}:
            raise ValidationError("task must be image, video, or audio")
        if self.output.type != self.task:
            raise ValidationError(
                "output.type must match task (%s != %s)"
                % (self.output.type, self.task)
            )
        if self.output.width <= 0 or self.output.height <= 0:
            raise ValidationError("output width and height must be positive")
        if self.output.frames <= 0 or self.output.fps <= 0:
            raise ValidationError("output frames and fps must be positive")
        if self.task == "image" and self.output.frames != 1:
            raise ValidationError("image generation requires output.frames=1")
        if self.task != "video" and self.output.audio:
            raise ValidationError("audio output is only valid for video generation")
        if self.sampling.steps is not None and self.sampling.steps <= 0:
            raise ValidationError("sampling steps must be positive")

    def as_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["policy"]["execution"] = self.policy.execution.value
        raw["policy"]["profile"] = self.policy.profile.value
        raw["policy"]["approximation"] = self.policy.approximation.value
        return raw


@dataclass(frozen=True)
class ExecutionPlan:
    id: str
    execution: ExecutionMode
    quality: ApproximationMode
    profile: str
    production: bool
    priority: int
    description: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExecutionPlan":
        return cls(
            id=str(raw["id"]),
            execution=ExecutionMode(str(raw["execution"])),
            quality=ApproximationMode(str(raw.get("quality", "exact"))),
            profile=str(raw.get("profile", "quality")),
            production=bool(raw.get("production", False)),
            priority=int(raw.get("priority", 0)),
            description=str(raw.get("description", "")),
            requirements=dict(raw.get("requirements", {})),
            environment={str(k): str(v) for k, v in raw.get("environment", {}).items()},
            metadata=dict(raw.get("metadata", {})),
        )

    def as_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["execution"] = self.execution.value
        raw["quality"] = self.quality.value
        return raw


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    name: str
    engine: str
    version: str
    capabilities: Dict[str, Any]
    config: Dict[str, Any]
    plans: List[ExecutionPlan]
    preparation: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "version": self.version,
            "capabilities": self.capabilities,
            "config": self.config,
            "plans": [plan.as_dict() for plan in self.plans],
            "preparation": self.preparation,
            "source": str(self.source) if self.source else None,
        }

    def public_dict(self) -> Dict[str, Any]:
        """API-safe descriptor without machine-local paths or configuration."""
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "version": self.version,
            "capabilities": self.capabilities,
            "plans": [
                {
                    "id": plan.id,
                    "execution": plan.execution.value,
                    "quality": plan.quality.value,
                    "profile": plan.profile,
                    "production": plan.production,
                    "description": plan.description,
                }
                for plan in self.plans
            ],
        }


@dataclass
class JobRecord:
    id: str
    request: GenerationRequest
    state: JobState = JobState.QUEUED
    plan_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    phase: str = "queued"
    estimated_total_seconds: Optional[float] = None
    output_paths: List[str] = field(default_factory=list)
    command: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    log_path: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JobRecord":
        request_raw = dict(raw["request"])
        if str(request_raw.get("task", "video")) != "video":
            output_raw = dict(request_raw.get("output", {}))
            output_raw["audio"] = False
            request_raw["output"] = output_raw
        return cls(
            id=str(raw["id"]),
            request=GenerationRequest.from_dict(request_raw),
            state=JobState(str(raw.get("state", "queued"))),
            plan_id=raw.get("plan_id"),
            created_at=float(raw.get("created_at", time.time())),
            started_at=(
                float(raw["started_at"])
                if raw.get("started_at") is not None
                else None
            ),
            finished_at=(
                float(raw["finished_at"])
                if raw.get("finished_at") is not None
                else None
            ),
            progress=float(raw.get("progress", 0.0)),
            phase=str(raw.get("phase", "queued")),
            estimated_total_seconds=(
                float(raw["estimated_total_seconds"])
                if raw.get("estimated_total_seconds") is not None
                else None
            ),
            output_paths=[str(path) for path in raw.get("output_paths", [])],
            command=[str(item) for item in raw.get("command", [])],
            metrics=dict(raw.get("metrics", {})),
            error=raw.get("error"),
            log_path=raw.get("log_path"),
        )

    def as_dict(self) -> Dict[str, Any]:
        elapsed_seconds = None
        estimated_remaining_seconds = None
        if self.started_at is not None:
            ended_at = self.finished_at if self.finished_at is not None else time.time()
            elapsed_seconds = max(0.0, ended_at - self.started_at)
            if self.estimated_total_seconds is not None and self.finished_at is None:
                estimated_remaining_seconds = max(
                    0.0, self.estimated_total_seconds - elapsed_seconds
                )
        return {
            "id": self.id,
            "request": self.request.as_dict(),
            "state": self.state.value,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "phase": self.phase,
            "elapsed_seconds": elapsed_seconds,
            "estimated_total_seconds": self.estimated_total_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "output_paths": list(self.output_paths),
            "command": list(self.command),
            "metrics": dict(self.metrics),
            "error": self.error,
            "log_path": self.log_path,
        }
