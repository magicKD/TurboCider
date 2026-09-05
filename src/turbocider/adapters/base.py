"""Common subprocess adapter contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from turbocider.models import ExecutionPlan, GenerationRequest, ModelDescriptor


@dataclass(frozen=True)
class CommandSpec:
    argv: List[str]
    cwd: Path
    environment: Dict[str, str] = field(default_factory=dict)
    unset_environment_prefixes: Sequence[str] = field(default_factory=tuple)
    output_paths: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["cwd"] = str(self.cwd)
        raw["output_paths"] = [str(path) for path in self.output_paths]
        return raw


class EngineAdapter:
    name = "base"

    def build_command(
        self,
        model: ModelDescriptor,
        request: GenerationRequest,
        plan: ExecutionPlan,
        output_path: Path,
    ) -> CommandSpec:
        raise NotImplementedError

    def doctor(self, model: ModelDescriptor) -> Dict[str, Any]:
        return {"engine": self.name, "available": True}


def append_extra_args(argv: List[str], options: Dict[str, Any]) -> None:
    extra = options.get("args", [])
    if not isinstance(extra, list) or not all(isinstance(item, (str, int, float)) for item in extra):
        raise ValueError("engine args must be a list of strings/numbers")
    argv.extend(str(item) for item in extra)


def extra_environment(options: Dict[str, Any]) -> Dict[str, str]:
    raw = options.get("env", {})
    if not isinstance(raw, dict):
        raise ValueError("engine env must be an object")
    return {str(key): str(value) for key, value in raw.items()}

