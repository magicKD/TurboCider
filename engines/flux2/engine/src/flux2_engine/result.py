"""Generation outputs and machine-readable timing metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    backend: str
    wall_seconds: float
    peak_mlx_gb: float | None = None
    actual_sequence_length: int | None = None
    bucket: int | None = None
    ane_call_count: int = 0
    ane_call_p50_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    image: Any
    metrics: GenerationMetrics
    output: Path | None = None

    def save(self, output: Path, *, overwrite: bool = True) -> Path:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.image.save(destination, overwrite=overwrite)
        except TypeError:
            self.image.save(destination)
        self.output = destination
        return destination
