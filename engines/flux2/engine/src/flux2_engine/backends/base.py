"""Backend protocol."""

from __future__ import annotations

from typing import Protocol

from flux2_engine.config import GenerationRequest
from flux2_engine.result import GenerationResult


class InferenceBackend(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def close(self) -> None: ...
