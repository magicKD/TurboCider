"""Backend-independent persistent FLUX.2 engine."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Self

from flux2_engine.backends.base import InferenceBackend
from flux2_engine.backends.hybrid import ANEHybridBackend
from flux2_engine.backends.mlx import MLXBackend
from flux2_engine.config import (
    EngineConfig,
    ExecutionMode,
    GenerationRequest,
    PipelineMode,
)
from flux2_engine.errors import ArtifactError, BackendUnavailableError
from flux2_engine.policy import BackendDecision, choose_backend
from flux2_engine.result import GenerationResult


class Flux2Engine:
    """Keep FLUX.2 weights resident and route requests to MLX or MLX+ANE."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.decision: BackendDecision = choose_backend(config)
        self._fallback: MLXBackend | None = None
        self.backend = self._make_backend(self.decision.mode)

    def _make_backend(self, mode: ExecutionMode) -> InferenceBackend:
        if mode is ExecutionMode.MLX:
            return MLXBackend(self.config, eager=self.config.mlx_eager)
        try:
            return ANEHybridBackend(self.config)
        except (ArtifactError, BackendUnavailableError):
            if self.config.mode is not ExecutionMode.AUTO or not self.config.allow_auto_fallback:
                raise
            self.decision = BackendDecision(ExecutionMode.MLX, "ANE unavailable; fell back to MLX")
            return MLXBackend(self.config, eager=self.config.mlx_eager)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.config.pipeline is PipelineMode.EDIT and not request.image_paths:
            raise ValueError("the edit pipeline requires one or more image_paths")
        if self.config.pipeline is PipelineMode.STANDARD and request.image_paths:
            raise ValueError("image_paths require pipeline=edit")
        try:
            return self.backend.generate(request)
        except ArtifactError:
            if self.config.mode is not ExecutionMode.AUTO or not self.config.allow_auto_fallback:
                raise
            if isinstance(self.backend, ANEHybridBackend):
                return self.backend.generate_mlx_fallback(request)
            if self._fallback is None:
                self._fallback = MLXBackend(self.config, eager=self.config.mlx_eager)
            return self._fallback.generate(request)

    def generate_many(self, requests: Iterable[GenerationRequest]) -> Iterator[GenerationResult]:
        for request in requests:
            yield self.generate(request)

    def close(self) -> None:
        self.backend.close()
        if self._fallback is not None:
            self._fallback.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
