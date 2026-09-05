"""Public API for the FLUX.2 Mac inference engine."""

from flux2_engine.config import (
    AttentionMode,
    EngineConfig,
    ExecutionMode,
    GenerationProfile,
    GenerationRequest,
    ModelVariant,
    PipelineMode,
)
from flux2_engine.engine import Flux2Engine
from flux2_engine.result import GenerationMetrics, GenerationResult

__all__ = [
    "AttentionMode",
    "EngineConfig",
    "ExecutionMode",
    "Flux2Engine",
    "GenerationMetrics",
    "GenerationProfile",
    "GenerationRequest",
    "GenerationResult",
    "ModelVariant",
    "PipelineMode",
]
