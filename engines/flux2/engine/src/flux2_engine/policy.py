"""Backend selection policy kept separate from backend implementations."""

from __future__ import annotations

from dataclasses import dataclass

from flux2_engine.config import AttentionMode, EngineConfig, ExecutionMode, PipelineMode
from flux2_engine.errors import AttentionPlanUnavailableError


@dataclass(frozen=True, slots=True)
class BackendDecision:
    mode: ExecutionMode
    reason: str


def validate_attention_mode(mode: AttentionMode) -> None:
    if mode in {AttentionMode.ANE_EXPERIMENTAL, AttentionMode.HEAD_SPLIT_EXPERIMENTAL}:
        raise AttentionPlanUnavailableError(
            f"{mode.value} has not passed the end-to-end latency and quality gate; use attention=mlx or attention=auto"
        )


def choose_backend(config: EngineConfig) -> BackendDecision:
    validate_attention_mode(config.attention)
    if config.pipeline is PipelineMode.EDIT:
        if config.mode is ExecutionMode.MLX_ANE:
            raise AttentionPlanUnavailableError(
                "FLUX.2 multi-reference edit changes the token sequence; "
                "the current fixed-shape ANE artifacts only support standard generation"
            )
        return BackendDecision(
            ExecutionMode.MLX,
            "multi-reference edit uses the variable-length pure MLX pipeline",
        )
    if config.mode is ExecutionMode.MLX:
        return BackendDecision(ExecutionMode.MLX, "pure MLX explicitly requested")
    if config.mode is ExecutionMode.MLX_ANE:
        return BackendDecision(ExecutionMode.MLX_ANE, "MLX+ANE explicitly requested")
    if config.persistent and config.ane_manifests:
        return BackendDecision(
            ExecutionMode.MLX_ANE,
            "persistent workload with available ANE artifacts",
        )
    return BackendDecision(
        ExecutionMode.MLX,
        "one-shot/portable auto policy avoids Core ML package startup",
    )
