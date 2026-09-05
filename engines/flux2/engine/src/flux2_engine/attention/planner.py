"""Static traffic/Amdahl analysis for FLUX.2 attention placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class AttentionCandidate:
    name: str
    status: str
    sequence_length: int
    qkv_output_bytes: int
    materialized_score_bytes: int
    reason: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def analyze_attention_candidates(
    *,
    width: int,
    height: int,
    text_tokens: int = 64,
    hidden_size: int = 3072,
    heads: int = 24,
) -> tuple[AttentionCandidate, ...]:
    image_tokens = (width // 16) * (height // 16)
    sequence = image_tokens + text_tokens
    qkv_bytes = 3 * sequence * hidden_size * 2
    score_bytes = heads * sequence * sequence * 2
    return (
        AttentionCandidate(
            name="mlx-fused-sdpa+ane-mlp-overlap",
            status="production",
            sequence_length=sequence,
            qkv_output_bytes=qkv_bytes,
            materialized_score_bytes=0,
            reason="keeps QKV and fused SDPA on Metal while the ANE independently computes the MLP branch",
        ),
        AttentionCandidate(
            name="ane-qkv+mlx-attention",
            status="rejected",
            sequence_length=sequence,
            qkv_output_bytes=qkv_bytes,
            materialized_score_bytes=0,
            reason="GPU attention must wait for ANE QKV, destroying current GPU/ANE overlap and returning a large QKV tensor",
        ),
        AttentionCandidate(
            name="ane-full-attention+mlx-mlp",
            status="rejected-measured",
            sequence_length=sequence,
            qkv_output_bytes=0,
            materialized_score_bytes=score_bytes,
            reason=(
                "public Core ML SDPA was 20.7x slower at sequence 1088 and 17.8x slower at sequence 4160 than MLX; "
                "the 4160 output also failed the FP32 reference gate"
            ),
        ),
        AttentionCandidate(
            name="gpu-ane-head-split",
            status="research",
            sequence_length=sequence,
            qkv_output_bytes=qkv_bytes,
            materialized_score_bytes=score_bytes,
            reason="can overlap head groups but competes with the already-profitable ANE MLP; requires a head-sharded fused Core ML model",
        ),
    )
