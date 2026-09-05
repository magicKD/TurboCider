"""Typed user configuration for the inference engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path


class ExecutionMode(StrEnum):
    AUTO = "auto"
    MLX = "mlx"
    MLX_ANE = "mlx-ane"


class AttentionMode(StrEnum):
    AUTO = "auto"
    MLX = "mlx"
    ANE_EXPERIMENTAL = "ane-experimental"
    HEAD_SPLIT_EXPERIMENTAL = "head-split-experimental"


class GenerationProfile(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    PREVIEW = "preview"

    @property
    def default_steps(self) -> int:
        return {
            GenerationProfile.QUALITY: 4,
            GenerationProfile.BALANCED: 3,
            GenerationProfile.PREVIEW: 2,
        }[self]


class ModelVariant(StrEnum):
    AUTO = "auto"
    KLEIN_4B = "flux2-klein-4b"
    KLEIN_9B = "flux2-klein-9b"
    KLEIN_9B_KV = "flux2-klein-9b-kv"


class PipelineMode(StrEnum):
    STANDARD = "standard"
    EDIT = "edit"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_path: Path
    mflux_root: Path
    model_variant: ModelVariant = ModelVariant.AUTO
    pipeline: PipelineMode = PipelineMode.STANDARD
    mode: ExecutionMode = ExecutionMode.AUTO
    attention: AttentionMode = AttentionMode.AUTO
    precision: str = "bf16"
    mlx_eager: bool = False
    ane_manifests: tuple[Path, ...] = ()
    bridge_dir: Path | None = None
    persistent: bool = False
    allow_auto_fallback: bool = True
    ane_variant: str = "int8_pc"
    ane_blocks: tuple[int, ...] | None = None
    reuse_ane_outputs: bool = True
    compile_quantized_gpu_attention: bool = True
    clear_mlx_cache_between_requests: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", ExecutionMode(self.mode))
        if isinstance(self.attention, str):
            object.__setattr__(self, "attention", AttentionMode(self.attention))
        if isinstance(self.model_variant, str):
            object.__setattr__(self, "model_variant", ModelVariant(self.model_variant))
        if isinstance(self.pipeline, str):
            object.__setattr__(self, "pipeline", PipelineMode(self.pipeline))
        object.__setattr__(self, "model_path", self.model_path.expanduser().resolve())
        object.__setattr__(self, "mflux_root", self.mflux_root.expanduser().resolve())
        object.__setattr__(
            self,
            "ane_manifests",
            tuple(path.expanduser().resolve() for path in self.ane_manifests),
        )
        if self.bridge_dir is not None:
            object.__setattr__(self, "bridge_dir", self.bridge_dir.expanduser().resolve())
        if self.precision not in {"bf16", "fp16"}:
            raise ValueError("precision must be 'bf16' or 'fp16'")
        if self.ane_variant not in {"int8_pc", "fp16"}:
            raise ValueError("ane_variant must be 'int8_pc' or 'fp16'")
        if self.mode is ExecutionMode.MLX_ANE and not self.ane_manifests:
            raise ValueError("mlx-ane mode requires at least one ANE manifest")

    @property
    def resolved_model_variant(self) -> ModelVariant:
        detected = self._detect_model_variant()
        if self.model_variant is ModelVariant.AUTO:
            return detected or ModelVariant.KLEIN_4B
        if detected is not None:
            explicit_family = self._architecture_family(self.model_variant)
            detected_family = self._architecture_family(detected)
            if explicit_family is not detected_family:
                raise ValueError(
                    "model variant does not match transformer/config.json: "
                    f"requested={self.model_variant.value}, detected={detected.value}"
                )
        return self.model_variant

    @staticmethod
    def _architecture_family(variant: ModelVariant) -> ModelVariant:
        if variant is ModelVariant.KLEIN_9B_KV:
            return ModelVariant.KLEIN_9B
        return variant

    def _detect_model_variant(self) -> ModelVariant | None:
        config_path = self.model_path / "transformer" / "config.json"
        if config_path.is_file():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                signature = (
                    int(raw["num_layers"]),
                    int(raw["num_single_layers"]),
                    int(raw["num_attention_heads"]),
                    int(raw["joint_attention_dim"]),
                )
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise ValueError(f"invalid FLUX.2 transformer config: {config_path}: {error}") from error
            variants = {
                (5, 20, 24, 7680): ModelVariant.KLEIN_4B,
                (8, 24, 32, 12288): ModelVariant.KLEIN_9B,
            }
            if signature not in variants:
                raise ValueError(
                    "unsupported FLUX.2 transformer configuration: "
                    f"layers={signature[0]}, single_layers={signature[1]}, "
                    f"heads={signature[2]}, joint_attention_dim={signature[3]}"
                )
            detected = variants[signature]
            if detected is ModelVariant.KLEIN_9B and "9b-kv" in self.model_path.name.lower():
                return ModelVariant.KLEIN_9B_KV
            return detected

        normalized = self.model_path.name.lower().replace("_", "-")
        if "9b-kv" in normalized:
            return ModelVariant.KLEIN_9B_KV
        if "9b" in normalized:
            return ModelVariant.KLEIN_9B
        if "4b" in normalized:
            return ModelVariant.KLEIN_4B
        return None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    output: Path | None = None
    width: int = 512
    height: int = 512
    steps: int | None = None
    seed: int = 42
    guidance: float = 1.0
    profile: GenerationProfile = GenerationProfile.QUALITY
    dynamic_text_length: bool = True
    image_path: Path | None = None
    image_paths: tuple[Path, ...] = ()
    image_strength: float = 0.75

    def __post_init__(self) -> None:
        if isinstance(self.profile, str):
            object.__setattr__(self, "profile", GenerationProfile(self.profile))
        if self.image_path is not None:
            object.__setattr__(
                self, "image_path", Path(self.image_path).expanduser().resolve()
            )
        object.__setattr__(
            self,
            "image_paths",
            tuple(Path(path).expanduser().resolve() for path in self.image_paths),
        )
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("width and height must be positive multiples of 16")
        if self.steps is not None and self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.image_path is not None and self.image_paths:
            raise ValueError("image_path and image_paths are mutually exclusive")
        if not 0.0 <= self.image_strength <= 1.0:
            raise ValueError("image_strength must be between 0 and 1")
        if self.output is not None:
            object.__setattr__(self, "output", self.output.expanduser().resolve())

    @property
    def resolved_steps(self) -> int:
        return self.steps if self.steps is not None else self.profile.default_steps

    @property
    def conditioning_mode(self) -> str:
        if self.image_paths:
            return "image_edit"
        if self.image_path is not None:
            return "image_to_image"
        return "text_to_image"

    def with_output(self, output: Path) -> GenerationRequest:
        return replace(self, output=output)
