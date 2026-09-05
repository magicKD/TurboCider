"""Persistent pure-MLX FLUX.2 backend."""

from __future__ import annotations

import time

from flux2_engine.config import EngineConfig, GenerationRequest
from flux2_engine.mflux_runtime import (
    MFluxSymbols,
    eager_mflux_predict,
    generation_kwargs,
    load_mflux_symbols,
    model_class_for,
    model_config_for,
)
from flux2_engine.result import GenerationMetrics, GenerationResult


class MLXBackend:
    name = "mlx"

    def __init__(self, config: EngineConfig, *, eager: bool = False):
        self.config = config
        self.eager = eager
        self.symbols: MFluxSymbols = load_mflux_symbols(config.mflux_root, config.precision)
        self.model = model_class_for(self.symbols, config)(
            model_config=model_config_for(self.symbols, config),
            model_path=str(config.model_path),
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.symbols.mx.reset_peak_memory()
        started = time.perf_counter()
        if self.eager:
            with eager_mflux_predict(self.symbols):
                image = self._generate(request)
        else:
            image = self._generate(request)
        wall_seconds = time.perf_counter() - started
        result = GenerationResult(
            image=image,
            metrics=GenerationMetrics(
                backend="mlx-eager" if self.eager else "mlx",
                wall_seconds=wall_seconds,
                peak_mlx_gb=self.symbols.mx.get_peak_memory() / 1e9,
                metadata={
                    "precision": self.config.precision,
                    "model_variant": self.config.resolved_model_variant.value,
                    "steps": request.resolved_steps,
                    "dynamic_text_length": request.dynamic_text_length,
                    "conditioning_mode": request.conditioning_mode,
                    "reference_image_count": len(request.image_paths),
                    "image_strength": (
                        request.image_strength if request.image_path is not None else None
                    ),
                },
            ),
        )
        if request.output is not None:
            result.save(request.output)
        if self.config.clear_mlx_cache_between_requests:
            self.symbols.mx.clear_cache()
        return result

    def _generate(self, request: GenerationRequest):
        return self.model.generate_image(**generation_kwargs(self.model, request))

    def close(self) -> None:
        self.model = None
