"""Persistent zero-copy MLX GPU + Core ML ANE backend."""

from __future__ import annotations

import importlib
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flux2_engine.artifacts import ANEArtifactCatalog, ResolvedANEPlan
from flux2_engine.backends.hybrid_ops import (
    HybridAttentionBinding,
    install_hybrid_attention_dispatch,
    make_compiled_gpu_attention,
)
from flux2_engine.config import EngineConfig, GenerationRequest
from flux2_engine.errors import BackendUnavailableError
from flux2_engine.mflux_runtime import (
    MFluxSymbols,
    configure_text_length,
    eager_mflux_predict,
    generation_kwargs,
    load_mflux_symbols,
    model_class_for,
    model_config_for,
)
from flux2_engine.result import GenerationMetrics, GenerationResult


@dataclass(slots=True)
class HybridSession:
    plan: ResolvedANEPlan
    bindings: dict[int, HybridAttentionBinding]
    load_seconds: float
    load_metrics: dict[int, dict[str, Any]]

    @property
    def calls(self) -> list[tuple[int, dict[str, Any]]]:
        calls: list[tuple[int, dict[str, Any]]] = []
        for binding in self.bindings.values():
            calls.extend(binding.calls)
        return calls


class ANEHybridBackend:
    name = "mlx-ane"

    def __init__(self, config: EngineConfig):
        self.config = config
        self.symbols: MFluxSymbols = load_mflux_symbols(config.mflux_root, config.precision)
        self.catalog = ANEArtifactCatalog.from_paths(config.ane_manifests)
        self.bridge_module = _load_bridge_module(config)
        install_hybrid_attention_dispatch(self.symbols)
        self.model = model_class_for(self.symbols, config)(
            model_config=model_config_for(self.symbols, config),
            model_path=str(config.model_path),
        )
        self.sessions: dict[tuple[Path, int, str, tuple[int, ...]], HybridSession] = {}
        self.lock = threading.RLock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self.lock:
            self.symbols.mx.reset_peak_memory()
            sequence_length = self._sequence_length(request)
            plan = self.catalog.resolve(
                sequence_length=sequence_length,
                variant=self.config.ane_variant,
                requested_blocks=self.config.ane_blocks,
                k=self.model.transformer.inner_dim,
                n=self.model.transformer.inner_dim,
                block_count=len(self.model.transformer.single_transformer_blocks),
            )
            session, session_was_loaded = self._session(plan)
            self._activate(session)
            calls_before = {block: len(binding.calls) for block, binding in session.bindings.items()}
            started = time.perf_counter()
            with eager_mflux_predict(self.symbols):
                image = self.model.generate_image(
                    **generation_kwargs(self.model, request)
                )
            wall_seconds = time.perf_counter() - started
            current_calls = []
            for block, binding in session.bindings.items():
                current_calls.extend(binding.calls[calls_before[block] :])
            elapsed = [result["elapsed_ms"] for _, result in current_calls]
            elapsed_by_block = {
                str(block): [
                    result["elapsed_ms"]
                    for call_block, result in current_calls
                    if call_block == block
                ]
                for block in plan.blocks
            }
            result = GenerationResult(
                image=image,
                metrics=GenerationMetrics(
                    backend=self.name,
                    wall_seconds=wall_seconds,
                    peak_mlx_gb=self.symbols.mx.get_peak_memory() / 1e9,
                    actual_sequence_length=sequence_length,
                    bucket=plan.bucket,
                    ane_call_count=len(current_calls),
                    ane_call_p50_ms=statistics.median(elapsed) if elapsed else None,
                    metadata={
                        "manifest": str(plan.manifest.path),
                        "model_variant": self.config.resolved_model_variant.value,
                        "conditioning_mode": request.conditioning_mode,
                        "reference_image_count": len(request.image_paths),
                        "image_strength": (
                            request.image_strength if request.image_path is not None else None
                        ),
                        "blocks": list(plan.blocks),
                        "variant": plan.variant,
                        "session_loaded_this_request": session_was_loaded,
                        "session_load_seconds": session.load_seconds if session_was_loaded else 0.0,
                        "attention_backend": "mlx",
                        "mlp_backend": "coreml-cpu-ane",
                        "output_backing_hit_rate": (
                            sum(call[1]["output_binding_cache_hit"] for call in current_calls) / len(current_calls)
                            if current_calls
                            else None
                        ),
                        "input_binding_hit_rate": (
                            sum(call[1]["input_binding_cache_hit"] for call in current_calls) / len(current_calls)
                            if current_calls
                            else None
                        ),
                        "ane_call_total_ms": sum(elapsed),
                        "ane_call_max_ms": max(elapsed) if elapsed else None,
                        "ane_block_p50_ms": {
                            block: statistics.median(values)
                            for block, values in elapsed_by_block.items()
                            if values
                        },
                    },
                ),
            )
            if request.output is not None:
                result.save(request.output)
            if self.config.clear_mlx_cache_between_requests:
                self.symbols.mx.clear_cache()
            return result

    def generate_mlx_fallback(self, request: GenerationRequest) -> GenerationResult:
        """Reuse the already loaded model when auto mode has no fitting ANE bucket."""
        with self.lock:
            self.symbols.mx.reset_peak_memory()
            self._deactivate()
            started = time.perf_counter()
            if self.config.mlx_eager:
                with eager_mflux_predict(self.symbols):
                    image = self._generate_mlx(request)
            else:
                image = self._generate_mlx(request)
            result = GenerationResult(
                image=image,
                metrics=GenerationMetrics(
                    backend="mlx-fallback",
                    wall_seconds=time.perf_counter() - started,
                    peak_mlx_gb=self.symbols.mx.get_peak_memory() / 1e9,
                    metadata={"reason": "no fitting ANE artifact"},
                ),
            )
            if request.output is not None:
                result.save(request.output)
            if self.config.clear_mlx_cache_between_requests:
                self.symbols.mx.clear_cache()
            return result

    def _generate_mlx(self, request: GenerationRequest):
        return self.model.generate_image(**generation_kwargs(self.model, request))

    def _sequence_length(self, request: GenerationRequest) -> int:
        configure_text_length(self.model, request.dynamic_text_length)
        tokens = self.model.tokenizers["qwen3"].tokenize(
            prompt=request.prompt,
            max_length=512,
            padding="longest",
        )
        text_tokens = int(tokens.input_ids.shape[1])
        image_tokens = (request.height // 16) * (request.width // 16)
        return text_tokens + image_tokens

    def _session(self, plan: ResolvedANEPlan) -> tuple[HybridSession, bool]:
        key = (plan.manifest.path, plan.bucket, plan.variant, plan.blocks)
        if key in self.sessions:
            return self.sessions[key], False
        started = time.perf_counter()
        bindings: dict[int, HybridAttentionBinding] = {}
        load_metrics: dict[int, dict[str, Any]] = {}
        for block in plan.blocks:
            bridge = self.bridge_module.Flux2ANEModel(
                model_path=str(plan.artifact_for(block)),
                function_name=plan.function_name,
                m=plan.bucket,
                k=plan.manifest.k,
                n=plan.manifest.n,
                cache_bindings=True,
                fast_prediction=False,
            )
            metrics = dict(bridge.load_metrics())
            load_metrics[block] = metrics
            attention = self.model.transformer.single_transformer_blocks[block].attn
            output = self.symbols.mx.zeros(
                (1, plan.bucket, attention.inner_dim),
                dtype=self.symbols.mx.float16,
            )
            bindings[block] = HybridAttentionBinding(
                bridge=bridge,
                block=block,
                bucket=plan.bucket,
                symbols=self.symbols,
                output=output,
                mlp_width=plan.manifest.mlp_width,
                ane_mlp_end=plan.manifest.ane_mlp_end,
            )
        self.symbols.mx.eval(*(binding.output for binding in bindings.values()))
        session = HybridSession(
            plan=plan,
            bindings=bindings,
            load_seconds=time.perf_counter() - started,
            load_metrics=load_metrics,
        )
        self.sessions[key] = session
        return session, True

    def _activate(self, session: HybridSession) -> None:
        for block in range(len(self.model.transformer.single_transformer_blocks)):
            attention = self.model.transformer.single_transformer_blocks[block].attn
            attention._flux2_engine_ane = session.bindings.get(block)
        if self.config.compile_quantized_gpu_attention:
            for block in session.plan.blocks:
                attention = self.model.transformer.single_transformer_blocks[block].attn
                binding = session.bindings[block]
                gpu_mlp_start = (
                    binding.ane_mlp_end
                    if binding.mlp_width is not None
                    and binding.ane_mlp_end is not None
                    and binding.ane_mlp_end < binding.mlp_width
                    else None
                )
                compiled_partition = getattr(
                    attention, "_flux2_engine_compiled_mlp_start", object()
                )
                should_compile = (
                    gpu_mlp_start is not None
                    or isinstance(
                        attention.to_qkv_mlp_proj,
                        self.symbols.nn.QuantizedLinear,
                    )
                )
                if should_compile and compiled_partition != gpu_mlp_start:
                    attention._flux2_engine_compiled_attention = make_compiled_gpu_attention(
                        self.symbols,
                        attention,
                        gpu_mlp_start=gpu_mlp_start,
                    )
                    attention._flux2_engine_compiled_mlp_start = gpu_mlp_start

    def _deactivate(self) -> None:
        for block in range(len(self.model.transformer.single_transformer_blocks)):
            attention = self.model.transformer.single_transformer_blocks[block].attn
            attention._flux2_engine_ane = None

    def close(self) -> None:
        self.sessions.clear()
        self.model = None


def _load_bridge_module(config: EngineConfig):
    candidates: list[Path] = []
    if config.bridge_dir is not None:
        candidates.append(config.bridge_dir)
    engine_root = Path(__file__).resolve().parents[3]
    candidates.append(engine_root / "build")
    candidates.append(config.mflux_root.parents[1] / "mac_local_ai" / "build")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("_flux2_ane_bridge*.so")):
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            try:
                return importlib.import_module("_flux2_ane_bridge")
            except ImportError:
                continue
    searched = ", ".join(str(path) for path in candidates)
    raise BackendUnavailableError(f"native ANE bridge was not found in: {searched}; run scripts/build_native.py")
