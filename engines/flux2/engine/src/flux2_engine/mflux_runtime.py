"""Lazy loading of the local mflux/MLX model kernel."""

from __future__ import annotations

import sys
import threading
from inspect import signature
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flux2_engine.errors import BackendUnavailableError

_EAGER_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class MFluxSymbols:
    mx: Any
    nn: Any
    model_config: Any
    flux2_class: Any
    flux2_edit_class: Any
    attention_class: Any
    attention_utils: Any
    apple_silicon_util: Any


def generation_kwargs(model: Any, request: Any) -> dict[str, Any]:
    """Map engine requests across compatible mflux FLUX.2 API revisions."""
    configure_text_length(model, request.dynamic_text_length)
    kwargs = {
        "seed": request.seed,
        "prompt": request.prompt,
        "num_inference_steps": request.resolved_steps,
        "width": request.width,
        "height": request.height,
        "guidance": request.guidance,
    }
    parameters = signature(model.generate_image).parameters
    image_paths = tuple(getattr(request, "image_paths", ()))
    image_path = getattr(request, "image_path", None)
    if image_paths:
        if "image_paths" not in parameters:
            raise BackendUnavailableError(
                "the selected FLUX.2 pipeline does not accept reference images"
            )
        kwargs["image_paths"] = list(image_paths)
    elif image_path is not None:
        if "image_path" not in parameters:
            raise BackendUnavailableError(
                "the selected FLUX.2 pipeline does not accept an init image"
            )
        kwargs["image_path"] = image_path
        if "image_strength" in parameters:
            kwargs["image_strength"] = getattr(request, "image_strength", 0.75)
    if "dynamic_text_length" in parameters:
        kwargs["dynamic_text_length"] = request.dynamic_text_length
    return kwargs


def configure_text_length(model: Any, dynamic_text_length: bool) -> None:
    """Preserve the engine's dynamic-text contract on newer mflux releases."""
    tokenizer = getattr(model, "tokenizers", {}).get("qwen3")
    if tokenizer is not None and hasattr(tokenizer, "padding"):
        tokenizer.padding = "longest" if dynamic_text_length else "max_length"


def model_config_for(symbols: MFluxSymbols, config: Any) -> Any:
    """Return the mflux configuration matching the resolved local checkpoint."""
    factories = {
        "flux2-klein-4b": "flux2_klein_4b",
        "flux2-klein-9b": "flux2_klein_9b",
        "flux2-klein-9b-kv": "flux2_klein_9b_kv",
    }
    variant = config.resolved_model_variant.value
    try:
        factory = getattr(symbols.model_config, factories[variant])
    except (KeyError, AttributeError) as error:
        raise BackendUnavailableError(
            f"the installed mflux source does not support {variant}"
        ) from error
    return factory()


def load_mflux_symbols(mflux_root: Path, precision: str) -> MFluxSymbols:
    source = (mflux_root / "src").resolve()
    if not source.is_dir():
        raise BackendUnavailableError(f"mflux source directory does not exist: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
        from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import (
            Flux2ParallelSelfAttention,
        )
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
        from mflux.utils.apple_silicon import AppleSiliconUtil
        from mlx import nn
    except ImportError as error:
        raise BackendUnavailableError(
            "mflux/MLX could not be imported; run with the mflux virtual environment"
        ) from error
    ModelConfig.precision = mx.bfloat16 if precision == "bf16" else mx.float16
    return MFluxSymbols(
        mx=mx,
        nn=nn,
        model_config=ModelConfig,
        flux2_class=Flux2Klein,
        flux2_edit_class=Flux2KleinEdit,
        attention_class=Flux2ParallelSelfAttention,
        attention_utils=AttentionUtils,
        apple_silicon_util=AppleSiliconUtil,
    )


def model_class_for(symbols: MFluxSymbols, config: Any) -> Any:
    return (
        symbols.flux2_edit_class
        if config.pipeline.value == "edit"
        else symbols.flux2_class
    )


@contextmanager
def eager_mflux_predict(symbols: MFluxSymbols) -> Iterator[None]:
    """Force the eager outer transformer boundary required by external Core ML."""
    with _EAGER_PATCH_LOCK:
        util = symbols.apple_silicon_util
        original = util.__dict__["is_m1_or_m2"]
        util.is_m1_or_m2 = classmethod(lambda cls: True)
        try:
            yield
        finally:
            util.is_m1_or_m2 = original
