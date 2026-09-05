"""MLX attention + Core ML MLP overlap used by the hybrid backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flux2_engine.mflux_runtime import MFluxSymbols

_ORIGINAL_ATTENTION_CALL = None
_INSTALLED_ATTENTION_CLASS = None


@dataclass(slots=True)
class HybridAttentionBinding:
    bridge: Any
    block: int
    bucket: int
    symbols: MFluxSymbols
    output: Any
    mlp_width: int | None = None
    ane_mlp_end: int | None = None
    calls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)


def linear_output_slice(symbols: MFluxSymbols, layer, inputs, output_features: int):
    """Apply leading output rows without unpacking MLX quantized weights."""
    mx = symbols.mx
    if isinstance(layer, symbols.nn.QuantizedLinear):
        return mx.quantized_matmul(
            inputs,
            layer.weight[:output_features],
            scales=layer.scales[:output_features],
            biases=layer.biases[:output_features],
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    output = mx.matmul(inputs, mx.transpose(layer.weight[:output_features]))
    if getattr(layer, "bias", None) is not None:
        output = output + layer.bias[:output_features]
    return output


def linear_output_range(
    symbols: MFluxSymbols,
    layer,
    inputs,
    start_feature: int,
    end_feature: int,
):
    """Apply a contiguous output-row range without unpacking quantized weights."""
    mx = symbols.mx
    if isinstance(layer, symbols.nn.QuantizedLinear):
        return mx.quantized_matmul(
            inputs,
            layer.weight[start_feature:end_feature],
            scales=layer.scales[start_feature:end_feature],
            biases=layer.biases[start_feature:end_feature],
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    output = mx.matmul(
        inputs,
        mx.transpose(layer.weight[start_feature:end_feature]),
    )
    if getattr(layer, "bias", None) is not None:
        output = output + layer.bias[start_feature:end_feature]
    return output


def linear_input_slice(symbols: MFluxSymbols, layer, inputs, input_features: int):
    """Apply leading input columns at a complete quantization-group boundary."""
    mx = symbols.mx
    if isinstance(layer, symbols.nn.QuantizedLinear):
        if input_features % layer.group_size:
            raise ValueError("quantized input slice must end on a complete quantization group")
        values_per_word = 32 // layer.bits
        packed_features = input_features // values_per_word
        groups = input_features // layer.group_size
        return mx.quantized_matmul(
            inputs,
            layer.weight[:, :packed_features],
            scales=layer.scales[:, :groups],
            biases=layer.biases[:, :groups],
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    return mx.matmul(inputs, mx.transpose(layer.weight[:, :input_features]))


def linear_input_range(
    symbols: MFluxSymbols,
    layer,
    inputs,
    start_feature: int,
    end_feature: int,
):
    """Apply a contiguous input-column range as one additive output partial."""
    mx = symbols.mx
    if isinstance(layer, symbols.nn.QuantizedLinear):
        if start_feature % layer.group_size or end_feature % layer.group_size:
            raise ValueError(
                "quantized input range must align to complete quantization groups"
            )
        values_per_word = 32 // layer.bits
        packed_start = start_feature // values_per_word
        packed_end = end_feature // values_per_word
        group_start = start_feature // layer.group_size
        group_end = end_feature // layer.group_size
        return mx.quantized_matmul(
            inputs,
            layer.weight[:, packed_start:packed_end],
            scales=layer.scales[:, group_start:group_end],
            biases=layer.biases[:, group_start:group_end],
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    return mx.matmul(
        inputs,
        mx.transpose(layer.weight[:, start_feature:end_feature]),
    )


def make_compiled_gpu_attention(
    symbols: MFluxSymbols,
    attention,
    *,
    gpu_mlp_start: int | None = None,
):
    mx = symbols.mx
    precision = symbols.model_config.precision
    attention_utils = symbols.attention_utils

    def gpu_attention(hidden_states, cos, sin):
        batch, seq_len, _ = hidden_states.shape
        qkv = linear_output_slice(
            symbols,
            attention.to_qkv_mlp_proj,
            hidden_states,
            attention.inner_dim * 3,
        )
        query, key, value = mx.split(qkv, 3, axis=-1)
        query = mx.transpose(
            mx.reshape(query, (batch, seq_len, attention.heads, attention.dim_head)),
            (0, 2, 1, 3),
        )
        key = mx.transpose(
            mx.reshape(key, (batch, seq_len, attention.heads, attention.dim_head)),
            (0, 2, 1, 3),
        )
        value = mx.transpose(
            mx.reshape(value, (batch, seq_len, attention.heads, attention.dim_head)),
            (0, 2, 1, 3),
        )
        query = attention.norm_q(query.astype(mx.float32)).astype(precision)
        key = attention.norm_k(key.astype(mx.float32)).astype(precision)
        query, key = attention_utils.apply_rope_bshd(query, key, cos, sin)
        attention_output = attention_utils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch,
            num_heads=attention.heads,
            head_dim=attention.dim_head,
        )
        attention_partial = linear_input_slice(
            symbols,
            attention.to_out,
            attention_output,
            attention.inner_dim,
        )
        mlp_partial = _mlx_mlp_complement(
            symbols,
            attention,
            hidden_states,
            gpu_mlp_start,
        )
        return (
            attention_partial
            if mlp_partial is None
            else attention_partial + mlp_partial
        )

    return mx.compile(gpu_attention)


def install_hybrid_attention_dispatch(symbols: MFluxSymbols) -> None:
    """Install one process-wide dispatcher; unbound mflux models remain unchanged."""
    global _INSTALLED_ATTENTION_CLASS, _ORIGINAL_ATTENTION_CALL
    attention_class = symbols.attention_class
    if _INSTALLED_ATTENTION_CLASS is attention_class:
        return
    if _INSTALLED_ATTENTION_CLASS is not None:
        raise RuntimeError("a different mflux attention class is already patched")
    _INSTALLED_ATTENTION_CLASS = attention_class
    _ORIGINAL_ATTENTION_CALL = attention_class.__call__
    attention_class.__call__ = _hybrid_attention_call


def _hybrid_attention_call(self, hidden_states, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
    binding: HybridAttentionBinding | None = getattr(self, "_flux2_engine_ane", None)
    if binding is None:
        return _ORIGINAL_ATTENTION_CALL(
            self,
            hidden_states,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_layer_idx=kv_cache_layer_idx,
        )
    symbols = binding.symbols
    mx = symbols.mx
    batch, seq_len, _ = hidden_states.shape
    if batch != 1 or seq_len > binding.bucket:
        raise ValueError(
            f"ANE attention binding expects batch=1 and sequence <= {binding.bucket}; got {hidden_states.shape}"
        )

    fp16_input = hidden_states.astype(mx.float16)
    if seq_len < binding.bucket:
        fp16_input = mx.pad(fp16_input, ((0, 0), (0, binding.bucket - seq_len), (0, 0)))
    mx.eval(hidden_states, fp16_input)

    gpu_mlp_start = (
        binding.ane_mlp_end
        if binding.mlp_width is not None
        and binding.ane_mlp_end is not None
        and binding.ane_mlp_end < binding.mlp_width
        else None
    )
    compiled = getattr(self, "_flux2_engine_compiled_attention", None)
    if compiled is not None and kv_cache is None and image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        attention_output = compiled(hidden_states, cos, sin)
    else:
        attention_output = _mlx_attention_branch(
            symbols,
            self,
            hidden_states,
            image_rotary_emb,
            kv_cache,
            kv_cache_layer_idx,
            gpu_mlp_start,
        )
    mx.async_eval(attention_output)
    result = binding.bridge.predict(fp16_input, binding.output)
    binding.calls.append((binding.block, result))
    if not result["output_backing_used"]:
        raise RuntimeError("Core ML rejected the caller-owned MLX output backing")
    return attention_output + binding.output[:, :seq_len].astype(attention_output.dtype)


def _mlx_attention_branch(
    symbols: MFluxSymbols,
    attention,
    hidden_states,
    image_rotary_emb,
    kv_cache,
    kv_cache_layer_idx,
    gpu_mlp_start: int | None = None,
):
    mx = symbols.mx
    batch, seq_len, _ = hidden_states.shape
    qkv = linear_output_slice(
        symbols,
        attention.to_qkv_mlp_proj,
        hidden_states,
        attention.inner_dim * 3,
    )
    query, key, value = mx.split(qkv, 3, axis=-1)
    shape = (batch, seq_len, attention.heads, attention.dim_head)
    query = mx.transpose(mx.reshape(query, shape), (0, 2, 1, 3))
    key = mx.transpose(mx.reshape(key, shape), (0, 2, 1, 3))
    value = mx.transpose(mx.reshape(value, shape), (0, 2, 1, 3))
    precision = symbols.model_config.precision
    query = attention.norm_q(query.astype(mx.float32)).astype(precision)
    key = attention.norm_k(key.astype(mx.float32)).astype(precision)
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        query, key = symbols.attention_utils.apply_rope_bshd(query, key, cos, sin)
    if kv_cache is not None:
        kv_cache.store_reference("single", kv_cache_layer_idx, key, value)
        key, value = kv_cache.append_reference("single", kv_cache_layer_idx, key, value)
        output = kv_cache.compute_extract_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch,
            num_heads=attention.heads,
            head_dim=attention.dim_head,
        )
    else:
        output = symbols.attention_utils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch,
            num_heads=attention.heads,
            head_dim=attention.dim_head,
        )
    attention_partial = linear_input_slice(
        symbols, attention.to_out, output, attention.inner_dim
    )
    mlp_partial = _mlx_mlp_complement(
        symbols,
        attention,
        hidden_states,
        gpu_mlp_start,
    )
    return (
        attention_partial
        if mlp_partial is None
        else attention_partial + mlp_partial
    )


def _mlx_mlp_complement(
    symbols: MFluxSymbols,
    attention,
    hidden_states,
    gpu_mlp_start: int | None,
):
    if gpu_mlp_start is None or gpu_mlp_start >= attention.mlp_hidden_dim:
        return None
    mlp_end = attention.mlp_hidden_dim
    projection_offset = 3 * attention.inner_dim
    gate = linear_output_range(
        symbols,
        attention.to_qkv_mlp_proj,
        hidden_states,
        projection_offset + gpu_mlp_start,
        projection_offset + mlp_end,
    )
    up = linear_output_range(
        symbols,
        attention.to_qkv_mlp_proj,
        hidden_states,
        projection_offset + mlp_end + gpu_mlp_start,
        projection_offset + 2 * mlp_end,
    )
    activated = attention.mlp_act(symbols.mx.concatenate((gate, up), axis=-1))
    return linear_input_range(
        symbols,
        attention.to_out,
        activated,
        attention.inner_dim + gpu_mlp_start,
        attention.inner_dim + mlp_end,
    )
