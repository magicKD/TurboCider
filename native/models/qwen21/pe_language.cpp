#include "pe_language.hpp"
#include <cmath>

namespace tc::qwen21::pe {
namespace {
Tensor zero_norm(const Tensor &x, const Tensor &weight, float epsilon) {
    auto f = mx::astype(x, mx::float32);
    return mx::astype(f * mx::rsqrt(mx::mean(f * f, -1, true) + epsilon) *
                      (1.f + mx::astype(weight, mx::float32)), x.dtype());
}
Tensor partial_rope(const Tensor &x, const Tensor &cosine, const Tensor &sine, int dim) {
    auto rotated = slice_axis(x, -1, 0, dim);
    auto half = mx::split(rotated, 2, -1);
    rotated = rotated * cosine + mx::concatenate({-half[1], half[0]}, -1) * sine;
    if (dim == x.shape(-1)) return rotated;
    return mx::concatenate({rotated, slice_axis(x, -1, dim, x.shape(-1))}, -1);
}
}

LanguageModel::LanguageModel(const Weights &weights, LanguageConfig config)
    : weights_(weights), config_(std::move(config)), prefix_(
        weights.has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.") {
    const auto &c = config_;
    require(c.hidden > 0 && c.layers > 0 && c.heads > 0 && c.kv_heads > 0 &&
            c.heads % c.kv_heads == 0 && c.head_dim > 0 && c.rotary_dim > 0 &&
            c.rotary_dim % 2 == 0 && c.rotary_dim <= c.head_dim &&
            c.linear_key_heads > 0 && c.linear_value_heads > 0 &&
            c.linear_value_heads % c.linear_key_heads == 0 &&
            c.linear_key_dim > 0 && c.linear_value_dim > 0 && c.conv_kernel > 0 &&
            std::isfinite(c.epsilon) && c.epsilon > 0 && std::isfinite(c.theta) && c.theta > 0,
            "invalid Qwen35 language geometry");
    require(c.mrope_sections[0] + c.mrope_sections[1] + c.mrope_sections[2] == c.rotary_dim / 2,
            "invalid Qwen35 mRoPE sections");
    for (int axis = 0; axis < 3; ++axis)
        require(c.mrope_sections[axis] >= 0 && (axis == 0 || c.mrope_sections[axis] == 0 ||
                    axis + 3 * (c.mrope_sections[axis] - 1) < c.rotary_dim / 2),
                "Qwen35 mRoPE section exceeds rotary dimension");
    if (config_.full_attention.empty())
        for (int i = 0; i < c.layers; ++i) config_.full_attention.push_back((i + 1) % 4 == 0);
    require(config_.full_attention.size() == size_t(c.layers), "Qwen35 layer-types length mismatch");
    const auto &embedding = weights_.at(prefix_ + "embed_tokens.weight");
    require(embedding.ndim() == 2 && embedding.shape(1) == c.hidden, "Qwen35 embedding geometry mismatch");
    reset();
}
void LanguageModel::reset() { tokens_ = 0; states_.clear(); states_.resize(config_.layers); }

Tensor LanguageModel::linear_attention(const Tensor &x, int layer, std::atomic<bool> &cancelled) {
    const auto &c = config_;
    auto &state = states_[layer];
    const auto p = prefix_ + "layers." + std::to_string(layer) + ".linear_attn";
    const int time = x.shape(1), keys = c.linear_key_heads * c.linear_key_dim;
    const int values = c.linear_value_heads * c.linear_value_dim, channels = keys * 2 + values;
    auto qkv = linear(x, weights_, p + ".in_proj_qkv");
    auto history = state.convolution ? *state.convolution :
        mx::zeros({1, c.conv_kernel - 1, channels}, qkv.dtype());
    auto joined = mx::concatenate({history, qkv}, 1);
    state.convolution = slice_axis(joined, 1, joined.shape(1) - (c.conv_kernel - 1), joined.shape(1));
    auto kernel = mx::transpose(weights_.at(p + ".conv1d.weight"), {0, 2, 1});
    qkv = silu(mx::conv1d(joined, kernel, 1, 0, 1, channels));
    auto q = mx::reshape(slice_axis(qkv, -1, 0, keys), {1, time, c.linear_key_heads, c.linear_key_dim});
    auto k = mx::reshape(slice_axis(qkv, -1, keys, keys * 2), q.shape());
    auto v = mx::reshape(slice_axis(qkv, -1, keys * 2, channels), {1, time, c.linear_value_heads, c.linear_value_dim});
    q = mx::repeat(q, c.linear_value_heads / c.linear_key_heads, 2);
    k = mx::repeat(k, c.linear_value_heads / c.linear_key_heads, 2);
    auto beta = mx::sigmoid(linear(x, weights_, p + ".in_proj_b"));
    auto a = mx::astype(linear(x, weights_, p + ".in_proj_a"), mx::float32) +
             mx::astype(weights_.at(p + ".dt_bias"), mx::float32);
    auto softplus = mx::maximum(a, Tensor(0.f)) + mx::log1p(mx::exp(-mx::abs(a)));
    auto g = -mx::exp(mx::astype(weights_.at(p + ".A_log"), mx::float32)) * softplus;
    auto delta = c.chunked_prefill && time > 1 ?
        chunked_delta(q, k, v, g, beta, state.recurrent, true, cancelled) :
        recurrent_delta(q, k, v, g, beta, state.recurrent, true, cancelled);
    state.recurrent = delta.state;
    auto output = mx::astype(delta.output, mx::float32);
    output = mx::astype(output * mx::rsqrt(mx::mean(output * output, -1, true) + c.epsilon), x.dtype());
    output = output * weights_.at(p + ".norm.weight");
    auto z = mx::astype(mx::reshape(linear(x, weights_, p + ".in_proj_z"), delta.output.shape()), mx::float32);
    output = mx::astype(mx::astype(output, mx::float32) * silu(z), x.dtype());
    return linear(mx::reshape(output, {1, time, values}), weights_, p + ".out_proj");
}

Tensor LanguageModel::full_attention(const Tensor &x, int layer, const Tensor &cosine, const Tensor &sine) {
    const auto &c = config_;
    auto &state = states_[layer];
    const auto p = prefix_ + "layers." + std::to_string(layer) + ".self_attn";
    const int time = x.shape(1);
    auto packed = mx::reshape(linear(x, weights_, p + ".q_proj"), {1, time, c.heads, c.head_dim * 2});
    auto pieces = mx::split(packed, 2, -1);
    auto gate = mx::reshape(pieces[1], {1, time, c.heads * c.head_dim});
    auto q = mx::transpose(zero_norm(pieces[0], weights_.at(p + ".q_norm.weight"), c.epsilon), {0, 2, 1, 3});
    auto k = zero_norm(heads(linear(x, weights_, p + ".k_proj"), c.kv_heads, c.head_dim),
                       weights_.at(p + ".k_norm.weight"), c.epsilon);
    auto v = heads(linear(x, weights_, p + ".v_proj"), c.kv_heads, c.head_dim);
    q = partial_rope(q, cosine, sine, c.rotary_dim);
    k = partial_rope(k, cosine, sine, c.rotary_dim);
    if (state.k) { k = mx::concatenate({*state.k, k}, 2); v = mx::concatenate({*state.v, v}, 2); }
    state.k = k; state.v = v;
    auto queries = mx::reshape(mx::arange(tokens_, tokens_ + time, mx::int32), {time, 1});
    auto keys = mx::reshape(mx::arange(tokens_ + time, mx::int32), {1, tokens_ + time});
    auto mask = mx::reshape(mx::where(keys > queries, Tensor(-INFINITY), Tensor(0.f)), {1, 1, time, tokens_ + time});
    auto output = attend(q, mx::repeat(k, c.heads / c.kv_heads, 1),
                         mx::repeat(v, c.heads / c.kv_heads, 1), true, mask);
    return linear(output * mx::sigmoid(gate), weights_, p + ".o_proj");
}

Tensor LanguageModel::forward_ids(const Tensor &ids, const Event &event, std::atomic<bool> &cancelled) try {
    checkpoint(cancelled);
    require(ids.ndim() == 2 && ids.shape(0) == 1 && ids.shape(1) > 0 && ids.dtype() == mx::int32,
            "Qwen35 token IDs must be nonempty int32 [1,sequence]");
    const auto &embedding = weights_.at(prefix_ + "embed_tokens.weight");
    require(mx::all(mx::logical_and(ids >= Tensor(0), ids < Tensor(embedding.shape(0)))).item<bool>(),
            "Qwen35 token ID outside vocabulary");
    auto positions = mx::broadcast_to(mx::reshape(mx::arange(tokens_, tokens_ + ids.shape(1), mx::int32),
                                                  {1, ids.shape(1)}), {3, ids.shape(1)});
    return forward_embeddings(mx::take(embedding, ids, 0), positions, event, cancelled);
} catch (...) { reset(); throw; }

Tensor LanguageModel::forward_embeddings(const Tensor &embeddings, const Tensor &positions,
    const Event &event, std::atomic<bool> &cancelled) try {
    checkpoint(cancelled);
    const auto &c = config_;
    require(embeddings.ndim() == 3 && embeddings.shape(0) == 1 && embeddings.shape(1) > 0 &&
            embeddings.shape(2) == c.hidden && positions.shape() == mx::Shape{3, embeddings.shape(1)} &&
            positions.dtype() == mx::int32, "Qwen35 embedding/position geometry mismatch");
    require(embeddings.dtype() == weights_.at(prefix_ + "embed_tokens.weight").dtype(),
            "Qwen35 embedding dtype differs from checkpoint");
    const int time = embeddings.shape(1);
    require(time <= 32768 && tokens_ <= 32768 - time, "Qwen35 native PE context exceeds 32768 tokens");
    auto frequency = Tensor(1.f) / mx::power(Tensor(c.theta), mx::arange(0, c.rotary_dim, 2, mx::float32) / float(c.rotary_dim));
    std::vector<Tensor> angles, columns;
    for (int axis = 0; axis < 3; ++axis)
        angles.push_back(mx::reshape(mx::astype(slice_axis(positions, 0, axis, axis + 1), mx::float32), {time, 1}) * frequency);
    for (int i = 0; i < c.rotary_dim / 2; ++i) {
        int axis = (i % 3 == 1 && i < c.mrope_sections[1] * 3) ? 1 :
                   ((i % 3 == 2 && i < c.mrope_sections[2] * 3) ? 2 : 0);
        columns.push_back(slice_axis(angles[axis], 1, i, i + 1));
    }
    auto angle = mx::concatenate(columns, -1);
    angle = mx::concatenate({angle, angle}, -1);
    auto cosine = mx::reshape(mx::astype(mx::cos(angle), embeddings.dtype()), {1, 1, time, c.rotary_dim});
    auto sine = mx::reshape(mx::astype(mx::sin(angle), embeddings.dtype()), {1, 1, time, c.rotary_dim});
    auto hidden = embeddings;
    for (int i = 0; i < c.layers; ++i) {
        checkpoint(cancelled);
        const auto p = prefix_ + "layers." + std::to_string(i);
        auto input = zero_norm(hidden, weights_.at(p + ".input_layernorm.weight"), c.epsilon);
        hidden = hidden + (c.full_attention[i] ? full_attention(input, i, cosine, sine) : linear_attention(input, i, cancelled));
        input = zero_norm(hidden, weights_.at(p + ".post_attention_layernorm.weight"), c.epsilon);
        hidden = hidden + linear(silu(linear(input, weights_, p + ".mlp.gate_proj")) *
                                  linear(input, weights_, p + ".mlp.up_proj"), weights_, p + ".mlp.down_proj");
        std::vector<Tensor> evaluated{hidden};
        const auto &state = states_[i];
        for (const auto *tensor : {&state.k, &state.v, &state.convolution, &state.recurrent})
            if (*tensor) evaluated.push_back(**tensor);
        mx::eval(evaluated);
        if (c.chunked_prefill && time > 1)
            require(mx::all(mx::isfinite(hidden)).item<bool>(),
                    "nonfinite Qwen35 chunked hidden at layer " + std::to_string(i) +
                    " prefix " + std::to_string(tokens_));
        if (event) event("qwen35_pe_language", i + 1, c.layers);
    }
    hidden = zero_norm(hidden, weights_.at(prefix_ + "norm.weight"), c.epsilon);
    mx::eval(hidden);
    checkpoint(cancelled);
    tokens_ += time;
    return hidden;
} catch (...) { reset(); throw; }
}
