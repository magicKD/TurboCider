#include "block.hpp"
#include "activation.hpp"

#include <cmath>

namespace tc::ltx098 {
namespace {

Tensor normalized(const Tensor &value, float epsilon) {
    auto input = mx::astype(value, mx::float32);
    return mx::astype(mx::fast::rms_norm(input, std::nullopt, epsilon), value.dtype());
}

Tensor interleaved_rope(const Tensor &value, const ltx_mlx::RopeFrequencies &rope) {
    require(value.ndim() == 4 && value.shape(-1) % 2 == 0,
            "LTX 0.9.8 RoPE requires [batch,heads,tokens,even head dimension]");
    const int half = value.shape(-1) / 2;
    require(rope.cosine.shape() == mx::Shape{value.shape(0), value.shape(1), value.shape(2), half} &&
                rope.sine.shape() == rope.cosine.shape(), "LTX 0.9.8 RoPE geometry mismatch");
    require(rope.cosine.dtype() == value.dtype() && rope.sine.dtype() == value.dtype(),
            "LTX 0.9.8 RoPE frequencies must match the input dtype");
    auto pairs = mx::reshape(value, {value.shape(0), value.shape(1), value.shape(2), half, 2});
    auto components = mx::split(pairs, 2, -1);
    auto even = mx::squeeze(components[0], -1);
    auto odd = mx::squeeze(components[1], -1);
    // Comfy's eager RoPE first stores the even-component product in BF16,
    // then applies a single-round addcmul for the odd component.
    auto first_even = even * rope.cosine;
    auto first_odd = even * rope.sine;
    auto rotated = mx::stack({addcmul(first_even, -rope.sine, odd),
                              addcmul(first_odd, rope.cosine, odd)}, -1);
    return mx::reshape(rotated, value.shape());
}

} // namespace

void Block::load(const std::filesystem::path &path, int index, BlockConfig config) {
    require(index >= 0 && index < 28, "LTX 0.9.8 block index must be in [0,28)");
    require(config.hidden > 0 && config.heads > 0 && config.hidden % config.heads == 0 &&
                (config.hidden / config.heads) % 2 == 0 && config.residual_epsilon > 0 && config.qk_epsilon > 0,
            "invalid LTX 0.9.8 block geometry");
    require(std::filesystem::is_regular_file(path), "LTX 0.9.8 checkpoint is missing");
    weights_.clear();
    config_ = config;
    weights_.load_file(path, "model.diffusion_model.transformer_blocks." + std::to_string(index) + ".");
    require(weights_.has("scale_shift_table") && weights_.at("scale_shift_table").shape() == mx::Shape{6, config.hidden},
            "LTX 0.9.8 block AdaLN table is missing or incompatible");
    for (const auto *name : {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                              "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                              "ff.net.0.proj", "ff.net.2"})
        require(weights_.has(std::string(name) + ".weight"), "LTX 0.9.8 block projection missing: " + std::string(name));
    for (const auto *name : {"attn1.q_norm.weight", "attn1.k_norm.weight",
                              "attn2.q_norm.weight", "attn2.k_norm.weight"})
        require(weights_.has(name) && weights_.at(name).shape() == mx::Shape{config.hidden},
                "LTX 0.9.8 block Q/K norm missing or incompatible: " + std::string(name));
    weights_.materialize();
}

Tensor Block::attention(const Tensor &input, const std::optional<Tensor> &context,
                        const std::string &prefix, const std::optional<ltx_mlx::RopeFrequencies> &rope,
                        const std::optional<Tensor> &mask, const CrossAttentionKV *cached) const {
    auto q = weights_.project(input, prefix + ".to_q");
    q = mx::fast::rms_norm(q, weights_.at(prefix + ".q_norm.weight"), config_.qk_epsilon);
    const int head_dim = config_.hidden / config_.heads;
    auto prepare_heads = [&](const Tensor &value) {
        return mx::transpose(mx::reshape(value, {1, value.shape(1), config_.heads, head_dim}), {0, 2, 1, 3});
    };
    q = prepare_heads(q);
    auto [k, v] = [&]() -> std::pair<Tensor, Tensor> {
        if (cached) {
            require(context && !rope && prefix == "attn2" &&
                        cached->key.shape() == mx::Shape{1, config_.heads, context->shape(1), head_dim} &&
                        cached->value.shape() == cached->key.shape() &&
                        cached->key.dtype() == input.dtype() && cached->value.dtype() == input.dtype(),
                    "LTX 0.9.8 cached cross-attention geometry mismatch");
            return {cached->key, cached->value};
        }
        const auto &kv_input = context ? *context : input;
        auto kv = weights_.project_many(kv_input, {prefix + ".to_k", prefix + ".to_v"});
        auto key = mx::fast::rms_norm(kv[0], weights_.at(prefix + ".k_norm.weight"), config_.qk_epsilon);
        return {prepare_heads(key), prepare_heads(kv[1])};
    }();
    if (rope) {
        require(!context, "LTX 0.9.8 cross-attention must not use video RoPE");
        q = interleaved_rope(q, *rope);
        k = interleaved_rope(k, *rope);
    }
    return weights_.project(tc::attend(q, k, v, false, mask), prefix + ".to_out.0");
}

CrossAttentionKV Block::prepare_cross_attention(const Tensor &context) const {
    require(context.ndim() == 3 && context.shape(0) == 1 && context.shape(1) > 0 &&
                context.shape(2) == config_.hidden, "LTX 0.9.8 cross-attention context mismatch");
    auto kv = weights_.project_many(context, {"attn2.to_k", "attn2.to_v"});
    auto key = mx::fast::rms_norm(kv[0], weights_.at("attn2.k_norm.weight"), config_.qk_epsilon);
    const int head_dim = config_.hidden / config_.heads;
    auto heads = [&](const Tensor &value) {
        return mx::transpose(mx::reshape(value, {1, value.shape(1), config_.heads, head_dim}), {0, 2, 1, 3});
    };
    return {heads(key), heads(kv[1])};
}

Tensor Block::forward(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                      const std::optional<ltx_mlx::RopeFrequencies> &rope,
                      const std::optional<Tensor> &text_mask) const {
    return forward_impl(hidden, context, modulation, rope, text_mask, nullptr);
}

Tensor Block::forward_cached(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                             const std::optional<ltx_mlx::RopeFrequencies> &rope,
                             const std::optional<Tensor> &text_mask, const CrossAttentionKV &cached) const {
    return forward_impl(hidden, context, modulation, rope, text_mask, &cached);
}

Tensor Block::forward_impl(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                           const std::optional<ltx_mlx::RopeFrequencies> &rope,
                           const std::optional<Tensor> &text_mask, const CrossAttentionKV *cached) const {
    const int dim = config_.hidden;
    require(hidden.ndim() == 3 && hidden.shape(0) == 1 && hidden.shape(1) > 0 && hidden.shape(2) == dim &&
                context.ndim() == 3 && context.shape(0) == 1 && context.shape(1) > 0 && context.shape(2) == dim,
            "LTX 0.9.8 block hidden/context geometry mismatch");
    require(modulation.ndim() == 3 && modulation.shape(0) == 1 &&
                (modulation.shape(1) == 1 || modulation.shape(1) == hidden.shape(1)) && modulation.shape(2) == 6 * dim,
            "LTX 0.9.8 block modulation geometry mismatch");
    require(hidden.dtype() == context.dtype() && hidden.dtype() == modulation.dtype(),
            "LTX 0.9.8 block input dtypes must match");
    if (text_mask)
        require(text_mask->shape() == mx::Shape{1, 1, 1, context.shape(1)}, "LTX 0.9.8 text mask geometry mismatch");
    auto table = mx::reshape(mx::astype(weights_.at("scale_shift_table"), hidden.dtype()), {1, 1, 6, dim});
    auto packed = mx::reshape(modulation, {1, modulation.shape(1), 6, dim}) + table;
    std::vector<Tensor> params;
    for (int index = 0; index < 6; ++index)
        params.push_back(mx::squeeze(mx::slice(packed, {0, 0, index, 0}, {1, modulation.shape(1), index + 1, dim}), 2));
    const Tensor one(1.f, hidden.dtype());
    auto x = hidden + attention(normalized(hidden, config_.residual_epsilon) * (one + params[1]) + params[0],
                                {}, "attn1", rope, {}, nullptr) * params[2];
    x = x + attention(x, context, "attn2", {}, text_mask, cached);
    auto feed_normalized = normalized(x, config_.residual_epsilon);
    auto feed_input = addcmul(feed_normalized, feed_normalized, params[4]) + params[3];
    auto feed = weights_.project(gelu_tanh(weights_.project(feed_input, "ff.net.0.proj")), "ff.net.2");
    return addcmul(x, feed, params[5]);
}

} // namespace tc::ltx098
