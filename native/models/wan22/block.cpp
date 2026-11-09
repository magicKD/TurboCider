#include "block.hpp"
#include "linear.hpp"

#include <cmath>

namespace tc::wan22 {
namespace {

Tensor addcmul(const Tensor &base, const Tensor &left, const Tensor &right) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        auto value = mx::astype(args[0], mx::float32) +
                     mx::astype(args[1], mx::float32) * mx::astype(args[2], mx::float32);
        return std::vector<Tensor>{mx::astype(value, args[0].dtype())};
    }, true);
    return graph({base, left, right})[0];
}

Tensor gelu_tanh(const Tensor &value) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        auto x = mx::astype(args[0], mx::float32);
        auto result = .5f * x * (1.f + mx::tanh(.7978845608028654f * (x + .044715f * x * x * x)));
        return std::vector<Tensor>{mx::astype(result, args[0].dtype())};
    }, true);
    return graph({value})[0];
}

Tensor norm(const Tensor &value, float epsilon) {
    auto fp32 = mx::astype(value, mx::float32);
    return mx::astype(mx::fast::layer_norm(fp32, std::nullopt, std::nullopt, epsilon), value.dtype());
}

Tensor rotate(const Tensor &value, const Tensor &cosine, const Tensor &sine, int heads) {
    const int rows = value.shape(1), hidden = value.shape(2), head_dim = hidden / heads;
    auto input = mx::reshape(mx::astype(value, mx::float32), {1, rows, heads, head_dim});
    auto pairs = mx::reshape(input, {1, rows, heads, head_dim / 2, 2});
    auto halves = mx::split(pairs, 2, -1);
    auto rotated = mx::reshape(mx::concatenate({-halves[1], halves[0]}, -1), {1, rows, heads, head_dim});
    auto c = mx::reshape(cosine, {1, rows, 1, head_dim});
    auto s = mx::reshape(sine, {1, rows, 1, head_dim});
    return mx::reshape(mx::astype(input * c + rotated * s, value.dtype()), value.shape());
}

} // namespace

void Block::load(const std::filesystem::path &path, int index, BlockConfig config) {
    require(std::filesystem::is_regular_file(path) && index >= 0 && index < 30,
            "Wan2.2 block checkpoint or index is invalid");
    require(config.hidden > 0 && config.heads > 0 && config.hidden % config.heads == 0 &&
                (config.hidden / config.heads) % 2 == 0 && config.intermediate > 0 && config.epsilon > 0,
            "invalid Wan2.2 block geometry");
    config_ = config;
    weights_.clear();
    weights_.load_file(path, "blocks." + std::to_string(index) + ".");
    require(weights_.has("scale_shift_table") &&
                weights_.at("scale_shift_table").shape() == mx::Shape{1, 6, config.hidden},
            "Wan2.2 modulation table is missing or incompatible");
    for (const auto *prefix : {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                               "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                               "ffn.net.0.proj", "ffn.net.2"})
        require(weights_.has(std::string(prefix) + ".weight"), "Wan2.2 projection is missing: " + std::string(prefix));
    for (const auto *name : {"attn1.norm_q.weight", "attn1.norm_k.weight", "attn2.norm_q.weight",
                              "attn2.norm_k.weight", "norm2.weight", "norm2.bias"})
        require(weights_.has(name) && weights_.at(name).shape() == mx::Shape{config.hidden},
                "Wan2.2 normalization is missing or incompatible: " + std::string(name));
    require(weights_.at("ffn.net.0.proj.weight").shape() == mx::Shape{config.intermediate, config.hidden} &&
                weights_.at("ffn.net.2.weight").shape() == mx::Shape{config.hidden, config.intermediate},
            "Wan2.2 feed-forward geometry mismatch");
    weights_.materialize();
}

Tensor Block::forward(const Tensor &hidden, const Tensor &context, const Tensor &modulation,
                      const Tensor &cosine, const Tensor &sine, int first_frame_tokens, const Trace &trace) const {
    const int dim = config_.hidden, heads = config_.heads, head_dim = dim / heads;
    require(hidden.ndim() == 3 && hidden.shape(0) == 1 && hidden.shape(1) > 0 && hidden.shape(2) == dim &&
                hidden.dtype() == mx::bfloat16 && context.ndim() == 3 && context.shape(0) == 1 &&
                context.shape(1) > 0 && context.shape(2) == dim && context.dtype() == hidden.dtype(),
            "Wan2.2 hidden/context must be BF16 [1,tokens,hidden]");
    const int rows = hidden.shape(1), groups = modulation.ndim() == 4 ? modulation.shape(1) : 0;
    require((groups == 1 || groups == 2) && modulation.shape() == mx::Shape{1, groups, 6, dim} &&
                modulation.dtype() == hidden.dtype() && first_frame_tokens > 0 && first_frame_tokens <= rows &&
                (groups != 2 || first_frame_tokens < rows), "invalid Wan2.2 timestep modulation geometry");
    require(cosine.shape() == mx::Shape{rows, head_dim} && sine.shape() == cosine.shape() &&
                cosine.dtype() == mx::float32 && sine.dtype() == mx::float32,
            "invalid Wan2.2 rotary frequencies");
    auto packed = modulation + mx::reshape(mx::astype(weights_.at("scale_shift_table"), hidden.dtype()), {1, 1, 6, dim});
    auto parameter = [&](int which) {
        auto entry = mx::squeeze(mx::slice(packed, {0, 0, which, 0}, {1, groups, which + 1, dim}), 2);
        if (groups == 1) return entry;
        auto first = mx::slice(entry, {0, 0, 0}, {1, 1, dim});
        auto rest = mx::slice(entry, {0, 1, 0}, {1, 2, dim});
        return mx::concatenate({mx::broadcast_to(first, {1, first_frame_tokens, dim}),
                                mx::broadcast_to(rest, {1, rows - first_frame_tokens, dim})}, 1);
    };
    auto shift_self = parameter(0), scale_self = parameter(1), gate_self = parameter(2);
    auto shift_ffn = parameter(3), scale_ffn = parameter(4), gate_ffn = parameter(5);
    auto modulate = [&](const Tensor &value, const Tensor &shift, const Tensor &scale) {
        return addcmul(shift, norm(value, config_.epsilon), Tensor(1.f, value.dtype()) + scale);
    };
    auto attention = [&](const Tensor &query_input, const Tensor &memory, const std::string &prefix, bool rope) {
        const auto label = rope ? std::string("self") : std::string("cross");
        if (trace) trace(label + "_input", query_input);
        auto q = linear(weights_, query_input, prefix + ".to_q");
        auto k = linear(weights_, memory, prefix + ".to_k");
        auto v = linear(weights_, memory, prefix + ".to_v");
        if (trace) { trace(label + "_q", q); trace(label + "_k", k); trace(label + "_v", v); }
        q = mx::fast::rms_norm(q, weights_.at(prefix + ".norm_q.weight"), config_.epsilon);
        k = mx::fast::rms_norm(k, weights_.at(prefix + ".norm_k.weight"), config_.epsilon);
        if (trace) { trace(label + "_q_norm", q); trace(label + "_k_norm", k); }
        if (rope) {
            q = rotate(q, cosine, sine, heads);
            k = rotate(k, cosine, sine, heads);
        }
        auto split_heads = [&](const Tensor &value) {
            return mx::transpose(mx::reshape(value, {1, value.shape(1), heads, head_dim}), {0, 2, 1, 3});
        };
        auto output = mx::fast::scaled_dot_product_attention(split_heads(q), split_heads(k), split_heads(v),
                                                             1.f / std::sqrt(float(head_dim)));
        output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {1, query_input.shape(1), dim});
        if (trace) trace(label + "_attended", output);
        output = linear(weights_, output, prefix + ".to_out.0");
        if (trace) trace(label + "_output", output);
        return output;
    };
    auto normalized = modulate(hidden, shift_self, scale_self);
    auto value = addcmul(hidden, attention(normalized, normalized, "attn1", true), gate_self);
    if (trace) trace("self_residual", value);
    auto cross_norm = mx::fast::layer_norm(mx::astype(value, mx::float32),
                    mx::astype(weights_.at("norm2.weight"), mx::float32),
                    mx::astype(weights_.at("norm2.bias"), mx::float32), config_.epsilon);
    value = value + attention(mx::astype(cross_norm, value.dtype()), context, "attn2", false);
    if (trace) trace("cross_residual", value);
    auto ffn_input = modulate(value, shift_ffn, scale_ffn);
    auto ffn_up = linear(weights_, ffn_input, "ffn.net.0.proj");
    auto ffn_activated = gelu_tanh(ffn_up);
    auto ffn = linear(weights_, ffn_activated, "ffn.net.2");
    if (trace) { trace("ffn_input", ffn_input); trace("ffn_up", ffn_up); trace("ffn_activated", ffn_activated); trace("ffn_output", ffn); }
    return addcmul(value, ffn, gate_ffn);
}

} // namespace tc::wan22
