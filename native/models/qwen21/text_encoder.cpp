#include "text_encoder.hpp"
#include <cmath>

namespace tc::qwen21 {
namespace {
// Qwen3-VL explicitly accumulates and scales in FP32, unlike DiT's BF16
// nn.RMSNorm. Do not share the DiT normalization implementation here.
Tensor vl_norm(const Tensor &x, const Tensor &weight, float eps) {
    auto f = mx::astype(x, mx::float32);
    return mx::astype(mx::astype(weight, mx::float32) *
                      (f * mx::rsqrt(mx::mean(f * f, -1, true) + eps)), x.dtype());
}
Tensor rotate_half(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto halves = mx::split(x, 2, -1);
    return x * cosine + mx::concatenate({-halves[1], halves[0]}, -1) * sine;
}
}

TextEncoder::TextEncoder(const Weights &weights, TextConfig config)
    : weights_(weights), config_(config), language_prefix_(
          weights.has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.") {
    require(config.layers > 0 && config.heads > 0 && config.kv_heads > 0 &&
            config.heads % config.kv_heads == 0 && config.head_dim > 0 &&
            config.head_dim % 2 == 0 && config.theta > 0,
            "invalid Qwen21 text encoder geometry");
    require(config.mrope_sections[0] + config.mrope_sections[1] + config.mrope_sections[2] == config.head_dim / 2,
            "invalid Qwen21 text mRoPE sections");
    for (int axis = 1; axis <= 2; ++axis)
        require(config.mrope_sections[axis] >= 0 &&
                (config.mrope_sections[axis] == 0 || axis + 3 * (config.mrope_sections[axis] - 1) < config.head_dim / 2),
                "Qwen21 interleaved mRoPE exceeds head dimension");
}

std::string TextEncoder::system_prefix() {
    return "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
}

std::string TextEncoder::prompt_template(const std::string &prompt) {
    const auto content = prompt.find_first_not_of(" \t\r\n") == std::string::npos ? " " : prompt;
    return system_prefix() + "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
}

Tensor TextEncoder::encode(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) const {
    require(!tokens.ids.empty(), "empty Qwen21 token sequence");
    const auto &embedding = weights_.at(language_prefix_ + "embed_tokens.weight");
    for (int id : tokens.ids) require(id >= 0 && id < embedding.shape(0), "Qwen21 token ID out of vocabulary");
    int count = int(tokens.ids.size());
    auto ids = Tensor(tokens.ids.data(), {1, count}, mx::int32);
    auto positions = mx::broadcast_to(mx::reshape(mx::arange(count, mx::int32), {1, count}), {3, count});
    return encode_embeddings(mx::take(embedding, ids, 0), positions, tokens.valid, event, cancelled);
}

Tensor TextEncoder::encode_embeddings(const Tensor &embeddings, const Tensor &positions,
                                      int valid_tokens, const Event &event,
                                      std::atomic<bool> &cancelled,
                                      const std::vector<Tensor> &deepstack_deltas) const {
    require(embeddings.ndim() == 3 && embeddings.shape(0) == 1 && embeddings.shape(1) > 0,
            "Qwen21 text embeddings must be [1,sequence,hidden]");
    int count = embeddings.shape(1);
    require(valid_tokens > 0 && valid_tokens <= count && positions.shape() == mx::Shape{3, count},
            "Qwen21 positions/valid-token count mismatch");
    require(deepstack_deltas.size() <= size_t(config_.layers), "too many visual deepstack residuals");
    for (const auto &delta : deepstack_deltas)
        require(delta.shape() == embeddings.shape() && delta.dtype() == embeddings.dtype(),
                "Qwen21 deepstack residual shape/dtype mismatch");
    auto frequency = Tensor(1.f) / mx::power(Tensor(config_.theta),
        mx::arange(0, config_.head_dim, 2, mx::float32) / float(config_.head_dim));
    std::vector<Tensor> angles;
    for (int axis = 0; axis < 3; ++axis)
        angles.push_back(mx::reshape(mx::astype(slice_axis(positions, 0, axis, axis + 1), mx::float32), {count, 1}) * frequency);
    std::vector<Tensor> columns;
    for (int column = 0; column < config_.head_dim / 2; ++column) {
        int axis = 0;
        if (column % 3 == 1 && column < config_.mrope_sections[1] * 3) axis = 1;
        if (column % 3 == 2 && column < config_.mrope_sections[2] * 3) axis = 2;
        columns.push_back(slice_axis(angles[axis], 1, column, column + 1));
    }
    auto angle = mx::concatenate(columns, -1);
    angle = mx::concatenate({angle, angle}, -1);
    auto cosine = mx::reshape(mx::astype(mx::cos(angle), embeddings.dtype()), {1, 1, count, config_.head_dim});
    auto sine = mx::reshape(mx::astype(mx::sin(angle), embeddings.dtype()), {1, 1, count, config_.head_dim});
    auto idx = mx::arange(count, mx::int32);
    auto key = mx::reshape(idx, {1, count}), query = mx::reshape(idx, {count, 1});
    auto mask = mx::reshape(mx::where(mx::logical_or(key > query, key >= Tensor(valid_tokens)),
                                      Tensor(-INFINITY), Tensor(0.f)), {1, 1, count, count});
    auto hidden = embeddings;
    for (int i = 0; i < config_.layers; ++i) {
        checkpoint(cancelled);
        auto p = language_prefix_ + "layers." + std::to_string(i);
        auto input = vl_norm(hidden, weights_.at(p + ".input_layernorm.weight"), config_.epsilon);
        auto q = heads(linear(input, weights_, p + ".self_attn.q_proj"), config_.heads, config_.head_dim);
        auto k = heads(linear(input, weights_, p + ".self_attn.k_proj"), config_.kv_heads, config_.head_dim);
        auto v = heads(linear(input, weights_, p + ".self_attn.v_proj"), config_.kv_heads, config_.head_dim);
        q = rotate_half(vl_norm(q, weights_.at(p + ".self_attn.q_norm.weight"), config_.epsilon), cosine, sine);
        k = rotate_half(vl_norm(k, weights_.at(p + ".self_attn.k_norm.weight"), config_.epsilon), cosine, sine);
        k = mx::repeat(k, config_.heads / config_.kv_heads, 1);
        v = mx::repeat(v, config_.heads / config_.kv_heads, 1);
        hidden = hidden + linear(attend(q, k, v, true, mask), weights_, p + ".self_attn.o_proj");
        input = vl_norm(hidden, weights_.at(p + ".post_attention_layernorm.weight"), config_.epsilon);
        hidden = hidden + linear(silu(linear(input, weights_, p + ".mlp.gate_proj")) *
                                  linear(input, weights_, p + ".mlp.up_proj"), weights_, p + ".mlp.down_proj");
        // Visual levels enter consecutive early language layers, not layers
        // 8/16/24. The prompt assembler zeros these deltas outside image spans.
        if (size_t(i) < deepstack_deltas.size()) hidden = hidden + deepstack_deltas[i];
        if ((i + 1) % 4 == 0 || i + 1 == config_.layers) mx::eval(hidden);
        if (event) event("qwen21_text_encode", i + 1, config_.layers);
    }
    return config_.final_norm ? vl_norm(hidden, weights_.at(language_prefix_ + "norm.weight"), config_.epsilon) : hidden;
}
} // namespace tc::qwen21
