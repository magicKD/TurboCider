#include "umt5.hpp"

#include <algorithm>
#include <cmath>

namespace tc::components {

UMT5Encoder::UMT5Encoder(const Weights &weights, UMT5Config config)
    : weights_(weights), config_(std::move(config)) {
    require(config_.layers > 0 && config_.hidden > 0 && config_.heads > 0 &&
                config_.head_dim > 0 && config_.intermediate > 0 &&
                config_.buckets >= 4 && config_.buckets % 4 == 0 &&
                config_.max_distance > config_.buckets / 4 && config_.epsilon > 0,
            "invalid UMT5 encoder geometry");
}

Tensor UMT5Encoder::weight(const std::string &name) const {
    // The shipped text checkpoint is FP16; the reference explicitly loads it
    // as BF16. Casting after matmul would implement a different model.
    return mx::astype(weights_.at(name), mx::bfloat16);
}

Tensor UMT5Encoder::project(const Tensor &x, const std::string &name) const {
    auto w = weight(name + ".weight");
    require(w.ndim() == 2 && w.shape(1) == x.shape(-1), "invalid UMT5 projection: " + name);
    return mx::matmul(x, mx::transpose(w));
}

Tensor UMT5Encoder::normalize(const Tensor &x, const std::string &name) const {
    auto f = mx::astype(x, mx::float32);
    auto y = f * mx::rsqrt(mx::mean(mx::square(f), -1, true) + config_.epsilon);
    return mx::astype(y, mx::bfloat16) * weight(name + ".weight");
}

Tensor UMT5Encoder::embed(const Tokens &tokens) const {
    require(!tokens.ids.empty() && tokens.ids.size() <= 512 && tokens.valid > 0 &&
                size_t(tokens.valid) <= tokens.ids.size(), "invalid UMT5 token sequence");
    const auto &embedding = weights_.at("shared.weight");
    require(embedding.ndim() == 2 && embedding.shape(1) == config_.hidden, "invalid UMT5 embedding");
    for (int id : tokens.ids)
        require(id >= 0 && id < embedding.shape(0), "UMT5 token outside vocabulary");
    // Take before casting avoids a second full-vocabulary allocation (~2GB).
    return mx::astype(mx::take(embedding, Tensor(tokens.ids.data(),
        {1, int(tokens.ids.size())}, mx::int32), 0), mx::bfloat16);
}

Tensor UMT5Encoder::relative_buckets(int count) const {
    require(count > 0 && count <= 512, "invalid UMT5 sequence length");
    const int half = config_.buckets / 2, exact = half / 2;
    std::vector<int> buckets(size_t(count) * count);
    for (int query = 0; query < count; ++query) {
        for (int key = 0; key < count; ++key) {
            const int relative = key - query, distance = std::abs(relative);
            const int bucket = distance < exact ? distance :
                std::min(half - 1, exact + int(std::log(float(distance) / float(exact)) /
                    std::log(float(config_.max_distance) / float(exact)) * float(half - exact)));
            buckets[size_t(query) * count + key] = bucket + (relative > 0 ? half : 0);
        }
    }
    return Tensor(buckets.data(), {count, count}, mx::int32);
}

Tensor UMT5Encoder::activation(const Tensor &input) {
    require(input.dtype() == mx::bfloat16, "UMT5 GELU expects BF16 input");
    // Eager BF16 operation boundaries, with FP32 scalar constants. Do not
    // replace this with Wan DiT's fused FP16 GELU.
    auto scalar_product = [](const Tensor &x, float value) {
        return mx::astype(mx::astype(x, mx::float32) * Tensor(value, mx::float32), mx::bfloat16);
    };
    auto cube = mx::power(input, Tensor(3.f, mx::bfloat16));
    auto inner = input + scalar_product(cube, .044715f);
    auto raised = Tensor(1.f, mx::bfloat16) + mx::tanh(
        scalar_product(inner, float(std::sqrt(2. / M_PI))));
    return scalar_product(input, .5f) * raised;
}

Tensor UMT5Encoder::block(const Tensor &input, const Tensor &buckets, int valid, int layer,
                          const Trace &trace) const {
    require(layer >= 0 && layer < config_.layers && input.ndim() == 3 && input.shape(0) == 1 &&
                input.shape(2) == config_.hidden && input.dtype() == mx::bfloat16,
            "invalid UMT5 block input");
    const int count = input.shape(1);
    require(valid > 0 && valid <= count && buckets.shape() == mx::Shape{count, count},
            "invalid UMT5 attention mask/buckets");
    const auto prefix = "encoder.block." + std::to_string(layer);
    const auto attention = prefix + ".layer.0.SelfAttention";
    auto normalized = normalize(input, prefix + ".layer.0.layer_norm");
    if (trace) trace("norm_attention", normalized);
    auto traced_project = [&](const Tensor &x, const std::string &name, const std::string &label) {
        auto output = project(x, name);
        if (trace) trace(label, output);
        return output;
    };
    auto q = heads(traced_project(normalized, attention + ".q", "q"), config_.heads, config_.head_dim);
    auto k = heads(traced_project(normalized, attention + ".k", "k"), config_.heads, config_.head_dim);
    auto v = heads(traced_project(normalized, attention + ".v", "v"), config_.heads, config_.head_dim);
    auto bias = mx::transpose(mx::take(weight(attention + ".relative_attention_bias.weight"),
                                       buckets, 0), {2, 0, 1});
    bias = mx::expand_dims(bias, 0);
    auto padding = mx::reshape(mx::arange(count, mx::int32) >= Tensor(valid), {1, 1, 1, count});
    auto mask = mx::where(padding, Tensor(-3.3895313892515355e38f, mx::bfloat16),
                           Tensor(0.f, mx::bfloat16));
    bias = bias + mask;
    // UMT5 does NOT divide attention scores by sqrt(head_dim). Preserve the
    // BF16 score and probability boundaries of the PyTorch reference.
    auto scores = mx::matmul(q, mx::swapaxes(k, -1, -2)) + bias;
    if (trace) trace("attention_scores", scores);
    auto probabilities = mx::astype(mx::softmax(mx::astype(scores, mx::float32), -1), mx::bfloat16);
    if (trace) trace("attention_probabilities", probabilities);
    auto attended = mx::reshape(mx::transpose(mx::matmul(probabilities, v), {0, 2, 1, 3}),
                                {1, count, config_.heads * config_.head_dim});
    if (trace) trace("attended", attended);
    auto residual = input + traced_project(attended, attention + ".o", "attention_output");
    normalized = normalize(residual, prefix + ".layer.1.layer_norm");
    if (trace) trace("norm_ffn", normalized);
    const auto ff = prefix + ".layer.1.DenseReluDense";
    auto gate = traced_project(normalized, ff + ".wi_0", "gate_input");
    gate = activation(gate);
    if (trace) trace("gate_activation", gate);
    return residual + traced_project(gate * traced_project(normalized, ff + ".wi_1", "up"),
                                      ff + ".wo", "ffn_output");
}

Tensor UMT5Encoder::finish(const Tensor &input, int valid) const {
    require(input.ndim() == 3 && valid > 0 && valid <= input.shape(1), "invalid UMT5 final mask");
    auto output = normalize(input, "encoder.final_layer_norm");
    auto mask = mx::reshape(mx::arange(input.shape(1), mx::int32) < Tensor(valid), {1, -1, 1});
    return mx::where(mask, output, Tensor(0.f, output.dtype()));
}

Tensor UMT5Encoder::encode(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    auto x = embed(tokens);
    auto buckets = relative_buckets(int(tokens.ids.size()));
    for (int layer = 0; layer < config_.layers; ++layer) {
        checkpoint(cancelled);
        event("umt5_text_encode", layer, config_.layers);
        x = block(x, buckets, tokens.valid, layer);
        mx::eval(x);
    }
    x = finish(x, tokens.valid);
    mx::eval(x);
    checkpoint(cancelled);
    event("umt5_text_encode", config_.layers, config_.layers);
    return x;
}

} // namespace tc::components
