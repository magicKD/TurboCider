#include "qwen3.hpp"

#include <algorithm>
#include <cmath>

namespace tc::components {

Qwen3Conditioning Qwen3Conditioning::flux_klein() {
    Qwen3Conditioning config;
    config.output_layers = {8, 17, 26};
    config.progress_phase = "text_encode";
    return config;
}

Qwen3Conditioning Qwen3Conditioning::z_image() {
    Qwen3Conditioning config;
    config.output_layers = {34};
    config.float32_residual = true;
    config.progress_phase = "z_image_text_encode";
    return config;
}

Tensor qwen3_conditioning(const Tokens &tokens, const Weights &weights,
                          const Qwen3Conditioning &config, const Event &event,
                          std::atomic<bool> &cancelled) {
    require(!config.output_layers.empty() && config.output_layers.front() >= 0 &&
                std::is_sorted(config.output_layers.begin(), config.output_layers.end()),
            "Qwen3 output layers must be nonempty and sorted");
    require(config.kv_heads > 0 && config.heads % config.kv_heads == 0 &&
                config.head_dim > 0 && config.head_dim % 2 == 0,
            "invalid Qwen3 attention geometry");
    const int count = int(tokens.ids.size());
    require(count > 0 && tokens.valid > 0 && tokens.valid <= count,
            "invalid Qwen3 token sequence");
    auto ids = Tensor(tokens.ids.data(), {1, count}, mx::int32);
    auto x = mx::take(weights.at("model.embed_tokens.weight"), ids, 0);
    if (config.float32_residual) x = mx::astype(x, mx::float32);
    const auto rope_dtype = config.float32_residual ? mx::float32 : mx::bfloat16;
    auto frequencies = 1.f / mx::power(Tensor(config.rope_theta),
        mx::arange(0, config.head_dim, 2, mx::float32) / float(config.head_dim));
    auto angles = mx::reshape(mx::arange(count, mx::float32), {1, count, 1}) *
        mx::reshape(frequencies, {1, 1, config.head_dim / 2});
    angles = mx::concatenate({angles, angles}, -1);
    auto cosine = mx::expand_dims(mx::astype(mx::cos(angles), rope_dtype), 1);
    auto sine = mx::expand_dims(mx::astype(mx::sin(angles), rope_dtype), 1);
    auto positions = mx::arange(count, mx::int32);
    auto query = mx::reshape(positions, {count, 1});
    auto key = mx::reshape(positions, {1, count});
    auto forbidden = mx::logical_or(key > query, key >= Tensor(tokens.valid));
    auto mask = mx::reshape(mx::where(forbidden, Tensor(-INFINITY, rope_dtype),
                                      Tensor(0.f, rope_dtype)), {1, 1, count, count});
    auto rotate = [&](const Tensor &value) {
        auto halves = mx::split(value, 2, -1);
        return value * cosine + mx::concatenate({-halves[1], halves[0]}, -1) * sine;
    };
    std::vector<Tensor> outputs;
    const int layers = config.output_layers.back() + 1;
    for (int layer = 0; layer < layers; ++layer) {
        checkpoint(cancelled);
        event(config.progress_phase, layer, layers);
        const auto prefix = "model.layers." + std::to_string(layer);
        auto normalized = rms(x, weights.at(prefix + ".input_layernorm.weight"), config.norm_epsilon);
        auto q = heads(linear(normalized, weights, prefix + ".self_attn.q_proj"), config.heads, config.head_dim);
        auto k = heads(linear(normalized, weights, prefix + ".self_attn.k_proj"), config.kv_heads, config.head_dim);
        auto v = heads(linear(normalized, weights, prefix + ".self_attn.v_proj"), config.kv_heads, config.head_dim);
        q = rotate(rms(q, weights.at(prefix + ".self_attn.q_norm.weight"), config.norm_epsilon));
        k = rotate(rms(k, weights.at(prefix + ".self_attn.k_norm.weight"), config.norm_epsilon));
        k = mx::repeat(k, config.heads / config.kv_heads, 1);
        v = mx::repeat(v, config.heads / config.kv_heads, 1);
        x = x + linear(attend(q, k, v, true, mask), weights, prefix + ".self_attn.o_proj");
        normalized = rms(x, weights.at(prefix + ".post_attention_layernorm.weight"), config.norm_epsilon);
        x = x + linear(silu(linear(normalized, weights, prefix + ".mlp.gate_proj")) *
                         linear(normalized, weights, prefix + ".mlp.up_proj"),
                         weights, prefix + ".mlp.down_proj");
        mx::eval(x);
        if (std::binary_search(config.output_layers.begin(), config.output_layers.end(), layer))
            outputs.push_back(x);
    }
    auto result = outputs.size() == 1 ? outputs.front() : mx::concatenate(outputs, -1);
    if (config.float32_residual) result = mx::astype(result, mx::bfloat16);
    mx::eval(result);
    event(config.progress_phase, layers, layers);
    return result;
}

} // namespace tc::components
