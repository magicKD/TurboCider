#include "conditioner.hpp"
#include "conditioner_math.hpp"
#include "prompt_cache.hpp"

#include <cmath>

namespace tc::h3_mlx {
namespace {
Tensor rms_fp32(const Tensor &value, const Tensor &weight, float epsilon) {
    auto input = mx::astype(value, mx::float32);
    return input / mx::sqrt(mx::mean(input * input, -1, true) + epsilon) *
           mx::astype(weight, mx::float32);
}

Tensor dense(const Tensor &value, const Tensor &weight) {
    require(weight.ndim() == 2 && value.shape(-1) == weight.shape(1),
            "invalid streamed H3 conditioner projection geometry");
    return mx::matmul(value, mx::transpose(weight));
}

Tensor rotate_half(const Tensor &value, const Tensor &cosine,
                   const Tensor &sine) {
    auto halves = mx::split(value, 2, -1);
    auto rotated = mx::concatenate({-halves[1], halves[0]}, -1);
    return value * cosine + rotated * sine;
}

} // namespace

Conditioner::Conditioner(const std::filesystem::path &component_root,
                         const std::filesystem::path &tokenizer_root)
    : config_(load_conditioner_config(component_root / "config.json")),
      weights_(component_root), tokenizer_(tokenizer_root),
      component_root_(component_root), tokenizer_root_(tokenizer_root) {
    require(config_.num_layers > output_layers,
            "H3 conditioner checkpoint must contain more than 50 layers");
    require(config_.hidden_size == 5120 && config_.num_heads == 64 &&
                config_.kv_heads == 8 && config_.head_dim == 128 &&
                config_.intermediate_size == 25600 &&
                config_.vocabulary_size == 151936,
            "unsupported H3 Qwen3-VL conditioner geometry");
    require(config_.num_heads % config_.kv_heads == 0 &&
                config_.head_dim % 2 == 0 &&
                config_.mrope_sections[0] + config_.mrope_sections[1] +
                    config_.mrope_sections[2] == config_.head_dim / 2,
            "invalid H3 Qwen3-VL attention geometry");
}

Tensor Conditioner::layer(int index, const Tensor &input,
                          const Tensor &cosine, const Tensor &sine,
                          ConditionerDebugTensors *debug) const {
    const auto prefix = "model.language_model.layers." +
                        std::to_string(index) + ".";
    auto weight = [&](const std::string &name) {
        return weights_.tensor(prefix + name);
    };
    const int sequence = input.shape(0);
    auto capture = [&](const std::string &name, const Tensor &value) {
        if (debug)
            debug->emplace(name, value);
    };

    auto residual = input;
    capture("layer0_input", input);
    auto normalized = rms_fp32(input, weight("input_layernorm.weight"),
                               config_.norm_epsilon);
    capture("layer0_input_norm", normalized);
    auto query = mx::reshape(dense(normalized, weight("self_attn.q_proj.weight")),
                             {sequence, config_.num_heads, config_.head_dim});
    auto key = mx::reshape(dense(normalized, weight("self_attn.k_proj.weight")),
                           {sequence, config_.kv_heads, config_.head_dim});
    auto value = mx::reshape(dense(normalized, weight("self_attn.v_proj.weight")),
                             {sequence, config_.kv_heads, config_.head_dim});
    capture("layer0_q_proj", query);
    capture("layer0_k_proj", key);
    capture("layer0_v_proj", value);
    query = rms_fp32(query, weight("self_attn.q_norm.weight"),
                     config_.norm_epsilon);
    key = rms_fp32(key, weight("self_attn.k_norm.weight"),
                   config_.norm_epsilon);
    capture("layer0_q_norm", query);
    capture("layer0_k_norm", key);
    query = rotate_half(query, cosine, sine);
    key = rotate_half(key, cosine, sine);
    capture("layer0_q_rope", query);
    capture("layer0_k_rope", key);
    key = mx::repeat(key, config_.num_heads / config_.kv_heads, 1);
    value = mx::repeat(value, config_.num_heads / config_.kv_heads, 1);
    capture("layer0_k_repeat", key);
    capture("layer0_v_repeat", value);

    auto query_heads = mx::transpose(query, {1, 0, 2});
    auto key_heads = mx::transpose(key, {1, 2, 0});
    auto value_heads = mx::transpose(value, {1, 0, 2});
    auto scores = mx::matmul(query_heads, key_heads) *
                  (1.f / std::sqrt(float(config_.head_dim)));
    capture("layer0_scores_unmasked", scores);
    auto positions = mx::arange(sequence, mx::int32);
    auto row = mx::reshape(positions, {sequence, 1});
    auto column = mx::reshape(positions, {1, sequence});
    auto forbidden = column > row;
    scores = mx::where(mx::expand_dims(forbidden, 0),
                       Tensor(-INFINITY, mx::float32), scores);
    capture("layer0_scores_masked", scores);
    auto probabilities = mx::softmax(scores, -1);
    capture("layer0_probabilities", probabilities);
    auto attended = mx::matmul(probabilities, value_heads);
    capture("layer0_attended_heads", attended);
    attended = mx::reshape(mx::transpose(attended, {1, 0, 2}),
                           {sequence, config_.num_heads * config_.head_dim});
    capture("layer0_attended", attended);
    auto attention_output = dense(attended, weight("self_attn.o_proj.weight"));
    capture("layer0_attention_output", attention_output);
    auto hidden = residual + attention_output;
    capture("layer0_post_attention", hidden);

    residual = hidden;
    normalized = rms_fp32(hidden, weight("post_attention_layernorm.weight"),
                          config_.norm_epsilon);
    capture("layer0_post_attention_norm", normalized);
    auto gate = dense(normalized, weight("mlp.gate_proj.weight"));
    auto up = dense(normalized, weight("mlp.up_proj.weight"));
    auto activated = gate * mx::sigmoid(gate) * up;
    capture("layer0_gate", gate);
    capture("layer0_up", up);
    capture("layer0_activated", activated);
    auto down = dense(activated, weight("mlp.down_proj.weight"));
    capture("layer0_down", down);
    auto output = residual + down;
    capture("layer0_output", output);
    return output;
}

ConditioningResult Conditioner::encode_prompt(const std::string &prompt,
                                              const Event &event,
                                              std::atomic<bool> &cancelled,
                                              std::vector<Tensor> *debug_layers,
                                              ConditionerDebugTensors *debug_tensors) const {
    auto tokens = tokenizer_.raw(prompt);
    return encode_tokens(tokens.ids, event, cancelled, debug_layers, debug_tensors);
}

ConditioningResult Conditioner::encode_prompt_cached(
    const std::string &prompt, const std::filesystem::path &cache_root,
    const Event &event, std::atomic<bool> &cancelled) const {
    const auto identity = prompt_cache_identity(component_root_, tokenizer_root_, prompt);
    if (auto cached = load_prompt_cache(cache_root, identity, event, cancelled))
        return std::move(*cached);
    auto result = encode_prompt(prompt, event, cancelled);
    save_prompt_cache(cache_root, identity, result, event, cancelled);
    return result;
}

ConditioningResult Conditioner::encode_tokens(const std::vector<int> &tokens,
                                              const Event &event,
                                              std::atomic<bool> &cancelled,
                                              std::vector<Tensor> *debug_layers,
                                              ConditionerDebugTensors *debug_tensors) const {
    require(!tokens.empty() && tokens.size() <= 512,
            "H3 conditioner requires 1...512 tokens");
    for (int token : tokens)
        require(token >= 0 && token < config_.vocabulary_size,
                "H3 conditioner token is outside the vocabulary");
    const int sequence = int(tokens.size());
    auto hidden = weights_.rows(
        "model.language_model.embed_tokens.weight", tokens);
    require(hidden.shape() == mx::Shape({sequence, config_.hidden_size}),
            "invalid H3 conditioner embedding rows");
    if (debug_tensors)
        debug_tensors->emplace("embedding", hidden);

    // T2VA text rows use the same monotonically increasing position on all
    // three MRoPE axes.  Interleaving equal axes is exactly this shared table;
    // multimodal vision spans remain outside the first profile's contract.
    // FastVideo builds the text-only MRoPE table in NumPy float32 before
    // handing it to MLX.  Construct the same table on the CPU here instead
    // of using an MLX power/cos graph: the latter can choose a different
    // intermediate precision and accumulate a visible error over 50 layers.
    const int half = config_.head_dim / 2;
    std::vector<float> frequencies(static_cast<size_t>(half));
    for (int index = 0; index < half; ++index) {
        const float exponent = static_cast<float>(index) /
                               static_cast<float>(half);
        frequencies[static_cast<size_t>(index)] =
            1.f / std::pow(config_.rope_theta, exponent);
    }
    std::vector<float> cosine_values(static_cast<size_t>(sequence) *
                                     static_cast<size_t>(config_.head_dim));
    std::vector<float> sine_values(cosine_values.size());
    for (int position = 0; position < sequence; ++position) {
        for (int index = 0; index < half; ++index) {
            const float angle = static_cast<float>(position) *
                                frequencies[static_cast<size_t>(index)];
            const auto offset = static_cast<size_t>(position) *
                                static_cast<size_t>(config_.head_dim);
            const auto trig = numpy_float32_sin_cos(angle);
            cosine_values[offset + static_cast<size_t>(index)] = trig.cosine;
            sine_values[offset + static_cast<size_t>(index)] = trig.sine;
            cosine_values[offset + static_cast<size_t>(half + index)] = trig.cosine;
            sine_values[offset + static_cast<size_t>(half + index)] = trig.sine;
        }
    }
    auto cosine = mx::expand_dims(
        Tensor(cosine_values.data(), {sequence, config_.head_dim}, mx::float32), 1);
    auto sine = mx::expand_dims(
        Tensor(sine_values.data(), {sequence, config_.head_dim}, mx::float32), 1);
    if (debug_tensors) {
        debug_tensors->emplace("mrope_cosine", cosine);
        debug_tensors->emplace("mrope_sine", sine);
    }
    mx::eval({hidden, cosine, sine});

    for (int index = 0; index < output_layers; ++index) {
        checkpoint(cancelled);
        event("h3_mlx_text_encode", index, output_layers);
        hidden = layer(index, hidden, cosine, sine,
                       index == 0 ? debug_tensors : nullptr);
        mx::eval(hidden);
        if (debug_layers)
            debug_layers->push_back(hidden);
    }
    checkpoint(cancelled);
    event("h3_mlx_text_encode", output_layers, output_layers);
    std::vector<int32_t> tags(tokens.size(), text_tag);
    return {std::move(hidden), std::move(tags), sequence};
}

} // namespace tc::h3_mlx
