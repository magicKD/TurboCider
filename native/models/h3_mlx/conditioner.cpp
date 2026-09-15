#include "conditioner.hpp"
#include "conditioner_math.hpp"
#include "prompt_cache.hpp"
#include "../../backends/coreml.hpp"
#include "../../components/text/qwen3.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

float quality_threshold(const char *name, float fallback) {
    const char *value = std::getenv(name);
    if (!value) return fallback;
    char *end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value, &end);
    require(errno == 0 && end != value && *end == '\0' &&
                std::isfinite(parsed),
            std::string("invalid H3 Qwen3 quality threshold: ") + name);
    return parsed;
}

bool fused_attention_enabled(int sequence) {
    // The MLX SDPA route is intentionally opt-in until a Metal host has
    // produced a matched sequence sweep.  Keep a sequence floor so short
    // prompts can remain on the small, predictable reference graph while
    // longer prefill workloads can exercise the fused kernel.
    if (!std::getenv("TURBOCIDER_H3_FUSED_SDPA") ||
        std::getenv("TURBOCIDER_H3_DISABLE_FUSED_SDPA"))
        return false;
    const char *value = std::getenv("TURBOCIDER_H3_FUSED_SDPA_MIN_TOKENS");
    if (!value || !*value)
        return sequence >= 64;
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    require(errno == 0 && end != value && *end == '\0' && parsed >= 1 &&
                parsed <= (1 << 20),
            "invalid TURBOCIDER_H3_FUSED_SDPA_MIN_TOKENS");
    return sequence >= parsed;
}

void validate_fused_attention(const Tensor &candidate, const Tensor &reference) {
    auto candidate_f32 = mx::astype(candidate, mx::float32);
    auto reference_f32 = mx::astype(reference, mx::float32);
    auto delta = candidate_f32 - reference_f32;
    auto scale = mx::maximum(
        mx::maximum(mx::max(mx::abs(candidate_f32)),
                    mx::max(mx::abs(reference_f32))),
        Tensor(1.f, mx::float32));
    auto scaled_delta = delta / scale;
    auto scaled_reference = reference_f32 / scale;
    auto scaled_candidate = candidate_f32 / scale;
    auto delta2 = mx::sum(scaled_delta * scaled_delta);
    auto reference2 = mx::sum(scaled_reference * scaled_reference);
    auto candidate2 = mx::sum(scaled_candidate * scaled_candidate);
    auto dot = mx::sum(scaled_candidate * scaled_reference);
    auto max_abs = mx::max(mx::abs(delta));
    auto reference_max = mx::max(mx::abs(reference_f32));
    auto finite = mx::logical_and(mx::all(mx::isfinite(candidate_f32)),
                                  mx::all(mx::isfinite(reference_f32)));
    mx::eval({delta2, reference2, candidate2, dot, max_abs, reference_max,
              finite});
    const bool all_finite = finite.item<bool>();
    const double reference_energy = reference2.item<float>();
    const double candidate_energy = candidate2.item<float>();
    const double delta_energy = delta2.item<float>();
    const double relative_l2 = all_finite && reference_energy > 0.0
        ? std::sqrt(delta_energy / reference_energy)
        : (all_finite && delta_energy == 0.0 ? 0.0 : 1e30);
    const double cosine_denominator =
        std::sqrt(candidate_energy * reference_energy);
    const double cosine = all_finite
        ? std::clamp(cosine_denominator > 0.0
                         ? dot.item<float>() / cosine_denominator
                         : (candidate_energy == 0.0 && reference_energy == 0.0
                                ? 1.0
                                : 0.0),
                     -1.0, 1.0)
        : -1.0;
    const double absolute = all_finite ? max_abs.item<float>() : 1e30;
    const double amplitude = reference_max.item<float>();
    const double relative_abs = all_finite
        ? (amplitude > 0.0 ? absolute / amplitude
                           : (absolute == 0.0 ? 0.0 : 1e30))
        : 1e30;
    const float max_relative_l2 = quality_threshold(
        "TURBOCIDER_H3_FUSED_SDPA_MAX_RELATIVE_L2", 0.0005f);
    const float min_cosine = quality_threshold(
        "TURBOCIDER_H3_FUSED_SDPA_MIN_COSINE", 0.99999f);
    const float max_relative_abs = quality_threshold(
        "TURBOCIDER_H3_FUSED_SDPA_MAX_RELATIVE_ABS", 0.005f);
    require(max_relative_l2 >= 0.f && min_cosine >= -1.f &&
                min_cosine <= 1.f && max_relative_abs >= 0.f,
            "invalid H3 fused SDPA quality threshold range");
    std::fprintf(stderr,
                 "h3_fused_sdpa rel_l2=%.9g cosine=%.9g max_abs=%.9g "
                 "relative_max_abs=%.9g finite=%d\n",
                 relative_l2, cosine, absolute, relative_abs,
                 all_finite ? 1 : 0);
    require(all_finite && relative_l2 <= max_relative_l2 &&
                cosine >= min_cosine && relative_abs <= max_relative_abs,
            "H3 fused SDPA quality gate failed");
}

std::filesystem::path conditioner_checkpoint(
    const std::filesystem::path &root) {
    const auto index = root / "model.safetensors.index.json";
    if (std::filesystem::is_regular_file(index) &&
        !std::filesystem::is_symlink(index))
        return std::filesystem::canonical(index);
    const auto single = root / "model.safetensors";
    require(std::filesystem::is_regular_file(single) &&
                !std::filesystem::is_symlink(single),
            "H3 conditioner checkpoint is missing");
    return std::filesystem::canonical(single);
}

} // namespace

Conditioner::Conditioner(const std::filesystem::path &component_root,
                         const std::filesystem::path &tokenizer_root,
                         const std::filesystem::path &hybrid_manifest,
                         int hybrid_warmup_iterations)
    : config_(load_conditioner_config(component_root / "config.json")),
      weights_(component_root), tokenizer_(tokenizer_root),
      component_root_(component_root), tokenizer_root_(tokenizer_root),
      hybrid_manifest_(hybrid_manifest.empty()
                           ? std::filesystem::path{}
                           : std::filesystem::absolute(hybrid_manifest)
                                 .lexically_normal()),
      hybrid_warmup_iterations_(hybrid_warmup_iterations) {
    require(hybrid_warmup_iterations_ >= 0 && hybrid_warmup_iterations_ <= 8,
            "H3 Qwen3 hybrid warmup iterations must be 0...8");
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

Conditioner::~Conditioner() = default;

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
    const bool use_fused_attention = fused_attention_enabled(sequence);
    const bool validate_attention = use_fused_attention &&
        std::getenv("TURBOCIDER_H3_FUSED_SDPA_VALIDATE");
    Tensor attended = query;
    if (use_fused_attention) {
        auto query_heads = mx::expand_dims(mx::transpose(query, {1, 0, 2}), 0);
        auto key_heads = mx::expand_dims(mx::transpose(key, {1, 0, 2}), 0);
        auto value_heads = mx::expand_dims(mx::transpose(value, {1, 0, 2}), 0);
        attended = mx::squeeze(
            tc::attend(query_heads, key_heads, value_heads, true, {}, true,
                       "causal"),
            0);
    }

    // Keep the explicit attention graph as the default and as the validation
    // reference. Debug captures retain their historical tensor ABI even when
    // the fused candidate is selected.
    if (!use_fused_attention || validate_attention || debug) {
        auto repeated_key = mx::repeat(
            key, config_.num_heads / config_.kv_heads, 1);
        auto repeated_value = mx::repeat(
            value, config_.num_heads / config_.kv_heads, 1);
        capture("layer0_k_repeat", repeated_key);
        capture("layer0_v_repeat", repeated_value);
        auto query_heads = mx::transpose(query, {1, 0, 2});
        auto key_heads = mx::transpose(repeated_key, {1, 2, 0});
        auto value_heads = mx::transpose(repeated_value, {1, 0, 2});
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
        auto reference_heads = mx::matmul(probabilities, value_heads);
        capture("layer0_attended_heads", reference_heads);
        auto reference = mx::reshape(mx::transpose(reference_heads, {1, 0, 2}),
                                     {sequence, config_.num_heads *
                                                    config_.head_dim});
        if (!use_fused_attention)
            attended = reference;
        else if (validate_attention)
            validate_fused_attention(attended, reference);
    }
    capture("layer0_attended", attended);
    auto attention_output = dense(attended, weight("self_attn.o_proj.weight"));
    capture("layer0_attention_output", attention_output);
    auto hidden = residual + attention_output;
    capture("layer0_post_attention", hidden);

    residual = hidden;
    normalized = rms_fp32(hidden, weight("post_attention_layernorm.weight"),
                          config_.norm_epsilon);
    capture("layer0_post_attention_norm", normalized);
    Tensor gate = normalized;
    Tensor up = normalized;
    Tensor activated = normalized;
    Tensor down = normalized;
    Tensor debug_gate = normalized;
    Tensor debug_up = normalized;
    Tensor debug_activated = normalized;
    Tensor debug_down = normalized;
    auto exact_mlp = [&] {
        gate = dense(normalized, weight("mlp.gate_proj.weight"));
        up = dense(normalized, weight("mlp.up_proj.weight"));
        activated = gate * mx::sigmoid(gate) * up;
        return dense(activated, weight("mlp.down_proj.weight"));
    };
    if (hybrid_ && hybrid_->runtime_available() && index < hybrid_->block_count) {
        const int ane_end = hybrid_->ane_mlp_end;
        require(hybrid_->hidden == config_.hidden_size &&
                    hybrid_->mlp_width == config_.intermediate_size &&
                    hybrid_->ane_mlp_start == 0 && ane_end > 0 &&
                    ane_end < config_.intermediate_size,
                "H3 Qwen3 Core ML manifest geometry is invalid");
        auto packed = mx::expand_dims(mx::astype(normalized, mx::float16), 0);
        if (sequence < hybrid_->rows)
            packed = mx::concatenate(
                {packed, mx::zeros({1, hybrid_->rows - sequence,
                                    config_.hidden_size}, mx::float16)}, 1);
        mx::eval({normalized, packed});

        Tensor gpu = normalized;
        {
            const int width = config_.intermediate_size;
            auto gate_weight = weights_.slice(
                prefix + "mlp.gate_proj.weight", ane_end, width,
                0, config_.hidden_size);
            auto up_weight = weights_.slice(
                prefix + "mlp.up_proj.weight", ane_end, width,
                0, config_.hidden_size);
            gate = dense(normalized, gate_weight);
            up = dense(normalized, up_weight);
            activated = gate * mx::sigmoid(gate) * up;
            auto down_weight = weights_.slice(
                prefix + "mlp.down_proj.weight", 0, config_.hidden_size,
                ane_end, width);
            gpu = dense(activated, down_weight);
            mx::async_eval({gpu});
            try {
                auto ane = hybrid_->predict(index, packed);
                ane = mx::squeeze(slice_axis(ane, 1, 0, sequence), 0);
                if (hybrid_->output_scale != 1.f)
                    ane = ane * Tensor(hybrid_->output_scale, ane.dtype());
                mx::eval({gpu, ane});
                down = gpu + mx::astype(ane, mx::float32);
            } catch (const std::exception &error) {
                std::fprintf(stderr,
                    "h3_qwen3_hybrid_runtime_fallback layer=%d reason=%s\n",
                    index, error.what());
                hybrid_->record_prefill_plan(
                    sequence, hybrid_->rows, sequence, 0, false,
                    "runtime_failure_latched");
                down = exact_mlp();
            }
        }
        if (hybrid_->runtime_available() &&
            std::getenv("TURBOCIDER_H3_QWEN3_HYBRID_VALIDATE")) {
            Tensor exact_prefix = normalized;
            {
                auto exact_gate_weight = weights_.slice(
                    prefix + "mlp.gate_proj.weight", 0, ane_end,
                    0, config_.hidden_size);
                auto exact_up_weight = weights_.slice(
                    prefix + "mlp.up_proj.weight", 0, ane_end,
                    0, config_.hidden_size);
                auto exact_gate = dense(normalized, exact_gate_weight);
                auto exact_up = dense(normalized, exact_up_weight);
                auto exact_activated =
                    exact_gate * mx::sigmoid(exact_gate) * exact_up;
                auto exact_down_weight = weights_.slice(
                    prefix + "mlp.down_proj.weight", 0,
                    config_.hidden_size, 0, ane_end);
                exact_prefix = dense(exact_activated, exact_down_weight);
                mx::eval({exact_prefix});
            }
            auto expected = gpu + exact_prefix;
            auto finite = mx::logical_and(mx::all(mx::isfinite(down)),
                                          mx::all(mx::isfinite(expected)));
            auto candidate = mx::astype(down, mx::float32);
            auto reference = mx::astype(expected, mx::float32);
            auto delta = candidate - reference;
            auto scale = mx::maximum(
                mx::maximum(mx::max(mx::abs(candidate)),
                            mx::max(mx::abs(reference))),
                Tensor(1.f, mx::float32));
            auto scaled_delta = delta / scale;
            auto scaled_reference = reference / scale;
            auto scaled_candidate = candidate / scale;
            auto delta2 = mx::sum(scaled_delta * scaled_delta);
            auto reference2 = mx::sum(scaled_reference * scaled_reference);
            auto candidate2 = mx::sum(scaled_candidate * scaled_candidate);
            auto dot = mx::sum(scaled_candidate * scaled_reference);
            auto max_abs = mx::max(mx::abs(delta));
            auto reference_max = mx::max(mx::abs(reference));
            mx::eval({finite, delta2, reference2, candidate2, dot,
                      max_abs, reference_max});
            const bool all_finite = finite.item<bool>();
            constexpr double invalid = 1e30;
            const double delta_energy = delta2.item<float>();
            const double reference_energy = reference2.item<float>();
            const double candidate_energy = candidate2.item<float>();
            const double relative_l2 = all_finite
                ? (reference_energy > 0
                    ? std::sqrt(delta_energy / reference_energy)
                    : (delta_energy == 0 ? 0 : invalid))
                : invalid;
            const double denominator =
                std::sqrt(candidate_energy * reference_energy);
            const double cosine = all_finite
                ? std::clamp(denominator > 0
                    ? dot.item<float>() / denominator
                    : (candidate_energy == 0 && reference_energy == 0 ? 1 : 0),
                    -1.0, 1.0)
                : -1;
            const double absolute = all_finite ? max_abs.item<float>() : invalid;
            const double amplitude = reference_max.item<float>();
            const double relative_abs = all_finite
                ? (amplitude > 0 ? absolute / amplitude
                                 : (absolute == 0 ? 0 : invalid))
                : invalid;
            const float max_relative_l2 = quality_threshold(
                "TURBOCIDER_H3_QWEN3_MAX_RELATIVE_L2", 0.025f);
            const float min_cosine = quality_threshold(
                "TURBOCIDER_H3_QWEN3_MIN_COSINE", 0.999f);
            const float max_relative_abs = quality_threshold(
                "TURBOCIDER_H3_QWEN3_MAX_RELATIVE_ABS", 0.05f);
            require(max_relative_l2 >= 0 && min_cosine >= -1 &&
                        min_cosine <= 1 && max_relative_abs >= 0,
                    "invalid H3 Qwen3 hybrid quality threshold range");
            const bool passed = all_finite &&
                relative_l2 <= max_relative_l2 && cosine >= min_cosine &&
                relative_abs <= max_relative_abs;
            hybrid_->record_quality(relative_l2, cosine, absolute,
                                    relative_abs, passed);
            std::fprintf(stderr,
                "h3_qwen3_hybrid layer=%d rel_l2=%.9g cosine=%.9g "
                "max_abs=%.9g relative_max_abs=%.9g passed=%d\n",
                index, relative_l2, cosine, absolute, relative_abs, passed);
            require(passed,
                    "H3 Qwen3 hybrid MLP quality gate failed at layer " +
                        std::to_string(index));
        }
        if (debug) {
            // Preserve the pre-hybrid debug ABI.  Diagnostic captures may pay
            // the cost of loading the complete MLP projections, while normal
            // execution keeps the streamed suffix-only path.  Keep these
            // separate from `down`, which must remain the hybrid candidate for
            // the layer output and quality gate above.
            debug_gate = dense(normalized, weight("mlp.gate_proj.weight"));
            debug_up = dense(normalized, weight("mlp.up_proj.weight"));
            debug_activated = debug_gate * mx::sigmoid(debug_gate) * debug_up;
            debug_down = dense(debug_activated, weight("mlp.down_proj.weight"));
        }
    } else {
        down = exact_mlp();
    }
    const bool exact_debug = hybrid_ && hybrid_->runtime_available() && debug;
    capture("layer0_gate", exact_debug ? debug_gate : gate);
    capture("layer0_up", exact_debug ? debug_up : up);
    capture("layer0_activated", exact_debug ? debug_activated : activated);
    capture("layer0_down", exact_debug ? debug_down : down);
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
    if (!hybrid_manifest_.empty()) {
        const auto prefill = ::tc::components::qwen3_prefill_plan(
            hybrid_manifest_, sequence);
        if (prefill.use_hybrid) {
            if (!hybrid_ || hybrid_->manifest != hybrid_manifest_.string())
                hybrid_ = std::make_unique<HybridSession>(
                    std::filesystem::absolute(hybrid_manifest_), component_root_,
                    sequence, event, cancelled, hybrid_warmup_iterations_,
                    conditioner_checkpoint(component_root_),
                    std::vector<LoRAAsset>{}, 0, 0);
            hybrid_->set_tokens(sequence);
            require(hybrid_->rows == prefill.compute_tokens &&
                        hybrid_->block_count > 0 &&
                        hybrid_->block_count <= output_layers,
                    "H3 Qwen3 encoder hybrid prefix is invalid");
            if (hybrid_->runtime_available())
                hybrid_->record_prefill_plan(
                    prefill.actual_tokens, prefill.selected_bucket,
                    prefill.compute_tokens, prefill.padding_tokens,
                    prefill.fixed_shape, prefill.reason);
            else
                hybrid_->record_prefill_plan(
                    sequence, hybrid_->rows, sequence, 0, false,
                    "runtime_failure_latched");
        } else {
            hybrid_.reset();
            event("h3_qwen3_encoder_gpu_" + prefill.reason, 1, 1);
        }
    } else {
        hybrid_.reset();
    }
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
