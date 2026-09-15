#include "qwen3.hpp"
#include "../../backends/coreml.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tc::components {

static int qwen3_integer_setting(const char *name, int fallback, int minimum,
                                 int maximum) {
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    require(errno == 0 && end != value && *end == '\0' &&
                parsed >= minimum && parsed <= maximum,
            std::string("invalid Qwen3 integer setting: ") + name);
    return int(parsed);
}

std::filesystem::path qwen3_checkpoint_path(const std::filesystem::path &path) {
    if (std::filesystem::is_regular_file(path))
        return std::filesystem::canonical(path);
    require(std::filesystem::is_directory(path),
            "Qwen3 text encoder path is neither a file nor directory: " + path.string());
    const auto single = path / "model.safetensors";
    int regular_safetensors = 0;
    for (const auto &entry : std::filesystem::directory_iterator(path))
        if (!entry.is_symlink() && entry.is_regular_file() &&
            entry.path().extension() == ".safetensors")
            ++regular_safetensors;
    // Match Weights::load(): a directory with one regular safetensors file is
    // loaded from that file even if a stale shard index remains beside it.
    if (regular_safetensors == 1 && std::filesystem::is_regular_file(single) &&
        !std::filesystem::is_symlink(single))
        return std::filesystem::canonical(single);
    const auto index = path / "model.safetensors.index.json";
    if (std::filesystem::is_regular_file(index) && !std::filesystem::is_symlink(index))
        return std::filesystem::canonical(index);
    if (std::filesystem::is_regular_file(single) && !std::filesystem::is_symlink(single))
        return std::filesystem::canonical(single);
    throw std::runtime_error("Qwen3 safetensors checkpoint/index missing under " +
                             path.string());
}

bool qwen3_hybrid_profitable(int tokens) {
    const int threshold = qwen3_integer_setting(
        "TURBOCIDER_QWEN3_ANE_MIN_TOKENS", 256, 1, 1 << 20);
    return tokens >= threshold;
}

Qwen3PrefillPlan qwen3_prefill_plan_rows(int actual_tokens, int bucket,
                                         int minimum_profitable_rows) {
    require(actual_tokens > 0, "Qwen3 prefill token count must be positive");
    require(bucket >= 0, "Qwen3 prefill bucket must be nonnegative");
    if (bucket > 0 && minimum_profitable_rows == 0)
        minimum_profitable_rows = bucket;
    require(minimum_profitable_rows >= 0 &&
                (!bucket || minimum_profitable_rows <= bucket),
            "Qwen3 minimum profitable rows must fit the selected bucket");
    Qwen3PrefillPlan plan;
    plan.actual_tokens = actual_tokens;
    plan.selected_bucket = bucket;
    plan.minimum_profitable_rows = minimum_profitable_rows;
    plan.compute_tokens = actual_tokens;
    if (!bucket) {
        plan.reason = "no_hybrid_bucket";
        return plan;
    }
    if (bucket < actual_tokens) {
        plan.reason = "manifest_capacity_exceeded";
        return plan;
    }
    if (bucket == actual_tokens) {
        plan.compute_tokens = bucket;
        plan.use_hybrid = true;
        plan.fixed_shape = true;
        plan.reason = "exact_bucket";
        return plan;
    }
    if (minimum_profitable_rows > 0 && actual_tokens < minimum_profitable_rows) {
        plan.reason = "below_min_profitable_rows";
        return plan;
    }
    if (minimum_profitable_rows == 0 && !qwen3_hybrid_profitable(actual_tokens)) {
        plan.reason = "below_min_tokens";
        return plan;
    }
    if (std::getenv("TURBOCIDER_QWEN3_DISABLE_MLP_TAIL_PADDING")) {
        plan.reason = "tail_padding_disabled";
        return plan;
    }
    // Default to at most 100% synthetic rows.  The bound is intentionally
    // configurable for device-specific qualification, but never implicit.
    const int max_padding_percent = qwen3_integer_setting(
        "TURBOCIDER_QWEN3_MLP_MAX_PADDING_PERCENT", 100, 0, 10000);
    const int padding = bucket - actual_tokens;
    if (int64_t(padding) * 100 >
        int64_t(actual_tokens) * max_padding_percent) {
        plan.reason = "padding_ratio_exceeded";
        return plan;
    }
    plan.compute_tokens = bucket;
    plan.padding_tokens = padding;
    plan.use_hybrid = true;
    plan.fixed_shape = true;
    plan.reason = "fixed_bucket";
    return plan;
}

Qwen3PrefillPlan qwen3_prefill_plan(const std::filesystem::path &manifest,
                                    int actual_tokens) {
    require(!manifest.empty(), "Qwen3 prefill manifest must not be empty");
    const auto bucket = hybrid_bucket_plan(manifest, actual_tokens);
    auto plan = qwen3_prefill_plan_rows(
        actual_tokens, bucket.supported ? bucket.selected_rows : 0,
        bucket.supported ? bucket.minimum_profitable_rows : 0);
    if (!bucket.supported && qwen3_hybrid_profitable(actual_tokens))
        plan.reason = "manifest_capacity_exceeded";
    return plan;
}

static bool qwen3_fused_attention_enabled(int tokens) {
    if (!std::getenv("TURBOCIDER_QWEN3_FUSED_SDPA") ||
        std::getenv("TURBOCIDER_QWEN3_DISABLE_FUSED_SDPA"))
        return false;
    const int threshold = qwen3_integer_setting(
        "TURBOCIDER_QWEN3_FUSED_SDPA_MIN_TOKENS", 128, 1, 1 << 20);
    return tokens >= threshold;
}

static float qwen3_validation_threshold(const char *name, float fallback) {
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    char *end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value, &end);
    require(errno == 0 && end != value && *end == '\0' && std::isfinite(parsed),
            std::string("invalid Qwen3 validation threshold: ") + name);
    return parsed;
}

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
                          std::atomic<bool> &cancelled, HybridSession *hybrid) {
    require(!config.output_layers.empty() && config.output_layers.front() >= 0 &&
                std::is_sorted(config.output_layers.begin(), config.output_layers.end()),
            "Qwen3 output layers must be nonempty and sorted");
    require(config.kv_heads > 0 && config.heads % config.kv_heads == 0 &&
                config.head_dim > 0 && config.head_dim % 2 == 0,
            "invalid Qwen3 attention geometry");
    const int count = int(tokens.ids.size());
    require(count > 0 && tokens.valid > 0 && tokens.valid <= count,
            "invalid Qwen3 token sequence");
    auto prefill = hybrid
        ? qwen3_prefill_plan_rows(count, hybrid->rows,
                                  hybrid->minimum_profitable_rows)
        : Qwen3PrefillPlan{count, 0, count, 0, 0, false, false,
                           "no_hybrid_bucket"};
    if (hybrid && !hybrid->runtime_available())
        prefill = Qwen3PrefillPlan{
            count, hybrid->rows, count, 0, hybrid->minimum_profitable_rows,
            false, false, "runtime_failure_latched"};
    if (hybrid)
        hybrid->record_prefill_plan(prefill.actual_tokens, prefill.selected_bucket,
                                    prefill.compute_tokens, prefill.padding_tokens,
                                    prefill.fixed_shape, prefill.reason);
    HybridSession *active_hybrid = hybrid && prefill.use_hybrid ? hybrid : nullptr;
    // Bound the lazy graph by token-rows instead of fencing every transformer
    // layer.  Four in-flight layers recover most of the short-prompt launch
    // overhead, while longer prompts reduce the interval to keep temporary
    // activation memory roughly stable.
    int eval_interval = qwen3_integer_setting(
        "TURBOCIDER_QWEN3_EVAL_INTERVAL", std::clamp(1024 / count, 1, 4),
        0, 1 << 20);
    if (std::getenv("TURBOCIDER_QWEN3_DEFER_LAYER_EVAL"))
        eval_interval = 0;
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
    std::optional<Tensor> mask;
    if (tokens.valid < count) {
        auto positions = mx::arange(count, mx::int32);
        auto query = mx::reshape(positions, {count, 1});
        auto key = mx::reshape(positions, {1, count});
        auto forbidden = mx::logical_or(key > query, key >= Tensor(tokens.valid));
        mask = mx::reshape(mx::where(forbidden, Tensor(-INFINITY, rope_dtype),
                                     Tensor(0.f, rope_dtype)), {1, 1, count, count});
    }
    auto rotate = [&](const Tensor &value) {
        auto halves = mx::split(value, 2, -1);
        return value * cosine + mx::concatenate({-halves[1], halves[0]}, -1) * sine;
    };
    const int layers = config.output_layers.back() + 1;
    int hybrid_mlp_width = 0;
    int hybrid_ane_end = 0;
    if (active_hybrid) {
        const int hidden = weights.at("model.embed_tokens.weight").shape(1);
        require(hybrid->hidden == hidden,
                "Qwen3 Core ML hidden dimension mismatch");
        require(hybrid->block_count >= layers,
                "Qwen3 Core ML manifest has too few MLP blocks");
        const auto &gate = weights.at("model.layers.0.mlp.gate_proj.weight");
        const auto &up = weights.at("model.layers.0.mlp.up_proj.weight");
        const auto &down = weights.at("model.layers.0.mlp.down_proj.weight");
        auto logical_columns = [&](const std::string &prefix, const Tensor &weight) {
            require(!weights.convrot(prefix) && !weights.nvfp4(prefix),
                    "Qwen3 hybrid supports dense or affine MLX weights only");
            return weights.quantized(prefix)
                       ? weights.at(prefix + ".scales").shape(1) * 32
                       : weight.shape(1);
        };
        require(gate.ndim() == 2 && up.ndim() == 2 && down.ndim() == 2 &&
                    gate.shape(0) == up.shape(0) && down.shape(0) == hidden &&
                    logical_columns("model.layers.0.mlp.gate_proj", gate) == hidden &&
                    logical_columns("model.layers.0.mlp.up_proj", up) == hidden &&
                    logical_columns("model.layers.0.mlp.down_proj", down) == gate.shape(0),
                "invalid Qwen3 MLP geometry for Core ML partition");
        hybrid_mlp_width = gate.shape(0);
        require(hybrid->mlp_width == hybrid_mlp_width && hybrid->ane_mlp_start == 0 &&
                    hybrid->ane_mlp_end > 0 && hybrid->ane_mlp_end < hybrid_mlp_width,
                "Qwen3 Core ML MLP partition geometry mismatch");
        hybrid_ane_end = hybrid->ane_mlp_end;
    }
    std::vector<Tensor> outputs;
    for (int layer = 0; layer < layers; ++layer) {
        checkpoint(cancelled);
        event(config.progress_phase, layer, layers);
        const auto prefix = "model.layers." + std::to_string(layer);
        auto normalized = rms(x, weights.at(prefix + ".input_layernorm.weight"), config.norm_epsilon);
        auto projections = weights.project_many(
            normalized, {prefix + ".self_attn.q_proj", prefix + ".self_attn.k_proj",
                         prefix + ".self_attn.v_proj"});
        auto q = heads(projections[0], config.heads, config.head_dim);
        auto k = heads(projections[1], config.kv_heads, config.head_dim);
        auto v = heads(projections[2], config.kv_heads, config.head_dim);
        q = rotate(rms(q, weights.at(prefix + ".self_attn.q_norm.weight"), config.norm_epsilon));
        k = rotate(rms(k, weights.at(prefix + ".self_attn.k_norm.weight"), config.norm_epsilon));
        // MLX SDPA accepts grouped-query attention directly. At 128+ tokens it
        // avoids materializing four times as many K/V rows; this MLX release is
        // still faster with the historical expansion for shorter prompts.
        // Keep the environment override as a diagnostic fallback.
        if (count < 128 || std::getenv("TURBOCIDER_QWEN3_EAGER_GQA")) {
            k = mx::repeat(k, config.heads / config.kv_heads, 1);
            v = mx::repeat(v, config.heads / config.kv_heads, 1);
        }
        x = x + linear(attend(q, k, v, true, mask,
                              qwen3_fused_attention_enabled(count),
                              mask ? "" : "causal"),
                         weights, prefix + ".self_attn.o_proj");
        normalized = rms(x, weights.at(prefix + ".post_attention_layernorm.weight"), config.norm_epsilon);
        Tensor feed = normalized;
        bool used_hybrid_output = false;
        if (active_hybrid) {
            const int actual_rows = normalized.shape(1);
            auto packed = mx::astype(normalized, mx::float16);
            if (actual_rows < active_hybrid->rows)
                packed = mx::concatenate(
                    {packed, mx::zeros({1, active_hybrid->rows - actual_rows,
                                       normalized.shape(-1)}, mx::float16)}, 1);
            mx::eval({normalized, packed});

            const auto gate_prefix = prefix + ".mlp.gate_proj";
            const auto up_prefix = prefix + ".mlp.up_proj";
            const auto down_prefix = prefix + ".mlp.down_proj";
            auto exact_mlp = [&] {
                return linear(
                    silu(linear(normalized, weights, gate_prefix)) *
                        linear(normalized, weights, up_prefix),
                    weights, down_prefix);
            };
            auto gate = weights.project_slice(normalized, gate_prefix,
                                              hybrid_ane_end, hybrid_mlp_width,
                                              0, normalized.shape(-1), true);
            auto up = weights.project_slice(normalized, up_prefix,
                                            hybrid_ane_end, hybrid_mlp_width,
                                            0, normalized.shape(-1), true);
            auto gpu = weights.project_slice(
                silu(gate) * up, down_prefix, 0, normalized.shape(-1),
                hybrid_ane_end, hybrid_mlp_width, true);
            if (std::getenv("TURBOCIDER_QWEN3_DISABLE_HYBRID_OVERLAP"))
                mx::eval(gpu);
            else
                mx::async_eval({gpu});
            try {
                auto ane = active_hybrid->predict(layer, packed);
                ane = slice_axis(ane, 1, 0, actual_rows);
                if (active_hybrid->output_scale != 1.f)
                    ane = ane * Tensor(active_hybrid->output_scale, ane.dtype());
                mx::eval({gpu, ane});
                feed = gpu + mx::astype(ane, gpu.dtype());
                used_hybrid_output = true;
            } catch (const std::exception &error) {
                std::fprintf(stderr,
                    "qwen3_hybrid_runtime_fallback layer=%d reason=%s\n",
                    layer, error.what());
                event("qwen3_hybrid_runtime_fallback", layer + 1, layers);
                active_hybrid->record_prefill_plan(
                    count, active_hybrid->rows, count, 0, false,
                    "runtime_failure_latched");
                active_hybrid = nullptr;
                feed = exact_mlp();
            }
            if (active_hybrid &&
                std::getenv("TURBOCIDER_QWEN3_HYBRID_VALIDATE")) {
                auto reference = exact_mlp();
                auto finite = mx::logical_and(mx::all(mx::isfinite(feed)),
                                              mx::all(mx::isfinite(reference)));
                auto candidate = mx::astype(feed, mx::float32);
                auto expected = mx::astype(reference, mx::float32);
                auto delta = candidate - expected;
                // Keep reductions in range for large FLUX 9B activations.
                // Relative L2 and cosine are invariant to a common positive
                // scale, while the unscaled FP32 sum-of-squares can overflow.
                auto scale = mx::maximum(
                    mx::maximum(mx::max(mx::abs(candidate)),
                                mx::max(mx::abs(expected))),
                    Tensor(1.f, mx::float32));
                auto scaled_delta = delta / scale;
                auto scaled_expected = expected / scale;
                auto scaled_candidate = candidate / scale;
                auto delta2 = mx::sum(scaled_delta * scaled_delta);
                auto expected2 = mx::sum(scaled_expected * scaled_expected);
                auto candidate2 = mx::sum(scaled_candidate * scaled_candidate);
                auto dot = mx::sum(scaled_candidate * scaled_expected);
                auto max_abs = mx::max(mx::abs(delta));
                auto reference_max = mx::max(mx::abs(expected));
                mx::eval({finite, scale, delta2, expected2, candidate2, dot,
                          max_abs, reference_max});
                const bool all_finite = finite.item<bool>();
                const double delta_energy = delta2.item<float>();
                const double expected_energy = expected2.item<float>();
                const double candidate_energy = candidate2.item<float>();
                constexpr double invalid_metric = 1e30;
                const double relative_l2 = all_finite
                    ? (expected_energy > 0
                        ? std::sqrt(delta_energy / expected_energy)
                        : (delta_energy == 0 ? 0 : invalid_metric))
                    : invalid_metric;
                const double cosine_denominator =
                    std::sqrt(candidate_energy * expected_energy);
                const double cosine = all_finite
                    ? std::clamp(cosine_denominator > 0
                        ? dot.item<float>() / cosine_denominator
                        : (candidate_energy == 0 && expected_energy == 0 ? 1 : 0),
                        -1.0, 1.0)
                    : -1;
                const double absolute = all_finite ? max_abs.item<float>() : invalid_metric;
                const double reference_amplitude = reference_max.item<float>();
                const double relative_abs = all_finite
                    ? (reference_amplitude > 0
                        ? absolute / reference_amplitude
                        : (absolute == 0 ? 0 : invalid_metric))
                    : invalid_metric;
                const float max_relative_l2 = qwen3_validation_threshold(
                    "TURBOCIDER_QWEN3_MAX_RELATIVE_L2", 0.025f);
                const float min_cosine = qwen3_validation_threshold(
                    "TURBOCIDER_QWEN3_MIN_COSINE", 0.999f);
                const float max_relative_abs = qwen3_validation_threshold(
                    "TURBOCIDER_QWEN3_MAX_RELATIVE_ABS", 0.05f);
                require(max_relative_l2 >= 0 && min_cosine >= -1 && min_cosine <= 1 &&
                            max_relative_abs >= 0,
                        "invalid Qwen3 hybrid quality threshold range");
                const bool passed = all_finite &&
                    relative_l2 <= max_relative_l2 && cosine >= min_cosine &&
                    relative_abs <= max_relative_abs;
                active_hybrid->record_quality(relative_l2, cosine, absolute,
                                              relative_abs, passed);
                std::fprintf(stderr,
                    "qwen3_hybrid layer=%d rel_l2=%.9g cosine=%.9g max_abs=%.9g "
                    "relative_max_abs=%.9g passed=%d\n",
                    layer, relative_l2, cosine, absolute, relative_abs, passed);
                require(passed,
                        "Qwen3 hybrid MLP quality gate failed at layer " +
                            std::to_string(layer));
                event("qwen3_hybrid_validate", layer + 1, layers);
            }
        } else {
            feed = linear(silu(linear(normalized, weights, prefix + ".mlp.gate_proj")) *
                              linear(normalized, weights, prefix + ".mlp.up_proj"),
                          weights, prefix + ".mlp.down_proj");
        }
        x = x + feed;
        // Every flexible Core ML branch copies into one session-wide storage
        // tensor.  Materialize its residual consumer before the next branch
        // reuses that storage; otherwise MLX's lazy graph can observe the next
        // layer's output instead of this layer's result.
        if (used_hybrid_output ||
            (eval_interval > 0 && (layer + 1) % eval_interval == 0))
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
