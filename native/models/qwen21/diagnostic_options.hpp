#pragma once
#include "../../core/contracts.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace tc::qwen21 {
// Keep the opt-in interpretation identical in planning, execution and result
// reporting. Unset and "0" are off; only the literal "1" enables a flag.
inline bool option_enabled(const char *value) {
    return value && std::string_view(value) == "1";
}

inline bool binary_option_or_unset(const char *value) {
    return !value || std::string_view(value) == "0" || option_enabled(value);
}

inline bool lora_base_ane(const Request &request) {
    if (request.hybrid_mlp_mode != "auto")
        return request.hybrid_mlp_mode == "lora_suffix" ||
               request.hybrid_mlp_mode == "lora_gate_up" ||
               request.hybrid_mlp_mode == "lora_fused";
    return option_enabled(std::getenv("TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC"));
}

inline bool gate_up_ane(const Request &request) {
    if (request.hybrid_mlp_mode != "auto")
        return request.hybrid_mlp_mode == "lora_gate_up";
    return option_enabled(std::getenv("TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC"));
}

inline bool fused_lora_ane(const Request &request) {
    return request.hybrid_mlp_mode == "lora_fused";
}

// Explicit six-step 1024px generation experiment. Keep this predicate shared
// by validation, plan labels and execution receipts; no edit/frozen/cache tier.
inline bool lora_1024_generation(const Request &request) {
    return option_enabled(std::getenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC")) &&
        request.model == "qwen-image-2.1" && request.width == 1024 && request.height == 1024 &&
        request.steps == 6 && request.operation == "image.generate" && request.inputs.empty() &&
        request.loras.size() == 1 && request.allow_approximation && request.residency == "resident" &&
        !request.prompt_enhance && !request.qwen21_w8a8 && !request.qwen21_gpu_w8a16 &&
        request.qwen21_gpu_full_ffn_blocks.empty() &&
        (request.execution == "gpu" || (request.execution == "gpu_ane" &&
                                       request.hybrid_mlp_mode == "runtime"));
}

// Return a negative sentinel for malformed values so planning and execution
// reject the same threshold rather than silently falling back to a default.
inline float db_cache_threshold(const char *value) {
    if (!value) return 0.08f;
    char *end = nullptr;
    const float threshold = std::strtof(value, &end);
    return end != value && *end == '\0' && std::isfinite(threshold) &&
           threshold > 0.f && threshold <= 0.5f ? threshold : -1.f;
}

// Additional explicit research control; the established policy remains two
// consecutive skips. Bound the range so a long decode periodically refreshes
// its middle residual even with a permissive difference threshold.
inline int db_cache_max_consecutive(const char *value) {
    if (!value) return 2;
    const std::string_view input(value);
    int limit = 0;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), limit);
    return error == std::errc{} && end == input.data() + input.size() &&
           limit >= 1 && limit <= 8 ? limit : -1;
}

// Shared by request planning and execution so an admitted shape cannot
// silently take a different number of W8A8 prefill blocks at runtime.
// Zero disables tiling; -1 denotes an unsupported option.
inline int tiled_prefill_layer_count(std::string_view option) {
    if (option == "0") return 0;
    if (option == "1") return 32; // original all-layer diagnostic
    if (option == "8") return 8;
    if (option == "16") return 16;
    if (option == "20") return 20;
    if (option == "24") return 24;
    return -1;
}

// Count Core ML predictions per request, not the session-cumulative metric.
// Kept independent of the MLX/Core ML runtime for small accounting tests.
struct W8A8CallBudget {
    int steps = 0;
    int decode_layers = 0;
    int decode_tiles = 1;
    int db_cached_steps = 0;
    int db_skipped_layers = 0;
    int final_reuse_layers = 0;
    int penultimate_reuse_layers = 0;
    int tiled_prefill_layers = 0;
    uint64_t prefix_tokens = 0;
    uint64_t total_tokens = 0;
    int tile_rows = 0;
    bool prefix_hit = false;
    bool tiled_prefix_reuse = false;
    bool prefix_target_only = false;
    bool last_target_only = false;
};

constexpr uint64_t expected_w8a8_calls(const W8A8CallBudget &budget) {
    const auto decode = (uint64_t(budget.steps - 1) * budget.decode_layers -
        uint64_t(budget.db_cached_steps) * budget.db_skipped_layers -
        budget.final_reuse_layers - budget.penultimate_reuse_layers) * budget.decode_tiles;
    if (!budget.tiled_prefill_layers) return decode;
    const uint64_t rows = budget.tile_rows;
    if (budget.prefix_hit && budget.tiled_prefix_reuse) {
        if (budget.prefix_target_only) return decode + budget.tiled_prefill_layers;
        const bool partial_prefix = budget.prefix_tokens % rows != 0;
        return decode + uint64_t(budget.tiled_prefill_layers) * (partial_prefix ? 2 : 1) -
            uint64_t(budget.last_target_only && partial_prefix);
    }
    const uint64_t tiles = (budget.total_tokens + rows - 1) / rows;
    return decode + uint64_t(budget.tiled_prefill_layers) * tiles -
        (budget.last_target_only ? tiles - 1 : 0);
}
} // namespace tc::qwen21
