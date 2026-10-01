#pragma once
#include "../../core/contracts.hpp"
#include "viggle_adapter.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
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

// Request opt-in for the locally verified r128 GPU editing route. Planning
// recognizes the release filename; Session still verifies its pinned SHA-256
// before binding. This does not qualify r256 or any runtime/frozen ANE route.
inline bool gpu_viggle_ref512_request(const Request &request) {
    if (request.model != "qwen-image-2.1" || request.execution != "gpu" ||
        request.hybrid_mlp_mode != "auto" || request.operation != "image.edit" ||
        request.width != 512 || request.height != 512 || request.steps != 6 ||
        request.qwen21_reference_size != 512 || !request.allow_approximation ||
        request.inputs.empty() || request.inputs.size() > 3 ||
        request.loras.size() != 1 || request.lora_strategy != "inference_time" ||
        request.prompt_enhance || request.prompt_enhance_edit_experimental ||
        request.qwen21_dit_cache != "off" ||
        request.qwen21_w8a8 || request.qwen21_gpu_w8a16 ||
        !request.qwen21_gpu_full_ffn_blocks.empty() || !request.ane_manifest.empty() ||
        !request.encoder_ane_manifest.empty()) return false;
    const auto &lora = request.loras.front();
    const auto *adapter = viggle_v021_adapter(
        std::filesystem::path(lora.path).filename().string());
    return adapter && adapter->rank == "r128" &&
        lora.role == "transformer" && lora.strength == 1.f;
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

// Restrict the SGLang comparison to its F1/W4 candidate and the established
// F8/W8 policy. These are diagnostic controls, never preset overrides.
inline int db_cache_front_blocks(const char *value) {
    if (!value) return 8;
    const std::string_view input(value);
    return input == "1" ? 1 : input == "8" ? 8 : -1;
}

inline int db_cache_warmup_steps(const char *value) {
    if (!value) return 8;
    const std::string_view input(value);
    return input == "4" ? 4 : input == "8" ? 8 : -1;
}

struct DbCacheOptions {
    bool enabled = false;
    float threshold = 0.08f;
    int max_consecutive = 2;
    int front_blocks = 8;
    int warmup_steps = 8;
    bool diagnostic = false;
    std::string mode = "off";
};

// Request presets own their parameters. Legacy environment diagnostics remain
// available with mode=off, but cannot silently override a saved App request.
inline DbCacheOptions db_cache_options(const Request &request) {
    // An explicit per-request Off wins over a process-wide diagnostic default.
    // Requests that omit the field retain the original environment behavior.
    if (request.qwen21_dit_cache_explicit && request.qwen21_dit_cache == "off")
        return {};
    const char *flag = std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC");
    const char *threshold = std::getenv("TURBOCIDER_QWEN21_DBCACHE_THRESHOLD");
    const char *maximum = std::getenv("TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE");
    const char *front = std::getenv("TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS");
    const char *warmup = std::getenv("TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS");
    if (!binary_option_or_unset(flag))
        throw std::invalid_argument("Qwen21 DBCache diagnostic accepts only 0 or 1");
    DbCacheOptions result;
    result.mode = request.qwen21_dit_cache;
    if (result.mode != "off") {
        if (result.mode != "conservative" && result.mode != "balanced" && result.mode != "fast")
            throw std::invalid_argument("Qwen21 DiT cache mode must be off, conservative, balanced or fast");
        if (option_enabled(flag) || threshold || maximum || front || warmup)
            throw std::invalid_argument("Qwen21 DiT cache preset conflicts with DBCache environment diagnostics; unset threshold/max/front/warmup overrides and disable the diagnostic flag");
        result.enabled = true;
        result.threshold = result.mode == "conservative" ? 0.15f : 0.25f;
        result.max_consecutive = result.mode == "conservative" ? 1 :
                                 result.mode == "balanced" ? 2 : 4;
        return result;
    }
    result.enabled = option_enabled(flag);
    result.diagnostic = result.enabled;
    result.threshold = db_cache_threshold(threshold);
    result.max_consecutive = db_cache_max_consecutive(maximum);
    result.front_blocks = db_cache_front_blocks(front);
    result.warmup_steps = db_cache_warmup_steps(warmup);
    if (result.threshold <= 0.f || result.max_consecutive <= 0 ||
        result.front_blocks <= 0 || result.warmup_steps <= 0 ||
        ((maximum || front || warmup) && !result.enabled))
        throw std::invalid_argument("Qwen21 DBCache requires a finite threshold in (0, 0.5], a max consecutive skip count of 1...8, front blocks 1 or 8, and warmup steps 4 or 8; max/front/warmup overrides require the diagnostic flag");
    return result;
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
