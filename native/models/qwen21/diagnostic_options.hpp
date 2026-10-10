#pragma once
#include "../../core/common.hpp"
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
inline bool compiled_encoder_gpu(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU");
    const char *prefill=std::getenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU");
    require(binary_option_or_unset(prefill),"Qwen fused GPU encoder prefill requires 0 or 1");
    require(binary_option_or_unset(raw),"Qwen compiled encoder GPU requires 0 or1");
    const bool enabled=option_enabled(raw) || option_enabled(prefill);
    if(option_enabled(prefill))require(r.encoder_ane_manifest.empty(),"fused encoder prefill is GPU-only");
    if(enabled)require(r.model=="qwen-image-2.1" && r.allow_approximation && r.residency=="resident" &&
        r.width==512 && r.height==512 && !r.prompt_enhance && !r.streaming.active() &&
        !r.memory_constrained.enabled && !r.memory_budget_bytes &&
        (r.operation=="image.generate" || (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=2 &&
            r.qwen21_reference_size==512)),
        "compiled Qwen encoder GPU requires approximate unconstrained resident512 generation/editing and at most two ref512");
    return enabled;
}
inline bool joint_bf16_lora_ab(const Request &request) {
    return request.model=="qwen-image-2.1" && !request.loras.empty() &&
        option_enabled(std::getenv("TURBOCIDER_QWEN21_LORA_BF16_AB"));
}

enum class RuntimeFfnPhase { Invalid, All, Prefill, Decode };
inline RuntimeFfnPhase runtime_ffn_phase(const char *value) {
    if (!value || std::string_view(value) == "all") return RuntimeFfnPhase::All;
    if (std::string_view(value) == "prefill") return RuntimeFfnPhase::Prefill;
    if (std::string_view(value) == "decode") return RuntimeFfnPhase::Decode;
    return RuntimeFfnPhase::Invalid;
}
inline const char *runtime_ffn_phase_name(RuntimeFfnPhase phase) {
    return phase == RuntimeFfnPhase::Prefill ? "prefill" :
           phase == RuntimeFfnPhase::Decode ? "decode" :
           phase == RuntimeFfnPhase::All ? "all" : "invalid";
}
inline bool runtime_ffn_phase_runs(RuntimeFfnPhase phase, bool prefix_reused) {
    return phase == RuntimeFfnPhase::All ||
           (prefix_reused ? phase == RuntimeFfnPhase::Decode : phase == RuntimeFfnPhase::Prefill);
}
inline std::string runtime_ffn_phase_identity(RuntimeFfnPhase phase) {
    // Preserve the unflagged executor identity. Nondefault request snapshots
    // must not borrow an executor/calibration/prefix bank from another policy.
    return phase == RuntimeFfnPhase::All ? "" :
        std::string(":ffn-phase-v1=") + runtime_ffn_phase_name(phase);
}

inline int student_ffn_reuse_layers(const char *value) {
    if(!value || std::string_view(value)=="0")return 0;
    if(std::string_view(value)=="1" || std::string_view(value)=="32")return 32;
    if(std::string_view(value)=="16")return 16;
    return -1;
}
inline bool student_final_ffn_reuse(const Request &request) {
    return request.model=="qwen-image-2.1" && !request.loras.empty() &&
        student_ffn_reuse_layers(std::getenv("TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"))>0;
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
