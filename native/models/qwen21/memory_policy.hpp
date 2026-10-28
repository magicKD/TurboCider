#pragma once
#include "../../core/contracts.hpp"
#include "diagnostic_options.hpp"
#include "../../platform/apple/platform.hpp"
#include <cstdlib>

namespace tc::qwen21 {
// Qualified small-canvas BF16 route: consume the language encoder one layer
// at a time, then stream DiT layers on every step, then load the VAE. Larger canvases,
// multimodal conditioning and PE retain their existing estimate.
inline bool layer_staged_hybrid_request(const Request &r, const DeviceInfo &device) {
    // Native currently owns all dense DiT weights during the complete graph.
    // It must never inherit the smaller per-layer MLX residency estimate.
    if (r.qwen21_denoiser_backend != "mlx") return false;
    if (!device.optimizations().qwen21_layer_streaming ||
        (r.hybrid_mlp_mode != "auto" && r.hybrid_mlp_mode != "base_fused")) return false;
    if (lora_base_ane(r) || gate_up_ane(r) || fused_lora_ane(r)) return false;
    // Existing diagnostic caches/fusions retain weights or change execution
    // geometry. Keep their established memory estimate and resident route.
    for (const char *flag : {"TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
            "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC"})
        if (option_enabled(std::getenv(flag))) return false;
    return r.model == "qwen-image-2.1" && r.residency == "component_staged" &&
        r.operation == "image.generate" && r.inputs.empty() &&
        r.execution == "gpu_ane" && r.qwen21_w8a8 &&
        r.allow_approximation && !r.ane_manifest.empty() &&
        r.loras.empty() && !r.prompt_enhance && !r.qwen21_gpu_w8a16 &&
        r.qwen21_gpu_full_ffn_blocks.empty() && r.width == 512 && r.height == 512 &&
        r.steps >= 2;
}

// This is the measured M5 Pro 24 GiB recipe, not a general M5 policy. Keep
// admission independent of optimization overrides so ablations can retain
// the same streamed residency and existing DBCache schedule.
inline bool qualified_staged_optimizations(const Request &r, const DeviceInfo &device) {
    if (!layer_staged_hybrid_request(r, device) || r.steps < 20 || r.steps > 40 ||
        !r.encoder_ane_manifest.empty() || !r.dump.empty() ||
        r.prompt_enhance_edit_experimental || !r.prompt_enhancer_path.empty() ||
        r.streaming.active() || r.streaming_offload || r.memory_constrained.enabled ||
        r.memory_budget_bytes || r.quantized_execution.active() ||
        (r.streaming_selector && r.streaming_selector->active()) ||
        (r.streaming_requested && r.streaming_requested->active()) ||
        (r.streaming_selector_requested && r.streaming_selector_requested->active()) ||
        r.compile_gpu || r.vsa || !r.qwen21_native_ane_cache.empty()) return false;
    for (const char *flag : {"TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
            "TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION",
            "TURBOCIDER_QWEN21_METAL_QK_ROPE",
            "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
            "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
            "TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_PROFILE_STEPS",
            "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS",
            "TURBOCIDER_QWEN21_PROFILE_GPU_OPS",
            "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS",
            "TURBOCIDER_QWEN21_PROFILE_RUNTIME_LORA_FFN",
            "TURBOCIDER_QWEN21_PROFILE_TILED_FFN",
            "TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS",
            "TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME",
            "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU",
            "TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
            "TURBOCIDER_QWEN21_BF16_STREAMING"}) {
        const char *value = std::getenv(flag);
        if (value && std::string_view(value) != "0") return false;
    }
    for (const char *path : {"TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST",
            "TURBOCIDER_QWEN21_ENCODER_CALIBRATION_OUT",
            "TURBOCIDER_QWEN21_ENCODER_S1_PROFILE",
            "TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE",
            "TURBOCIDER_QWEN21_NATIVE_PREFILL_ANE_CACHE",
            "TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS"}) {
        const char *value = std::getenv(path);
        if (value && *value) return false;
    }
    const char *encoder = std::getenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");
    if (encoder && std::string_view(encoder) != "mlx") return false;
    const char *ffn_phase = std::getenv("TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE");
    if (ffn_phase && std::string_view(ffn_phase) != "all") return false;
    const auto cache = db_cache_options(r);
    return !cache.enabled || (cache.front_blocks == 8 && cache.warmup_steps == 8);
}

struct StagedOptimizationOptions {
    int suffix_bits = 0;
    int attention_bits = 0;
    bool qk_norm_rope = false;
    bool dit_w8a8 = false;
};

inline std::string staged_quantized_precision(const StagedOptimizationOptions &options) {
    if (options.dit_w8a8) return "w8a8_dit_pc_fp16_compute_bf16_io+w8a8_mlp_prefix_fp16_io";
    if (!options.suffix_bits && !options.attention_bits) return {};
    return std::string("bf16_prefill+") +
        (options.attention_bits ? "w" + std::to_string(options.attention_bits) + "_attention_bf16_io"
                                : "bf16_attention") + "+" +
        (options.suffix_bits ? "w" + std::to_string(options.suffix_bits) + "_ffn_suffix_fp16_io"
                             : "bf16_ffn_suffix") + "+w8a8_mlp_prefix_fp16_io";
}

inline StagedOptimizationOptions staged_optimization_options(const Request &r, const DeviceInfo &device) {
    const char *defaults = std::getenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS");
    require(binary_option_or_unset(defaults), "Qwen staged fast defaults require 0 or 1");
    const bool qualified = qualified_staged_optimizations(r, device);
    require(!r.qwen21_dit_smoothquant_convrot || r.qwen21_dit_w8a8,
            "SmoothQuant+ConvRot requires DiT W8A8 with its staged ANE eligibility conditions");
    if (r.qwen21_dit_w8a8) {
        require(qualified && r.frames == 1 && !r.audio,
                "DiT W8A8 requires M5 Pro 24 GiB, no LoRA, 512x512/20-40 steps, layer-streamed component_staged, GPU+ANE W8A8 and original MLX text/VAE precision");
        require(!defaults || option_enabled(defaults),
                "DiT W8A8 cannot combine with disabled staged defaults");
        for (const char *name : {"TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS", "TURBOCIDER_QWEN21_STAGED_ATTN_BITS"}) {
            const char *value = std::getenv(name);
            require(!value || std::string_view(value) == "8",
                    "DiT W8A8 cannot combine with a different diagnostic weight precision");
        }
        return {8, 8, option_or_default(std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"), true), true};
    }
    if (r.qwen21_staged_fast == false) return {};
    const bool use_defaults = qualified && r.qwen21_staged_fast.value_or(true) &&
        (!defaults || option_enabled(defaults));
    auto bits = [qualified, use_defaults](const char *name) {
        const char *value = std::getenv(name);
        if (!value) return use_defaults ? 4 : 0;
        const std::string_view input(value);
        require(input == "0" || input == "4" || input == "8",
                std::string(name) + " requires 0, 4 or 8");
        const int result = input == "4" ? 4 : input == "8" ? 8 : 0;
        require(!result || qualified,
                std::string(name) + " requires the qualified M5 Pro 24 GiB staged W8A8 request");
        return result;
    };
    return {bits("TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS"),
            bits("TURBOCIDER_QWEN21_STAGED_ATTN_BITS"),
            // Explicit legacy QK selection remains valid outside this profile.
            option_or_default(std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"), use_defaults)};
}

inline bool layer_staged_hybrid_t2i(const Request &r, const DeviceInfo &device) {
    if (!layer_staged_hybrid_request(r, device)) return false;
    return !db_cache_options(r).enabled || qualified_staged_optimizations(r, device);
}
inline bool layer_staged_t2i(const Request &r, const DeviceInfo &device) {
    if (r.qwen21_denoiser_backend != "mlx") return false;
    if (!device.optimizations().qwen21_layer_streaming) return false;
    if (layer_staged_hybrid_t2i(r, device)) return true;
    return r.model == "qwen-image-2.1" && r.residency == "component_staged" &&
        r.operation == "image.generate" && r.inputs.empty() &&
        (r.execution == "gpu" || r.execution == "auto") &&
        r.loras.empty() && !r.prompt_enhance && !r.allow_approximation &&
        r.width <= 512 && r.height <= 512;
}
} // namespace tc::qwen21
