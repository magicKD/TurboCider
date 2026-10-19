#pragma once
#include "../../core/contracts.hpp"
#include "viggle_adapter.hpp"
#include "../../core/common.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace tc::qwen21 {
// Literal flag parsing stays identical in planning, execution and reporting.
// Source-qualified GGUF defaults are resolved separately once sources are known.
inline bool option_enabled(const char *value) {
    return value && std::string_view(value) == "1";
}

inline bool binary_option_or_unset(const char *value) {
    return !value || std::string_view(value) == "0" || option_enabled(value);
}
// Ordinary style adapters must retain the base Euler schedule. Legacy
// experimental runtime adapters keep their six-step admission unless this
// separate, default-off research mode is explicitly selected.
inline bool base_schedule_runtime_lora_request(const Request &r,const char *raw) {
    require(binary_option_or_unset(raw),"Qwen base-schedule runtime LoRA requires0 or1");
    if(!option_enabled(raw) || r.loras.empty())return false;
    require(r.model=="qwen-image-2.1" && r.model_variant!="qwen-image-2.1-turbo" &&
        r.execution=="gpu_ane" && r.hybrid_mlp_mode=="runtime" && r.allow_approximation &&
        r.residency=="resident" && r.width==512 && r.height==512 && r.steps>=20 && r.steps<=40 &&
        r.frames==1 && !r.audio && r.loras.size()==1 && r.loras[0].role=="transformer" &&
        std::isfinite(r.loras[0].strength) && r.loras[0].strength>=-8.f && r.loras[0].strength<=8.f &&
        !viggle_v021_adapter(std::filesystem::path(r.loras[0].path).filename().string()) &&
        r.lora_strategy=="inference_time" && !r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
        !r.prompt_enhance && !r.prompt_enhance_edit_experimental && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        r.qwen21_gpu_full_ffn_blocks.empty() && r.qwen21_dit_cache=="off" &&
        !r.memory_constrained.enabled && !r.memory_budget_bytes && !r.streaming.active() &&
        r.qwen21_reference_size==1024 &&
        ((r.operation=="image.generate" && r.inputs.empty()) ||
         (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=3)),
        "base-schedule runtime LoRA requires explicit resident512 BF16 runtime, one ordinary transformer adapter,20...40 steps, full references and no other backend/cache/memory recipe");
    return true;
}
inline bool configured_base_schedule_runtime_lora_request(const Request &r) {
    return base_schedule_runtime_lora_request(r,std::getenv("TURBOCIDER_QWEN21_RUNTIME_BASE_LORA"));
}
inline bool student_lora_schedule(const Request &r,bool frozen_student,bool ordinary_runtime) {
    return !r.loras.empty() &&
        (viggle_v021_adapter(std::filesystem::path(r.loras[0].path).filename().string()) || frozen_student ||
         (r.hybrid_mlp_mode=="runtime" && !ordinary_runtime));
}
// CPU processor reuse only: the ordinary encoder's tokens, GPU graph,
// precision and model weights do not change. Independent of fused prefill.
inline bool configured_tokenizer_reuse_request(const Request &r) {
    const char *raw = std::getenv("TURBOCIDER_QWEN21_TOKENIZER_REUSE");
    require(binary_option_or_unset(raw), "Qwen21 tokenizer reuse requires 0 or 1");
    if (!option_enabled(raw)) return false;
    require(r.model == "qwen-image-2.1" && r.execution == "gpu" && r.residency == "resident" &&
            r.width == 512 && r.height == 512 && r.hybrid_mlp_mode == "auto" &&
            r.qwen21_denoiser_backend == "mlx" && !r.prompt_enhance && !r.prompt_enhance_edit_experimental &&
            r.encoder_ane_manifest.empty() && r.ane_manifest.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
            !r.memory_constrained.enabled && !r.memory_budget_bytes && !r.streaming.active() &&
            ((r.operation == "image.generate" && r.inputs.empty()) ||
             (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3 &&
              (r.qwen21_reference_size == 512 || r.qwen21_reference_size == 1024))),
            "Qwen21 tokenizer reuse requires resident512 ordinary GPU/MLX generation or 1...3 ref512/ref1024 edits without PE, ANE, streaming or memory constraints");
    for (const char *name : {"TURBOCIDER_QWEN21_BF16_STREAMING", "TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
                             "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU"}) {
        const char *value = std::getenv(name);
        require(!value || std::string_view(value) == "0",
                std::string("Qwen21 standalone tokenizer reuse conflicts with encoder pilot ") + name);
    }
    const char *backend = std::getenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");
    require(!backend || std::string_view(backend) == "mlx", "Qwen21 tokenizer reuse requires the ordinary MLX encoder");
    for (const char *name : {"TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST", "TURBOCIDER_QWEN21_ENCODER_CALIBRATION_OUT"}) {
        const char *value = std::getenv(name);
        require(!value || !*value, "Qwen21 tokenizer reuse excludes encoder runtime/calibration pilots");
    }
    return true;
}
inline bool ordinary_gpu_vae_decode_request(const Request &r) {
    return r.model == "qwen-image-2.1" && r.execution == "gpu" && r.residency == "resident" &&
            r.allow_approximation && r.width == 512 && r.height == 512 &&
            r.hybrid_mlp_mode == "auto" && r.qwen21_denoiser_backend == "mlx" &&
            !r.prompt_enhance && !r.prompt_enhance_edit_experimental && !r.streaming.active() &&
            !r.memory_constrained.enabled && !r.memory_budget_bytes && r.ane_manifest.empty() &&
            r.encoder_ane_manifest.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
            ((r.operation == "image.generate" && r.inputs.empty()) ||
             (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3));
}
// Author-supported optional five-node sampler, not equivalent six-step work.
// Keep the historical six-step admission/observer predicates unchanged. The
// filename is metadata admission only; Session still verifies the pinned SHA.
inline bool student_five_step_request(const Request &r) {
    if (!ordinary_gpu_vae_decode_request(r) || r.steps != 5 || r.frames != 1 || r.audio ||
        r.loras.size() != 1 || r.lora_strategy != "inference_time" ||
        r.qwen21_dit_cache != "off" || !r.qwen21_gpu_full_ffn_blocks.empty() ||
        r.qwen21_reference_size != (r.operation == "image.edit" ? 512 : 1024)) return false;
    const auto &lora = r.loras.front();
    const auto *adapter = viggle_v021_adapter(std::filesystem::path(lora.path).filename().string());
    return adapter && adapter->rank == "r128" && lora.role == "transformer" && lora.strength == 1.f;
}
inline bool configured_student_five_step_request(const Request &r) {
    const char *raw = std::getenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP");
    require(binary_option_or_unset(raw), "Qwen21 student five-step sampler requires 0 or 1");
    if (!option_enabled(raw)) return false;
    require(student_five_step_request(r),
            "Qwen21 student five-step sampler requires explicit approximate resident512 pinned r128 ordinary GPU/MLX generation or 1...3 ref512 edits, strength1/inference_time and steps5");
    // Original/joint arithmetic stays valid. Corrected direct QKV has its
    // own bounded admission below; other approximation pilots remain excluded.
    for (const char *name : {
            "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC", "TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
            "TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE", "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
            "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC", "TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC", "TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_VIGGLE_LORA_FP16", "TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS",
            "TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE", "TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH",
            "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC", "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE",
            "TURBOCIDER_QWEN21_METAL_QK_ROPE", "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION",
            "TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_BF16_STREAMING", "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU",
            "TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
            "TURBOCIDER_QWEN21_VAE_FP16_CONV", "TURBOCIDER_QWEN21_VAE_FUSED_NORM"}) {
        const char *value = std::getenv(name);
        require(!value || std::string_view(value) == "0",
                std::string("Qwen21 student five-step sampler conflicts with experimental option ") + name);
    }
    const char *encoder = std::getenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");
    require(!encoder || std::string_view(encoder) == "mlx",
            "Qwen21 student five-step sampler requires the ordinary MLX encoder");
    for (const char *name : {"TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST",
                             "TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE"}) {
        const char *value = std::getenv(name);
        require(!value || !*value, "Qwen21 student five-step sampler excludes encoder ANE pilots");
    }
    return true;
}
inline bool configured_fp16_vae_decode_request(const Request &r) {
    const char *raw = std::getenv("TURBOCIDER_QWEN21_VAE_FP16_CONV");
    require(binary_option_or_unset(raw), "Qwen21 VAE FP16 convolution requires 0 or 1");
    if (!option_enabled(raw)) return false;
    require(ordinary_gpu_vae_decode_request(r),
            "Qwen21 VAE FP16 convolution requires explicit approximate resident512 ordinary GPU generation/editing without PE, streaming or memory constraints");
    return true;
}
inline bool configured_fused_vae_norm_request(const Request &r) {
    const char *raw = std::getenv("TURBOCIDER_QWEN21_VAE_FUSED_NORM");
    require(binary_option_or_unset(raw), "Qwen21 VAE fused norm requires 0 or 1");
    if (!option_enabled(raw)) return false;
    require(ordinary_gpu_vae_decode_request(r),
            "Qwen21 VAE fused norm requires explicit approximate resident512 ordinary GPU generation/editing without PE, streaming or memory constraints");
    return true;
}
inline bool gguf_fast_defaults_enabled() {
    const auto *flag=std::getenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS");
    require(binary_option_or_unset(flag),"Qwen GGUF fast defaults require0 or1");
    return !flag || option_enabled(flag);
}
inline bool option_or_default(const char *value,bool fallback) {
    require(binary_option_or_unset(value),"Qwen optimization override requires0 or1");
    return value ? option_enabled(value) : fallback;
}
inline bool gguf_fast_encoder_request(const Request &r,bool gguf_encoder,bool gguf_dit) {
    return gguf_encoder && gguf_dit && r.model=="qwen-image-2.1" && r.allow_approximation &&
        r.operation=="image.generate" && r.inputs.empty() && r.loras.empty() &&
        r.width==512 && r.height==512 && r.residency=="resident" && !r.prompt_enhance &&
        !r.streaming.active() && !r.memory_constrained.enabled && !r.memory_budget_bytes &&
        r.encoder_ane_manifest.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        (r.execution=="gpu" || (r.execution=="gpu_ane" && r.hybrid_mlp_mode=="runtime"));
}
inline bool compiled_encoder_gpu(const Request &r,bool default_prefill=false) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU");
    const char *prefill=std::getenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU");
    require(binary_option_or_unset(prefill),"Qwen fused GPU encoder prefill requires 0 or 1");
    require(binary_option_or_unset(raw),"Qwen compiled encoder GPU requires 0 or1");
    const bool use_prefill=option_or_default(prefill,default_prefill);
    const bool enabled=option_enabled(raw) || use_prefill;
    if(use_prefill)require(r.encoder_ane_manifest.empty(),"fused encoder prefill is GPU-only");
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
inline bool frozen_fp16(const Request &request) {
    return request.execution == "gpu_ane" && fused_lora_ane(request) && !request.qwen21_w8a8;
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

// Bounded original-source CPU import scratch; zero rejects malformed options.
inline int gguf_read_buffer_mib(const char *value) {
    if (!value || std::string_view(value) == "1") return 1;
    if (std::string_view(value) == "4") return 4;
    if (std::string_view(value) == "8") return 8;
    return 0;
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

// Explicit composition candidate, not automatic device selection. Keep the
// ordinary GPU presets unchanged. Only the measured base40 resident recipe
// may combine the conservative residual cache with frozen W8A8. The loader
// independently verifies BF16 source identity and the 6144-channel graph.
// Full-reference editing still needs its existing separate diagnostic gate.
inline bool frozen_conservative_cache_request(const Request &r) {
    return r.model == "qwen-image-2.1" && r.qwen21_dit_cache == "conservative" &&
        r.execution == "gpu_ane" && r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        r.qwen21_gpu_full_ffn_blocks.empty() && r.hybrid_mlp_mode == "auto" &&
        r.qwen21_denoiser_backend == "mlx" && r.allow_approximation &&
        !r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
        r.residency == "resident" && r.width == 512 && r.height == 512 && r.steps == 40 &&
        r.loras.empty() && !r.prompt_enhance && r.qwen21_reference_size == 1024 &&
        !r.memory_constrained.enabled && !r.memory_budget_bytes && !r.streaming.active() &&
        ((r.operation == "image.generate" && r.inputs.empty()) ||
         (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3));
}

// Only immutable conditioning K/V are shared. Frozen decode can use this
// bank because its first step remains ordinary BF16 GPU; the loader verifies
// its source/partition independently. Temporal residuals are request-local.
inline bool db_prefix_snapshot_request(const Request &r, const DbCacheOptions &options,
                                       bool explicit_snapshot) {
    const bool gpu = r.execution == "gpu" && r.ane_manifest.empty() &&
        !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty();
    return explicit_snapshot && options.enabled && !options.diagnostic &&
        options.mode == "conservative" && r.qwen21_dit_cache == "conservative" &&
        r.model == "qwen-image-2.1" && (gpu || frozen_conservative_cache_request(r)) &&
        r.residency == "resident" && r.allow_approximation &&
        r.width == 512 && r.height == 512 && r.steps >= 20 && r.steps <= 40 &&
        r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3 &&
        r.qwen21_reference_size == 1024 && r.loras.empty() && !r.prompt_enhance &&
        r.encoder_ane_manifest.empty() && r.hybrid_mlp_mode == "auto" &&
        r.qwen21_denoiser_backend == "mlx" && !r.memory_constrained.enabled &&
        !r.memory_budget_bytes && !r.streaming.active();
}

// Source admission only; Session verifies the released adapter's pinned SHA
// before binding. This opt-in shares conditioning K/V, never student target
// outputs or temporal FFN state. Ref512 keeps its independent request gate.
inline bool pinned_student_gpu_request(const Request &r) {
    if (r.model != "qwen-image-2.1" || r.execution != "gpu" ||
        r.residency != "resident" || !r.allow_approximation ||
        r.width != 512 || r.height != 512 || r.steps != 6 ||
        r.loras.size() != 1 ||
        r.lora_strategy != "inference_time" || r.qwen21_dit_cache != "off" ||
        r.prompt_enhance || r.prompt_enhance_edit_experimental ||
        r.hybrid_mlp_mode != "auto" || r.qwen21_denoiser_backend != "mlx" ||
        r.qwen21_w8a8 || r.qwen21_gpu_w8a16 || !r.qwen21_gpu_full_ffn_blocks.empty() ||
        !r.ane_manifest.empty() || !r.encoder_ane_manifest.empty() ||
        r.memory_constrained.enabled || r.memory_budget_bytes || r.streaming.active()) return false;
    const auto &lora = r.loras.front();
    return viggle_v021_adapter(std::filesystem::path(lora.path).filename().string()) &&
        lora.role == "transformer" && lora.strength == 1.f;
}

inline bool student_prefix_snapshot_request(const Request &r, bool explicit_snapshot) {
    return explicit_snapshot &&
        (pinned_student_gpu_request(r) || configured_student_five_step_request(r)) && r.operation == "image.edit" &&
        !r.inputs.empty() && r.inputs.size() <= 3 &&
        (r.qwen21_reference_size == 512 || r.qwen21_reference_size == 1024);
}

// An installed C++ observer, not a request-selectable acceleration route.
// Keep the additional real-LoRA replay scope bounded to the pinned r128 GPU
// recipe and one ref512 edit. Session still verifies content before binding.
// Source capture only, not an approximate model route: the diagnostic aborts
// before image decoding. Keep full base40 geometry independent of students.
inline bool base_block_observer_request(const Request &r) {
    return r.model=="qwen-image-2.1" && r.residency=="resident" && r.execution=="gpu" &&
        r.width==512 && r.height==512 && r.steps==40 && r.frames==1 && !r.audio &&
        r.loras.empty() && !r.allow_approximation && !r.prompt_enhance &&
        r.hybrid_mlp_mode=="auto" && r.qwen21_denoiser_backend=="mlx" &&
        r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
        !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && r.qwen21_dit_cache=="off" &&
        r.qwen21_gpu_full_ffn_blocks.empty() && !r.memory_constrained.enabled &&
        !r.memory_budget_bytes && !r.streaming.active() && r.dump.empty() && r.noise_path.empty() &&
        ((r.operation=="image.generate" && r.inputs.empty()) ||
         (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=2 && r.qwen21_reference_size==1024));
}
inline bool student_block_observer_request(const Request &r) {
    if (!pinned_student_gpu_request(r) || !r.dump.empty() || !r.noise_path.empty()) return false;
    const auto *adapter = viggle_v021_adapter(std::filesystem::path(r.loras[0].path).filename().string());
    return adapter && adapter->rank == "r128" &&
        ((r.operation == "image.generate" && r.inputs.empty() && r.qwen21_reference_size == 1024) ||
         (r.operation == "image.edit" && r.inputs.size() == 1 && r.qwen21_reference_size == 512));
}

inline bool configured_corrected_qkv_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_CORRECTED_QKV");
    const char *direct=std::getenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT");
    const char *prefix=std::getenv("TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX");
    require(binary_option_or_unset(raw),"Qwen corrected QKV requires 0 or 1");
    require(binary_option_or_unset(direct),"Qwen direct corrected QKV requires 0 or 1");
    require(binary_option_or_unset(prefix),"Qwen direct QKV prefix composition requires 0 or 1");
    if(!option_enabled(raw)){require(!option_enabled(direct)&&!option_enabled(prefix),"direct QKV/prefix requires the corrected-QKV opt-in");return false;}
    const char *five_flag=std::getenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP");
    require(binary_option_or_unset(five_flag),"Qwen student five-step sampler requires 0 or 1");
    const bool five=r.steps==5&&option_enabled(five_flag);
    require(!option_enabled(five_flag)||five,"five-step flag requires an actual five-step request");
    Request admission=r;if(five)admission.steps=6; // only reuse unchanged six-step shape/source checks
    require(pinned_student_gpu_request(admission) && (!five || (option_enabled(direct)&&student_five_step_request(r))) && r.frames==1 && !r.audio &&
        viggle_v021_adapter(std::filesystem::path(r.loras[0].path).filename().string())->rank=="r128" &&
        ((r.operation=="image.generate" && r.inputs.empty() && r.qwen21_reference_size==1024) ||
         (r.operation=="image.edit" && r.inputs.size()==1 && r.qwen21_reference_size==512)) &&
        joint_bf16_lora_ab(r),
        "corrected QKV requires approximate resident512 pinned r128 strength1/inference-time six-step GPU generation/single-ref512 edit with joint BF16 A/B, or explicit authored five-step composition with direct QKV");
    const char *snapshot=std::getenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT");
    require(binary_option_or_unset(snapshot),"Qwen prefix snapshot requires 0 or 1");
    require(!option_enabled(prefix) || (option_enabled(direct)&&option_enabled(snapshot)&&r.operation=="image.edit"&&r.inputs.size()==1&&r.dump.empty()),
        "direct QKV prefix composition requires explicit direct QKV and prefix snapshot with a single ref512 edit");
    require(!option_enabled(snapshot)||option_enabled(prefix),"corrected QKV prefix snapshot needs its independent composition opt-in");
    for(const char *name:{"TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC","TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC","TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE",
        "TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION","TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
        "TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH","TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
        "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN","TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
        "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC","TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
        "TURBOCIDER_QWEN21_BF16_STREAMING","TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU","TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
        "TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS","TURBOCIDER_QWEN21_TOKENIZER_REUSE","TURBOCIDER_QWEN21_VAE_FP16_CONV",
        "TURBOCIDER_QWEN21_VAE_FUSED_NORM","TURBOCIDER_QWEN21_VIGGLE_LORA_FP16",
        "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS","TURBOCIDER_QWEN21_PROFILE_GPU_OPS","TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS"}) {
        const char *value=std::getenv(name);
        require(!value || std::string_view(value)=="0",std::string("corrected QKV conflicts with ")+name);
    }
    return true;
}

inline void require_bounded_joint_student_decode(const Request &r) {
    const bool five=configured_student_five_step_request(r);
    Request admission=r;if(five)admission.steps=6;
    require(pinned_student_gpu_request(admission)&&r.frames==1&&!r.audio&&joint_bf16_lora_ab(r)&&
        viggle_v021_adapter(std::filesystem::path(r.loras[0].path).filename().string())->rank=="r128"&&
        ((r.operation=="image.generate"&&r.inputs.empty()&&r.qwen21_reference_size==1024)||
         (r.operation=="image.edit"&&r.inputs.size()==1&&r.qwen21_reference_size==512)),
        "typed LoRA SwiGLU requires approximate resident512 pinned r128 strength1/inference-time joint GPU generation/single-ref512 editing with default six or explicit authored five steps");
    const bool qkv=configured_corrected_qkv_request(r);
    require(!qkv||option_enabled(std::getenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT")),"typed LoRA SwiGLU excludes packed QKV storage");
    const char *snapshot=std::getenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT");
    require(binary_option_or_unset(snapshot),"Qwen prefix snapshot requires 0 or 1");
    require(!option_enabled(snapshot)||(r.operation=="image.edit"&&r.inputs.size()==1&&r.dump.empty()),
        "typed LoRA SwiGLU prefix composition requires a complete single-ref512 edit without dumps");
    const char *encoder=std::getenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");
    require(!encoder||std::string_view(encoder)=="mlx","typed LoRA SwiGLU requires the ordinary MLX encoder");
    for(const char *name:{"TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST","TURBOCIDER_QWEN21_NATIVE_ENCODER_ANE_CACHE","TURBOCIDER_QWEN21_ENCODER_CALIBRATION_OUT"}){
        const char *value=std::getenv(name);require(!value||!*value,"typed LoRA SwiGLU excludes encoder runtime/calibration pilots");}
    for(const char *name:{"TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC","TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC","TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE",
        "TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION","TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
        "TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH","TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
        "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN","TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
        "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC","TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
        "TURBOCIDER_QWEN21_BF16_STREAMING","TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU","TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU",
        "TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS","TURBOCIDER_QWEN21_TOKENIZER_REUSE","TURBOCIDER_QWEN21_VAE_FP16_CONV",
        "TURBOCIDER_QWEN21_VAE_FUSED_NORM","TURBOCIDER_QWEN21_VIGGLE_LORA_FP16",
        "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS","TURBOCIDER_QWEN21_PROFILE_GPU_OPS","TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS"}) {
        const char *value=std::getenv(name);require(!value||std::string_view(value)=="0",std::string("typed LoRA SwiGLU conflicts with ")+name);
    }
}

inline bool configured_lora_swiglu_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_LORA_SWIGLU");
    require(binary_option_or_unset(raw),"Qwen typed LoRA SwiGLU requires 0 or 1");
    if(!option_enabled(raw))return false;
    require_bounded_joint_student_decode(r);return true;
}

inline bool configured_async_step_pair_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_ASYNC_STEP_PAIR");
    require(binary_option_or_unset(raw),"Qwen bounded async step pair requires 0 or 1");
    if(!option_enabled(raw))return false;
    require_bounded_joint_student_decode(r);
    (void)configured_lora_swiglu_request(r);
    for(const char *name:{"TURBOCIDER_QWEN21_PROFILE_STEPS","TURBOCIDER_ANE_CALIBRATION_REPLAY","TURBOCIDER_ANE_CALIBRATION_CAPTURE"}){
        const char *value=std::getenv(name);require(!value||std::string_view(value)=="0",std::string("bounded async steps exclude step profiling/calibration: ")+name);
    }
    const char *capture=std::getenv("TURBOCIDER_ANE_CALIBRATION_DIR");
    require(!capture||!*capture,"bounded async steps exclude calibration output directories");
    return true;
}

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

inline bool configured_student_graph_request(const Request &) {
    const char *raw = std::getenv("TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH");
    require(binary_option_or_unset(raw), "student graph retention requires 0 or 1");
    require(!option_enabled(raw), "student graph retention retired: no measured net benefit; use the archived research binary only");
    return false;
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
