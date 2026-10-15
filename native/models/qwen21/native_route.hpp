#pragma once

#include "../../core/contracts.hpp"
#include "viggle_adapter.hpp"
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace tc::qwen21 {

inline bool native_denoiser_requested(const Request& request) noexcept {
    return request.qwen21_denoiser_backend == "native-metal-graph";
}
inline bool native_w1_requested(const Request& request) noexcept {
    return native_denoiser_requested(request) && request.execution == "gpu_ane";
}
// Metadata admission only: the known filename selects a candidate whose
// content MUST still match the r128 SHA-256 pin from the same held descriptor
// used for loading. No path lookup/hash/model read occurs here. Generated
// numerical coverage does not grant real-model quality/performance status;
// this route remains experimental-unqualified.
// Raw "auto" uses this model's existing inference_time default; make_plan
// normalizes it before execution. Neither merged strategy is admitted.
inline bool native_viggle_lora_requested(const Request& r) {
    if (!native_denoiser_requested(r) || r.execution!="gpu" ||
        r.model!="qwen-image-2.1" || r.operation!="image.generate" || !r.inputs.empty() ||
        r.width!=512 || r.height!=512 || r.frames!=1 || r.steps!=6 ||
        r.loras.size()!=1 || (r.lora_strategy!="inference_time" && r.lora_strategy!="auto")) return false;
    const auto& adapter=r.loras.front();
    if (adapter.role!="transformer" || adapter.strength!=1.f || adapter.path.empty() ||
        adapter.path.size()>4096 || adapter.path.find('\0')!=std::string::npos ||
        adapter.path.find("://")!=std::string::npos) return false;
    const auto* pin=viggle_v021_adapter(std::filesystem::path(adapter.path).filename().string());
    return pin && pin->rank=="r128";
}

// This is an explicit executor choice, not another interpretation of a
// device/precision/cache flag. Expand the request domain only with matching
// numerical, source, residency and lifecycle evidence.
inline void validate_native_denoiser_request(const Request& r) {
    const auto check=[](bool valid,const char* error) {
        if (!valid) throw std::invalid_argument(error);
    };
    check(r.qwen21_denoiser_backend=="mlx" || native_denoiser_requested(r),
          "qwen21_denoiser_backend must be mlx or native-metal-graph");
    check(r.qwen21_native_w8_staging=="generic" || r.qwen21_native_w8_staging=="register",
          "qwen21_native_w8_staging must be generic or register");
    check(r.qwen21_native_lora_kernel=="scalar" || r.qwen21_native_lora_kernel=="sgmma",
          "qwen21_native_lora_kernel must be scalar or sgmma");
    if (!native_viggle_lora_requested(r))
        check(!r.qwen21_native_lora_kernel_specified && r.qwen21_native_lora_kernel=="scalar",
              "qwen21_native_lora_kernel is legal only for native GPU Viggle r128");
    if (!native_w1_requested(r)) {
        check(!r.qwen21_native_scale_cache_specified && !r.qwen21_native_scale_cache,
              "qwen21_native_scale_cache is legal only for native-metal-graph private W1");
        check(!r.qwen21_native_w8_staging_specified && r.qwen21_native_w8_staging=="generic",
              "qwen21_native_w8_staging is legal only for native-metal-graph private W1");
    }
    if (!native_denoiser_requested(r)) {
        check(r.qwen21_native_ane_cache.empty(), "qwen21_native_ane_cache requires native-metal-graph W1");
        return;
    }
    const bool viggle=native_viggle_lora_requested(r);
    check(r.model=="qwen-image-2.1" && r.operation=="image.generate" && r.inputs.empty() &&
          r.width==512 && r.height==512 && r.frames==1 &&
          ((r.loras.empty() && r.steps==20) || viggle),
          "native-metal-graph requires 512px BaseOnly/20-step generation or explicit GPU Viggle r128/strength1/inference_time/6-step generation");
    check((r.execution=="gpu" || native_w1_requested(r)) && r.allow_approximation &&
          (r.loras.empty() || viggle) &&
          r.hybrid_mlp_mode=="auto" && r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
          !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty(),
          "native-metal-graph requires explicit experimental base GPU/W1 or candidate GPU Viggle execution without legacy manifests or runtime modes");
    if (native_w1_requested(r)) {
        check(r.qwen21_w8a8 && !r.qwen21_native_ane_cache.empty() && r.qwen21_native_ane_cache.size()<=4096 &&
              r.qwen21_native_ane_cache.find('\0')==std::string::npos,
              "native W1 requires qwen21_w8a8=true and a bounded explicit qwen21_native_ane_cache");
        const std::filesystem::path cache(r.qwen21_native_ane_cache);
        check(cache.is_absolute() && cache!=cache.root_path(), "native W1 cache must be an absolute non-root directory");
        for (const auto& component:cache)
            check(component!=".." && component!=".", "native W1 cache cannot contain dot traversal components");
#if !defined(TURBOCIDER_ENABLE_PRIVATE_ANE) || !TURBOCIDER_ENABLE_PRIVATE_ANE
        check(false, "native W1 requires a private-ANE-enabled build; public builds do not contain this executor");
#endif
    } else {
        check(!r.qwen21_w8a8 && r.qwen21_native_ane_cache.empty(),
              "native GPU requires qwen21_w8a8=false and no native ANE cache");
    }
    check((r.residency=="component_staged" || r.residency=="resident") &&
          !r.memory_constrained.enabled && !r.streaming.specified() && !r.streaming_selector &&
          !r.quantized_execution.specified() && !r.streaming_offload && r.quantized_cache.empty(),
          "native-metal-graph uses dense full-DiT residency and has no full-request constrained/streaming certification");
    check(!r.prompt_enhance && !r.prompt_enhance_edit_experimental && r.prompt_enhancer_path.empty() &&
          !r.compile_gpu && !r.vsa && r.qwen21_dit_cache=="off",
          "native-metal-graph does not silently ignore MLX compile, PE or approximation caches");
    // Existing process flags describe another executor's recipe. Reject a
    // mixed request rather than silently running a different native recipe.
    for (const auto* key:{"TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE",
         "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION","TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV",
         "TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
         "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
         "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC",
         "TURBOCIDER_QWEN21_VIGGLE_LORA_FP16",
         "TURBOCIDER_QWEN21_LORA_BF16_AB","TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS",
         "TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE","TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
         "TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS","TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS",
         "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS","TURBOCIDER_QWEN21_PROFILE_GPU_OPS",
         "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS"}) {
        const char* value=std::getenv(key);
        if (value && *value && std::string_view(value)!="0")
            throw std::invalid_argument(std::string("native-metal-graph conflicts with executor diagnostic: ")+key);
    }
    for (const auto* key:{"TURBOCIDER_ANE_CALIBRATION_DIR","TURBOCIDER_ANE_CALIBRATION_REPLAY"}) {
        const char* value=std::getenv(key);
        check(!value || !*value,"native-metal-graph calibration capture is not wired to this executor");
    }
}

} // namespace tc::qwen21
