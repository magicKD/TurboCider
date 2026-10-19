#pragma once
#include "diagnostic_options.hpp"

namespace tc::qwen21 {
// Separate approximate tier. Off requests and the existing full-work route
// remain unchanged; this opt-in never changes the actual sampler schedule.
inline bool runtime_conservative_cache_domain(const Request &r) {
    return r.model=="qwen-image-2.1" && (r.model_variant=="auto" || r.model_variant=="qwen-image-2.1") &&
        r.qwen21_dit_cache=="conservative" && r.execution=="gpu_ane" && r.hybrid_mlp_mode=="runtime" &&
        r.qwen21_denoiser_backend=="mlx" && r.allow_approximation && r.loras.empty() &&
        r.width==512 && r.height==512 && r.steps==40 && r.frames==1 && !r.audio && r.residency=="resident" &&
        !r.ane_manifest.empty() && r.encoder_ane_manifest.empty() && !r.memory_budget_bytes &&
        !r.memory_constrained.enabled && !r.streaming.active() && !r.prompt_enhance &&
        r.dump.empty() && r.noise_path.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        r.qwen21_gpu_full_ffn_blocks.empty() && r.qwen21_reference_size==1024 &&
        ((r.operation=="image.generate" && r.inputs.empty()) ||
         (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=2));
}
inline bool configured_runtime_conservative_cache_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE");
    if(!binary_option_or_unset(raw))throw std::invalid_argument("runtime conservative cache requires0 or1");
    if(!option_enabled(raw) || r.qwen21_dit_cache=="off")return false;
    if(!runtime_conservative_cache_domain(r))throw std::invalid_argument("runtime conservative cache requires explicit original-base40 resident512 joint runtime without adapters/other encodings");
    for(const auto &[name,want]:{std::pair{"TURBOCIDER_ANE_BACKEND","private"},
        std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV","1"},std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH","1"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1"},std::pair{"TURBOCIDER_PRIVATE_ANE_CHANNELS","6144"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_W8_PROJECTION","descriptor"},
        std::pair{"TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1"},std::pair{"TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1"},
        std::pair{"TURBOCIDER_RUNTIME_ANE_CHUNKS","1"}}) {
        const auto value=std::getenv(name);
        if(!value || std::string_view(value)!=want)throw std::invalid_argument(std::string("runtime cache requires fixed option ")+name+"="+want);
    }
    for(const char *name:{"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","TURBOCIDER_QWEN21_COMPILED_COMPLEMENT", "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC"}) {
        const auto value=std::getenv(name);
        if(value && std::string_view(value)!="0")throw std::invalid_argument(std::string("runtime cache excludes ")+name);
    }
    if(std::getenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS"))throw std::invalid_argument("runtime cache does not combine with function families");
    const auto edit=std::getenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0");
    if(edit && std::string_view(edit)!="ane")throw std::invalid_argument("runtime cache keeps the original edit first-step QKV policy");
    return true;
}
inline int runtime_conservative_saved_blocks(int cached_steps) {
    // Eligible steps8...38, at most one consecutive reuse, final step full.
    if(cached_steps<0 || cached_steps>16)throw std::invalid_argument("invalid conservative runtime cache step coverage");
    return 24*cached_steps;
}
}
