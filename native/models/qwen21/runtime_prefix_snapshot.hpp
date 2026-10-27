#pragma once
#include "diagnostic_options.hpp"
#include "../../runtime/session.hpp"
#include "runtime_function_options.hpp"
#include "runtime_cache_options.hpp"

namespace tc::qwen21 {
// Explicit base40 / authored Turbo8 pilot. Shape/backend health is checked AFTER the
// actual executor is selected, and after its submitted work is drained.
inline bool runtime_prefix_snapshot_base_request(const Request &r) {
    const bool sampler=(r.steps==40 && (r.model_variant=="auto" || r.model_variant=="qwen-image-2.1")) ||
        (r.steps==8 && r.model_variant=="qwen-image-2.1-turbo");
    return r.model == "qwen-image-2.1" &&
        r.execution == "gpu_ane" && r.hybrid_mlp_mode == "runtime" &&
        r.qwen21_denoiser_backend == "mlx" &&
        r.allow_approximation && r.loras.empty() && !r.ane_manifest.empty() &&
        r.encoder_ane_manifest.empty() && r.width == 512 && r.height == 512 &&
        sampler &&
        r.qwen21_reference_size == 1024 && r.residency == "resident" &&
        !r.memory_budget_bytes && !r.memory_constrained.enabled && !r.streaming.active() &&
        !r.prompt_enhance && r.dump.empty() && !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        (r.qwen21_dit_cache == "off" || configured_runtime_conservative_cache_request(r));
}
inline bool runtime_prefix_snapshot_request(const Request &r) {
    return runtime_prefix_snapshot_base_request(r) && r.operation=="image.edit" &&
        !r.inputs.empty() && r.inputs.size()<=2;
}
inline bool runtime_text_prefix_snapshot_request(const Request &r) {
    return runtime_prefix_snapshot_base_request(r) && r.operation=="image.generate" && r.inputs.empty();
}
inline bool configured_runtime_text_prefix_snapshot_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT");
    if(!binary_option_or_unset(raw))throw std::invalid_argument("Qwen runtime text prefix snapshot requires0 or1");
    // A process may also edit: this independent flag never enables edit reuse.
    if(!option_enabled(raw) || r.operation!="image.generate")return false;
    if(!runtime_text_prefix_snapshot_request(r))throw std::invalid_argument(
        "runtime text prefix snapshot requires explicit resident512 base40 or authored Turbo8 runtime generation without references or other encodings");
    return true;
}
enum class RuntimeTextQkvStep0 { Ane, Gpu };
inline RuntimeTextQkvStep0 configured_runtime_text_qkv_step0(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0");
    if(raw && std::string_view(raw)!="ane" && std::string_view(raw)!="gpu")
        throw std::invalid_argument("runtime text-prefix first QKV requires ane or gpu");
    if(!raw || std::string_view(raw)=="ane" || r.operation!="image.generate")return RuntimeTextQkvStep0::Ane;
    if(!configured_runtime_text_prefix_snapshot_request(r))
        throw std::invalid_argument("GPU first-step QKV policy requires explicit runtime text-prefix generation");
    return RuntimeTextQkvStep0::Gpu;
}
// Separate default-off edit experiment; do not silently reinterpret the
// existing generation-only flag or change the authored Turbo/student domain.
inline RuntimeTextQkvStep0 configured_runtime_edit_qkv_step0(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0");
    if(raw && std::string_view(raw)!="ane" && std::string_view(raw)!="gpu")
        throw std::invalid_argument("runtime edit-prefix first QKV requires ane or gpu");
    if(!raw || std::string_view(raw)=="ane" || r.operation!="image.edit")return RuntimeTextQkvStep0::Ane;
    if(!runtime_prefix_snapshot_request(r) || r.steps!=40 || r.frames!=1 || r.audio ||
       !option_enabled(std::getenv("TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT")))
        throw std::invalid_argument("edit GPU first-step QKV requires explicit base40 resident512 runtime edit prefix with1...2 full references");
    return RuntimeTextQkvStep0::Gpu;
}
inline std::string_view runtime_edit_qkv_recipe(RuntimeTextQkvStep0 policy) {
    return policy==RuntimeTextQkvStep0::Gpu?"base-private-w8-edit-prefix-gpu-qkv-step0-v1:":"base-private-w8-prefix-v1:";
}
inline bool runtime_text_qkv_permitted(RuntimeTextQkvStep0 policy,bool text_prefix_hit,int step) {
    if(step<0)throw std::invalid_argument("runtime text-prefix sampler step cannot be negative");
    return !(policy==RuntimeTextQkvStep0::Gpu && text_prefix_hit && step==0);
}

inline bool runtime_prefix_snapshot_healthy(const HybridMetrics &m) {
    const auto&functions=m.runtime_weight_function_rows;
    const bool family_healthy=functions.empty() || (functions.size()>=2 && functions.size()<=4 &&
        m.runtime_weight_row_alignment==512 && m.bucket==functions.back() &&
        m.runtime_weight_capability_calls==3*functions.size() && [&]{
            for(size_t i=0;i<functions.size();++i)if(functions[i]!=int(i+1)*512)return false;
            return m.runtime_weight_source_recipe.find(ane::w8_function_identity(functions))!=std::string::npos;
        }());
    return family_healthy && !m.runtime_failed && (!m.runtime_weight_fixed_program_requested ||
        (m.runtime_weight_fixed_program_enabled && !m.runtime_weight_program_disabled &&
         m.runtime_weight_program_instances==1 && m.runtime_weight_blocked_rebuilds==0)) && m.runtime_weight_backend == "private_ane" &&
        m.runtime_weight_data_path == "w8a8_hadamard" &&
        m.runtime_weight_partition_axis == "intermediate_channels" &&
        (m.runtime_weight_ane_channels == 4096 || m.runtime_weight_ane_channels == 5120 ||
         m.runtime_weight_ane_channels == 6144) &&
        m.runtime_weight_gpu_channels + m.runtime_weight_ane_channels == 12288 &&
        m.runtime_weight_fp32_channel_join_enabled && m.runtime_weight_gpu_layers.empty() &&
        m.runtime_weight_fallback_blocks == 0 && m.runtime_weight_overflow_retries == 0 &&
        m.runtime_weight_headroom == 1.f && m.runtime_weight_s1_digest.empty();
}
inline bool runtime_text_prefix_snapshot_healthy(const HybridMetrics &m) {
    return runtime_prefix_snapshot_healthy(m) && m.runtime_weight_fixed_program_requested &&
        m.runtime_weight_fixed_program_enabled && m.runtime_weight_program_instances==1 &&
        m.runtime_weight_blocked_rebuilds==0 && !m.runtime_weight_program_disabled;
}
} // namespace tc::qwen21
