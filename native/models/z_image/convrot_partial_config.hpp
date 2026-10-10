#pragma once

#include "../../backends/ane_backend.hpp"

namespace tc::z_image {
inline bool configured_convrot_runtime_lora(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_Z_RUNTIME_CONVROT_LORA");
    require(!raw || std::string_view(raw)=="0" || std::string_view(raw)=="1",
            "ConvRot runtime LoRA requires 0 or 1");
    if(!raw || std::string_view(raw)!="1" || r.loras.empty() || r.execution!="gpu_ane" || r.hybrid_mlp_mode!="runtime")return false;
    const auto *convrot=std::getenv("TURBOCIDER_Z_RUNTIME_CONVROT");
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *fp32=std::getenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
    require(r.allow_approximation && r.residency=="resident" && r.width==512 && r.height==512 &&
        r.lora_strategy=="inference_time" && !r.memory_constrained.enabled && !r.streaming.active() &&
        r.encoder_ane_manifest.empty() && convrot && std::string_view(convrot)=="1" &&
        path && std::string_view(path)=="convrot_w8a8" && fp32 && std::string_view(fp32)=="1" &&
        ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(10240)>0 &&
        !ane::configured_convrot_bf16_boundaries() && !std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES"),
        "ConvRot runtime LoRA requires explicit resident512 unmerged Private fixed ConvRot W8A8/F32, original BF16 scales and hidden ABI");
    return true;
}
// Shared plan/runtime gates. Valid flags on ordinary GPU controls are inert;
// only an explicitly authorized resident512 Private ConvRot F32 channel route
// can snapshot the candidate into its current source owner.
inline bool configured_convrot_partial_mpp(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_Z_CONVROT_FP32_MPP");
    require(!raw || std::string_view(raw)=="0" || std::string_view(raw)=="1",
            "ConvRot MPP partial requires 0 or 1");
    const char *narrow=std::getenv("TURBOCIDER_Z_CONVROT_BF16_PARTIAL");
    require(!narrow || std::string_view(narrow)=="0" || std::string_view(narrow)=="1",
            "ConvRot BF16 partial requires 0 or 1");
    require(!raw || std::string_view(raw)!="1" || !narrow || std::string_view(narrow)!="1",
            "ConvRot MPP and BF16 partial recipes are mutually exclusive");
    if(!raw || std::string_view(raw)!="1" || r.execution!="gpu_ane" || r.hybrid_mlp_mode!="runtime")return false;
    const auto *convrot=std::getenv("TURBOCIDER_Z_RUNTIME_CONVROT");
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *fp32=std::getenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
    require(r.allow_approximation && r.residency=="resident" && r.width==512 && r.height==512 &&
        (r.loras.empty() || configured_convrot_runtime_lora(r)) && !r.memory_constrained.enabled && !r.streaming.active() &&
        r.encoder_ane_manifest.empty() && convrot && std::string_view(convrot)=="1" &&
        path && std::string_view(path)=="convrot_w8a8" && fp32 && std::string_view(fp32)=="1" &&
        ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(10240)>0,
        "ConvRot MPP partial requires explicit resident512 Private fixed ConvRot W8A8/F32 channel join, with separate LoRA opt-in");
    return true;
}
inline bool configured_convrot_partial_bf16(const Request &r) {
    // Validate both selections even on an inert ordinary-GPU control.
    (void)configured_convrot_partial_mpp(r);
    const char *raw=std::getenv("TURBOCIDER_Z_CONVROT_BF16_PARTIAL");
    if(!raw || std::string_view(raw)!="1" || r.execution!="gpu_ane" || r.hybrid_mlp_mode!="runtime")return false;
    const auto *convrot=std::getenv("TURBOCIDER_Z_RUNTIME_CONVROT");
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *fp32=std::getenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
    require(r.allow_approximation && r.residency=="resident" && r.width==512 && r.height==512 &&
        (r.loras.empty() || configured_convrot_runtime_lora(r)) && !r.memory_constrained.enabled && !r.streaming.active() &&
        r.encoder_ane_manifest.empty() && convrot && std::string_view(convrot)=="1" &&
        path && std::string_view(path)=="convrot_w8a8" && fp32 && std::string_view(fp32)=="1" &&
        ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(10240)>0 &&
        !std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES"),
        "ConvRot BF16 partial requires explicit resident512 Private fixed ConvRot W8A8/F32 join and original BF16-scale source, with separate LoRA opt-in");
    return true;
}
} // namespace tc::z_image
