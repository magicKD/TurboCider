#pragma once

#include "../../backends/ane_backend.hpp"

namespace tc::z_image {
// Shared plan/runtime gates. Valid flags on ordinary GPU controls are inert;
// only an explicitly authorized resident512 Private ConvRot F32 channel route
// can snapshot the candidate into its current source owner.
inline bool configured_convrot_partial_mpp(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_Z_CONVROT_FP32_MPP");
    require(!raw || std::string_view(raw)=="0" || std::string_view(raw)=="1",
            "ConvRot MPP partial requires 0 or 1");
    if(!raw || std::string_view(raw)!="1" || r.execution!="gpu_ane" || r.hybrid_mlp_mode!="runtime")return false;
    const auto *convrot=std::getenv("TURBOCIDER_Z_RUNTIME_CONVROT");
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *fp32=std::getenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");
    require(r.allow_approximation && r.residency=="resident" && r.width==512 && r.height==512 &&
        r.loras.empty() && !r.memory_constrained.enabled && !r.streaming.active() &&
        r.encoder_ane_manifest.empty() && convrot && std::string_view(convrot)=="1" &&
        path && std::string_view(path)=="convrot_w8a8" && fp32 && std::string_view(fp32)=="1" &&
        ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(10240)>0,
        "ConvRot MPP partial requires explicit resident512 base Private fixed ConvRot W8A8/F32 channel join");
    return true;
}
} // namespace tc::z_image
