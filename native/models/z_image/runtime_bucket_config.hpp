#pragma once
#include "../../backends/ane_backend.hpp"
#include "../../core/common.hpp"

namespace tc::z_image {
inline bool configured_runtime_match_rows(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_Z_RUNTIME_MATCH_ROWS");
    require(!raw || std::string_view(raw)=="0" || std::string_view(raw)=="1","Z runtime row matching requires0 or1");
    if(!raw || std::string_view(raw)!="1" || r.execution!="gpu_ane" || r.hybrid_mlp_mode!="runtime")return false;
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *chunks=std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
    const auto *async=std::getenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
    require((r.model=="z-image-turbo" || r.model=="z-image-turbo-gguf") && r.operation=="image.generate" &&
        r.width==512 && r.height==512 && r.residency=="resident" && r.allow_approximation &&
        !r.memory_constrained.enabled && !r.memory_budget_bytes && !r.streaming.active() && r.encoder_ane_manifest.empty() &&
        ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(10240)>0 &&
        path && (std::string_view(path)=="w8a8" || std::string_view(path)=="convrot_w8a8") &&
        chunks && std::string_view(chunks)=="1" && async && std::string_view(async)=="1",
        "Z row matching requires approximate unconstrained resident512 explicit fixed Private W8 channel/fixed-async runtime");
    return true;
}
inline int matched_runtime_bucket(int image_rows,int caption_rows,int template_rows=0) {
    require(image_rows==1024 && caption_rows>0 && caption_rows%32==0 && caption_rows<=7168,
        "Z row matching requires original512 image and bounded padded caption rows");
    const int rows=image_rows+caption_rows;
    require(template_rows>=0 && template_rows<=8192,"Z row matching template extent exceeds bound");
    if(template_rows>=rows && template_rows%32==0)return template_rows;
    // Small bounded grid prevents the demonstrated c1056 caption cliff and
    // shares one program between nearby captions. This is not profitability
    // calibration or a guarantee that every longer-caption workload wins.
    return ((rows+127)/128)*128;
}
}
