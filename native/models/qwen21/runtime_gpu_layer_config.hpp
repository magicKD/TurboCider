#pragma once
#include "../../core/common.hpp"
#include "diagnostic_options.hpp"
#include "../../backends/ane_backend.hpp"
#include "../../backends/ane_gpu_layer_policy.hpp"

namespace tc::qwen21 {
inline std::string prefill_gpu_layer_list(const std::vector<int> &layers) {
    std::string result;
    for(int layer:layers)result+=(result.empty()?"":",")+std::to_string(layer);
    return result;
}
inline std::string prefill_gpu_layer_identity(const std::vector<int> &layers) {
    return layers.empty()?"":":prefill-gpu-ffn-layers-v1="+prefill_gpu_layer_list(layers);
}
inline std::vector<int> configured_prefill_gpu_layers(const Request &r) {
    auto layers=ane::parse_gpu_layers(std::getenv("TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS"),32);
    if(layers.empty() || r.execution!="gpu_ane")return {};
    const auto *path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
    const auto *chunks=std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
    const auto *async=std::getenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
    const auto backend=ane::configured_backend();
    require(r.model=="qwen-image-2.1" && r.hybrid_mlp_mode=="runtime" && r.operation=="image.edit" && r.width==512 && r.height==512 &&
        !r.inputs.empty() && r.inputs.size()<=2 && r.qwen21_reference_size==512 && r.allow_approximation &&
        r.residency=="resident" && !r.prompt_enhance && !r.streaming.active() && !r.memory_constrained.enabled &&
        !r.memory_budget_bytes && r.encoder_ane_manifest.empty() && layers.size()<32 &&
        runtime_ffn_phase(std::getenv("TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"))==RuntimeFfnPhase::Prefill &&
        backend.preferred==ane::BackendPreference::Private && backend.allow_private && ane::private_channel_count(12288)>0 &&
        path && std::string_view(path)=="w8a8" && chunks && std::string_view(chunks)=="1" &&
        async && std::string_view(async)=="1",
        "Qwen prefill GPU layers require partial fixed Private W8 channels/async, approximate unconstrained resident512 editing with one/two ref512, prefill-only and GPU encoder");
    return layers;
}
}
