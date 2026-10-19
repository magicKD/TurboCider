#pragma once
#include "../../core/contracts.hpp"
#include "../../core/common.hpp"
#include "../../backends/ane_w8_functions.hpp"

namespace tc::qwen21 {
inline std::vector<int> runtime_function_request(const Request&r,const char*raw) {
    auto rows=ane::parse_w8_function_rows(raw);
    if(rows.empty())return rows;
    require(r.model=="qwen-image-2.1" && (r.model_variant=="auto"||r.model_variant=="qwen-image-2.1") &&
        r.execution=="gpu_ane" && r.hybrid_mlp_mode=="runtime" && r.allow_approximation &&
        r.residency=="resident" && r.width==512 && r.height==512 && r.steps==40 && r.frames==1 && !r.audio &&
        r.qwen21_denoiser_backend=="mlx" && r.loras.empty() && !r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
        !r.prompt_enhance && !r.prompt_enhance_edit_experimental && !r.memory_constrained.enabled &&
        !r.memory_budget_bytes && !r.streaming.active() && r.dump.empty() && r.noise_path.empty() &&
        !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
        r.qwen21_dit_cache=="off" && r.qwen21_reference_size==1024 &&
        ((r.operation=="image.generate" && r.inputs.empty()) ||
         (r.operation=="image.edit" && !r.inputs.empty() && r.inputs.size()<=2)),
        "Qwen fixed-function family requires explicit approximate base40 resident512 BF16 runtime without LoRA/encoder/PE/cache/streaming recipes");
    require(rows.size()>=2 && rows.size()<=4,"Qwen family requires2...4 fixed procedures");
    for(size_t i=0;i<rows.size();++i)require(rows[i]==int(i+1)*512,"Qwen family requires consecutive512-row buckets beginning512,1024");
    return rows;
}
inline std::vector<int> configured_runtime_function_request(const Request&r) {
    return runtime_function_request(r,std::getenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS"));
}
inline bool runtime_function_descriptor(const ane::GraphShape&s,std::span<const int> functions) {
    return s.kind==ane::Kind::SwiGLU && s.rows==(functions.empty()?1024:functions.back()) &&
        s.hidden==4096 && s.width==12288 && !s.lora_inputs;
}
} // namespace tc::qwen21
