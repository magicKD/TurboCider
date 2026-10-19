#pragma once
#include "runtime_prefix_snapshot.hpp"
namespace tc::qwen21 {
inline bool private_qkv_joint_request(const Request &r) {
    return r.model=="qwen-image-2.1"&&r.execution=="gpu_ane"&&r.hybrid_mlp_mode=="runtime"&&
        r.qwen21_denoiser_backend=="mlx"&&r.allow_approximation&&r.loras.empty()&&
        r.width==512&&r.height==512&&r.steps==40&&r.frames==1&&!r.audio&&r.residency=="resident"&&
        !r.ane_manifest.empty()&&r.encoder_ane_manifest.empty()&&!r.memory_budget_bytes&&
        !r.memory_constrained.enabled&&!r.streaming.active()&&!r.prompt_enhance&&
        r.dump.empty()&&r.noise_path.empty()&&(r.qwen21_dit_cache=="off"||configured_runtime_conservative_cache_request(r))&&!r.qwen21_w8a8&&!r.qwen21_gpu_w8a16&&
        r.qwen21_gpu_full_ffn_blocks.empty()&&r.qwen21_reference_size==1024&&
        ((r.operation=="image.generate"&&r.inputs.empty())||
         (r.operation=="image.edit"&&!r.inputs.empty()&&r.inputs.size()<=2));
}
inline bool private_qkv_joint_ffn(const HybridMetrics &m) {
    return runtime_prefix_snapshot_healthy(m)&&!m.runtime_weight_prefetch_enabled;
}
inline bool configured_private_qkv_prefetch_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH");
    if(!binary_option_or_unset(raw))throw std::invalid_argument("Private QKV prefetch requires0 or1");
    if(!option_enabled(raw))return false;
    if(!private_qkv_joint_request(r) || !option_enabled(std::getenv("TURBOCIDER_QWEN21_PRIVATE_QKV")))
        throw std::invalid_argument("QKV prefetch requires explicit base40 resident512 joint runtime");
    return true;
}
struct QkvLookaheadTarget {int layer=-1,step=-1,rows=0;bool permitted=false;};
inline QkvLookaheadTarget qkv_lookahead_target(int layer,int step,int rows,RuntimeTextQkvStep0 policy,bool text_hit) {
    if(layer<0||layer>=32||step<0||step>=40||rows<=0)throw std::invalid_argument("invalid QKV future geometry/step");
    if(layer==31 && step==39)return {};
    const int next_layer=(layer+1)%32,next_step=step+(layer==31),next_rows=layer==31?1024:rows;
    return {next_layer,next_step,next_rows,next_rows==1024 && runtime_text_qkv_permitted(policy,text_hit,next_step)};
}
} // namespace tc::qwen21
