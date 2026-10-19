#include "../../native/models/qwen21/runtime_private_qkv_options.hpp"
#include <cassert>

int main() {
    using namespace tc;
    using namespace tc::qwen21;
    Request r;
    r.model="qwen-image-2.1";r.operation="image.generate";r.execution="gpu_ane";
    r.hybrid_mlp_mode="runtime";r.allow_approximation=true;r.ane_manifest="fixed.json";
    r.width=r.height=512;r.steps=40;r.audio=false;r.residency="resident";r.qwen21_reference_size=1024;
    assert(private_qkv_joint_request(r));
    unsetenv("TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH");assert(!configured_private_qkv_prefetch_request(r));
    setenv("TURBOCIDER_QWEN21_PRIVATE_QKV","1",1);setenv("TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH","1",1);
    assert(configured_private_qkv_prefetch_request(r));
    setenv("TURBOCIDER_QWEN21_PRIVATE_QKV","0",1);
    bool rejected=false;try{(void)configured_private_qkv_prefetch_request(r);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    unsetenv("TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH");unsetenv("TURBOCIDER_QWEN21_PRIVATE_QKV");
    for(int step=0;step<40;++step)for(int layer=0;layer<32;++layer){
        auto next=qkv_lookahead_target(layer,step,1024,RuntimeTextQkvStep0::Gpu,true);
        if(step==39&&layer==31){assert(!next.permitted&&next.layer==-1);continue;}
        assert(next.layer==(layer+1)%32&&next.step==step+(layer==31));
        assert(next.permitted==(next.step>0));
    }
    auto long_prefix=qkv_lookahead_target(0,0,5155,RuntimeTextQkvStep0::Ane,false);
    assert(!long_prefix.permitted&&long_prefix.rows==5155);
    auto wrap=qkv_lookahead_target(31,0,5155,RuntimeTextQkvStep0::Ane,false);
    assert(wrap.permitted&&wrap.rows==1024&&wrap.layer==0&&wrap.step==1);
    auto no=[&](auto change){auto x=r;change(x);assert(!private_qkv_joint_request(x));};
    no([](auto&x){x.model="z-image";});no([](auto&x){x.steps=6;});
    no([](auto&x){x.width=1024;});no([](auto&x){x.frames=2;});no([](auto&x){x.audio=true;});
    no([](auto&x){x.execution="gpu";});no([](auto&x){x.hybrid_mlp_mode="auto";});
    no([](auto&x){x.qwen21_denoiser_backend="native_metal";});
    no([](auto&x){x.allow_approximation=false;});no([](auto&x){x.ane_manifest.clear();});
    no([](auto&x){x.encoder_ane_manifest="other";});no([](auto&x){x.memory_budget_bytes=1;});
    no([](auto&x){x.memory_constrained.enabled=true;});no([](auto&x){x.residency="component_staged";});
    no([](auto&x){x.loras.push_back({"adapter",1.f,"transformer"});});
    no([](auto&x){x.prompt_enhance=true;});no([](auto&x){x.dump="capture";});
    no([](auto&x){x.noise_path="noise";});no([](auto&x){x.qwen21_w8a8=true;});
    no([](auto&x){x.qwen21_gpu_w8a16=true;});no([](auto&x){x.qwen21_dit_cache="conservative";});
    no([](auto&x){x.qwen21_gpu_full_ffn_blocks={0};});
    no([](auto&x){x.qwen21_reference_size=512;});no([](auto&x){x.inputs.push_back({});});
    no([](auto&x){x.operation="image.edit";});
    for(int count:{1,2}){auto edit=r;edit.operation="image.edit";edit.inputs.resize(count);assert(private_qkv_joint_request(edit));}
    no([](auto&x){x.operation="image.edit";x.inputs.resize(3);});
    HybridMetrics m;
    m.runtime_weight_backend="private_ane";m.runtime_weight_data_path="w8a8_hadamard";
    m.runtime_weight_partition_axis="intermediate_channels";m.runtime_weight_fp32_channel_join_enabled=true;
    for(int channels:{4096,5120,6144}){
        m.runtime_weight_ane_channels=channels;m.runtime_weight_gpu_channels=12288-channels;
        assert(private_qkv_joint_ffn(m));
    }
    auto bad=[&](auto change){auto x=m;change(x);assert(!private_qkv_joint_ffn(x));};
    bad([](auto&x){x.runtime_failed=true;});bad([](auto&x){x.runtime_weight_prefetch_enabled=true;});
    bad([](auto&x){x.runtime_weight_backend="coreml";});bad([](auto&x){x.runtime_weight_data_path="fp16";});
    bad([](auto&x){x.runtime_weight_partition_axis="rows";});bad([](auto&x){x.runtime_weight_gpu_channels=4096;});
    bad([](auto&x){x.runtime_weight_fp32_channel_join_enabled=false;});bad([](auto&x){x.runtime_weight_gpu_layers={0};});
    bad([](auto&x){x.runtime_weight_fallback_blocks=1;});bad([](auto&x){x.runtime_weight_overflow_retries=1;});
    bad([](auto&x){x.runtime_weight_headroom=4.f;});bad([](auto&x){x.runtime_weight_s1_digest="new";});
}
