#include "../../native/models/qwen21/runtime_private_qkv_options.hpp"
#include <cassert>
#include <optional>

int main() {
    using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";
    r.allow_approximation=true;r.ane_manifest="fixed";r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.residency="resident";
    r.qwen21_reference_size=1024;r.qwen21_dit_cache="conservative";
    unsetenv("TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE");assert(!configured_runtime_conservative_cache_request(r));
    assert(!private_qkv_joint_request(r)&&!runtime_prefix_snapshot_base_request(r));
    setenv("TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE","1",1);
    for(const auto &[name,value]:{std::pair{"TURBOCIDER_ANE_BACKEND","private"},std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV","1"},
        std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH","1"},std::pair{"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_CHANNELS","6144"},std::pair{"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_W8_PROJECTION","descriptor"},std::pair{"TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1"},
        std::pair{"TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1"},std::pair{"TURBOCIDER_RUNTIME_ANE_CHUNKS","1"}})setenv(name,value,1);
    for(const auto name:{"TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS","TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0"})unsetenv(name);
    for(const auto name:{"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC"})setenv(name,"0",1);
    assert(configured_runtime_conservative_cache_request(r)&&private_qkv_joint_request(r)&&runtime_text_prefix_snapshot_request(r));
    for(int n:{1,2}){auto e=r;e.operation="image.edit";e.inputs.resize(n);assert(configured_runtime_conservative_cache_request(e)&&runtime_prefix_snapshot_request(e));}
    auto deny=[&](auto change){auto q=r;change(q);bool bad=false;try{(void)configured_runtime_conservative_cache_request(q);}catch(const std::invalid_argument&){bad=true;}assert(bad);};
    deny([](auto&q){q.steps=20;});deny([](auto&q){q.width=768;});deny([](auto&q){q.model_variant="qwen-image-2.1-turbo";});
    deny([](auto&q){q.loras.push_back({"adapter",1.f,"transformer"});});deny([](auto&q){q.qwen21_dit_cache="fast";});
    deny([](auto&q){q.encoder_ane_manifest="encoder";});deny([](auto&q){q.execution="gpu";});
    const char *required="TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH";setenv(required,"0",1);deny([](auto&){});setenv(required,"1",1);
    setenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","1",1);deny([](auto&){});setenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","0",1);
    auto off=r;off.qwen21_dit_cache="off";assert(!configured_runtime_conservative_cache_request(off)&&private_qkv_joint_request(off));
    setenv("TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE","yes",1);deny([](auto&){});
    for(int s=0;s<=16;++s)assert(runtime_conservative_saved_blocks(s)==24*s);
    for(int s:{-1,17}){bool bad=false;try{(void)runtime_conservative_saved_blocks(s);}catch(const std::invalid_argument&){bad=true;}assert(bad);}
    // Exhaust every permitted skip pattern phase and both first-step routes.
    for(bool hit:{false,true})for(bool text:{false,true})for(int offset:{0,1}) {
        const auto policy=text?RuntimeTextQkvStep0::Gpu:RuntimeTextQkvStep0::Ane;
        std::optional<std::pair<int,int>> future;
        int predictions=0,claims=0,submissions=0,discards=0,skips=0;
        for(int step=0;step<40;++step) {
            const bool skip=step>=8&&step<39&&step%2==offset;
            for(int block=0;block<32;++block) {
                if(skip&&block==8) {
                    assert(future==std::make_pair(step,8));future.reset();++discards;++skips;
                    future=std::make_pair(step+1,0);++submissions;break;
                }
                const int rows=step==0&&!hit?5155:1024;
                if(rows==1024&&runtime_text_qkv_permitted(policy,hit,step)) {
                    ++predictions;if(future){assert(future==std::make_pair(step,block));future.reset();++claims;}
                }
                const auto next=qkv_lookahead_target(block,step,rows,policy,hit);
                if(next.permitted){assert(!future);future=std::make_pair(next.step,next.layer);++submissions;}
            }
        }
        assert(!future&&skips<=16&&discards==skips&&submissions==claims+discards);
        assert(predictions==(hit&&!text?1280:1248)-24*skips);
        assert(claims==predictions-(hit&&!text?1:0));
    }
}
