#include "../../native/models/qwen21/runtime_gpu_dual.hpp"
#include <cassert>

int main(){
    using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";
    r.allow_approximation=true;r.ane_manifest="fixed.json";r.width=r.height=512;r.steps=40;r.audio=false;r.residency="resident";r.qwen21_reference_size=1024;
    unsetenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU");assert(!configured_gpu_dual_swiglu_request(r));
    setenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","0",1);assert(!configured_gpu_dual_swiglu_request(r));
    setenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","1",1);setenv("TURBOCIDER_QWEN21_PRIVATE_QKV","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","6144",1);
    setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement",1);assert(configured_gpu_dual_swiglu_request(r));
    for(int count:{1,2}){auto e=r;e.operation="image.edit";e.inputs.resize(count);assert(configured_gpu_dual_swiglu_request(e));}
    auto deny=[&](auto change){auto x=r;change(x);bool bad=false;try{(void)configured_gpu_dual_swiglu_request(x);}catch(const std::invalid_argument&){bad=true;}assert(bad);};
    deny([](auto&x){x.steps=6;});deny([](auto&x){x.execution="gpu";});deny([](auto&x){x.width=1024;});
    deny([](auto&x){x.residency="component_staged";});deny([](auto&x){x.qwen21_dit_cache="conservative";});
    deny([](auto&x){x.loras.push_back({"adapter",1.f,"transformer"});});deny([](auto&x){x.qwen21_gpu_w8a16=true;});
    for(const auto &[key,value]:{std::pair{"TURBOCIDER_PRIVATE_ANE_CHANNELS","5632"},std::pair{"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","exp"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","0"},std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV","0"}}){
        const std::string old=std::getenv(key);setenv(key,value,1);deny([](auto&){});setenv(key,old.c_str(),1);
    }
    setenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","yes",1);deny([](auto&){});
    RuntimeGpuDualState state;state.bind("source-a");assert(state.available()&&!state.healthy());++state.validation_calls;state.accept();assert(state.healthy());
    state.bind("source-a");assert(state.healthy()&&state.validation_calls==1);state.bind("source-b");assert(!state.available()&&!state.healthy());
    const auto reason=state.failure_reason;state.bind("source-a");state.accept();assert(!state.healthy()&&state.failure_reason==reason);
    state={};state.bind("");assert(!state.available());
}
