#include "../../native/models/qwen21/runtime_compiled_complement_options.hpp"
#include <cassert>

int main() {
    using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";
    r.allow_approximation=true;r.ane_manifest="fixed.json";r.width=r.height=512;r.steps=40;r.audio=false;r.residency="resident";
    r.qwen21_reference_size=1024;
    unsetenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT");assert(!configured_compiled_complement_request(r));
    setenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","0",1);assert(!configured_compiled_complement_request(r));
    setenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","1",1);
    setenv("TURBOCIDER_QWEN21_PRIVATE_QKV","1",1);setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","6144",1);setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement",1);
    setenv("TURBOCIDER_PRIVATE_ANE_W8_PROJECTION","descriptor",1);unsetenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS");
    setenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","0",1);assert(configured_compiled_complement_request(r));
    for(int count:{1,2}){auto e=r;e.operation="image.edit";e.inputs.resize(count);assert(configured_compiled_complement_request(e));}
    auto deny=[&](auto change){auto x=r;change(x);bool rejected=false;
        try{(void)configured_compiled_complement_request(x);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);};
    deny([](auto&x){x.steps=8;});deny([](auto&x){x.width=768;});deny([](auto&x){x.execution="gpu";});
    deny([](auto&x){x.loras.push_back({"adapter",1.f,"transformer"});});deny([](auto&x){x.qwen21_dit_cache="conservative";});
    deny([](auto&x){x.encoder_ane_manifest="encoder";});deny([](auto&x){x.residency="component_staged";});
    for(const auto &[key,value]:{std::pair{"TURBOCIDER_QWEN21_PRIVATE_QKV","0"},std::pair{"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","0"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_CHANNELS","5120"},std::pair{"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","exp"},
        std::pair{"TURBOCIDER_PRIVATE_ANE_W8_PROJECTION","single"},std::pair{"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU","1"}}) {
        const std::string old=std::getenv(key);setenv(key,value,1);deny([](auto&){});setenv(key,old.c_str(),1);
    }
    setenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS","512,1024",1);deny([](auto&){});unsetenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS");
    setenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT","yes",1);deny([](auto&){});
    RuntimeCompiledComplementState state;state.bind("a");assert(state.available()&&!state.healthy());
    ++state.validation_calls;++state.traces;state.accept();assert(state.healthy());state.bind("a");assert(state.healthy()&&state.traces==1);
    state.bind("b");assert(!state.available());const auto reason=state.failure_reason;state.accept();state.bind("a");
    assert(!state.healthy()&&state.failure_reason==reason);state={};state.bind("");assert(!state.available());
}
