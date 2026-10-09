#include "../../native/models/qwen21/runtime_gpu_layer_config.hpp"
#include <iostream>
using namespace tc;
int main(){try {
    require(ane::parse_gpu_layers("31,0",32)==std::vector<int>({0,31}) &&
        qwen21::prefill_gpu_layer_identity({})=="" &&
        qwen21::prefill_gpu_layer_identity({0,31})==":prefill-gpu-ffn-layers-v1=0,31","canonical prefill layer identity mismatch");
    for(const char *raw:{"","-1","32","0,0","0,"," 0","0;1","1.0"}) {
        bool rejected=false;try{ane::parse_gpu_layers(raw,32);}catch(const std::exception&){rejected=true;}
        require(rejected,"malformed prefill GPU layers accepted");
    }
    Request r;r.model="qwen-image-2.1";r.operation="image.edit";r.width=r.height=512;r.residency="resident";
    r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";r.allow_approximation=true;r.inputs.resize(1);r.qwen21_reference_size=512;
    setenv("TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS","31,0",1);setenv("TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE","prefill",1);
    setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","5120",1);setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
    require(qwen21::configured_prefill_gpu_layers(r)==std::vector<int>({0,31}),"valid prefill policy declined");
    r.inputs.resize(2);require(qwen21::configured_prefill_gpu_layers(r).size()==2,"second reference declined");
    for(auto bad:std::vector<Request>{[&]{auto v=r;v.width=1024;return v;}(),[&]{auto v=r;v.allow_approximation=false;return v;}(),
        [&]{auto v=r;v.memory_budget_bytes=1<<20;return v;}(),[&]{auto v=r;v.inputs.resize(3);return v;}(),
        [&]{auto v=r;v.operation="image.generate";v.inputs.clear();return v;}(),[&]{auto v=r;v.encoder_ane_manifest="unused";return v;}()}) {
        bool rejected=false;try{qwen21::configured_prefill_gpu_layers(bad);}catch(const std::exception&){rejected=true;}
        require(rejected,"unsupported workload inherited prefill layer policy");
    }
    for(const auto &value:std::vector<std::pair<const char*,const char*>>{{"TURBOCIDER_ANE_BACKEND","public"},
        {"TURBOCIDER_ALLOW_PRIVATE_ANE","0"},{"TURBOCIDER_PRIVATE_ANE_CHANNELS","auto"},
        {"TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE","decode"},{"TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE","all"},
        {"TURBOCIDER_PRIVATE_ANE_DATA_PATH","fp16"},{"TURBOCIDER_RUNTIME_ANE_CHUNKS","auto"},
        {"TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","0"}}) {
        const auto old=std::string(std::getenv(value.first));setenv(value.first,value.second,1);bool rejected=false;
        try{qwen21::configured_prefill_gpu_layers(r);}catch(const std::exception&){rejected=true;}
        setenv(value.first,old.c_str(),1);require(rejected,"unsupported backend/phase inherited prefill layer policy");
    }
    r.execution="gpu";require(qwen21::configured_prefill_gpu_layers(r).empty(),"ordinary GPU control acquired split policy");
    std::cout<<"PASS Qwen prefill GPU layer policy: bounded canonical ordinals, stable identity, explicit one/two-ref Private guards and inert GPU control\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
