#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <iostream>

using namespace tc;
using namespace tc::qwen21;
template<typename Function> void rejects(Function fn,const char *reason) {
    try { fn(); }catch(const std::invalid_argument &error) {
        require(std::string(error.what()).find(reason)!=std::string::npos,"wrong timestep policy rejection");return;
    }
    throw std::runtime_error("invalid timestep-cache policy accepted");
}
int main() {try {
    Request r;r.model="qwen-image-2.1";r.residency="streamed";r.execution="gpu";
    r.operation="image.generate";r.steps=8;r.frames=1;r.audio=false;r.hybrid_mlp_mode="auto";r.qwen21_denoiser_backend="mlx";
    r.qwen21_dit_cache="off";
    auto selected=[&](const Request &request,const char *raw) {
        return stream_timestep_cache_request(request,true,true,true,true,false,raw);
    };
    require(selected(r,nullptr)&&!selected(r,"0")&&selected(r,"1"),"qualified automatic timestep cache or explicit control changed");
    auto unqualified=r;unqualified.steps=7;
    require(!selected(unqualified,"0"),"disabled flag altered old request admission");
    for(const char *bad:{"","true","2","01"," 1","1\n"})rejects([&]{selected(r,bad);},"0 or 1");
    for(int changed=0;changed<12;++changed) {
        auto bad=r;
        switch(changed) {
            case 0:bad.residency="resident";break;
            case 1:bad.execution="gpu_ane";break;
            case 2:bad.hybrid_mlp_mode="runtime";break;
            case 3:bad.steps=7;break;
            case 4:bad.loras.push_back({"fake.safetensors",1.f,"transformer"});break;
            case 5:bad.operation="image.edit";break;
            case 6:bad.ane_manifest="fake.json";break;
            case 7:bad.qwen21_dit_cache="conservative";break;
            case 8:bad.qwen21_w8a8=true;break;
            case 9:bad.prompt_enhance=true;break;
            case 10:bad.frames=2;break;
            case 11:bad.memory_budget_bytes=1;break;
        }
        require(!selected(bad,nullptr),"automatic timestep cache selected an incompatible request");
        rejects([&]{selected(bad,"1");},"ordinary GPU Turbo8");
    }
    rejects([&]{stream_timestep_cache_request(r,false,true,true,true,false,"1");},"ordinary GPU Turbo8");
    rejects([&]{stream_timestep_cache_request(r,true,false,true,true,false,"1");},"ordinary GPU Turbo8");
    rejects([&]{stream_timestep_cache_request(r,true,true,false,true,false,"1");},"ordinary GPU Turbo8");
    rejects([&]{stream_timestep_cache_request(r,true,true,true,false,false,"1");},"ordinary GPU Turbo8");
    rejects([&]{stream_timestep_cache_request(r,true,true,true,true,true,"1");},"ordinary GPU Turbo8");
    require(!stream_timestep_cache_request(r,false,true,true,true,false,nullptr) &&
        !stream_timestep_cache_request(r,true,false,true,true,false,nullptr) &&
        !stream_timestep_cache_request(r,true,true,false,true,false,nullptr) &&
        !stream_timestep_cache_request(r,true,true,true,false,false,nullptr) &&
        !stream_timestep_cache_request(r,true,true,true,true,true,nullptr),
        "automatic cache must preserve base, non-dual, Float-A/older-device and Runtime ANE routes");
    r.qwen21_stream_weight_budget_bytes=1280ull<<20;
    require(selected(r,nullptr),"qualified optional weight cache disabled automatic time retirement");
    r.execution="auto";require(selected(r,nullptr)&&selected(r,"1"),"ordinary auto GPU route was rejected");
    auto resident=r;resident.residency="resident";
    validate_stream_timestep_cache_route(resident,nullptr);validate_stream_timestep_cache_route(resident,"0");
    rejects([&]{validate_stream_timestep_cache_route(resident,"1");},"streamed generation");
    std::cout<<"PASS timestep cache strict request policy\n";return 0;
}catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}}
