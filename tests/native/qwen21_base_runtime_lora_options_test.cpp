#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main(){
    using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.model_variant="qwen-image-2.1";r.operation="image.generate";
    r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";r.residency="resident";r.allow_approximation=true;
    r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.qwen21_reference_size=1024;r.ane_manifest="fixed.json";
    r.lora_strategy="inference_time";r.loras.push_back({"ordinary-style.safetensors",1.f,"transformer"});
    auto denied=[](const Request&v,const char*flag="1"){try{(void)base_schedule_runtime_lora_request(v,flag);}catch(const std::invalid_argument&){return true;}return false;};
    assert(!base_schedule_runtime_lora_request(r,nullptr));assert(!base_schedule_runtime_lora_request(r,"0"));
    assert(student_lora_schedule(r,false,false)); // Legacy runtime domain is preserved.
    for(int steps:{20,28,40}){r.steps=steps;assert(base_schedule_runtime_lora_request(r,"1"));assert(!student_lora_schedule(r,false,true));}
    assert(denied(r,"yes"));
    for(int steps:{1,5,6,19,41}){auto v=r;v.steps=steps;assert(denied(v));}
    for(auto name:viggle_v021_adapters){auto v=r;v.loras[0].path=std::string(name.filename);assert(denied(v));assert(student_lora_schedule(v,false,false));}
    {auto v=r;v.execution="gpu";v.hybrid_mlp_mode="auto";assert(denied(v));assert(!student_lora_schedule(v,false,false));}
    {auto v=r;v.model_variant="qwen-image-2.1-turbo";assert(denied(v));}
    {auto v=r;v.width=768;assert(denied(v));}
    {auto v=r;v.loras[0].role="encoder";assert(denied(v));}
    {auto v=r;v.loras[0].strength=std::numeric_limits<float>::quiet_NaN();assert(denied(v));}
    {auto v=r;v.loras[0].strength=-8;assert(base_schedule_runtime_lora_request(v,"1"));v.loras[0].strength=9;assert(denied(v));}
    {auto v=r;v.lora_strategy="in_memory_merge";assert(denied(v));}
    {auto v=r;v.encoder_ane_manifest="encoder.json";assert(denied(v));}
    {auto v=r;v.qwen21_dit_cache="fast";assert(denied(v));}
    {auto v=r;v.allow_approximation=false;assert(denied(v));}
    {auto v=r;v.memory_budget_bytes=uint64_t(32)<<30;assert(denied(v));}
    {auto v=r;v.operation="image.edit";assert(denied(v));v.inputs.resize(1);v.inputs[0].kind="image";v.inputs[0].role="reference";v.inputs[0].path="ref.png";assert(base_schedule_runtime_lora_request(v,"1"));v.qwen21_reference_size=512;assert(denied(v));}
    {auto v=r;v.loras.clear();assert(!base_schedule_runtime_lora_request(v,"1"));assert(!student_lora_schedule(v,true,true));}
    assert(student_lora_schedule(r,true,false));
    std::cout<<"PASS ordinary runtime base schedule opt-in, strict domain and legacy student preservation\n";
}
