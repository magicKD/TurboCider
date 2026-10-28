#include "../../native/models/qwen21/runtime_function_options.hpp"
#include "../../native/models/qwen21/runtime_prefix_snapshot.hpp"
#include <cassert>
#include <iostream>

int main(){using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.model_variant="qwen-image-2.1";r.operation="image.generate";
    r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.residency="resident";r.execution="gpu_ane";
    r.hybrid_mlp_mode="runtime";r.allow_approximation=true;r.ane_manifest="shape.json";r.qwen21_reference_size=1024;
    assert(runtime_function_request(r,nullptr).empty());assert(runtime_function_request(r,"0").empty());
    assert(runtime_function_request(r,"512,1024")==std::vector<int>({512,1024}));
    assert(runtime_function_request(r,"512,1024,1536,2048").size()==4);
    auto denied=[](auto&&f){try{f();}catch(const std::exception&){return true;}return false;};
    for(const char*bad:{"1024","512,512","512,2048","32,64","512,1024,","512, 1024","512,1025","512,1024,1536,2048,2560"})
        assert(denied([&]{runtime_function_request(r,bad);}));
    auto check=[&](auto change){auto v=r;change(v);assert(denied([&]{runtime_function_request(v,"512,1024");}));};
    check([](auto&v){v.execution="gpu";});check([](auto&v){v.width=768;});check([](auto&v){v.steps=8;});
    check([](auto&v){v.model_variant="qwen-image-2.1-turbo";});check([](auto&v){v.loras.push_back({"a",1.f,"transformer"});});
    check([](auto&v){v.encoder_ane_manifest="encoder";});check([](auto&v){v.qwen21_dit_cache="fast";});
    check([](auto&v){v.allow_approximation=false;});check([](auto&v){v.memory_budget_bytes=1;});
    {auto v=r;v.operation="image.edit";v.inputs.resize(2);assert(runtime_function_request(v,"512,1024").size()==2);v.inputs.resize(3);assert(denied([&]{runtime_function_request(v,"512,1024");}));}
    assert(ane::w8_function_identity({}).empty());
    const auto a=ane::parse_w8_function_rows("512,1024"),b=ane::parse_w8_function_rows("512,1024,1536,2048");
    assert(ane::w8_function_identity(a)!=ane::w8_function_identity(b));
    const ane::GraphShape s1{ane::Kind::SwiGLU,1024,4096,12288,4096,512,false},s2{ane::Kind::SwiGLU,2048,4096,12288,4096,512,false};
    assert(runtime_function_descriptor(s1,{})&&runtime_function_descriptor(s2,b));
    assert(!runtime_function_descriptor(s2,{})&&!runtime_function_descriptor(s1,b));
    HybridMetrics m;m.runtime_weight_backend="private_ane";m.runtime_weight_data_path="w8a8_hadamard";
    m.runtime_weight_partition_axis="intermediate_channels";m.runtime_weight_ane_channels=m.runtime_weight_gpu_channels=6144;
    m.runtime_weight_fp32_channel_join_enabled=true;m.runtime_weight_fixed_program_requested=m.runtime_weight_fixed_program_enabled=true;
    m.runtime_weight_program_instances=1;m.runtime_weight_function_rows=b;m.runtime_weight_row_alignment=512;
    m.bucket=2048;m.runtime_weight_capability_calls=12;m.runtime_weight_source_recipe="source"+ane::w8_function_identity(b);
    assert(runtime_text_prefix_snapshot_healthy(m));
    auto bad_health=[&](auto change){auto v=m;change(v);assert(!runtime_prefix_snapshot_healthy(v));};
    bad_health([](auto&v){v.runtime_weight_capability_calls=3;});
    bad_health([](auto&v){v.runtime_weight_row_alignment=2048;});
    bad_health([](auto&v){v.bucket=1024;});
    bad_health([](auto&v){v.runtime_weight_source_recipe="old";});
    std::cout<<"PASS explicit Qwen base40 function domain, immutable list identity and rejected unsupported recipes\n";
}
