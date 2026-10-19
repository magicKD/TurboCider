#include "../../native/backends/ane_backend.hpp"
#include "../../native/models/qwen21/runtime_prefix_snapshot.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc::ane;
    assert(!parse_fixed_program(nullptr)&&!parse_fixed_program("0")&&parse_fixed_program("1"));
    for(const auto *value:{"","true","2"," 1","1 ","01"}) {
        bool rejected=false;try{(void)parse_fixed_program(value);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    assert(program_creation_allowed(true,false)&&!program_creation_allowed(true,true));
    assert(program_creation_allowed(false,false)&&program_creation_allowed(false,true));
    for(auto policy:{BackendPolicy{BackendPreference::Public,false},BackendPolicy{BackendPreference::Auto,true},
                     BackendPolicy{BackendPreference::Off,false},BackendPolicy{BackendPreference::Private,false}}) {
        validate_fixed_program_backend(false,policy,"fp16");
        bool rejected=false;try{validate_fixed_program_backend(true,policy,"w8a8");}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    for(const auto *path:{"fp16","","unknown"}) {
        bool rejected=false;try{validate_fixed_program_backend(true,{BackendPreference::Private,true},path);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    validate_fixed_program_backend(true,{BackendPreference::Private,true},"w8a8");
    validate_fixed_program_backend(true,{BackendPreference::Private,true},"convrot_w8a8");
    tc::HybridMetrics m;
    m.runtime_weight_backend="private_ane";m.runtime_weight_data_path="w8a8_hadamard";
    m.runtime_weight_partition_axis="intermediate_channels";
    m.runtime_weight_ane_channels=m.runtime_weight_gpu_channels=6144;
    m.runtime_weight_fp32_channel_join_enabled=true;
    assert(tc::qwen21::runtime_prefix_snapshot_healthy(m));
    m.runtime_weight_fixed_program_requested=true;
    assert(!tc::qwen21::runtime_prefix_snapshot_healthy(m));
    m.runtime_weight_fixed_program_enabled=true;m.runtime_weight_program_instances=1;
    assert(tc::qwen21::runtime_prefix_snapshot_healthy(m));
    for(int which=0;which<3;++which) {
        auto bad=m;if(which==0)bad.runtime_weight_program_disabled=true;
        if(which==1)bad.runtime_weight_program_instances=2;
        if(which==2)bad.runtime_weight_blocked_rebuilds=1;
        assert(!tc::qwen21::runtime_prefix_snapshot_healthy(bad));
    }
    std::cout<<"PASS fixed Program policy: explicit authorization, one construction and unhealthy prefix rejection\n";
}
