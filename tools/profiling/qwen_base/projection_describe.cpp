#include "../../../native/backends/private/ane_mil.hpp"
#include "../../../native/runtime/memory_manifest.hpp"
#include "turbocider/turbocider.h"
#include <iostream>

#ifndef TC_W8_DESCRIBE_CPP_SHA
#define TC_W8_DESCRIBE_CPP_SHA "unbound"
#endif

int main() {
    using namespace tc::ane;
    char *id=tc_runtime_build_identity();if(!id)return 1;
    std::cout<<"{\"runtime_build_id\":\""<<id<<"\",\"cpp_sha256\":\""<<TC_W8_DESCRIBE_CPP_SHA<<"\",\"cpu_only\":true,\"cases\":[";
    tc_string_free(id);
    bool comma=false;
    for(int variant=0;variant<6;++variant){
        const bool comfy=variant>=4,rounded=variant==5,lora=variant==3;
        const int rows=variant==1||variant==2?1024:33;
        const int hidden=variant==1||variant==2?4096:comfy?512:variant==3?384:128;
        const int width=variant==1?6144:variant==2?5120:512;
        auto result=private_api::w8_swiglu_program({Kind::SwiGLU,rows,hidden,width,4096,512,lora},
            comfy?0:20260930,1.f,comfy?W8Basis::ComfyH256:W8Basis::SylvesterDH,0,0,rounded);
        const auto mil=tc::memory_sha256_hex(result.mil);
        const auto constants=tc::memory_sha256_hex({reinterpret_cast<const char*>(result.constants.data()),result.constants.size()});
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"hidden\":"<<hidden<<",\"width\":"<<width
            <<",\"mil_sha256\":\""<<mil<<"\",\"constants_sha256\":\""<<constants<<"\",\"mil_bytes\":"<<result.mil.size()
            <<",\"constants_bytes\":"<<result.constants.size()<<",\"packed_rows\":"<<result.packed_rows<<'}';
    }
    std::cout<<"]}\n";
}
