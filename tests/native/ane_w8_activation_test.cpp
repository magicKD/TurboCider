#include "../../native/backends/ane_backend.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

std::string without_activation(const std::string &mil) {
    std::istringstream stream(mil);std::string line,result;
    while(std::getline(stream,line)) {
        bool activation=false;
        for(const auto *name:{"neg","eg","denom","sigmoid","silu","sigmoid_negative","complement"})
            activation=activation || line.find(" "+std::string(name)+" = ")!=std::string::npos;
        if(!activation)result+=line+"\n";
    }
    return result;
}
int main() {
    using namespace tc::ane;using namespace tc::ane::private_api;
    assert(parse_w8_activation(nullptr)==W8Activation::ExpDivide);
    for(const auto *raw:{"exp","sigmoid","silu","sigmoid-complement"})assert(w8_activation_name(parse_w8_activation(raw))==raw);
    for(const auto *raw:{"","0","SiLU","silu "," sigmoid","exp-divide"}) {
        bool bad=false;try{(void)parse_w8_activation(raw);}catch(const std::runtime_error&){bad=true;}assert(bad);
    }
    for(const auto shape:{GraphShape{Kind::SwiGLU,1024,4096,6144,4096,512,false},GraphShape{Kind::SwiGLU,33,128,512,4096,512,false}}) {
        const auto original=w8_swiglu_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false);
        for(auto projection:{W8Projection::Descriptor,W8Projection::K4096,W8Projection::Single,W8Projection::UpSingle,W8Projection::DownSingle}) {
            const auto old=w8_swiglu_projection_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false,projection);
            const auto same=w8_swiglu_activation_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false,projection,W8Activation::ExpDivide);
            assert(old.mil==same.mil&&old.constants==same.constants&&old.packed_rows==same.packed_rows);
        }
        for(auto activation:{W8Activation::Sigmoid,W8Activation::SiLU,W8Activation::SigmoidComplement}) {
            const auto changed=w8_swiglu_activation_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false,W8Projection::Descriptor,activation);
            assert(changed.constants==original.constants&&changed.packed_rows==original.packed_rows&&changed.headroom==1);
            assert(without_activation(changed.mil)==without_activation(original.mil));
            assert(changed.mil.find("exp(x = neg)")==std::string::npos);
            assert(changed.mil.find(activation==W8Activation::Sigmoid?"sigmoid(x = gt)":activation==W8Activation::SiLU?"silu(x = gt)":"sigmoid(x = neg)")!=std::string::npos);
            validate_w8_activation_backend(activation,W8Projection::Descriptor,true,{BackendPreference::Private,true},"w8a8",false);
            for(int which=0;which<7;++which) {
                bool bad=false;try{validate_w8_activation_backend(activation,which==6?W8Projection::Single:W8Projection::Descriptor,
                    which!=0,{which==1?BackendPreference::Public:which==5?BackendPreference::Auto:BackendPreference::Private,which!=2},
                    which==3?"fp16":"w8a8",which==4);}catch(const std::runtime_error&){bad=true;}assert(bad);
            }
            for(int which=0;which<6;++which) {
                auto unsupported=shape;unsupported.lora_inputs=which==0;bool bad=false;
                try{(void)w8_swiglu_activation_program(unsupported,which==1?0:20260930,which==5?2.f:1.f,
                    which==1?W8Basis::ComfyH256:W8Basis::SylvesterDH,which==2?256:0,0,which==3,
                    which==4?W8Projection::UpSingle:W8Projection::Descriptor,activation);}catch(const CapabilityError&){bad=true;}assert(bad);
            }
        }
    }
    std::cout<<"PASS W8 activation: legacy/projection bytes unchanged, exact nonactivation subgraph/ABI/constants, rejected unqualified combinations\n";
}
