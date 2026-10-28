#include "../../tools/profiling/qwen_base/hidden_quant_mil.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cassert>
#include <iostream>

std::string original_normalization(const std::string&s,int c,int m) {
    const auto type="tensor<fp16, "+tc::qwen_rotation_lab::shape(1,1,c,m)+"> ";
    const auto begin=s.find("        "+type+"habs = ");assert(begin!=std::string::npos);
    const auto last=s.find(" hq = quantize(",begin);assert(last!=std::string::npos);
    return s.substr(begin,s.find('\n',last)-begin+1);
}
int main(){
    using namespace tc;namespace lab=qwen_hidden_quant_lab;
    for(int c:{512,6144})for(int m:{32,1024}) {
        const auto original=ane::private_api::w8_swiglu_activation_program({ane::Kind::SwiGLU,m,4096,c,4096,512,false},
            20260930,1,ane::W8Basis::SylvesterDH,0,0,false,ane::W8Projection::Descriptor,ane::W8Activation::SigmoidComplement);
        for(bool rotate:{false,true})for(int policy=0;policy<3;++policy) {
            auto p=lab::program(c,m,rotate,policy);
            assert(rotate?p.constants==original.constants:p.constants.empty());
            assert(p.mil.find("shape=[1, 1, "+std::to_string(c+1)+", "+std::to_string(m)+"]")!=std::string::npos);
            assert(p.mil.find("axis = int32(2)")!=std::string::npos);
            if(policy==0&&rotate) {
                // The isolated rotation returns its value under a different
                // SSA name. Compare the exact production subgraph at the
                // same explicitly named boundary, not a variable spelling.
                assert(original_normalization(lab::normalization("hr",c,m,0),c,m)==original_normalization(original.mil,c,m));
                assert(p.mil.find("habs = abs(x = value)")!=std::string::npos);
                assert(p.mil.find("ratio = real_div(x = value, y = floor)")!=std::string::npos);
            }
            if(policy==1)assert(p.mil.find("inv_floor = real_div(x = fp16(1), y = floor)")!=std::string::npos);
            if(policy==2)assert(p.mil.find("inverse(x = floor, epsilon = fp16(0))")!=std::string::npos);
            assert(p.mil.find("maximum(x = peak, y = fp16(0x1p-12))")!=std::string::npos);
            assert(p.mil.find("quantize(input = a8, scale = fp16(1), output_dtype = string(\"int8\"))")!=std::string::npos);
        }
    }
    assert(float(_Float16(1.f/_Float16(0x1p-12f)))==4096.f);
    for(int policy:{-1,3}){bool bad=false;try{(void)lab::program(6144,1024,true,policy);}catch(const std::invalid_argument&){bad=true;}assert(bad);}
    std::cout<<"PASS hidden quant: original normalization/rotation constants exact; reciprocal bounded; explicit inverse epsilon0; same q/scale consumer\n";
}
