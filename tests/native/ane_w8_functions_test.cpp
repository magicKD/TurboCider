#include "../../native/backends/ane_w8_functions.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cassert>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    using namespace tc::ane;
    const GraphShape shape{Kind::SwiGLU,64,128,512,128,512,false};
    const std::vector<int> rows{32,64};
    assert(checked_w8_function_rows(shape,rows)==rows);
    assert(w8_function_chunk(32,rows)==32 && w8_function_chunk(64,rows)==64 && w8_function_chunk(96,rows)==64);
    const auto denied=[](auto &&f){try{f();}catch(const CapabilityError&){return true;}return false;};
    for(const auto &bad:std::vector<std::vector<int>>{{},{64},{64,32},{32,32},{16,64},{32,96},{32,48,64}})
        assert(denied([&]{(void)checked_w8_function_rows(shape,bad);}));
    for(int bad:{0,-32,31,65})assert(denied([&]{(void)w8_function_chunk(bad,rows);}));
    {auto s=shape;s.lora_inputs=true;assert(denied([&]{checked_w8_function_rows(s,rows);}));}
    for(auto activation:{W8Activation::ExpDivide,W8Activation::SigmoidComplement}) {
        const auto family=private_api::w8_swiglu_function_family_program(shape,rows,activation);
        const auto single=private_api::w8_swiglu_activation_program(shape,20260930,1.f,W8Basis::SylvesterDH,0,0,false,W8Projection::Descriptor,activation);
        assert(family.rows==rows && family.program.constants==single.constants && family.program.packed_rows==129 && family.program.headroom==1.f);
        assert(family.program.mil.find("func ffn32<ios18>")!=std::string::npos);
        assert(family.program.mil.find("func ffn64<ios18>")!=std::string::npos);
        assert(family.program.mil.find("x_storage")!=std::string::npos && family.program.mil.find("tx_storage")!=std::string::npos);
        assert(family.program.mil.find("shape=[1, 1, 129, 32], strides=[8256, 8256, 64, 1]")!=std::string::npos);
        auto expected=single.mil;
        expected.replace(expected.find("main_ane"),8,"ffn64");
        expected.erase(0,std::string("program(1.3)\n{\n").size());
        assert(family.program.mil.ends_with(expected));
    }
    if(argc==2) {
        const std::filesystem::path out(argv[1]);
        for(int i=0;i<5;++i) {
            const GraphShape s=i<2?GraphShape{Kind::SwiGLU,i?2048:1024,4096,6144,4096,512,false}:
                GraphShape{Kind::SwiGLU,32,i==4?256:128,512,128,512,i==3};
            const auto value=private_api::w8_swiglu_activation_program(s,i==4?0:20260930,1.f,
                i==4?W8Basis::ComfyH256:W8Basis::SylvesterDH,0,0,false,W8Projection::Descriptor,
                i<2?W8Activation::SigmoidComplement:W8Activation::ExpDivide);
            {std::ofstream f(out/(std::to_string(i)+".mil"));f<<value.mil;assert(bool(f));}
            {std::ofstream f(out/(std::to_string(i)+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(value.constants.data()),value.constants.size());assert(bool(f));}
        }
    }
    std::cout<<"PASS fixed W8 function layout, shared constants, full-size default body and admission/selection negatives\n";
}
