// CPU-only native emitter golden. No device, model loading or inference.
#include "../../native/backends/private/ane_mil.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    using namespace tc::ane;
    if(argc!=2 || std::filesystem::exists(argv[1]))return 2;
    const std::filesystem::path out(argv[1]);std::filesystem::create_directories(out);
    char *identity=tc_runtime_build_identity();if(!identity)return 3;
    {std::ofstream f(out/"native-id.txt");f<<identity<<'\n';}tc_string_free(identity);
    for(int i=0;i<5;++i) {
        const GraphShape shape=i<2?GraphShape{Kind::SwiGLU,i?2048:1024,4096,6144,4096,512,false}:
            GraphShape{Kind::SwiGLU,32,i==4?256:128,512,128,512,i==3};
        const auto value=private_api::w8_swiglu_activation_program(shape,i==4?0:20260930,1.f,
            i==4?W8Basis::ComfyH256:W8Basis::SylvesterDH,0,0,false,W8Projection::Descriptor,
            i<2?W8Activation::SigmoidComplement:W8Activation::ExpDivide);
        {std::ofstream f(out/(std::to_string(i)+".mil"));f<<value.mil;if(!f)return 4;}
        {std::ofstream f(out/(std::to_string(i)+".bin"),std::ios::binary);
         f.write(reinterpret_cast<const char*>(value.constants.data()),value.constants.size());if(!f)return 5;}
    }
    std::cout<<"exported five original emitter recipes\n";
}
