// Isolate composition failures; never relax a model/quality gate.
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace tc::ane;
using namespace tc::ane::private_api;
void replace_all(std::string &text,const std::string &from,const std::string &to) {
    size_t at=0;while((at=text.find(from,at))!=std::string::npos){text.replace(at,from.size(),to);at+=to.size();}
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        Device device;GraphShape shape{Kind::SwiGLU,32,128,512,64,128,true};
        uint64_t timeline=0;
        for(int mode=0;mode<4;++mode) {
            const char *name=mode==0?"legacy":mode==1?"full":mode==2?"without-carrier-output":"tensor-numerator";
            auto mil=fp16_program(shape,mode!=0);
            if(mode==2)replace_all(mil,"add(x = down_bf16_out, y = carrier_invalid)","mul(x = down_bf16_out, y = fp16(1))");
            if(mode==3) {
                const std::string declaration="tensor<fp16, [1, 1, 512, 32]> sigmoid =";
                replace_all(mil,declaration,
                    "tensor<fp16, [1, 1, 512, 32]> sigmoid_zero = mul(x = gate_bf16_out, y = fp16(0));\n"
                    "        tensor<fp16, [1, 1, 512, 32]> sigmoid_one = add(x = sigmoid_zero, y = fp16(1));\n"
                    "        "+declaration);
                replace_all(mil,"real_div(x = fp16(1), y = denom)","real_div(x = sigmoid_one, y = denom)");
            }
            try {
                Program program(device,mil,{},std::filesystem::path(argv[1])/name);
                std::vector<std::pair<std::string,Surface>> inputs;
                for(const auto *slot:{"x","wg","wu","wd","dg","du"}) {
                    const bool x=std::string(slot)=="x",down=std::string(slot)=="wd",delta=std::string(slot)=="dg"||std::string(slot)=="du";
                    inputs.emplace_back(slot,Surface(device,x||down?128:512,x||delta?32:down?512:128,Element::FP16));
                    auto &surface=inputs.back().second;
                    std::memset(surface.data(),0,surface.rows()*surface.pitch());
                }
                std::pair<std::string,Surface> output[]{ {"y",Surface(device,640,32,Element::FP16)} };
                const auto ready=++timeline,done=++timeline;
                auto job=program.enqueue(inputs,output,ready,done);device.signal(ready);
                auto result=job.finish();if(!result.ok)throw CapabilityError(result.error);
                std::cout<<"{\"mode\":\""<<name<<"\",\"compiled\":true,\"executed\":true,\"scope\":\"small composition/capability only\"}"<<std::endl;
            }catch(const CapabilityError &error) {
                std::cout<<"{\"mode\":\""<<name<<"\",\"compiled\":false,\"reason\":"<<std::quoted(error.what())<<"}"<<std::endl;
                if(!Program::healthy())return 1;
            }
        }
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
