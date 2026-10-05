// Capability diagnostic only. A parsed dtype is NOT proof of ANE support.
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/core/gguf_decode.hpp"
#include <bit>
#include <cstring>
#include <iostream>
#include "private_ane_bf16_emulation.hpp"

using namespace tc::ane;
using namespace tc::ane::private_api;
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        Device device;
        Surface x(device,1,256,Element::FP16),y(device,1,256,Element::FP16);
        auto *data=static_cast<uint16_t*>(x.data());
        for(int i=0;i<256;++i)data[i]=uint16_t(0x3800+i);
        uint64_t timeline=0;
        for(const std::string dtype:{"fp16","bf16","bfloat16"}) {
            const auto mil=std::string("program(1.3)\n{\n func main_ane<ios18>(tensor_buffer<fp16, shape=[1, 1, 1, 256], strides=[256, 256, 256, 1], interleave_factors=[1, 1, 1, 1]> x) {\n")+
                " tensor<fp16, [1, 1, 1, 256]> xt = tensor_buffer_to_tensor<ios17>(input = x);\n"+
                " tensor<"+dtype+", [1, 1, 1, 256]> b = cast(x = xt, dtype = string(\""+dtype+"\"));\n"+
                " tensor<fp16, [1, 1, 1, 256]> h = cast(x = b, dtype = string(\"fp16\"));\n"+
                " tensor_buffer<fp16, shape=[1, 1, 1, 256], strides=[256, 256, 256, 1], interleave_factors=[1, 1, 1, 1]> y = tensor_to_tensor_buffer<ios17>(input = h, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([256, 256, 256, 1]));\n } -> (y);\n}\n";
            try {
                Program program(device,mil,{},std::filesystem::path(argv[1])/dtype);
                std::pair<std::string,Surface> inputs[]{{"x",x}},outputs[]{{"y",y}};
                const auto ready=++timeline,done=++timeline;
                auto job=program.enqueue(inputs,outputs,ready,done);device.signal(ready);
                const auto result=job.finish();if(!result.ok)throw CapabilityError(result.error);
                int wrong=0;
                for(int i=0;i<256;++i) {
                    const float v=tc::gguf::fp16_to_float(data[i]);
                    const float bf=std::bit_cast<float>(uint32_t(tc::gguf::float_to_bf16_rne(v))<<16);
                    const auto expected=dtype=="fp16"?data[i]:tc::gguf::float_to_fp16_rne(bf);
                    if(static_cast<const uint16_t*>(y.data())[i]!=expected)++wrong;
                }
                std::cout<<"CAST dtype="<<dtype<<" compiled=1 oracle_mismatches="<<wrong<<std::endl;
                if(wrong)return 1;
            } catch(const CapabilityError &e) {
                std::cout<<"CAST dtype="<<dtype<<" supported=0 reason="<<e.what()<<std::endl;
                if(dtype=="fp16" || !Program::healthy())return 1;
            }
        }
        if(!probe_bf16_emulation(device,std::filesystem::path(argv[1]),timeline))return 1;
        std::cout<<"scope=actual compiler/driver capability, not native BF16 arithmetic, a model route or performance claim\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
