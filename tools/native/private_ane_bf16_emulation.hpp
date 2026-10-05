#pragma once
// Diagnostic graph only. Native BF16 cast support is tested separately.
#include <cmath>
#include <iomanip>
#include <sstream>

inline bool probe_bf16_emulation(tc::ane::private_api::Device &device,
    const std::filesystem::path &cache,uint64_t &timeline) {
    using namespace tc::ane;
    using namespace tc::ane::private_api;
    constexpr int columns=4096;
    const std::string shape="[1, 1, 1, 4096]",strides="[4096, 4096, 4096, 1]";
    const std::string type="tensor<fp16, "+shape+">",boolean="tensor<bool, "+shape+">";
    const std::string buffer="tensor_buffer<fp16, shape="+shape+", strides="+strides+", interleave_factors=[1, 1, 1, 1]>";
    auto power2=[](int exponent) {
        std::ostringstream out;out<<std::hexfloat<<std::ldexp(1.f,exponent);
        return "fp16("+out.str()+")";
    };
    // Exact exponent bins, NOT rounded FP16 log2/floor near powers of two.
    // FP16 values below 2^-16 already fit BF16 exactly, including signed zero.
    std::string body=" "+type+" magnitude = abs(x = xt);\n";
    body+=" "+type+" zero = mul(x = magnitude, y = fp16(0));\n";
    body+=" "+type+" step0 = add(x = zero, y = fp16(1));\n";
    std::string previous="step0";
    for(int exponent=-16;exponent<=15;++exponent) {
        const auto suffix=std::to_string(exponent+17),next="step"+suffix;
        body+=" "+boolean+" bin"+suffix+" = greater_equal(x = magnitude, y = "+power2(exponent)+");\n";
        body+=" "+type+" "+next+" = select(cond = bin"+suffix+", a = "+power2(exponent-7)+", b = "+previous+");\n";
        previous=next;
    }
    body+=" "+type+" scaled = real_div(x = xt, y = "+previous+");\n";
    body+=" "+type+" rounded = round(x = scaled);\n";
    body+=" "+type+" restored = mul(x = rounded, y = "+previous+");\n";
    body+=" "+boolean+" tiny = less(x = magnitude, y = "+power2(-16)+");\n";
    body+=" "+type+" h = select(cond = tiny, a = xt, b = restored);\n";
    const auto mil="program(1.3)\n{\n func main_ane<ios18>("+buffer+" x) {\n "+type+
        " xt = tensor_buffer_to_tensor<ios17>(input = x);\n"+body+" "+buffer+
        " y = tensor_to_tensor_buffer<ios17>(input = h, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>("+strides+"));\n } -> (y);\n}\n";
    try {
        Surface x(device,1,columns,Element::FP16),y(device,1,columns,Element::FP16);
        std::vector<std::pair<uint16_t,uint16_t>> pairs;
        int overflow_excluded=0;
        for(uint32_t raw=0;raw<65536;++raw) {
            if((raw&0x7c00)==0x7c00)continue;
            const float value=tc::gguf::fp16_to_float(uint16_t(raw));
            const float bf=std::bit_cast<float>(uint32_t(tc::gguf::float_to_bf16_rne(value))<<16);
            if(std::abs(bf)>65504.f) {++overflow_excluded;continue;}
            pairs.emplace_back(uint16_t(raw),tc::gguf::float_to_fp16_rne(bf));
        }
        Program model(device,mil,{},cache/"fp16-bf16-rne-bins-v1");
        size_t wrong=0,calls=0;
        for(size_t first=0;first<pairs.size();first+=columns) {
            const auto count=std::min<size_t>(columns,pairs.size()-first);
            auto *input=static_cast<uint16_t*>(x.data());
            for(int i=0;i<columns;++i)input[i]=size_t(i)<count?pairs[first+i].first:0;
            std::pair<std::string,Surface> inputs[]{{"x",x}},outputs[]{{"y",y}};
            const auto ready=++timeline,done=++timeline;
            auto job=model.enqueue(inputs,outputs,ready,done);device.signal(ready);
            const auto result=job.finish();if(!result.ok)throw CapabilityError(result.error);
            ++calls;
            for(int i=0;i<columns;++i) {
                const auto expected=size_t(i)<count?pairs[first+i].second:0;
                const auto actual=static_cast<const uint16_t*>(y.data())[i];
                if(actual!=expected) {
                    if(wrong<8)std::cout<<"MISMATCH input="<<input[i]<<" expected="<<expected<<" actual="<<actual<<std::endl;
                    ++wrong;
                }
            }
        }
        std::cout<<"EMULATION compiled=1 finite_half_cases="<<pairs.size()<<" half_overflow_excluded="<<overflow_excluded
            <<" driver_calls="<<calls<<" bf16_rne_oracle_mismatches="<<wrong<<std::endl;
        return wrong==0;
    } catch(const CapabilityError &error) {
        std::cout<<"EMULATION supported=0 reason="<<error.what()<<std::endl;
        return Program::healthy();
    }
}
