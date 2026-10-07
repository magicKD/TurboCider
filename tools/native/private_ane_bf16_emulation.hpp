#pragma once
// Diagnostic graph only. Native BF16 cast support is tested separately.
#include <cmath>
#include <iomanip>
#include <sstream>
#include "../../native/backends/private/ane_mil_round.hpp"
#include "../../native/backends/private/ane_mil_round_compact.hpp"

inline bool probe_bf16_emulation(tc::ane::private_api::Device &device,
    const std::filesystem::path &cache,uint64_t &timeline,bool split_powers=false,bool ties_even=false,
    bool compact=false,bool numeric_policy=false,bool compact_floor=false) {
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
    if(!split_powers)body+=" "+type+" step0 = add(x = zero, y = fp16(1));\n";
    if(split_powers) for(const char *name:{"normalize_a","normalize_b","restore_a","restore_b"})
        body+=" "+type+" "+name+"0 = add(x = zero, y = fp16(1));\n";
    std::string previous="step0";
    for(int exponent=-16;exponent<=15;++exponent) {
        const auto suffix=std::to_string(exponent+17),next="step"+suffix;
        body+=" "+boolean+" bin"+suffix+" = greater_equal(x = magnitude, y = "+power2(exponent)+");\n";
        if(!split_powers)body+=" "+type+" "+next+" = select(cond = bin"+suffix+", a = "+power2(exponent-7)+", b = "+previous+");\n";
        if(split_powers) {
            // No reciprocal and no subnormal multiplier. Each normalization
            // factor is finite normal FP16; the intermediate remains <=256.
            // Restore via two normal factors, retaining the same exact power.
            const int powers[]={std::min(7-exponent,10),std::max(7-exponent-10,0),
                std::max(exponent-7,-14),std::min(exponent+7,0)};
            const char *names[]={"normalize_a","normalize_b","restore_a","restore_b"};
            for(int factor=0;factor<4;++factor) {
                const auto prior=std::string(names[factor])+std::to_string(exponent+16);
                body+=" "+type+" "+names[factor]+suffix+" = select(cond = bin"+suffix+", a = "+
                    power2(powers[factor])+", b = "+prior+");\n";
            }
        }
        previous=next;
    }
    if(split_powers) {
        body+=" "+type+" scaled_first = mul(x = xt, y = normalize_a32);\n";
        body+=" "+type+" scaled = mul(x = scaled_first, y = normalize_b32);\n";
    } else body+=" "+type+" scaled = real_div(x = xt, y = "+previous+");\n";
    if(ties_even) {
        // Do not trust MIL round's lowering: the actual driver rounds ties
        // away on this tuple. Resolve exact binary fractions and parity.
        body+=" "+type+" scaled_abs = abs(x = scaled);\n";
        body+=" "+type+" lower = floor(x = scaled_abs);\n";
        body+=" "+type+" fraction = sub(x = scaled_abs, y = lower);\n";
        body+=" "+type+" lower_half = mul(x = lower, y = fp16(0.5));\n";
        body+=" "+type+" parity_floor = floor(x = lower_half);\n";
        body+=" "+type+" even_lower = add(x = parity_floor, y = parity_floor);\n";
        body+=" "+type+" parity = sub(x = lower, y = even_lower);\n";
        body+=" "+boolean+" above = greater(x = fraction, y = fp16(0.5));\n";
        body+=" "+boolean+" tie = equal(x = fraction, y = fp16(0.5));\n";
        body+=" "+type+" above_add = select(cond = above, a = fp16(1), b = zero);\n";
        body+=" "+type+" tie_add = select(cond = tie, a = parity, b = zero);\n";
        body+=" "+type+" round_first = add(x = lower, y = above_add);\n";
        body+=" "+type+" round_abs = add(x = round_first, y = tie_add);\n";
        body+=" "+type+" round_negative = mul(x = round_abs, y = fp16(-1));\n";
        body+=" "+boolean+" negative = less(x = scaled, y = fp16(0));\n";
        body+=" "+type+" rounded = select(cond = negative, a = round_negative, b = round_abs);\n";
    } else body+=" "+type+" rounded = round(x = scaled);\n";
    if(split_powers) {
        body+=" "+type+" restored_first = mul(x = rounded, y = restore_a32);\n";
        body+=" "+type+" restored = mul(x = restored_first, y = restore_b32);\n";
    } else body+=" "+type+" restored = mul(x = rounded, y = "+previous+");\n";
    body+=" "+boolean+" tiny = less(x = magnitude, y = "+power2(-16)+");\n";
    body+=" "+type+" h = select(cond = tiny, a = xt, b = restored);\n";
    if(ties_even || compact) {
        body.clear();
        const auto value=compact?tc::ane::private_api::emit_bf16_value_round_compact(body,"xt","diagnostic_bf16_value",shape,!compact_floor):
            tc::ane::private_api::emit_bf16_value_round(body,"xt","diagnostic_bf16_value",shape);
        body+=" "+type+" h = mul(x = "+value+", y = fp16(1));\n";
    }
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
        Program model(device,mil,{},cache/(compact?(compact_floor?"fp16-bf16-value-compact-floor-v2":"fp16-bf16-value-compact-magic-v1"):
            ties_even?"fp16-bf16-value-native-emitter-v6":split_powers?"fp16-bf16-rne-split-powers-v3":"fp16-bf16-rne-bins-v1"));
        size_t wrong=0,calls=0,zero_sign_mismatches=0;
        std::array<size_t,32> mismatches_by_exponent{};
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
                    ++mismatches_by_exponent[(input[i]>>10)&31];
                    if((actual&0x7fff)==0 && (expected&0x7fff)==0)++zero_sign_mismatches;
                    ++wrong;
                }
            }
        }
        std::cout<<"EMULATION recipe="<<(compact?(compact_floor?"compact-floor-v2":"compact-magic-v1"):ties_even?"native-value-emitter-v6":split_powers?"split-powers-v3":"divide-v1")<<" compiled=1 finite_half_cases="<<pairs.size()<<" half_overflow_excluded="<<overflow_excluded
            <<" driver_calls="<<calls<<" bf16_rne_oracle_mismatches="<<wrong
            <<" numeric_mismatches="<<(wrong-zero_sign_mismatches)<<" zero_sign_mismatches="<<zero_sign_mismatches
            <<" declared_policy="<<(numeric_policy?"numeric-canonical-zero":"strict-bits")<<std::endl;
        for(size_t exponent=0;exponent<mismatches_by_exponent.size();++exponent)
            if(mismatches_by_exponent[exponent])std::cout<<"ERROR_BIN half_exponent="<<exponent<<" mismatches="<<mismatches_by_exponent[exponent]<<std::endl;
        return wrong==0 || (compact && numeric_policy && wrong==zero_sign_mismatches);
    } catch(const CapabilityError &error) {
        std::cout<<"EMULATION supported=0 reason="<<error.what()<<std::endl;
        return !compact && Program::healthy();
    }
}
