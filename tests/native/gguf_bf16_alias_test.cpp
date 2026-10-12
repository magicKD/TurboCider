#include "gguf_decode.hpp"
#include <bit>
#include <cstring>
#include <vector>
#include <iostream>
using namespace tc::gguf;
void check(bool value,const char *reason) {if (!value) throw std::runtime_error(reason);}
int main() {
    try {
        for (uint32_t bits=0;bits<=65535;++bits) {
            const float value=std::bit_cast<float>(bits<<16);
            bool valid=true;uint16_t expected=0;
            try {expected=float_to_fp16_rne(value);} catch (const std::exception &) {valid=false;}
            for (bool simd:{false,true}) {
                std::vector<std::byte> data(18);
                for (size_t i=0;i<9;++i) {const uint16_t source=uint16_t(bits);std::memcpy(data.data()+i*2,&source,2);}
                bool accepted=true;
                try {bf16_to_fp16_inplace(data,nullptr,{simd});} catch (const std::exception &) {accepted=false;}
                check(accepted==valid,"alias finite/overflow verdict changed");
                if (valid) for (size_t i=0;i<9;++i) {uint16_t actual;std::memcpy(&actual,data.data()+i*2,2);check(actual==expected,"alias RNE bit pattern changed");}
            }
        }
        std::vector<std::byte> data(16);std::atomic<bool> stop{true};bool rejected=false;
        try {bf16_to_fp16_inplace(data,&stop);} catch (...) {rejected=true;}
        check(rejected,"cancelled alias published");
        rejected=false;try {bf16_to_fp16_inplace({data.data(),3});} catch (...) {rejected=true;}
        check(rejected,"odd alias span accepted");
        std::cout<<"PASS BF16 alias: exhaustive 65536 scalar/SIMD cases, tail, finite/overflow and cancellation\n";
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
