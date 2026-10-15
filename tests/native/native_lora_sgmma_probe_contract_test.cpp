#include "../../tools/native/native_lora_sgmma_probe_contract.hpp"
#include <array>
#include <iostream>
using namespace tc::lora_probe;
int main() try {
    const std::array<uint16_t,4> a{0,0x8000,0x3f80,0xbf80};
    auto b=a;b[0]=0x8000;
    View x{reinterpret_cast<const std::byte*>(a.data()),8,2,2,4,Type::BF16};
    View y{reinterpret_cast<const std::byte*>(b.data()),8,2,2,4,Type::BF16};
    auto require=[](bool v){if(!v)throw std::runtime_error("CPU probe contract failed");};
    const auto same=compare(x,x),sign=compare(x,y);
    require(accepted(same,same,true) && sign.rrms==0 && sign.bit_differences==1 && !accepted(sign,sign,true));
    size_t rejects=0;
    auto reject=[&](auto f){try{f();}catch(const std::exception&){++rejects;return;}throw std::runtime_error("invalid view accepted");};
    reject([&]{auto v=x;v.size=7;compare(v,x);});
    reject([&]{auto v=x;v.pitch=3;compare(v,x);});
    reject([&]{auto v=x;v.data=nullptr;compare(v,x);});
    reject([&]{auto v=x;v.rows=SIZE_MAX;compare(v,x);});
    for(uint16_t invalid:{uint16_t(0x7f80),uint16_t(0xff80),uint16_t(0x7fc0)}) {
        b[0]=invalid;reject([&]{compare(x,y);});
    }
    const std::array<float,2> f{1.0001220703125f,8193.f};
    View z{reinterpret_cast<const std::byte*>(f.data()),8,1,2,8,Type::FP32};
    require(compare(z,z).bit_differences==0);
    require(rejects==7);
    std::cout<<"lora_sgmma_probe_cpu=PASS invalid_cases="<<rejects<<" devices=0 models=0\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
