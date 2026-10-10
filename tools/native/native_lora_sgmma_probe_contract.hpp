#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tc::lora_probe {
enum class Type { BF16, FP32 };
struct View { const std::byte* data; size_t size,rows,columns,pitch; Type type; };
struct Comparison { double rrms=0,max_abs=0; size_t bit_differences=0,count=0; };
inline void validate(View x) {
    const size_t item=x.type==Type::BF16?2:4;
    if(!x.data || !x.rows || !x.columns || x.columns>SIZE_MAX/item ||
       x.pitch<x.columns*item || x.pitch%item ||
       (x.rows-1)>(SIZE_MAX-x.columns*item)/x.pitch ||
       (x.rows-1)*x.pitch+x.columns*item>x.size)
        throw std::invalid_argument("LoRA probe invalid CPU matrix extent");
}
inline Comparison compare(View a,View b) {
    validate(a);validate(b);
    if(a.rows!=b.rows || a.columns!=b.columns || a.type!=b.type)
        throw std::invalid_argument("LoRA probe comparison shape/dtype differs");
    const size_t item=a.type==Type::BF16?2:4;
    Comparison out;double difference=0,energy=0;
    for(size_t r=0;r<a.rows;++r)for(size_t c=0;c<a.columns;++c) {
        uint32_t ab=0,bb=0;
        std::memcpy(&ab,a.data+r*a.pitch+c*item,item);
        std::memcpy(&bb,b.data+r*b.pitch+c*item,item);
        const float av=std::bit_cast<float>(item==2?ab<<16:ab),bv=std::bit_cast<float>(item==2?bb<<16:bb);
        if(!std::isfinite(av)||!std::isfinite(bv))throw std::runtime_error("LoRA probe nonfinite result");
        const double d=double(av)-bv;difference+=d*d;energy+=double(bv)*bv;
        out.max_abs=std::max(out.max_abs,std::abs(d));out.bit_differences+=ab!=bb;++out.count;
    }
    out.rrms=energy?std::sqrt(difference/energy):(difference?std::numeric_limits<double>::infinity():0);
    return out;
}
inline bool accepted(const Comparison& low,const Comparison& output,bool exact) {
    return std::isfinite(low.rrms) && std::isfinite(output.rrms) &&
        (exact ? low.bit_differences==0 && output.bit_differences==0 : low.rrms<=1e-4 && output.rrms<=1e-3);
}
} // namespace tc::lora_probe
