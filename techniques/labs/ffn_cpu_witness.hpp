// Original independent BF16-source/full-K,F FP64 sample oracle. No device API.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace tc_witness {
inline double bf16(uint16_t code) {
    uint32_t bits=uint32_t(code)<<16; float value=0; std::memcpy(&value,&bits,4);
    if(!std::isfinite(value))throw std::invalid_argument("nonfinite original BF16 source");
    return double(value);
}
struct Result {
    std::array<size_t,4> rows,cols;
    std::array<double,16> output{};
    double gate_min=0,gate_max=0; size_t gate_elements=0,gate_abs_over_11=0;
};
inline Result selected(std::span<const uint16_t> x,std::span<const uint16_t> gu,std::span<const uint16_t> down,
                       size_t rows,size_t hidden,size_t width,size_t first,size_t count) {
    if(rows<4 || rows>1024 || hidden<4 || hidden>4096 || width<1 || width>12288 || !count || first>width-count || count>width ||
       x.size()!=rows*hidden || gu.size()!=2*width*hidden || down.size()!=hidden*width)
        throw std::invalid_argument("bounded complete BF16 witness geometry required");
    Result result; result.rows={0,rows/3,2*rows/3,rows-1}; result.cols={0,hidden/3,2*hidden/3,hidden-1};
    bool seen=false;
    for(size_t r=0;r<4;++r)for(size_t j=first;j<first+count;++j) {
        double gate=0,up=0;
        for(size_t k=0;k<hidden;++k) {
            double value=bf16(x[result.rows[r]*hidden+k]);
            gate+=value*bf16(gu[j*hidden+k]); up+=value*bf16(gu[(width+j)*hidden+k]);
        }
        if(!seen){result.gate_min=result.gate_max=gate;seen=true;}
        result.gate_min=std::min(result.gate_min,gate);result.gate_max=std::max(result.gate_max,gate);
        ++result.gate_elements;result.gate_abs_over_11+=std::abs(gate)>11;
        double small=std::exp(-std::abs(gate)); double sigmoid=gate>=0?1/(1+small):small/(1+small);
        double intermediate=(gate*sigmoid)*up;
        for(size_t c=0;c<4;++c)result.output[r*4+c]+=intermediate*bf16(down[result.cols[c]*width+j]);
    }
    return result;
}
}
