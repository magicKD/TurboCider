#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace tc {
struct Float32Comparison {
    double rel_l2=0,cosine=1,max_abs=0,max_norm_error=0;
    bool zero_reference=false;
    bool n1_layer() const { return zero_reference ? max_abs<=1e-7 : rel_l2<=.01 && cosine>=.9995 && max_norm_error<=.05; }
    bool n1_final() const { return zero_reference ? max_abs<=1e-7 : rel_l2<=.03 && cosine>=.999; }
};
struct Float32ComparisonAccumulator {
    double aa=0,bb=0,error=0,dot=0,maximum=0,max_reference=0;
    size_t elements=0;
    void add(std::span<const float> candidate,std::span<const float> reference) {
    if (candidate.empty() || candidate.size()!=reference.size()) throw std::invalid_argument("tensor comparison geometry mismatch");
    for (size_t i=0;i<candidate.size();++i) {
        const double a=candidate[i],b=reference[i];
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::invalid_argument("tensor comparison nonfinite input");
        const double delta=a-b;
        aa+=a*a;bb+=b*b;error+=delta*delta;dot+=a*b;
        maximum=std::max(maximum,std::abs(delta));max_reference=std::max(max_reference,std::abs(b));
    }
    elements+=candidate.size();
    }
    Float32Comparison result() const {
    if (!elements) throw std::invalid_argument("empty tensor comparison");
    return {std::sqrt(error)/std::max(std::sqrt(bb),1e-12),
        aa && bb ? dot/(std::sqrt(aa)*std::sqrt(bb)) : !aa && !bb ? 1. : 0.,
        maximum,maximum/std::max(max_reference,1e-6),!bb};
    }
};
inline Float32Comparison compare_float32(std::span<const float> candidate,std::span<const float> reference) {
    Float32ComparisonAccumulator accumulated;accumulated.add(candidate,reference);return accumulated.result();
}
} // namespace tc
