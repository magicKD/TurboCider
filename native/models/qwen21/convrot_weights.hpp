#pragma once
#include "../../backends/ane_ffn.hpp"

namespace tc::qwen21 {
// Preserve the original signed codes and F32 row scales in either source
// representation. The streamed bank supplies signed rows directly.
inline std::vector<ane::FfnWeight> convrot_ffn_sources(const Weights& weights,const std::string& prefix) {
    require(!weights.has_runtime_loras(),"ConvRot runtime initial source excludes encoder/transformer LoRA");
    auto matrix=[&](const std::string& name,int rows,int cols) {
        require(weights.convrot(name) && !weights.nvfp4(name),"original Comfy ConvRot source required");
        const auto& q=weights.at(name+".weight");
        constexpr auto basis=ane::FfnWeight::Transform::ComfyH256Inverse;
        if (q.dtype()==mx::int8) {
            const auto& s=weights.at(name+".weight_scale");
            require(q.shape()==mx::Shape{rows,cols} && s.shape()==mx::Shape{rows,1} && s.dtype()==mx::float32 &&
                    !weights.has(name+".bias"),"signed ConvRot runtime requires bias-free I8 rows and original F32 row scales");
            return ane::FfnWeight{q,s,std::nullopt,128,8,basis};
        }
        const auto& s=weights.at(name+".scales");const auto& b=weights.at(name+".biases");
        require(q.dtype()==mx::uint32 && q.shape()==mx::Shape{rows,cols/4} && s.shape()==mx::Shape{rows,cols/128} &&
            b.shape()==s.shape() && s.dtype()==mx::float32 && b.dtype()==mx::float32,
            "original signed-q+128, exact F32 repeated row scale/bias and group128 geometry required");
        return ane::FfnWeight{q,s,b,128,8,basis};
    };
    auto gu=matrix(prefix+"gate_up",24576,4096);
    auto half=[&](int first,int last) {
        auto view=gu;
        view.values=slice_axis(gu.values,0,first,last);
        view.scales=slice_axis(*gu.scales,0,first,last);
        if(gu.offsets)view.offsets=slice_axis(*gu.offsets,0,first,last);
        return view;
    };
    return {half(0,12288),half(12288,24576),matrix(prefix+"out",4096,12288)};
}
} // namespace tc::qwen21
