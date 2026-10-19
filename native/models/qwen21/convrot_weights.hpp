#pragma once
#include "../../backends/ane_ffn.hpp"

namespace tc::qwen21 {
// These are shallow views of one packed original-code model, not per-layer
// dense reconstruction or a frozen ANE checkpoint. Packing keeps the F32
// row scales; ComfyH256Inverse explicitly describes the published source basis.
inline std::vector<ane::FfnWeight> convrot_ffn_sources(const Weights& weights,const std::string& prefix) {
    require(!weights.has_runtime_loras(),"ConvRot runtime initial source excludes encoder/transformer LoRA");
    auto matrix=[&](const std::string& name,int rows,int cols) {
        require(weights.convrot(name) && !weights.nvfp4(name),"original Comfy ConvRot source required");
        const auto& q=weights.at(name+".weight");const auto& s=weights.at(name+".scales");const auto& b=weights.at(name+".biases");
        require(q.dtype()==mx::uint32 && q.shape()==mx::Shape{rows,cols/4} && s.shape()==mx::Shape{rows,cols/128} &&
            b.shape()==s.shape() && s.dtype()==mx::float32 && b.dtype()==mx::float32,
            "original signed-q+128, exact F32 repeated row scale/bias and group128 geometry required");
        return ane::FfnWeight{q,s,b,128,8,ane::FfnWeight::Transform::ComfyH256Inverse,std::nullopt};
    };
    auto gu=matrix(prefix+"gate_up",24576,4096);
    auto q=mx::split(gu.values,2,0),s=mx::split(*gu.scales,2,0),b=mx::split(*gu.offsets,2,0);
    return {{q[0],s[0],b[0],128,8,ane::FfnWeight::Transform::ComfyH256Inverse,std::nullopt},
        {q[1],s[1],b[1],128,8,ane::FfnWeight::Transform::ComfyH256Inverse,std::nullopt},matrix(prefix+"out",4096,12288)};
}
} // namespace tc::qwen21
