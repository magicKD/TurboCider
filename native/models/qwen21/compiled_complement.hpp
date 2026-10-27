#pragma once
#include "../../backends/mlx.hpp"
#include "../../backends/dense_gpu_projection.hpp"

namespace tc::qwen21::compiled_complement {
inline void admit(const std::vector<Tensor>& a) {
    require(a.size()==3,"explicit X/gate_up/down inputs required");
    require(a[0].shape()==mx::Shape{1,1024,4096} &&
        a[1].shape()==mx::Shape{24576,4096} && a[2].shape()==mx::Shape{4096,12288},
        "fixed original base512 complement geometry required");
    for(const auto& x:a)require(x.dtype()==mx::bfloat16,"original BF16 source required");
}
inline std::vector<Tensor> body(const std::vector<Tensor>& a) {
    // Same row slices and BF16 stages as pipeline.cpp ordinary_gpu callback.
    auto g=mx::matmul(a[0],mx::transpose(slice_axis(a[1],0,0,6144)));
    auto u=mx::matmul(a[0],mx::transpose(slice_axis(a[1],0,12288,18432)));
    auto hidden=silu(g)*u;
    auto partial=dense_gpu::projection_range(hidden,a[2],0,4096,0,6144,32,true);
    return {partial,hidden};
}
}
