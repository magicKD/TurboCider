#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tc::qwen21::metal {
namespace mx = mlx::core;

// Qwen image-VAE channel L2 norm, optionally followed by SiLU. One SIMD group
// handles a spatial pixel, consuming its ACTUAL NCHW strides and producing
// compact NHWC for the next convolution. No input compaction, half state,
// changed epsilon, altered gamma, or whole-frame statistics approximation.
// FP32 sum order differs from MLX Reduce: this is not a bitwise claim.
inline mx::array vae_norm_silu(const mx::array &x, const mx::array &gamma, bool activation) {
    if (x.ndim()!=4 || x.shape(0)!=1 || x.dtype()!=mx::float32 ||
        x.shape(1)<=0 || x.shape(1)>4096 || x.shape(2)<=0 || x.shape(3)<=0 ||
        (gamma.dtype()!=mx::bfloat16 && gamma.dtype()!=mx::float32) ||
        gamma.size()!=size_t(x.shape(1)) || x.size()>(1u<<28) ||
        int64_t(x.shape(2))*x.shape(3)>(1u<<24))
        throw std::invalid_argument("Qwen VAE fused norm requires bounded batch1 FP32 NCHW and BF16/F32 channel gamma");
    for (auto stride:x.strides())
        if (stride<0 || stride>std::numeric_limits<int>::max())
            throw std::invalid_argument("Qwen VAE fused norm stride exceeds bounded index width");
    const int c=x.shape(1),h=x.shape(2),w=x.shape(3);
    static auto kernel=mx::fast::metal_kernel("tc_qwen21_vae_norm_silu_nhwc",
        {"x","gamma","root_channels"},{"out"},R"metal(
        uint pixel=thread_position_in_grid.x/32;
        uint lane=thread_index_in_simdgroup;
        if(pixel>=H*W)return;
        // MLX can choose/change the physical strides when a lazy producer is
        // evaluated. Read its execution-time metadata, never a host snapshot.
        size_t origin=(pixel/W)*x_strides[2]+(pixel%W)*x_strides[3];
        float sum=0.0f;
        for(uint c=lane;c<C;c+=32) {
            float v=x[origin+c*x_strides[1]];
            sum+=v*v;
        }
        float denominator=max(metal::precise::sqrt(simd_sum(sum)),1e-12f);
        for(uint c=lane;c<C;c+=32) {
            float value=(x[origin+c*x_strides[1]]/denominator)*root_channels;
            value=value*float(gamma[c]);
            if(ACTIVATION)value=value*(1.0f/(1.0f+metal::precise::exp(-value)));
            out[pixel*C+c]=value;
        }
        )metal", "", false);
    // A tiny owned F32 parameter keeps a sliced BF16 gamma's value/origin
    // explicit. Widening BF16 coefficients is exact; x is never compacted.
    auto coefficients=mx::astype(mx::reshape(gamma,{c}),mx::float32,std::optional<bool>{true});
    auto output=kernel({x,coefficients,mx::array(std::sqrt(float(c)),mx::float32)},
        {{1,h,w,c}},{mx::float32},{h*w*32,1,1},{128,1,1},
        {{"C",c},{"H",h},{"W",w},{"ACTIVATION",activation}}, {},false,{})[0];
    return mx::transpose(output,{0,3,1,2});
}
} // namespace tc::qwen21::metal
