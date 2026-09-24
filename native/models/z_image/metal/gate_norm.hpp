#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// Combine the attention residual epilogue and the following FFN input norm.
// Keep both outputs: the residual is needed again after the FFN projection.
// The 960-thread reduction and explicit BF16 boundaries match norm_mod.
inline std::vector<mx::array> gate_norm(const mx::array &x, const mx::array &residual,
    const mx::array &gate_weight, const mx::array &gate_mod,
    const mx::array &scale_weight, const mx::array &scale_mod) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        residual.shape() != x.shape() || residual.dtype() != x.dtype() ||
        gate_weight.shape() != mx::Shape{3840} || scale_weight.shape() != mx::Shape{3840} ||
        gate_mod.size() != 3840 || scale_mod.size() != 3840 ||
        gate_mod.dtype() != x.dtype() || scale_mod.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image gate/norm geometry mismatch");
    const bool precomputed = x.dtype() != mx::float32;
    auto gate = precomputed ? mx::tanh(gate_mod) : gate_mod;
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_gate_norm_3840", {"x", "res", "gw", "gate", "sw", "scale"},
        {"value", "feed"}, R"metal(
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;
        threadgroup float sums[32];
        float values[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            values[i] = float(x[row*3840 + tid*4+i]);
            sum += values[i]*values[i];
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = lane < 30 ? sums[lane] : 0.0f;
        float inv = metal::precise::rsqrt(simd_sum(sum)/3840.0f+1e-5f);
        // No warp may overwrite the shared sums until every warp has read them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint d = tid*4+i, index = row*3840+d;
            T normalized = T((values[i]*inv)*float(gw[d]));
            T g = PRECOMPUTED ? T(gate[d]) : T(metal::precise::tanh(float(gate[d])));
            T product = T(float(normalized)*float(g));
            T v = T(float(res[index])+float(product));
            value[index] = v;
            values[i] = float(v);
            sum += values[i]*values[i];
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = lane < 30 ? sums[lane] : 0.0f;
        inv = metal::precise::rsqrt(simd_sum(sum)/3840.0f+1e-5f);
        for (uint i = 0; i < 4; ++i) {
            uint d = tid*4+i;
            T normalized = T((values[i]*inv)*float(sw[d]));
            T factor = T(1.0f+float(scale[d]));
            feed[row*3840+d] = T(float(normalized)*float(factor));
        }
        )metal");
    return kernel({x,residual,gate_weight,gate,scale_weight,scale_mod},
                  {x.shape(),x.shape()}, {x.dtype(),x.dtype()},
                  {x.shape(1)*960,1,1}, {960,1,1},
                  {{"T",x.dtype()},{"PRECOMPUTED",precomputed}}, {}, false, {});
}
} // namespace tc::z_metal
