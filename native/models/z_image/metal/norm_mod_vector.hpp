#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace tc::z_metal {
// Internal candidate, called after norm_mod's geometry checks. Four-wide
// memory accesses retain the scalar kernel's per-lane accumulation order.
inline mlx::core::array norm_mod_vector(const mlx::core::array &x,
    const mlx::core::array &weight, const mlx::core::array &mod,
    const mlx::core::array &residual, bool gated, bool precompute_gate) {
    namespace mx = mlx::core;
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_norm_mod_vector_3840", {"x", "w", "mod", "res"}, {"out"}, R"metal(
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;
        uint index = row * 960 + tid;
        threadgroup float sums[32];
        float4 values = float4(reinterpret_cast<device const vec<T,4>*>(x)[index]);
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) sum += values[i] * values[i];
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = lane < 30 ? sums[lane] : 0.0f;
        float inv = metal::precise::rsqrt(simd_sum(sum) / 3840.0f + 1e-5f);
        float4 weights = float4(reinterpret_cast<device const vec<W,4>*>(w)[tid]);
        vec<T,4> modulation = reinterpret_cast<device const vec<T,4>*>(mod)[tid];
        vec<T,4> result;
        vec<T,4> residual;
        if (GATED) residual = reinterpret_cast<device const vec<T,4>*>(res)[index];
        for (uint i = 0; i < 4; ++i) {
            T n = T((values[i] * inv) * weights[i]);
            if (GATED) {
                T gate = PRECOMPUTED ? modulation[i] :
                    T(metal::precise::tanh(float(modulation[i])));
                T product = T(float(n) * float(gate));
                result[i] = T(float(residual[i]) + float(product));
            } else {
                T scale = T(1.0f + float(modulation[i]));
                result[i] = T(float(n) * float(scale));
            }
        }
        reinterpret_cast<device vec<T,4>*>(out)[index] = result;
        )metal");
    auto modulation = gated && precompute_gate ? mx::tanh(mod) : mod;
    return kernel({x,weight,modulation,residual}, {x.shape()}, {x.dtype()},
                  {x.shape(1)*960,1,1}, {960,1,1},
                  {{"T",x.dtype()},{"W",weight.dtype()},{"GATED",gated},
                   {"PRECOMPUTED",precompute_gate}}, {}, false, {})[0];
}
} // namespace tc::z_metal
