#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
// Experimental physical-thread multiplexing. Each virtual SIMD group owns
// the same 128 consecutive channels as the original 960-thread kernel.
// Never accumulate across virtual groups before the final 30-way reduction.
inline mlx::core::array norm_mod_virtual(const mlx::core::array &x,
    const mlx::core::array &w, const mlx::core::array &mod,
    const mlx::core::array &res, bool gated, bool precomputed, int threads) {
    namespace mx = mlx::core;
    if ((threads != 128 && threads != 256 && threads != 512) ||
        x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        w.shape() != mx::Shape{3840} || mod.size() != 3840 ||
        res.shape() != x.shape() || res.dtype() != x.dtype() || mod.dtype() != x.dtype())
        throw std::invalid_argument("invalid virtual norm geometry");
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_norm_mod_virtual", {"x", "w", "mod", "res"}, {"out"}, R"metal(
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x;
        constexpr uint Chunks = (960 + Threads - 1) / Threads;
        threadgroup float sums[32];
        float4 values[Chunks];
        for (uint chunk = 0; chunk < Chunks; ++chunk) {
            uint virtual_tid = chunk * Threads + tid;
            // Uniform within each SIMD group, including the tail groups.
            if (virtual_tid < 960) {
                values[chunk] = float4(reinterpret_cast<device const vec<T,4>*>(x)
                                      [row * 960 + virtual_tid]);
                float sum = 0.0f;
                for (uint i = 0; i < 4; ++i)
                    sum += values[chunk][i] * values[chunk][i];
                sum = simd_sum(sum);
                if (lane == 0) sums[virtual_tid / 32] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = lane < 30 ? sums[lane] : 0.0f;
        float inv = metal::precise::rsqrt(simd_sum(sum) / 3840.0f + 1e-5f);
        for (uint chunk = 0; chunk < Chunks; ++chunk) {
            uint virtual_tid = chunk * Threads + tid;
            if (virtual_tid >= 960) continue;
            uint index = row * 960 + virtual_tid;
            float4 weights = float4(reinterpret_cast<device const vec<W,4>*>(w)[virtual_tid]);
            vec<T,4> modulation = reinterpret_cast<device const vec<T,4>*>(mod)[virtual_tid];
            vec<T,4> residual, result;
            if (GATED) residual = reinterpret_cast<device const vec<T,4>*>(res)[index];
            for (uint i = 0; i < 4; ++i) {
                T n = T((values[chunk][i] * inv) * weights[i]);
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
        }
        )metal");
    auto modulation = gated && precomputed ? mx::tanh(mod) : mod;
    return kernel({x,w,modulation,res}, {x.shape()}, {x.dtype()},
                  {x.shape(1)*threads,1,1}, {threads,1,1},
                  {{"T",x.dtype()},{"W",w.dtype()},{"GATED",gated},
                   {"PRECOMPUTED",precomputed},{"Threads",threads}}, {}, false, {})[0];
}
} // namespace tc::z_metal
