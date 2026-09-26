#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>
#include <cstdlib>
#include "norm_mod_vector.hpp"

namespace tc::z_metal {
namespace mx = mlx::core;

// RMSNorm + AdaLN scale, or RMSNorm + tanh gate + residual. Use the same
// four-consecutive-elements-per-lane reduction as MLX's single-row RMSNorm.
// All intermediate activation conversions are explicit: fusion must not
// silently change BF16 multiplication/addition into a single FP32 expression.
inline mx::array norm_mod(const mx::array &x, const mx::array &weight,
                           const mx::array &mod, const mx::array &residual,
                           bool gated, bool precompute_gate = false,
                           bool vectorized = false) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(2) != 3840 ||
        weight.shape() != mx::Shape{3840} || mod.size() != 3840 ||
        residual.shape() != x.shape() || residual.dtype() != x.dtype() ||
        mod.dtype() != x.dtype())
        throw std::invalid_argument("Z-Image Metal norm/mod geometry mismatch");
    int threads = 960;
    if (const char *value = std::getenv("TURBOCIDER_Z_NORM_THREADS")) {
        threads = std::atoi(value);
        if (threads != 128 && threads != 256 && threads != 512 && threads != 960)
            throw std::invalid_argument("Z-Image norm threads must be 128/256/512/960");
    }
    if (vectorized) {
        if (threads != 960)
            throw std::invalid_argument("Z-Image vector norm requires 960 threads");
        return norm_mod_vector(x,weight,mod,residual,gated,precompute_gate);
    }
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_norm_mod_3840", {"x", "w", "mod", "res"}, {"out"}, R"metal(
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;
        threadgroup float sums[32];
        constexpr uint Items = ((3840 + Threads * 4 - 1) / (Threads * 4)) * 4;
        float values[Items];
        float sum = 0.0f;
        for (uint i = 0; i < Items; ++i) {
            uint d = (i / 4) * Threads * 4 + tid * 4 + i % 4;
            values[i] = d < 3840 ? float(x[row * 3840 + d]) : 0.0f;
            sum += values[i] * values[i];
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = lane < Threads / 32 ? sums[lane] : 0.0f;
        float inv = metal::precise::rsqrt(simd_sum(sum) / 3840.0f + 1e-5f);
        for (uint i = 0; i < Items; ++i) {
            uint d = (i / 4) * Threads * 4 + tid * 4 + i % 4;
            if (d >= 3840) continue;
            uint index = row * 3840 + d;
            T n = T((values[i] * inv) * float(w[d]));
            if (GATED) {
                T gate = PRECOMPUTED ? T(mod[d]) : T(metal::precise::tanh(float(mod[d])));
                T product = T(float(n) * float(gate));
                out[index] = T(float(res[index]) + float(product));
            } else {
                T scale = T(1.0f + float(mod[d]));
                out[index] = T(float(n) * float(scale));
            }
        }
        )metal");
    auto modulation = gated && precompute_gate ? mx::tanh(mod) : mod;
    return kernel({x, weight, modulation, residual}, {x.shape()}, {x.dtype()},
                  {x.shape(1) * threads, 1, 1}, {threads, 1, 1},
                  {{"T", x.dtype()}, {"GATED", gated}, {"PRECOMPUTED", precompute_gate},
                   {"Threads", threads}}, {}, false, {})[0];
}

} // namespace tc::z_metal
