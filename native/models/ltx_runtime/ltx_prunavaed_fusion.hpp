#pragma once

#include <mlx/mlx.h>
#include <limits>
#include <stdexcept>

namespace ltx_prunavaed {
namespace mx = mlx::core;

// Match the MLX 0.32 four-values-per-thread RMS reduction, precise rsqrt,
// and BF16-typed sigmoid arithmetic. Promoting the entire sigmoid to float
// changes many BF16 values even though the real-number formula is identical.
// Reference: ml-explore/mlx v0.32.0, rms_norm.metal and unary_ops.h (MIT).
inline bool can_fuse_norm_silu(const mx::array& input) {
    if (input.ndim() == 0 || input.dtype() != mx::bfloat16) return false;
    const int channels = input.shape(-1);
    if (channels != 64 && channels != 128 && channels != 256 &&
        channels != 384 && channels != 512) return false;
    const size_t threads = ((channels + 127) / 128) * 32;
    return input.size() && input.size() / channels <=
        static_cast<size_t>(std::numeric_limits<int>::max()) / threads;
}

inline mx::array norm_silu(const mx::array& input) {
    if (!can_fuse_norm_silu(input))
        throw std::invalid_argument("unsupported PrunaVAED fused RMS/SiLU shape or dtype");
    static auto kernel = mx::fast::metal_kernel(
        "tc_prunavaed_norm_silu", {"inp"}, {"out"}, R"MSL(
const uint row = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint group = simdgroup_index_in_threadgroup;
threadgroup float partial[32];
threadgroup float inverse[1];
float values[4];
float sum = 0.0f;
for (uint j = 0; j < 4; ++j) {
    const uint col = tid * 4 + j;
    values[j] = col < C ? float(inp[size_t(row) * C + col]) : 0.0f;
    sum += values[j] * values[j];
}
sum = simd_sum(sum);
if (group == 0) partial[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) partial[group] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0) {
    const float total = simd_sum(partial[lane]);
    if (lane == 0) inverse[0] = metal::precise::rsqrt(total / float(C) + 1.0e-8f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < 4; ++j) {
    const uint col = tid * 4 + j;
    if (col < C) {
        const bfloat normalized = bfloat(values[j] * inverse[0]);
        out[size_t(row) * C + col] = normalized * tc_pruna_sigmoid(normalized);
    }
}
)MSL", R"MSL(
inline bfloat tc_pruna_sigmoid(bfloat x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
)MSL");
    const int channels = input.shape(-1);
    const int threads = ((channels + 127) / 128) * 32;
    const int rows = static_cast<int>(input.size() / channels);
    return kernel({input}, {input.shape()}, {input.dtype()},
                  {rows * threads, 1, 1}, {threads, 1, 1}, {{"C", channels}},
                  std::nullopt, false, {})[0];
}
} // namespace ltx_prunavaed
