#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::convrot_kernel {
namespace mx = mlx::core;
enum class Rotation { Shared, Simd, SimdQuad };

// Exact Comfy H4^k ordering, NOT the Sylvester H256 used by ANE staging.
// Both paths retain FP32 radix-4 arithmetic and one final dtype rounding.
inline mx::array rotate(const mx::array &x, Rotation kind) {
    if (!x.ndim() || x.shape(-1) <= 0 || x.shape(-1) % 256 ||
        (x.dtype() != mx::bfloat16 && x.dtype() != mx::float16 && x.dtype() != mx::float32))
        throw std::invalid_argument("ConvRot Metal requires floating-point complete H256 groups");
    static auto shared = mx::fast::metal_kernel("tc_convrot_h256_shared_control", {"x", "columns"}, {"out"}, R"metal(
        threadgroup float current[256]; threadgroup float next[256];
        uint lane = thread_index_in_threadgroup, K = uint(columns);
        uint base = threadgroup_position_in_grid.y * K + threadgroup_position_in_grid.x * 256;
        current[lane] = float(x[base + lane]); threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 256; stride *= 4) {
            uint digit = (lane / stride) & 3, first = lane - digit * stride;
            float a = current[first], b = current[first + stride], c = current[first + 2 * stride], d = current[first + 3 * stride];
            switch (digit) {
                case 0: next[lane] = a + b + c - d; break;
                case 1: next[lane] = a + b - c + d; break;
                case 2: next[lane] = a - b + c + d; break;
                default: next[lane] = -a + b + c + d; break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); current[lane] = next[lane];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        out[base + lane] = T(current[lane] * 0.0625f);
    )metal");
    static auto simd = mx::fast::metal_kernel("tc_convrot_h256_simd_candidate", {"x", "columns"}, {"out"}, R"metal(
        threadgroup float scratch[256];
        uint lane = thread_index_in_threadgroup, K = uint(columns);
        uint base = threadgroup_position_in_grid.y * K + threadgroup_position_in_grid.x * 256;
        float value = float(x[base + lane]);
        for (uint stride = 1; stride < 256; stride *= 4) {
            uint digit = (lane / stride) & 3, first = lane - digit * stride;
            float a, b, c, d;
            if (stride < 8) {
                uint local = first & 31;
                a = simd_shuffle(value, local); b = simd_shuffle(value, local + stride);
                c = simd_shuffle(value, local + 2 * stride); d = simd_shuffle(value, local + 3 * stride);
            } else {
                scratch[lane] = value; threadgroup_barrier(mem_flags::mem_threadgroup);
                a = scratch[first]; b = scratch[first + stride]; c = scratch[first + 2 * stride]; d = scratch[first + 3 * stride];
            }
            switch (digit) {
                case 0: value = a + b + c - d; break;
                case 1: value = a + b - c + d; break;
                case 2: value = a - b + c + d; break;
                default: value = -a + b + c + d; break;
            }
            // Protect stride16 reads before stride64 writes. No final
            // scratch reuse, and the first two stages stay in SIMD registers.
            if (stride == 16) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        out[base + lane] = T(value * 0.0625f);
    )metal");
    static auto quad = mx::fast::metal_kernel("tc_convrot_h256_simd_quad_candidate", {"x", "columns"}, {"out"}, R"metal(
        threadgroup float4 scratch[256];
        uint lane = thread_index_in_threadgroup, K = uint(columns);
        uint row = threadgroup_position_in_grid.y * K;
        uint offset = threadgroup_position_in_grid.x * 1024 + lane;
        float4 value = float4(0);
        for (uint group = 0; group < 4; ++group)
            if (offset + group * 256 < K) value[group] = float(x[row + offset + group * 256]);
        for (uint stride = 1; stride < 256; stride *= 4) {
            uint digit = (lane / stride) & 3, first = lane - digit * stride;
            float4 a, b, c, d;
            if (stride < 8) {
                uint local = first & 31;
                a = simd_shuffle(value, local); b = simd_shuffle(value, local + stride);
                c = simd_shuffle(value, local + 2 * stride); d = simd_shuffle(value, local + 3 * stride);
            } else {
                scratch[lane] = value; threadgroup_barrier(mem_flags::mem_threadgroup);
                a = scratch[first]; b = scratch[first + stride]; c = scratch[first + 2 * stride]; d = scratch[first + 3 * stride];
            }
            switch (digit) {
                case 0: value = a + b + c - d; break;
                case 1: value = a + b - c + d; break;
                case 2: value = a - b + c + d; break;
                default: value = -a + b + c + d; break;
            }
            if (stride == 16) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint group = 0; group < 4; ++group)
            if (offset + group * 256 < K) out[row + offset + group * 256] = T(value[group] * 0.0625f);
    )metal");
    const int columns = x.shape(-1), rows = int(x.size() / columns);
    if (kind == Rotation::SimdQuad)
        return quad({x, mx::array(columns)}, {x.shape()}, {x.dtype()}, {((columns + 1023) / 1024) * 256, rows, 1}, {256, 1, 1},
                    {{"T", x.dtype()}}, {}, false, {})[0];
    auto &kernel = kind == Rotation::Simd ? simd : shared;
    return kernel({x, mx::array(columns)}, {x.shape()}, {x.dtype()}, {columns, rows, 1}, {256, 1, 1},
                  {{"T", x.dtype()}}, {}, false, {})[0];
}
} // namespace tc::convrot_kernel
