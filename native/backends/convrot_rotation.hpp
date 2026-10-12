#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::convrot_kernel {
namespace mx = mlx::core;
enum class Rotation { Shared, Simd, SimdQuad, SimdRegister };

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
    // One SIMD group owns all 256 coefficients: each lane keeps eight
    // values, 32 columns apart. The first two digits use lane shuffles;
    // digit 16 crosses two registers and two lanes, digit 64 is entirely
    // register-local. Four independent groups per TG, no shared scratch or
    // barriers. Keep the SAME left-to-right radix-4 expressions as Shared.
    static auto registers = mx::fast::metal_kernel("tc_convrot_h256_simd_register_candidate", {"x", "columns"}, {"out"}, R"metal(
        uint lane = thread_index_in_simdgroup, warp = simdgroup_index_in_threadgroup;
        uint K = uint(columns), group = threadgroup_position_in_grid.x * 4 + warp;
        uint base = threadgroup_position_in_grid.y * K + group * 256 + lane;
        float value[8], next[8];
        for (uint j = 0; j < 8; ++j)
            value[j] = group * 256 < K ? float(x[base + j * 32]) : 0;
        for (uint stride = 1; stride <= 4; stride *= 4) {
            uint digit = (lane / stride) & 3, first = lane - digit * stride;
            for (uint j = 0; j < 8; ++j) {
                float a = simd_shuffle(value[j], first), b = simd_shuffle(value[j], first + stride);
                float c = simd_shuffle(value[j], first + 2 * stride), d = simd_shuffle(value[j], first + 3 * stride);
                switch (digit) {
                    case 0: next[j] = a + b + c - d; break;
                    case 1: next[j] = a + b - c + d; break;
                    case 2: next[j] = a - b + c + d; break;
                    default: next[j] = -a + b + c + d; break;
                }
            }
            for (uint j = 0; j < 8; ++j) value[j] = next[j];
        }
        for (uint j = 0; j < 8; ++j) {
            uint even = j & ~1u, first = lane & 15u;
            uint digit = (lane / 16) + (j & 1u) * 2;
            float a = simd_shuffle(value[even], first), b = simd_shuffle(value[even], first + 16);
            float c = simd_shuffle(value[even + 1], first), d = simd_shuffle(value[even + 1], first + 16);
            switch (digit) {
                case 0: next[j] = a + b + c - d; break;
                case 1: next[j] = a + b - c + d; break;
                case 2: next[j] = a - b + c + d; break;
                default: next[j] = -a + b + c + d; break;
            }
        }
        for (uint j = 0; j < 8; ++j) value[j] = next[j];
        for (uint j = 0; j < 8; ++j) {
            uint first = j & 1u, digit = j / 2;
            float a = value[first], b = value[first + 2], c = value[first + 4], d = value[first + 6];
            float result;
            switch (digit) {
                case 0: result = a + b + c - d; break;
                case 1: result = a + b - c + d; break;
                case 2: result = a - b + c + d; break;
                default: result = -a + b + c + d; break;
            }
            if (group * 256 < K) out[base + j * 32] = T(result * 0.0625f);
        }
    )metal");
    const int columns = x.shape(-1), rows = int(x.size() / columns);
    if (kind == Rotation::SimdRegister)
        return registers({x, mx::array(columns)}, {x.shape()}, {x.dtype()}, {((columns + 1023) / 1024) * 128, rows, 1}, {128, 1, 1},
                         {{"T", x.dtype()}}, {}, false, {})[0];
    if (kind == Rotation::SimdQuad)
        return quad({x, mx::array(columns)}, {x.shape()}, {x.dtype()}, {((columns + 1023) / 1024) * 256, rows, 1}, {256, 1, 1},
                    {{"T", x.dtype()}}, {}, false, {})[0];
    auto &kernel = kind == Rotation::Simd ? simd : shared;
    return kernel({x, mx::array(columns)}, {x.shape()}, {x.dtype()}, {columns, rows, 1}, {256, 1, 1},
                  {{"T", x.dtype()}}, {}, false, {})[0];
}
} // namespace tc::convrot_kernel
