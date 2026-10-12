#pragma once
// Integer FP16/BF16 RNE conversion avoids Metal half subnormal flushing and
// preserves the CPU conversion contract. FP32 scaling has fast math disabled.
namespace tc::ane::gpu {
inline constexpr const char *transfer_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Params { uint rows, cols, source_pitch, target_pitch, dtype, validate_only; float scale; uint row_scale_pitch, scaled, second_scaled, signed_row_scale; };
inline float from_half(ushort h) {
    uint sign = uint(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    if (!e) {
        if (!m) return as_type<float>(sign);
        uint shift = clz(m) - 21;
        return as_type<float>(sign | ((113 - shift) << 23) | ((m << shift & 1023) << 13));
    }
    return as_type<float>(sign | ((e == 31 ? 255 : e + 112) << 23) | (m << 13));
}
inline ushort to_half(float x) {
    uint bits = as_type<uint>(x), sign = (bits >> 16) & 0x8000;
    uint e = (bits >> 23) & 255, m = bits & 0x7fffff;
    if (e == 255) return ushort(sign | 0x7c00 | (m ? 0x0200 : 0));
    if (e > 142) return ushort(sign | 0x7c00);
    if (e >= 113) {
        uint rounded = (bits & 0x7fffffff) + 0xfff + ((bits >> 13) & 1);
        return ushort(sign | ((rounded >> 13) - 0x1c000));
    }
    if (e < 102) return ushort(sign);
    uint shift = 126 - e, value = m | 0x800000;
    uint result = value >> shift, rest = value & ((1u << shift) - 1);
    uint tie = 1u << (shift - 1);
    return ushort(sign | (result + uint(rest > tie || (rest == tie && (result & 1)))));
}
inline ushort to_bfloat(float x) {
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
kernel void tc_ane_upload(device const uchar *src [[buffer(0)]], device ushort *dst [[buffer(1)]],
                          device atomic_uint *status [[buffer(2)]], constant Params &p [[buffer(3)]],
                          uint2 tile [[threadgroup_position_in_grid]], uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup ushort block[32][33];
    for (uint j = t.y; j < 32; j += 8) {
        uint r = tile.x * 32 + j, c = tile.y * 32 + t.x;
        ushort h = 0;
        if (r < p.rows && c < p.cols) {
            device const uchar *at = src + r * p.source_pitch;
            float value = p.dtype == 2 ? ((device const float *)at)[c] :
                p.dtype == 1 ? as_type<float>(uint(((device const ushort *)at)[c]) << 16) :
                from_half(((device const ushort *)at)[c]);
            h = to_half(value * p.scale);
            if (!isfinite(value) || (h & 0x7c00) == 0x7c00) {
                atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); h = 0;
            }
        }
        block[j][t.x] = h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t.y; j < 32; j += 8) {
        uint r = tile.x * 32 + t.x, c = tile.y * 32 + j;
        if (r < p.rows && c < p.cols) dst[c * p.target_pitch / 2 + r] = block[t.x][j];
    }
}
kernel void tc_ane_restore(device const ushort *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                           device atomic_uint *status [[buffer(2)]], constant Params &p [[buffer(3)]],
                           device const uint *failed [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
                           device const ushort *row_scales [[buffer(5)]], device const ushort *token_scales [[buffer(6)]],
                           device const ushort *second_scales [[buffer(7)]],
                           uint2 t [[thread_position_in_threadgroup]]) {
    if (*failed) return; // MUST precede any IOSurface read
    threadgroup ushort block[32][33];
    for (uint j = t.y; j < 32; j += 8) {
        uint r = tile.x * 32 + t.x, c = tile.y * 32 + j;
        ushort h = 0;
        if (r < p.rows && c < p.cols) {
            h = src[c * p.source_pitch / 2 + r];
            if ((h & 0x7c00) == 0x7c00) { atomic_fetch_or_explicit(status, 2u, memory_order_relaxed); h = 0; }
        }
        block[j][t.x] = h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = t.y; j < 32; j += 8) {
        uint r = tile.x * 32 + j, c = tile.y * 32 + t.x;
        if (r < p.rows && c < p.cols) {
            float value = from_half(block[t.x][j]) * p.scale;
            if (p.scaled) {
                float w = from_half(row_scales[c * p.row_scale_pitch / 2]), x = from_half(token_scales[r]);
                if (!isfinite(w) || !isfinite(x) || (!p.signed_row_scale && w <= 0) || x <= 0) {
                    atomic_fetch_or_explicit(status, 2u, memory_order_relaxed); w = x = 1;
                }
                value = (value * w) * x;
            }
            if (p.second_scaled) {
                float x = from_half(second_scales[r]);
                if (!isfinite(x) || x <= 0) { atomic_fetch_or_explicit(status, 2u, memory_order_relaxed); x = 1; }
                value = value * x;
            }
            if (p.dtype == 2) {
                if (!isfinite(value)) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
                if (!p.validate_only) ((device float *)(dst+r*p.target_pitch))[c]=value;
            } else {
                ushort out = p.dtype == 1 ? to_bfloat(value) : to_half(value);
                uint mask = p.dtype == 1 ? 0x7f80 : 0x7c00;
                if ((out & mask) == mask) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
                if (!p.validate_only) ((device ushort *)(dst + r * p.target_pitch))[c] = out;
            }
        }
    }
}
)metal";
}
