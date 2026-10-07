#pragma once
namespace tc::ane::gpu {
// Experimental A8 path is a SEPARATE library: the default two-pass library
// never parses or compiles this source. Function constants bound private
// retention to 30/32 FP32 values per lane. Device::stage_w8 selects only dense
// full-width transpose H128 activations, not W banks or H512 down staging.
inline constexpr const char *a8_single_pass_source = R"metal(
constant uint tc_a8_columns [[function_constant(0)]];
struct A8Params {
    uint source_pitch, physical_cols, encoding, dtype, group_size;
    uint meta_pitch, meta_dtype, offset_pitch, offset_dtype, has_offset;
    uint row_begin, rows, column_begin, columns, block, code_pitch, scale_pitch, transpose;
    uint source_aligned, seed_low, seed_high, basis, activation_group;
    uint has_column_scale, inverse_column_scale;
    float norm;
};
inline uint a8_u16(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint a8_u32(device const uchar *p) { return a8_u16(p) | (a8_u16(p + 2) << 16); }
inline float a8_dense(device const uchar *at, uint dtype) {
    return dtype == 2 ? as_type<float>(a8_u32(at)) : dtype == 1 ? as_type<float>(a8_u16(at) << 16) : from_half(ushort(a8_u16(at)));
}
inline float a8_rotate(float value, threadgroup float *v, uint lane, constant A8Params &p,
                        constant float *signs) {
    value = value * signs[lane];
    for (uint span = 1; span < 128; span <<= 1) {
        float other;
        if (span < 32) other = simd_shuffle_xor(value, span);
        else {
            v[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            other = v[lane ^ span];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = (lane & span) ? other - value : value + other;
    }
    return value * p.norm;
}
kernel void tc_ane_a8_single_pass(device const uchar *src [[buffer(0)]], device ushort *row_scales [[buffer(3)]],
    device char *dst [[buffer(4)]], device atomic_uint *status [[buffer(5)]], constant A8Params &p [[buffer(6)]],
    constant float *signs [[buffer(7)]], device const float *column_scale [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float v[128];
    threadgroup ushort row_scale;
    thread float rotated[32];
    float peak = 0;
    // All 128 lanes execute every barrier; invalid lanes substitute zero.
    // H3840 has exactly 30 active blocks: its two array tails are never read.
    #pragma unroll
    for (uint block = 0; block < tc_a8_columns / 128; ++block) {
        const uint column = p.column_begin + block * 128 + lane;
        float x = a8_dense(src + ulong(p.row_begin + row) * p.source_pitch + ulong(column) * (p.dtype == 2 ? 4 : 2), p.dtype);
        if (!isfinite(x)) { atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); x = 0; }
        if (p.has_column_scale) {
            const float scale = column_scale[column];
            if (!isfinite(scale) || scale < 0.0625f || scale > 16.f) {
                atomic_fetch_or_explicit(status, 16u, memory_order_relaxed); x = 0;
            } else {
                x = x / scale;
                if (!isfinite(x)) { atomic_fetch_or_explicit(status, 16u, memory_order_relaxed); x = 0; }
            }
        }
        const float value = a8_rotate(x, v, lane, p, signs);
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        rotated[block] = value;
        peak = max(peak, abs(value));
    }
    peak = simd_max(peak);
    if (!(lane % 32)) v[lane / 32] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        float total = v[0];
        for (uint warp = 1; warp < 4; ++warp) total = max(total, v[warp]);
        float norm = total == 0 ? 128.f : max((total / 127.f) * 128.f, 0x1p-24f);
        ushort out = to_half(norm);
        if (!out || (out & 0x7c00) == 0x7c00) { atomic_fetch_or_explicit(status, 4u, memory_order_relaxed); out = 0x5800; }
        row_scales[row * p.scale_pitch / 2] = out;
        row_scale = out;
    }
    // Quantize with the same rounded FP16 scale as the two-pass recipe.
    // TG visibility is explicit; no device-memory read-after-write barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = from_half(row_scale);
    #pragma unroll
    for (uint block = 0; block < tc_a8_columns / 128; ++block) {
        float q = clamp((rotated[block] / scale) * 128.f, -127.f, 127.f);
        if (!isfinite(q)) { atomic_fetch_or_explicit(status, 8u, memory_order_relaxed); q = 0; }
        const uint col = block * 128 + lane;
        dst[ulong(col) * p.code_pitch + row] = char(rint(q));
    }
}
)metal";
}
