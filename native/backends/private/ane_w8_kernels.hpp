#pragma once
// Decoder math follows the pinned GGML layouts in native/core/gguf_decode.cpp.
// No full dense intermediate: two decode/rotate passes, <=512 floats/group.
namespace tc::ane::private_api {
inline constexpr const char *w8_source = R"metal(
// Immutable per-pipeline format/block metadata, never checkpoint values.
// The generic pipeline remains available for same-library ablation.
constant bool tc_w8_specialize [[function_constant(0)]];
constant uint tc_w8_encoding [[function_constant(1)]];
constant uint tc_w8_dtype [[function_constant(2)]];
constant uint tc_w8_block [[function_constant(3)]];
struct W8Params {
    uint source_pitch, physical_cols, encoding, dtype, group_size;
    uint meta_pitch, meta_dtype, offset_pitch, offset_dtype, has_offset;
    uint row_begin, rows, column_begin, columns, block, code_pitch, scale_pitch, transpose;
    uint source_aligned, seed_low, seed_high;
    float norm;
};
inline uint w8_u16(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint w8_u32(device const uchar *p) { return w8_u16(p) | (w8_u16(p + 2) << 16); }
inline float w8_float(device const uchar *p, uint dtype) {
    return dtype == 2 ? as_type<float>(w8_u32(p)) : dtype == 1 ? as_type<float>(w8_u16(p) << 16) : from_half(ushort(w8_u16(p)));
}
inline float w8_decode(device const uchar *src, device const uchar *scales, device const uchar *offsets,
                        uint row, uint col, constant W8Params &p) {
    device const uchar *a = src + ulong(row) * p.source_pitch;
    const uint encoding = tc_w8_specialize ? tc_w8_encoding : p.encoding;
    const uint dtype = tc_w8_specialize ? tc_w8_dtype : p.dtype;
    if (!encoding) {
        // Only the specialized path uses typed loads, and only after the
        // host checked BOTH the binding offset and physical row pitch.
        // Keep byte decoding for unaligned views and the generic oracle;
        // half subnormals still use our exact integer FP16 conversion.
        if (tc_w8_specialize && p.source_aligned) {
            if (dtype == 2) return ((device const float *)a)[col];
            ushort bits = ((device const ushort *)a)[col];
            return dtype == 1 ? as_type<float>(uint(bits) << 16) : from_half(bits);
        }
        return w8_float(a + ulong(col) * (dtype == 2 ? 4 : 2), dtype);
    }
    if (encoding == 1 || encoding == 2) {
        uint bits = encoding == 1 ? 4 : 8;
        uint code = (w8_u32(a + (col / (32 / bits)) * 4) >> ((col % (32 / bits)) * bits)) & ((1u << bits) - 1);
        uint group = col / p.group_size;
        float scale = w8_float(scales + ulong(row) * p.meta_pitch + group * (p.meta_dtype == 2 ? 4 : 2), p.meta_dtype);
        float value = float(code) * scale;
        if (p.has_offset) value = value + w8_float(offsets + ulong(row) * p.offset_pitch + group * (p.offset_dtype == 2 ? 4 : 2), p.offset_dtype);
        return value;
    }
    if (encoding == 3) { // GGML Q4_0: d, 16 low/high nibbles
        a += (col / 32) * 18; uint i = col % 32;
        uint q = a[2 + i % 16]; return float(int(i < 16 ? q & 15 : q >> 4) - 8) * from_half(ushort(w8_u16(a)));
    }
    if (encoding == 5) { // GGML Q8_0, retain signed -128
        a += (col / 32) * 34; return float(char(a[2 + col % 32])) * from_half(ushort(w8_u16(a)));
    }
    if (encoding == 4) { // GGML Q4_K, 256 values / 144 bytes
        a += (col / 256) * 144; uint i = col % 256, group = i / 32, j = i % 32;
        device const uchar *s = a + 4;
        uint sd = group < 4 ? s[group] & 63 : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
        uint sm = group < 4 ? s[group + 4] & 63 : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
        uint q = a[16 + (group / 2) * 32 + j]; q = group & 1 ? q >> 4 : q & 15;
        float d = from_half(ushort(w8_u16(a))) * float(sd), m = from_half(ushort(w8_u16(a + 2))) * float(sm);
        return d * float(q) - m;
    }
    // GGML Q6_K, 256 values / 210 bytes
    a += (col / 256) * 210; uint i = col % 256, h = i / 128, group = (i % 128) / 32, j = i % 32;
    uint low = a[h * 64 + j + (group & 1) * 32], high = a[128 + h * 32 + j];
    int code = int((group < 2 ? low & 15 : low >> 4) | (((high >> (2 * group)) & 3) << 4)) - 32;
    float d = from_half(ushort(w8_u16(a + 208))) * float(char(a[192 + h * 8 + j / 16 + group * 2]));
    return d * float(code);
}
// The exact recipe's +/-1 signs are immutable 2KiB CPU-generated metadata.
// Never recompute the same 64-bit hash for every value/rotation block on GPU.
inline float w8_rotate(float value, threadgroup float *v, uint lane, constant W8Params &p,
                       constant float *signs) {
    value = value * signs[lane];
    const uint block = tc_w8_specialize ? tc_w8_block : p.block;
    for (uint span = 1; span < block; span <<= 1) {
        float other;
        if (span < 32) {
            // Apple GPU SIMD width is checked before dispatch. Register
            // shuffles preserve the exact butterfly operation/order while
            // removing ten threadgroup barriers per rotation block.
            other = simd_shuffle_xor(value, span);
        } else {
            v[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            other = v[lane ^ span];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = (lane & span) ? other - value : value + other;
    }
    return value * p.norm;
}
kernel void tc_ane_w8_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]], constant float *signs [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float v[512]; float peak = 0;
    const uint block = tc_w8_specialize ? tc_w8_block : p.block;
    for (uint col = 0; col < p.columns; col += block) {
        float x = w8_decode(src, scales, offsets, p.row_begin + row, p.column_begin + col + lane, p);
        if (!isfinite(x)) { atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); x = 0; }
        float rotated = w8_rotate(x, v, lane, p, signs);
        if (!isfinite(rotated)) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        peak = max(peak, abs(rotated));
    }
    peak = simd_max(peak);
    if (!(lane % 32)) v[lane / 32] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        float total = v[0];
        for (uint warp = 1; warp < block / 32; ++warp) total = max(total, v[warp]);
        float norm = total == 0 ? 128.f : max((total / 127.f) * 128.f, 0x1p-24f);
        ushort out = to_half(norm);
        if (!out || (out & 0x7c00) == 0x7c00) { atomic_fetch_or_explicit(status, 4u, memory_order_relaxed); out = 0x5800; }
        dst[row * p.scale_pitch / 2] = out;
    }
}
kernel void tc_ane_w8_codes(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device const ushort *row_scales [[buffer(3)]], device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant W8Params &p [[buffer(6)]], constant float *signs [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint block = tc_w8_specialize ? tc_w8_block : p.block;
    threadgroup float v[512]; uint row = tile.x, col = tile.y * block + lane;
    float x = w8_decode(src, scales, offsets, p.row_begin + row, p.column_begin + col, p);
    if (!isfinite(x)) { atomic_fetch_or_explicit(status,1u,memory_order_relaxed); x=0; }
    float rotated = w8_rotate(x, v, lane, p, signs);
    if (!isfinite(rotated)) atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    float scale = from_half(row_scales[row * p.scale_pitch / 2]);
    if (!(scale>0) || !isfinite(scale)) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    float q = clamp((rotated / scale) * 128.f, -127.f, 127.f);
    if (!isfinite(q)) { atomic_fetch_or_explicit(status, 8u, memory_order_relaxed); q = 0; }
    dst[p.transpose ? ulong(col) * p.code_pitch + row : ulong(row) * p.code_pitch + col] = char(rint(q));
}
// Only compact FP16 scale metadata is retained, never a W8/dense matrix.
kernel void tc_ane_w8_scale_copy(device const ushort *src [[buffer(0)]],device ushort *dst [[buffer(1)]],
    constant uint3 &p [[buffer(2)]],uint row [[thread_position_in_grid]]) {
    if(row<p.x)dst[ulong(row)*p.z]=src[ulong(row)*p.y];
}
)metal";
}
