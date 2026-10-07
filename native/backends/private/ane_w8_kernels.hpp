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
    uint source_aligned, seed_low, seed_high, basis, activation_group;
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
    if (p.basis == 1) {
        // Comfy H4^4, no random signs and NOT Sylvester ordering. Preserve
        // the GPU ConvRot radix-4 arithmetic and original dtype boundary.
        for (uint stride = 1; stride < 256; stride *= 4) {
            uint digit = (lane / stride) & 3, first = lane - digit * stride;
            float a,b,c,d;
            if (stride < 8) {
                uint local = first & 31;
                a=simd_shuffle(value,local); b=simd_shuffle(value,local+stride);
                c=simd_shuffle(value,local+2*stride); d=simd_shuffle(value,local+3*stride);
            } else {
                v[lane]=value; threadgroup_barrier(mem_flags::mem_threadgroup);
                a=v[first]; b=v[first+stride]; c=v[first+2*stride]; d=v[first+3*stride];
            }
            switch (digit) {
                case 0: value=a+b+c-d; break;
                case 1: value=a+b-c+d; break;
                case 2: value=a-b+c+d; break;
                default: value=-a+b+c+d; break;
            }
            // The scale pass reuses this scratch for the NEXT H256 group.
            if (stride >= 16) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value *= .0625f;
        return p.dtype == 0 ? from_half(to_half(value)) :
            p.dtype == 1 ? as_type<float>(uint(to_bfloat(value)) << 16) : value;
    }
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
// Dense/affine/raw-GGUF Sylvester H128/H512: one SIMD owns the block. Preserve original
// butterfly FP32 add/sub order; high levels exchange private registers.
inline void w8_sylvester_register_load(device const uchar *src,device const uchar *scales,
    device const uchar *offsets,device atomic_uint *status,constant W8Params &p,
    constant float *signs,uint row,uint col,uint lane,thread float (&value)[16]) {
    const uint count=tc_w8_block/32;
    for(uint j=0;j<count;++j) {
        float x=w8_decode(src,scales,offsets,p.row_begin+row,p.column_begin+col+lane+j*32,p);
        if(!isfinite(x)){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);x=0;}
        value[j]=x*signs[lane+j*32];
    }
    for(uint span=1;span<32;span<<=1)for(uint j=0;j<count;++j) {
        float other=simd_shuffle_xor(value[j],span);
        value[j]=(lane&span)?other-value[j]:value[j]+other;
    }
    float next[16];
    for(uint span=1;span<count;span<<=1) {
        for(uint j=0;j<count;++j) {
            float other=value[j^span];next[j]=(j&span)?other-value[j]:value[j]+other;
        }
        for(uint j=0;j<count;++j)value[j]=next[j];
    }
    for(uint j=0;j<count;++j) {
        value[j]*=p.norm;
        if(!isfinite(value[j]))atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    }
}
kernel void tc_ane_sylvester_register_scales(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device ushort *dst [[buffer(3)]],device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]],constant float *signs [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float peak=0;
    for(uint col=0;col<p.columns;col+=tc_w8_block) {
        float value[16];w8_sylvester_register_load(src,scales,offsets,status,p,signs,row,col,lane,value);
        for(uint j=0;j<tc_w8_block/32;++j)peak=max(peak,abs(value[j]));
    }
    peak=simd_max(peak);
    if(!lane) {
        float norm=peak==0?128.f:max((peak/127.f)*128.f,0x1p-24f);
        ushort out=to_half(norm);
        if(!out || (out&0x7c00)==0x7c00){atomic_fetch_or_explicit(status,4u,memory_order_relaxed);out=0x5800;}
        dst[ulong(row)*p.scale_pitch/2]=out;
    }
}
kernel void tc_ane_sylvester_register_codes(device const uchar *src [[buffer(0)]],device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]],device const ushort *row_scales [[buffer(3)]],device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],constant W8Params &p [[buffer(6)]],constant float *signs [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float value[16];w8_sylvester_register_load(src,scales,offsets,status,p,signs,tile.x,tile.y*tc_w8_block,lane,value);
    float scale=from_half(row_scales[ulong(tile.x)*p.scale_pitch/2]);
    if(!(scale>0) || !isfinite(scale))atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    for(uint j=0;j<tc_w8_block/32;++j) {
        uint col=tile.y*tc_w8_block+lane+j*32;float q=clamp((value[j]/scale)*128.f,-127.f,127.f);
        if(!isfinite(q)){atomic_fetch_or_explicit(status,8u,memory_order_relaxed);q=0;}
        dst[p.transpose?ulong(col)*p.code_pitch+tile.x:ulong(tile.x)*p.code_pitch+col]=char(rint(q));
    }
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
    float scale = from_half(row_scales[p.activation_group ? ulong(tile.y)*p.scale_pitch/2+row : ulong(row)*p.scale_pitch/2]);
    if (!(scale>0) || !isfinite(scale)) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    float q = clamp((rotated / scale) * 128.f, -127.f, 127.f);
    if (!isfinite(q)) { atomic_fetch_or_explicit(status, 8u, memory_order_relaxed); q = 0; }
    dst[p.transpose ? ulong(col) * p.code_pitch + row : ulong(row) * p.code_pitch + col] = char(rint(q));
}
kernel void tc_ane_w8_group_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]], constant float *signs [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float v[512];
    float x=w8_decode(src,scales,offsets,p.row_begin+tile.x,p.column_begin+tile.y*256+lane,p);
    if(!isfinite(x)) {atomic_fetch_or_explicit(status,1u,memory_order_relaxed);x=0;}
    float rotated=w8_rotate(x,v,lane,p,signs);
    if(!isfinite(rotated))atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    float peak=simd_max(abs(rotated));
    if(!(lane%32))v[lane/32]=peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!lane) {
        float total=v[0];for(uint warp=1;warp<8;++warp)total=max(total,v[warp]);
        float norm=total==0?128.f:max((total/127.f)*128.f,0x1p-24f);
        ushort out=to_half(norm);
        if(!out || (out&0x7c00)==0x7c00) {atomic_fetch_or_explicit(status,4u,memory_order_relaxed);out=0x5800;}
        dst[ulong(tile.y)*p.scale_pitch/2+tile.x]=out;
    }
}
// Comfy A8 only: one SIMD group owns H256, eight values per lane. Keep
// the generic radix-4 expression order and ONE source-dtype boundary.
// No shared scratch/barriers; W codes, scales and normalized math unchanged.
inline void w8_comfy_register_rotate(thread float (&value)[8], uint lane, constant W8Params &p) {
    float next[8];
    for (uint stride=1; stride<=4; stride*=4) {
        uint digit=(lane/stride)&3, first=lane-digit*stride;
        for (uint j=0; j<8; ++j) {
            float a=simd_shuffle(value[j],first), b=simd_shuffle(value[j],first+stride);
            float c=simd_shuffle(value[j],first+2*stride), d=simd_shuffle(value[j],first+3*stride);
            switch (digit) {
                case 0: next[j]=a+b+c-d; break;
                case 1: next[j]=a+b-c+d; break;
                case 2: next[j]=a-b+c+d; break;
                default: next[j]=-a+b+c+d; break;
            }
        }
        for (uint j=0; j<8; ++j) value[j]=next[j];
    }
    for (uint j=0; j<8; ++j) {
        uint even=j&~1u, first=lane&15u, digit=lane/16+(j&1u)*2;
        float a=simd_shuffle(value[even],first), b=simd_shuffle(value[even],first+16);
        float c=simd_shuffle(value[even+1],first), d=simd_shuffle(value[even+1],first+16);
        switch (digit) {
            case 0: next[j]=a+b+c-d; break;
            case 1: next[j]=a+b-c+d; break;
            case 2: next[j]=a-b+c+d; break;
            default: next[j]=-a+b+c+d; break;
        }
    }
    for (uint j=0; j<8; ++j) value[j]=next[j];
    for (uint j=0; j<8; ++j) {
        uint first=j&1u, digit=j/2;
        float a=value[first], b=value[first+2], c=value[first+4], d=value[first+6];
        switch (digit) {
            case 0: next[j]=a+b+c-d; break;
            case 1: next[j]=a+b-c+d; break;
            case 2: next[j]=a-b+c+d; break;
            default: next[j]=-a+b+c+d; break;
        }
        next[j]*=.0625f;
        next[j]=p.dtype==0 ? from_half(to_half(next[j])) :
            p.dtype==1 ? as_type<float>(uint(to_bfloat(next[j]))<<16) : next[j];
    }
    for (uint j=0; j<8; ++j) value[j]=next[j];
}
inline void w8_comfy_register_load(device const uchar *src, device const uchar *scales,
    device const uchar *offsets, device atomic_uint *status, constant W8Params &p,
    uint row, uint col, uint lane, thread float (&value)[8]) {
    for (uint j=0; j<8; ++j) {
        float x=w8_decode(src,scales,offsets,p.row_begin+row,p.column_begin+col+lane+j*32,p);
        if (!isfinite(x)) {atomic_fetch_or_explicit(status,1u,memory_order_relaxed);x=0;}
        value[j]=x;
    }
    w8_comfy_register_rotate(value,lane,p);
    for (uint j=0; j<8; ++j)
        if (!isfinite(value[j])) atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
}
kernel void tc_ane_comfy_register_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]], uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float peak=0;
    const uint begin=p.activation_group ? tile.y*256 : 0, end=p.activation_group ? begin+256 : p.columns;
    for (uint col=begin; col<end; col+=256) {
        float value[8];w8_comfy_register_load(src,scales,offsets,status,p,tile.x,col,lane,value);
        for (uint j=0; j<8; ++j) peak=max(peak,abs(value[j]));
    }
    peak=simd_max(peak);
    if (!lane) {
        float norm=peak==0 ? 128.f : max((peak/127.f)*128.f,0x1p-24f);
        ushort out=to_half(norm);
        if (!out || (out&0x7c00)==0x7c00) {atomic_fetch_or_explicit(status,4u,memory_order_relaxed);out=0x5800;}
        dst[p.activation_group ? ulong(tile.y)*p.scale_pitch/2+tile.x : ulong(tile.x)*p.scale_pitch/2]=out;
    }
}
kernel void tc_ane_comfy_register_codes(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device const ushort *row_scales [[buffer(3)]], device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant W8Params &p [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float value[8];w8_comfy_register_load(src,scales,offsets,status,p,tile.x,tile.y*256,lane,value);
    float scale=from_half(row_scales[p.activation_group ? ulong(tile.y)*p.scale_pitch/2+tile.x : ulong(tile.x)*p.scale_pitch/2]);
    if (!(scale>0) || !isfinite(scale)) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    for (uint j=0; j<8; ++j) {
        float q=clamp((value[j]/scale)*128.f,-127.f,127.f);
        if (!isfinite(q)) {atomic_fetch_or_explicit(status,8u,memory_order_relaxed);q=0;}
        dst[ulong(tile.y*256+lane+j*32)*p.code_pitch+tile.x]=char(rint(q));
    }
}
// Only compact FP16 scale metadata is retained, never a W8/dense matrix.
kernel void tc_ane_w8_scale_copy(device const ushort *src [[buffer(0)]],device ushort *dst [[buffer(1)]],
    constant uint3 &p [[buffer(2)]],uint row [[thread_position_in_grid]]) {
    if(row<p.x)dst[ulong(row)*p.z]=src[ulong(row)*p.y];
}
// Direct W staging never rotates/requantizes W. Validate ALL physical row
// metadata, not only a selected channel range; legacy packed is q+128.
kernel void tc_ane_convrot_scales(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device ushort *dst [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
    constant W8Params &p [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint physical_row = p.row_begin + row;
    device const uchar *s = scales + ulong(physical_row) * p.meta_pitch;
    const uint item = p.meta_dtype == 2 ? 4 : 2;
    float scale = w8_float(s,p.meta_dtype);
    if (!isfinite(scale)) { atomic_fetch_or_explicit(status,1u,memory_order_relaxed); scale=0; }
    if (p.encoding == 8) {
        device const uchar *b = offsets + ulong(physical_row) * p.offset_pitch;
        for (uint group=0; group<p.physical_cols/p.group_size; ++group) {
            float sg=w8_float(s+group*item,p.meta_dtype);
            float bg=w8_float(b+group*(p.offset_dtype==2?4:2),p.offset_dtype);
            if (!isfinite(sg) || !isfinite(bg) || sg!=scale || bg!=-128.f*scale)
                atomic_fetch_or_explicit(status,16u,memory_order_relaxed);
        }
    }
    ushort normalized=to_half(scale*128.f);
    if ((normalized&0x7c00)==0x7c00 || (scale!=0 && !(normalized&0x7fff))) {
        atomic_fetch_or_explicit(status,4u,memory_order_relaxed); normalized=0;
    }
    dst[ulong(row)*p.scale_pitch/2]=normalized;
}
kernel void tc_ane_convrot_codes(device const uchar *src [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *offsets [[buffer(2)]], device const ushort *row_scales [[buffer(3)]], device char *dst [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant W8Params &p [[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x>=p.columns || index.y>=p.rows) return;
    uchar code=src[ulong(p.row_begin+index.y)*p.source_pitch+p.column_begin+index.x];
    dst[ulong(index.y)*p.code_pitch+index.x]=p.encoding==7 ? char(code) : char(int(code)-128);
}
)metal";
}
