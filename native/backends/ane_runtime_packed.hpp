#pragma once

#include "ane_runtime_convert.hpp"
#include "ane_runtime_quant.hpp"
#include "../core/gguf_decode.hpp"
#include <array>

namespace tc::ane {

// Row kernels require validated immutable sources and caller-owned capacity
// for cols aligned uint16_t elements; no allocation or retained spans. Runtime
// publishes staged=true only after EVERY row succeeds.
inline void validate_packed_storage(const void *data, size_t bytes, int rows,
                                    int cols, size_t pitch, size_t row_bytes) {
    if (!data || rows <= 0 || rows > 1048576 || cols <= 0 || cols > 32768 ||
        pitch < row_bytes || pitch > SIZE_MAX / size_t(rows) ||
        bytes < size_t(rows - 1) * pitch + row_bytes ||
        bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(data))
        throw std::runtime_error("invalid runtime ANE packed storage");
}
inline bool packed_target_disjoint(const void *source, size_t bytes,
                                   const uint16_t *target, size_t columns) {
    const auto begin = reinterpret_cast<uintptr_t>(source);
    const auto output = reinterpret_cast<uintptr_t>(target);
    const size_t written = columns * sizeof(uint16_t);
    return output % alignof(uint16_t) == 0 && output <= UINTPTR_MAX - written &&
           (output + written <= begin || begin + bytes <= output);
}
inline size_t gguf_row_bytes(const GgufView &view) {
    const auto &type = gguf::type_info(view.ggml_type);
    if (view.cols <= 0 || size_t(view.cols) % type.elements)
        throw std::runtime_error("invalid ANE GGUF block geometry");
    return size_t(view.cols) / type.elements * type.bytes;
}
inline void validate_gguf_view(const GgufView &view) {
    const size_t row_bytes = gguf_row_bytes(view);
    const size_t pitch = view.row_stride_bytes ? view.row_stride_bytes : row_bytes;
    validate_packed_storage(view.data, view.bytes, view.rows, view.cols, pitch, row_bytes);
}
inline bool gguf_fp16_row(const GgufView &view, size_t row, uint16_t *target,
                          bool scalar_only = false, float headroom = 1.f) {
    if (row >= size_t(view.rows) || !target || !std::isfinite(headroom) || headroom <= 0 ||
        !packed_target_disjoint(view.data, view.bytes, target, size_t(view.cols))) return false;
    const auto &type = gguf::type_info(view.ggml_type);
    const size_t row_bytes = gguf_row_bytes(view);
    const size_t pitch = view.row_stride_bytes ? view.row_stride_bytes : row_bytes;
    const auto *source = static_cast<const std::byte *>(view.data) + row * pitch;
    try {
        if (headroom == 1.f) {
            gguf::decode_cpu_into({{source, row_bytes}, view.ggml_type, 1, uint64_t(view.cols)},
                {0, 1, 0, uint64_t(view.cols)},
                {{reinterpret_cast<std::byte *>(target), size_t(view.cols) * 2},
                 gguf::DecodeDType::f16, size_t(view.cols) * 2, 2}, nullptr, {!scalar_only});
        } else {
            // Scale BEFORE the only FP16 rounding, including values that would
            // overflow FP16 without headroom. Scratch <=2 KiB per active worker
            // (1 KiB here + <=1 KiB decoder), independent of layer size.
            std::array<float, 256> scratch;
            for (size_t begin = 0; begin < size_t(view.cols); begin += type.elements) {
                gguf::decode_cpu_into(
                    {{source + (begin / type.elements) * type.bytes, type.bytes},
                     view.ggml_type, 1, type.elements}, {0, 1, 0, type.elements},
                    {{reinterpret_cast<std::byte *>(scratch.data()), type.elements * 4},
                     gguf::DecodeDType::f32, type.elements * 4, 4}, nullptr, {!scalar_only});
                if (!convert_fp16_row(scratch.data(), target + begin, type.elements,
                                      DType::FP32, scalar_only, headroom)) return false;
            }
        }
        return true;
    } catch (const std::exception &) { return false; }
}
inline void validate_convrot_view(const ConvrotView &view) {
    if (view.cols <= 0 || view.cols % 256)
        throw std::runtime_error("ConvRot columns require complete Comfy H256 groups");
    const size_t pitch = view.row_stride_bytes ? view.row_stride_bytes : size_t(view.cols);
    validate_packed_storage(view.data, view.bytes, view.rows, view.cols, pitch, size_t(view.cols));
    const auto &scales = view.row_scales;
    const size_t scale_pitch = scales.row_stride_bytes ? scales.row_stride_bytes : 4;
    if (scales.dtype != DType::FP32 || scales.cols != 1 || scales.rows != view.rows)
        throw std::runtime_error("ConvRot requires original FP32 per-output-row scales");
    validate_packed_storage(scales.data, scales.bytes, scales.rows, 1, scale_pitch, 4);
}
inline void convrot_integer_h256(std::array<int32_t, 256> &values, bool scalar_only) {
    // Four Comfy H4 Kronecker factors, NOT Sylvester ordering. Integer sums
    // are exact, bounded by 256*128, then normalized once by 1/16.
#if defined(__aarch64__)
    if (!scalar_only) {
        auto h4 = [](int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d) {
            const auto ab = vaddq_s32(a, b), cd = vsubq_s32(c, d);
            const auto am = vsubq_s32(a, b), cp = vaddq_s32(c, d);
            return int32x4x4_t{{vaddq_s32(ab, cd), vsubq_s32(ab, cd),
                               vaddq_s32(am, cp), vsubq_s32(cp, am)}};
        };
        for (int base = 0; base < 256; base += 16) {
            const auto x = vld4q_s32(values.data() + base);
            vst4q_s32(values.data() + base, h4(x.val[0], x.val[1], x.val[2], x.val[3]));
        }
        for (int stride = 4; stride < 256; stride *= 4)
            for (int base = 0; base < 256; base += stride * 4)
                for (int lane = 0; lane < stride; lane += 4) {
                    auto *p = values.data() + base + lane;
                    const auto out = h4(vld1q_s32(p), vld1q_s32(p + stride),
                                        vld1q_s32(p + 2 * stride), vld1q_s32(p + 3 * stride));
                    for (int i = 0; i < 4; ++i) vst1q_s32(p + i * stride, out.val[i]);
                }
        return;
    }
#else
    (void)scalar_only;
#endif
    for (int stride = 1; stride < 256; stride *= 4)
        for (int base = 0; base < 256; base += stride * 4)
            for (int lane = 0; lane < stride; ++lane) {
                const int p = base + lane;
                const int32_t a = values[p], b = values[p + stride];
                const int32_t c = values[p + 2 * stride], d = values[p + 3 * stride];
                values[p] = a + b + c - d;
                values[p + stride] = a + b - c + d;
                values[p + 2 * stride] = a - b + c + d;
                values[p + 3 * stride] = -a + b + c + d;
            }
}
inline bool convrot_integer_fp16_block(std::array<int32_t,256> &integer,float scale,float headroom,
                                       uint16_t *target,bool scalar_only) {
    convrot_integer_h256(integer,scalar_only);
    int offset = 0;
#if defined(__aarch64__)
    if (!scalar_only) {
        uint16x8_t invalid = vdupq_n_u16(0);
        for (; offset < 256; offset += 8) {
            auto convert = [&](int at) {
                auto value = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(integer.data() + at)), .0625f);
                value = vmulq_n_f32(value, scale);
                return vcvt_f16_f32(vmulq_n_f32(value, headroom));
            };
            const auto result = vreinterpretq_u16_f16(vcombine_f16(convert(offset), convert(offset + 4)));
            vst1q_u16(target + offset, result);
            invalid = vorrq_u16(invalid, vceqq_u16(vandq_u16(result, vdupq_n_u16(0x7c00)),
                                                  vdupq_n_u16(0x7c00)));
        }
        if (vmaxvq_u16(invalid)) return false;
    }
#endif
    try {
        for (; offset < 256; ++offset) {
            const float value = ((float(integer[offset]) * .0625f) * scale) * headroom;
            target[offset] = gguf::float_to_fp16_rne(value);
        }
    } catch (const std::exception &) { return false; }
    return true;
}
// Recipe: integer H256 -> /16 -> original F32 scale -> headroom -> RNE FP16.
// Model quality needs separate qualification. Scratch: 1 KiB per worker.
inline bool convrot_fp16_row(const ConvrotView &view,size_t row,uint16_t *target,
                             bool scalar_only=false,float headroom=1.f) {
    if (row>=size_t(view.rows) || !target || !std::isfinite(headroom) || headroom<=0 ||
        !packed_target_disjoint(view.data,view.bytes,target,size_t(view.cols)) ||
        !packed_target_disjoint(view.row_scales.data,view.row_scales.bytes,target,size_t(view.cols))) return false;
    const size_t pitch=view.row_stride_bytes ? view.row_stride_bytes : size_t(view.cols);
    const size_t scale_pitch=view.row_scales.row_stride_bytes ? view.row_scales.row_stride_bytes : 4;
    float scale;std::memcpy(&scale,static_cast<const char *>(view.row_scales.data)+row*scale_pitch,4);
    if (!std::isfinite(scale)) return false;
    std::array<int32_t,256> integer;
    const int8_t *source=view.data+row*pitch;
    for (int begin=0;begin<view.cols;begin+=256) {
        for (int i=0;i<256;++i) integer[i]=source[begin+i];
        if (!convrot_integer_fp16_block(integer,scale,headroom,target+begin,scalar_only)) return false;
    }
    return true;
}
inline void validate_convrot_affine_view(const ConvrotAffineView &view) {
    const auto &v=view.packed;validate_affine_view(v);
    if (v.bits!=8 || v.cols%256 || !v.offsets)
        throw std::runtime_error("packed ConvRot requires Q8, complete H256 and signed offsets");
    validate_packed_storage(v.data,v.bytes,v.rows,v.cols,affine_row_stride(v),size_t(v.cols));
    for (const auto *m : {&v.scales,&*v.offsets}) {
        const size_t item=m->dtype==DType::FP32 ? 4 : 2;
        const size_t pitch=m->row_stride_bytes ? m->row_stride_bytes : size_t(m->cols)*item;
        validate_packed_storage(m->data,m->bytes,m->rows,m->cols,pitch,size_t(m->cols)*item);
    }
}
// This consumes legacy rounded scales. Do not label it source-F32-scale.
// One 256-int scratch, no signed-code bank or dense intermediate allocation.
inline bool convrot_affine_fp16_row(const ConvrotAffineView &view,size_t row,uint16_t *target,
                                    bool scalar_only=false,float headroom=1.f) {
    const auto &v=view.packed;
    if (row>=size_t(v.rows) || !target || !std::isfinite(headroom) || headroom<=0 || !v.offsets ||
        !packed_target_disjoint(v.data,v.bytes,target,size_t(v.cols)) ||
        !packed_target_disjoint(v.scales.data,v.scales.bytes,target,size_t(v.cols)) ||
        !packed_target_disjoint(v.offsets->data,v.offsets->bytes,target,size_t(v.cols))) return false;
    const auto *scales=affine_metadata_row(v.scales,row),*biases=affine_metadata_row(*v.offsets,row);
    const float scale=load_scalar(scales,0,v.scales.dtype);
    if (!std::isfinite(scale)) return false;
    for (int group=0;group<v.cols/v.group_size;++group) {
        const float s=load_scalar(scales,size_t(group),v.scales.dtype);
        const float b=load_scalar(biases,size_t(group),v.offsets->dtype);
        if (!std::isfinite(s) || !std::isfinite(b) || s!=scale || b!=-128.f*scale) return false;
    }
    const auto *codes=static_cast<const uint8_t *>(v.data)+row*affine_row_stride(v);
    std::array<int32_t,256> integer;
    for (int begin=0;begin<v.cols;begin+=256) {
        for (int i=0;i<256;++i) integer[i]=int32_t(codes[begin+i])-128;
        if (!convrot_integer_fp16_block(integer,scale,headroom,target+begin,scalar_only)) return false;
    }
    return true;
}
} // namespace tc::ane
