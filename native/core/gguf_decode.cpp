#include "gguf_decode.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// GGML block layouts/mathematics, pinned reference:
// ggml-org/llama.cpp 64e9bceb2c3a856efed96feda784a50947049feb.
// Q4/Q5/K decoding expressions are adapted from ggml-quants.c. Attribution and
// the upstream MIT license are retained in THIRD_PARTY_NOTICES/ggml-quantization.md.
namespace tc::gguf {
namespace {
void decode_check(bool valid, const char *reason) {
    if (!valid) throw DecodeError(reason);
}
uint64_t decode_add(uint64_t a, uint64_t b) {
    decode_check(b <= UINT64_MAX - a, "gguf_decode: addition overflow"); return a + b;
}
uint64_t decode_mul(uint64_t a, uint64_t b) {
    decode_check(!a || b <= UINT64_MAX / a, "gguf_decode: multiplication overflow"); return a * b;
}
const TypeInfo &decode_type(uint32_t id) {
    for (const auto &type : types) if (type.id == id) return type;
    throw DecodeError("gguf_decode: unsupported tensor type");
}
uint16_t u16(const uint8_t *p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint32_t u32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
int signed_byte(uint8_t value) { return value < 128 ? int(value) : int(value) - 256; }
float half(const uint8_t *p) { return fp16_to_float(u16(p)); }
uint32_t round_right_even(uint32_t value, uint32_t shift) {
    const uint32_t quotient = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}
void finite(float value) { decode_check(std::isfinite(value), "nonfinite weight/scale or conversion overflow"); }
void cancelled(const std::atomic<bool> *cancel) {
    if (cancel && cancel->load(std::memory_order_acquire))
        throw DecodeError("gguf_decode_cancelled");
}
void scale_min(uint32_t group, const uint8_t *p, uint8_t &scale, uint8_t &minimum) {
    if (group < 4) { scale = p[group] & 63; minimum = p[group + 4] & 63; }
    else {
        scale = (p[group + 4] & 15) | ((p[group - 4] >> 6) << 4);
        minimum = (p[group + 4] >> 4) | ((p[group] >> 6) << 4);
    }
}
void decode_block(uint32_t type, const uint8_t *p, float *out) {
    if (type == 0) { out[0] = std::bit_cast<float>(u32(p)); finite(out[0]); return; }
    if (type == 1) { out[0] = half(p); finite(out[0]); return; }
    if (type == 30) { out[0] = std::bit_cast<float>(uint32_t(u16(p)) << 16); finite(out[0]); return; }
    if (type == 2 || type == 3 || type == 6 || type == 7 || type == 8) {
        const float scale = half(p); finite(scale);
        if (type == 8) {
            for (uint32_t i = 0; i < 32; ++i) out[i] = float(signed_byte(p[2 + i])) * scale;
        } else {
            const bool affine = type == 3 || type == 7, five = type == 6 || type == 7;
            const float minimum = affine ? half(p + 2) : 0.f; finite(minimum);
            const uint32_t offset = affine ? 4 : 2;
            const uint32_t high = five ? u32(p + offset) : 0;
            const uint8_t *low = p + offset + (five ? 4 : 0);
            for (uint32_t i = 0; i < 16; ++i) {
                int a = (low[i] & 15) | (five ? int((high >> i) & 1) << 4 : 0);
                int b = (low[i] >> 4) | (five ? int((high >> (i + 16)) & 1) << 4 : 0);
                if (!affine) { a -= five ? 16 : 8; b -= five ? 16 : 8; }
                out[i] = affine ? float(a) * scale + minimum : float(a) * scale;
                out[i + 16] = affine ? float(b) * scale + minimum : float(b) * scale;
            }
        }
    } else if (type == 12 || type == 13) {
        const float scale = half(p), minimum = half(p + 2); finite(scale); finite(minimum);
        const uint8_t *scales = p + 4, *high = p + 16;
        const uint8_t *low = p + (type == 13 ? 48 : 16);
        for (uint32_t group = 0; group < 8; ++group) {
            uint8_t subscale, subminimum; scale_min(group, scales, subscale, subminimum);
            const float d = scale * subscale, m = minimum * subminimum;
            for (uint32_t j = 0; j < 32; ++j) {
                const uint8_t q = low[(group / 2) * 32 + j];
                int code = (group & 1) ? q >> 4 : q & 15;
                if (type == 13 && (high[j] & (uint8_t(1) << group))) code += 16;
                out[group * 32 + j] = d * code - m;
            }
        }
    } else if (type == 14) {
        const float scale = half(p + 208); finite(scale);
        for (uint32_t half_index = 0; half_index < 2; ++half_index) {
            const uint8_t *low = p + half_index * 64, *high = p + 128 + half_index * 32;
            const uint8_t *scales = p + 192 + half_index * 8;
            for (uint32_t j = 0; j < 32; ++j) {
                for (uint32_t group = 0; group < 4; ++group) {
                    const uint8_t q = low[j + (group & 1) * 32];
                    const int code = int((group < 2 ? q & 15 : q >> 4) |
                                         (((high[j] >> (2 * group)) & 3) << 4)) - 32;
                    out[half_index * 128 + group * 32 + j] =
                        scale * float(signed_byte(scales[j / 16 + group * 2])) * float(code);
                }
            }
        }
    } else {
        (void)decode_type(type);
        throw DecodeError("gguf_decode: decoder missing for registered type");
    }
    for (uint32_t i = 0; i < decode_type(type).elements; ++i) finite(out[i]);
}
void write_value(std::byte *p, DecodeDType dtype, float value) {
    if (dtype == DecodeDType::f32) { std::memcpy(p, &value, 4); return; }
    const uint16_t bits = dtype == DecodeDType::bf16 ? float_to_bf16_rne(value) : float_to_fp16_rne(value);
    std::memcpy(p, &bits, 2);
}
struct Geometry { uint64_t row_bytes, row_extent, target_extent; const TypeInfo *type; };

bool identical_dtype(uint32_t source, DecodeDType target) {
    return (source == 0 && target == DecodeDType::f32) ||
           (source == 1 && target == DecodeDType::f16) ||
           (source == 30 && target == DecodeDType::bf16);
}

uint64_t copy_finite(const uint8_t *source, std::byte *target, uint64_t elements,
                     uint32_t type, bool use_simd, const std::atomic<bool> *cancel) {
    const uint32_t width = type == 0 ? 4 : 2;
    const uint32_t mask = type == 0 ? 0x7f800000 : type == 1 ? 0x7c00 : 0x7f80;
    uint64_t copied = 0, vector_elements = 0;
#if defined(__aarch64__)
    if (use_simd) {
        const uint32_t lanes = 16 / width;
        for (; elements - copied >= lanes; copied += lanes) {
            cancelled(cancel);
            const uint8x16_t bytes = vld1q_u8(source + copied * width);
            if (width == 4) {
                const uint32x4_t bits = vreinterpretq_u32_u8(bytes);
                decode_check(!vmaxvq_u32(vceqq_u32(vandq_u32(bits, vdupq_n_u32(mask)), vdupq_n_u32(mask))),
                             "nonfinite source float");
            } else {
                const uint16x8_t bits = vreinterpretq_u16_u8(bytes);
                decode_check(!vmaxvq_u16(vceqq_u16(vandq_u16(bits, vdupq_n_u16(uint16_t(mask))),
                                                   vdupq_n_u16(uint16_t(mask)))), "nonfinite source float");
            }
            vst1q_u8(reinterpret_cast<uint8_t *>(target + copied * width), bytes);
        }
        vector_elements = copied;
    }
#else
    (void)use_simd;
#endif
    for (; copied < elements; ++copied) {
        cancelled(cancel);
        const uint32_t bits = width == 4 ? u32(source + copied * width) : u16(source + copied * width);
        decode_check((bits & mask) != mask, "nonfinite source float");
        std::memcpy(target + copied * width, source + copied * width, width);
    }
    return vector_elements;
}
Geometry validate(const PackedMatrix &source, uint64_t target_rows,
                  uint64_t column_begin, uint64_t columns, const DecodeTarget &target) {
    const auto &type = decode_type(source.type);
    decode_check(source.bytes.data() && target.bytes.data(), "null source/target storage");
    decode_check(source.rows && source.columns && source.columns % type.elements == 0, "invalid source geometry");
    decode_check(target_rows && columns && column_begin <= source.columns &&
              columns <= source.columns - column_begin, "invalid decode slice");
    const uint64_t row_bytes = decode_mul(source.columns / type.elements, type.bytes);
    const uint64_t source_bytes = decode_mul(source.rows, row_bytes);
    decode_check(source_bytes <= source.bytes.size(), "packed source too short");
    const uint32_t item = dtype_bytes(target.dtype);
    decode_check(target.column_stride >= item, "invalid/overlapping column stride");
    const uint64_t row_extent = decode_add(decode_mul(columns - 1, target.column_stride), item);
    decode_check(target.row_stride >= row_extent, "invalid/overlapping row stride");
    const uint64_t target_extent = decode_add(decode_mul(target_rows - 1, target.row_stride), row_extent);
    decode_check(target_extent <= target.bytes.size(), "decode target too short");
    const uintptr_t source_start = reinterpret_cast<uintptr_t>(source.bytes.data());
    const uintptr_t target_start = reinterpret_cast<uintptr_t>(target.bytes.data());
    decode_check(source_bytes <= UINTPTR_MAX - source_start && target_extent <= UINTPTR_MAX - target_start,
          "pointer range overflow");
    decode_check(source_start + source_bytes <= target_start || target_start + target_extent <= source_start,
          "packed source overlaps target");
    return {row_bytes, row_extent, target_extent, &type};
}
#if defined(__aarch64__)
void q8_dense_block(const uint8_t *source, std::byte *target, DecodeDType dtype) {
    const float scale = half(source); finite(scale);
    // Q8 codes * finite FP16 scale cannot overflow FP32/BF16. Preserve the
    // FP16 scale exactly, multiply in FP32, then apply integer RNE once.
    for (uint32_t j = 0; j < 32; j += 8) {
        const int16x8_t codes = vmovl_s8(vld1_s8(reinterpret_cast<const int8_t *>(source + 2 + j)));
        const float32x4_t a = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes))), scale);
        const float32x4_t b = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes))), scale);
        auto round = [](float32x4_t values) {
            const uint32x4_t bits = vreinterpretq_u32_f32(values);
            const uint32x4_t tie = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
            return vshrn_n_u32(vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), tie)), 16);
        };
        if (dtype == DecodeDType::f32) {
            vst1q_f32(reinterpret_cast<float *>(target + j * 4), a);
            vst1q_f32(reinterpret_cast<float *>(target + j * 4 + 16), b);
        } else if (dtype == DecodeDType::f16) {
            decode_check(vmaxvq_f32(vabsq_f32(a)) < 65520.f && vmaxvq_f32(vabsq_f32(b)) < 65520.f,
                         "FP16 conversion overflow");
            const uint16x8_t result = vcombine_u16(vreinterpret_u16_f16(vcvt_f16_f32(a)),
                                                  vreinterpret_u16_f16(vcvt_f16_f32(b)));
            vst1q_u16(reinterpret_cast<uint16_t *>(target + j * 2), result);
        } else {
            const uint16x8_t result = vcombine_u16(round(a), round(b));
            vst1q_u16(reinterpret_cast<uint16_t *>(target + j * 2), result);
        }
    }
}
#endif
DecodeReceipt row_into(const PackedMatrix &source, uint64_t source_row, uint64_t target_row,
                       uint64_t column_begin, uint64_t columns, const DecodeTarget &target,
                       const Geometry &geometry, float *scratch, const std::atomic<bool> *cancel,
                       DecodeOptions options) {
    decode_check(source_row < source.rows, "gather row out of bounds");
    const auto &type = *geometry.type;
    const auto *bytes = reinterpret_cast<const uint8_t *>(source.bytes.data());
    if (identical_dtype(source.type, target.dtype) && target.column_stride == type.bytes) {
        const uint64_t vectors = copy_finite(bytes + source_row * geometry.row_bytes + column_begin * type.bytes,
            target.bytes.data() + target_row * target.row_stride, columns, source.type, options.use_simd, cancel);
        return {decode_mul(columns, type.bytes), columns, decode_mul(columns, type.bytes), 1024, vectors};
    }
    const uint64_t first = column_begin / type.elements;
    const uint64_t last = (column_begin + columns - 1) / type.elements;
    uint64_t simd_blocks = 0;
    for (uint64_t block = first; block <= last; ++block) {
        cancelled(cancel);
        const uint64_t begin = std::max(column_begin, block * type.elements);
        const uint64_t end = std::min(column_begin + columns, (block + 1) * type.elements);
#if defined(__aarch64__)
        if (options.use_simd && source.type == 8 && target.column_stride == dtype_bytes(target.dtype) &&
            begin == block * 32 && end == (block + 1) * 32 &&
            (reinterpret_cast<uintptr_t>(target.bytes.data()) + target_row * target.row_stride) % dtype_bytes(target.dtype) == 0) {
            q8_dense_block(bytes + source_row * geometry.row_bytes + block * type.bytes,
                          target.bytes.data() + target_row * target.row_stride + (begin - column_begin) * dtype_bytes(target.dtype), target.dtype);
            ++simd_blocks;
            continue;
        }
#else
        (void)options;
#endif
        decode_block(source.type, bytes + source_row * geometry.row_bytes + block * type.bytes, scratch);
        for (uint64_t column = begin; column < end; ++column)
            write_value(target.bytes.data() + target_row * target.row_stride +
                        (column - column_begin) * target.column_stride,
                        target.dtype, scratch[column - block * type.elements]);
    }
    return {decode_mul(last - first + 1, type.bytes), columns,
            decode_mul(columns, dtype_bytes(target.dtype)), 1024, simd_blocks};
}
void accumulate(DecodeReceipt &a, const DecodeReceipt &b) {
    a.source_bytes_processed = decode_add(a.source_bytes_processed, b.source_bytes_processed);
    a.elements_written = decode_add(a.elements_written, b.elements_written);
    a.bytes_written = decode_add(a.bytes_written, b.bytes_written);
    a.scratch_bytes = std::max(a.scratch_bytes, b.scratch_bytes);
    a.simd_blocks = decode_add(a.simd_blocks, b.simd_blocks);
}
} // namespace

float fp16_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 31, fraction = value & 1023, bits;
    if (!exponent && fraction) {
        int unbiased = -14;
        while (!(fraction & 1024)) { fraction <<= 1; --unbiased; }
        bits = sign | (uint32_t(unbiased + 127) << 23) | ((fraction & 1023) << 13);
    } else if (!exponent) bits = sign;
    else if (exponent == 31) bits = sign | 0x7f800000 | (fraction << 13);
    else bits = sign | ((exponent + 112) << 23) | (fraction << 13);
    return std::bit_cast<float>(bits);
}
uint16_t float_to_bf16_rne(float value) {
    finite(value);
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint16_t result = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    decode_check((result & 0x7f80) != 0x7f80, "BF16 conversion overflow"); return result;
}
uint16_t float_to_fp16_rne(float value) {
    finite(value);
    const uint32_t bits = std::bit_cast<uint32_t>(value), fraction = bits & 0x7fffff;
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    const int exponent = int((bits >> 23) & 255) - 127;
    if (exponent < -25) return sign;
    decode_check(exponent <= 15, "FP16 conversion overflow");
    if (exponent < -14) return sign | uint16_t(round_right_even(0x800000 | fraction, uint32_t(-exponent - 1)));
    const uint32_t result = uint32_t(exponent + 15) * 1024 + round_right_even(fraction, 13);
    decode_check(result < 0x7c00, "FP16 conversion overflow"); return sign | uint16_t(result);
}
DecodeReceipt decode_cpu_into(const PackedMatrix &source, const DecodeSlice &slice,
                              const DecodeTarget &target, const std::atomic<bool> *cancel,
                              DecodeOptions options) {
    cancelled(cancel);
    decode_check(slice.row_begin <= source.rows && slice.rows <= source.rows - slice.row_begin, "row slice out of bounds");
    const auto geometry = validate(source, slice.rows, slice.column_begin, slice.columns, target);
    std::array<float, 256> scratch{};
    DecodeReceipt receipt;
    for (uint64_t row = 0; row < slice.rows; ++row)
        accumulate(receipt, row_into(source, slice.row_begin + row, row, slice.column_begin,
                                    slice.columns, target, geometry, scratch.data(), cancel, options));
    cancelled(cancel); return receipt;
}
DecodeReceipt decode_cpu_gather(const PackedMatrix &source, std::span<const uint64_t> rows,
                                uint64_t column_begin, uint64_t columns,
                                const DecodeTarget &target, const std::atomic<bool> *cancel,
                                DecodeOptions options) {
    cancelled(cancel);
    const auto geometry = validate(source, rows.size(), column_begin, columns, target);
    const uintptr_t indices_begin = reinterpret_cast<uintptr_t>(rows.data());
    const uint64_t indices_bytes = decode_mul(rows.size(), sizeof(uint64_t));
    const uintptr_t target_begin = reinterpret_cast<uintptr_t>(target.bytes.data());
    decode_check(indices_bytes <= UINTPTR_MAX - indices_begin &&
              (indices_begin + indices_bytes <= target_begin || target_begin + geometry.target_extent <= indices_begin),
          "gather indices overlap target");
    for (uint64_t row : rows) decode_check(row < source.rows, "gather row out of bounds");
    std::array<float, 256> scratch{};
    DecodeReceipt receipt;
    for (size_t row = 0; row < rows.size(); ++row)
        accumulate(receipt, row_into(source, rows[row], row, column_begin, columns,
                                    target, geometry, scratch.data(), cancel, options));
    cancelled(cancel); return receipt;
}
} // namespace tc::gguf
