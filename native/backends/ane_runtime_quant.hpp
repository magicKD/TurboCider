#pragma once

#include "ane_runtime_convert.hpp"

namespace tc::ane {

inline size_t affine_row_stride(const AffineView &v) {
    return v.row_stride_bytes ? v.row_stride_bytes : size_t(v.cols) * v.bits / 8;
}

// Validate before pointer arithmetic or conversion. Separate strides permit
// padded/sliced rows without a transpose or a dense temporary allocation.
inline void validate_affine_view(const AffineView &v) {
    auto check = [](bool valid, const char *reason) {
        if (!valid) throw std::runtime_error(reason);
    };
    check(v.data && v.rows > 0 && v.rows <= 1048576 && v.cols > 0 && v.cols <= 32768 &&
              (v.bits == 4 || v.bits == 8) &&
              (v.group_size == 32 || v.group_size == 64 || v.group_size == 128),
          "invalid runtime ANE affine geometry");
    check(v.cols % v.group_size == 0, "runtime ANE affine columns must be group aligned");
    const size_t packed_bytes = size_t(v.cols) * v.bits / 8;
    const size_t pitch = affine_row_stride(v);
    check(pitch >= packed_bytes && pitch <= SIZE_MAX / size_t(v.rows) &&
              v.bytes >= size_t(v.rows - 1) * pitch + packed_bytes,
          "runtime ANE affine packed storage is too small");
    auto metadata = [&](const MatrixView &m) {
        check(m.data && m.rows == v.rows && m.cols == v.cols / v.group_size &&
                  (m.dtype == DType::FP16 || m.dtype == DType::BF16 || m.dtype == DType::FP32),
              "invalid runtime ANE affine metadata geometry/dtype");
        const size_t row_bytes = size_t(m.cols) * (m.dtype == DType::FP32 ? 4 : 2);
        const size_t stride = m.row_stride_bytes ? m.row_stride_bytes : row_bytes;
        check(stride >= row_bytes && stride <= SIZE_MAX / size_t(m.rows) &&
                  m.bytes >= size_t(m.rows - 1) * stride + row_bytes,
              "runtime ANE affine metadata storage is too small");
    };
    metadata(v.scales);
    if (v.offsets) metadata(*v.offsets);
}

inline const void *affine_metadata_row(const MatrixView &m, size_t row) {
    const size_t stride = m.row_stride_bytes ? m.row_stride_bytes :
        size_t(m.cols) * (m.dtype == DType::FP32 ? 4 : 2);
    return static_cast<const char *>(m.data) + row * stride;
}

// Requires a validated view and row < rows; one row maps directly to an
// IOSurface row. Q4 has a 16-entry LUT per group; Q8 uses NEON on Apple Silicon.
// FMA in FP32 followed by ONE FP16 rounding, including headroom scaling.
// Only selected codes are checked for FP16 overflow (unused LUT values may
// overflow). Nonfinite scales/offsets always reject, including zero codes.
inline bool affine_fp16_row(const AffineView &v, size_t row, uint16_t *out,
                            bool scalar_only = false, float headroom = 1.f) {
    const auto *codes = static_cast<const char *>(v.data) + row * affine_row_stride(v);
    const auto *scales = affine_metadata_row(v.scales, row);
    const void *offsets = v.offsets ? affine_metadata_row(*v.offsets, row) : nullptr;
    const unsigned per_word = 32 / unsigned(v.bits), mask = (1u << v.bits) - 1;
    bool finite = std::isfinite(headroom) && headroom > 0;
    for (int group = 0; group < v.cols / v.group_size; ++group) {
        const float scale = load_scalar(scales, group, v.scales.dtype);
        const float bias = offsets ? load_scalar(offsets, group, v.offsets->dtype) : 0.f;
        finite &= std::isfinite(scale) && std::isfinite(bias);
        const size_t begin = size_t(group) * v.group_size;
        uint16_t table[16];
        if (v.bits == 4 && !scalar_only)
            for (unsigned code = 0; code < 16; ++code)
                table[code] = std::bit_cast<uint16_t>(_Float16(std::fma(scale, float(code), bias) * headroom));
        size_t column = begin;
        if (v.bits == 4 && !scalar_only) {
            // Decode one packed word at a time: no division or repeated word
            // load per element. This path is useful without ARM intrinsics.
            for (; column < begin + v.group_size; column += 8) {
                uint32_t word;
                std::memcpy(&word, codes + column / 2, 4);
                for (unsigned lane = 0; lane < 8; ++lane, word >>= 4) {
                    const auto value = table[word & 15];
                    out[column + lane] = value;
                    finite &= (value & 0x7c00) != 0x7c00;
                }
            }
        }
#if defined(__aarch64__)
        if (v.bits == 8 && !scalar_only) {
            auto invalid = vdupq_n_u16(0);
            const auto exponent = vdupq_n_u16(0x7c00);
            for (; column + 8 <= begin + v.group_size; column += 8) {
                // Little-endian bytes equal the uint32 low-bit-first codes.
                const auto decoded = vmovl_u8(vld1_u8(reinterpret_cast<const uint8_t *>(codes) + column));
                const auto lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(decoded)));
                const auto hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(decoded)));
                const auto value = vreinterpretq_u16_f16(vcombine_f16(
                    vcvt_f16_f32(vmulq_n_f32(vfmaq_n_f32(vdupq_n_f32(bias), lo, scale), headroom)),
                    vcvt_f16_f32(vmulq_n_f32(vfmaq_n_f32(vdupq_n_f32(bias), hi, scale), headroom))));
                vst1q_u16(out + column, value);
                invalid = vorrq_u16(invalid, vceqq_u16(vandq_u16(value, exponent), exponent));
            }
            finite &= vmaxvq_u16(invalid) == 0;
        }
#endif
        for (; column < begin + v.group_size; ++column) {
            uint32_t word;
            std::memcpy(&word, codes + (column / per_word) * 4, 4);
            const unsigned code = (word >> (v.bits * (column % per_word))) & mask;
            const uint16_t value = v.bits == 4 && !scalar_only ? table[code] :
                std::bit_cast<uint16_t>(_Float16(std::fma(scale, float(code), bias) * headroom));
            out[column] = value;
            finite &= (value & 0x7c00) != 0x7c00;
        }
    }
    return finite;
}

} // namespace tc::ane
