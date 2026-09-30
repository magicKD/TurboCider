#pragma once
#include "ane_runtime.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tc::ane {

inline float load_scalar(const void *data, size_t index, DType dtype) {
    if (dtype == DType::FP32) {
        float value;
        std::memcpy(&value, static_cast<const char *>(data) + 4 * index, 4);
        return value;
    }
    uint16_t value;
    std::memcpy(&value, static_cast<const char *>(data) + 2 * index, 2);
    return dtype == DType::BF16 ? std::bit_cast<float>(uint32_t(value) << 16) :
                                 float(std::bit_cast<_Float16>(value));
}

// Pure row conversion: no allocation, transpose, model state, or GPU dispatch.
// The scalar path remains available for numerical and same-binary A/B tests.
inline bool convert_fp16_row(const void *source, uint16_t *destination, size_t count,
                             DType dtype, bool scalar_only = false, float scale = 1.f) {
    size_t offset = 0;
    bool finite = true;
#if defined(__aarch64__)
    if (!scalar_only) {
        uint16x8_t invalid = vdupq_n_u16(0);
        const uint16x8_t mask = vdupq_n_u16(0x7c00);
        for (; offset + 8 <= count; offset += 8) {
            uint16x8_t result;
            if (dtype == DType::FP16) {
                result = vld1q_u16(static_cast<const uint16_t *>(source) + offset);
                if (scale != 1.f) {
                    const auto half = vreinterpretq_f16_u16(result);
                    result = vreinterpretq_u16_f16(vcombine_f16(
                        vcvt_f16_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(half)), scale)),
                        vcvt_f16_f32(vmulq_n_f32(vcvt_f32_f16(vget_high_f16(half)), scale))));
                }
            } else if (dtype == DType::BF16) {
                const auto bf = vld1q_u16(static_cast<const uint16_t *>(source) + offset);
                const auto lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
                const auto hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
                result = vreinterpretq_u16_f16(vcombine_f16(
                    vcvt_f16_f32(vmulq_n_f32(lo, scale)), vcvt_f16_f32(vmulq_n_f32(hi, scale))));
            } else {
                const auto *fp = static_cast<const float *>(source) + offset;
                result = vreinterpretq_u16_f16(vcombine_f16(
                    vcvt_f16_f32(vmulq_n_f32(vld1q_f32(fp), scale)),
                    vcvt_f16_f32(vmulq_n_f32(vld1q_f32(fp + 4), scale))));
            }
            vst1q_u16(destination + offset, result);
            invalid = vorrq_u16(invalid, vceqq_u16(vandq_u16(result, mask), mask));
        }
        finite = vmaxvq_u16(invalid) == 0;
    }
#else
    (void)scalar_only;
#endif
    for (; offset < count; ++offset) {
        const _Float16 value = _Float16(load_scalar(source, offset, dtype) * scale);
        destination[offset] = std::bit_cast<uint16_t>(value);
        finite &= std::isfinite(float(value));
    }
    return finite;
}

inline uint16_t round_bf16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    // Round-to-nearest-even, preserving NaN as NaN rather than infinity.
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return uint16_t(bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

struct OutputRowStatus { bool source_finite = true, restored_finite = true; };

// Restore power-of-two headroom and narrow in one pass. Separate failure
// flags preserve the distinction between retryable graph overflow and an
// output dtype that cannot hold the restored result. Scalar is the oracle.
// A null destination validates an unused graph output without materializing
// a copy. It must retain BOTH failure checks, including headroom restoration.
inline OutputRowStatus restore_fp16_row(const uint16_t *source, uint16_t *destination,
                                       size_t count, DType dtype, float scale,
                                       bool scalar_only = false) {
    OutputRowStatus status;
    size_t offset = 0;
    // An unused BF16 hidden needs no conversion at the runtime's bounded
    // headroom: max finite FP16 * 4096 is far below BF16's finite range.
    // NaN comparisons also reject the fast path; other scales/dtypes retain
    // the general restoration oracle below. Still inspect EVERY source bit.
    if (!destination && dtype == DType::BF16 && scale >= -4096.f && scale <= 4096.f) {
#if defined(__aarch64__)
        if (!scalar_only) {
            uint16x8_t invalid = vdupq_n_u16(0);
            const auto mask = vdupq_n_u16(0x7c00);
            for (; offset + 8 <= count; offset += 8)
                invalid = vorrq_u16(invalid, vceqq_u16(vandq_u16(vld1q_u16(source + offset), mask), mask));
            status.source_finite = vmaxvq_u16(invalid) == 0;
        }
#endif
        for (; offset < count; ++offset)
            status.source_finite &= (source[offset] & 0x7c00u) != 0x7c00u;
        status.restored_finite = status.source_finite;
        return status;
    }
#if defined(__aarch64__)
    if (!scalar_only) {
        uint16x8_t bad_source = vdupq_n_u16(0), bad_output = vdupq_n_u16(0);
        const auto source_mask = vdupq_n_u16(0x7c00);
        const auto target_mask = vdupq_n_u16(dtype == DType::BF16 ? 0x7f80 : 0x7c00);
        auto bf16 = [](float32x4_t value) {
            const auto bits = vreinterpretq_u32_f32(value);
            const auto exponent = vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000)), vdupq_n_u32(0x7f800000));
            const auto nan = vmvnq_u32(vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffff)), vdupq_n_u32(0)));
            const auto special = vorrq_u32(vshrq_n_u32(bits, 16), vandq_u32(nan, vdupq_n_u32(0x40)));
            const auto rounded = vshrq_n_u32(vaddq_u32(vaddq_u32(bits, vdupq_n_u32(0x7fff)),
                vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1))), 16);
            return vmovn_u32(vbslq_u32(exponent, special, rounded));
        };
        for (; offset + 8 <= count; offset += 8) {
            const auto raw = vld1q_u16(source + offset);
            const auto half = vreinterpretq_f16_u16(raw);
            const auto lo = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(half)), scale);
            const auto hi = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(half)), scale);
            const auto narrowed = dtype == DType::BF16 ? vcombine_u16(bf16(lo), bf16(hi)) :
                vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            if (destination) vst1q_u16(destination + offset, narrowed);
            bad_source = vorrq_u16(bad_source, vceqq_u16(vandq_u16(raw, source_mask), source_mask));
            bad_output = vorrq_u16(bad_output, vceqq_u16(vandq_u16(narrowed, target_mask), target_mask));
        }
        status.source_finite = vmaxvq_u16(bad_source) == 0;
        status.restored_finite = vmaxvq_u16(bad_output) == 0;
    }
#else
    (void)scalar_only;
#endif
    for (; offset < count; ++offset) {
        const auto value = source[offset];
        const float restored = float(std::bit_cast<_Float16>(value)) * scale;
        const auto narrowed = dtype == DType::BF16 ? round_bf16(restored) :
            std::bit_cast<uint16_t>(_Float16(restored));
        if (destination) destination[offset] = narrowed;
        status.source_finite &= (value & 0x7c00u) != 0x7c00u;
        const uint16_t mask = dtype == DType::BF16 ? 0x7f80 : 0x7c00;
        status.restored_finite &= (narrowed & mask) != mask;
    }
    return status;
}

// Matrix wrapper for Core ML's scoped read access, including padded rows and
// scattered columns. Keep this host-testable and serial: parallel output
// restoration reduced its own span but did not improve matched whole requests.
inline OutputRowStatus restore_fp16_matrix(const uint16_t *source, uint16_t *destination,
                                          size_t rows, size_t cols, size_t row_stride,
                                          size_t col_stride, DType dtype, float scale,
                                          bool scalar_only = false) {
    OutputRowStatus status;
    for (size_t r = 0; r < rows; ++r) {
        if (col_stride == 1) {
            const auto row = restore_fp16_row(source + r * row_stride,
                destination ? destination + r * cols : nullptr, cols, dtype, scale, scalar_only);
            status.source_finite &= row.source_finite;
            status.restored_finite &= row.restored_finite;
        } else for (size_t c = 0; c < cols; ++c) {
            const auto value = source[r * row_stride + c * col_stride];
            const float restored = float(std::bit_cast<_Float16>(value)) * scale;
            const uint16_t narrowed = dtype == DType::BF16 ? round_bf16(restored) :
                std::bit_cast<uint16_t>(_Float16(restored));
            if (destination) destination[r * cols + c] = narrowed;
            status.source_finite &= (value & 0x7c00u) != 0x7c00u;
            const uint16_t mask = dtype == DType::BF16 ? 0x7f80 : 0x7c00;
            status.restored_finite &= (narrowed & mask) != mask;
        }
    }
    return status;
}
} // namespace tc::ane
