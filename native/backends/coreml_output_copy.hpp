#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tc {
#if defined(__aarch64__)
inline void transpose_coreml_fp16_8x8(uint16_t *destination, const uint16_t *source,
                                    size_t destination_stride, size_t source_stride) {
    const auto a = vtrnq_u16(vld1q_u16(source), vld1q_u16(source + source_stride));
    const auto b = vtrnq_u16(vld1q_u16(source + 2 * source_stride), vld1q_u16(source + 3 * source_stride));
    const auto c = vtrnq_u16(vld1q_u16(source + 4 * source_stride), vld1q_u16(source + 5 * source_stride));
    const auto d = vtrnq_u16(vld1q_u16(source + 6 * source_stride), vld1q_u16(source + 7 * source_stride));
    const auto e = vtrnq_u32(vreinterpretq_u32_u16(a.val[0]), vreinterpretq_u32_u16(b.val[0]));
    const auto f = vtrnq_u32(vreinterpretq_u32_u16(a.val[1]), vreinterpretq_u32_u16(b.val[1]));
    const auto g = vtrnq_u32(vreinterpretq_u32_u16(c.val[0]), vreinterpretq_u32_u16(d.val[0]));
    const auto h = vtrnq_u32(vreinterpretq_u32_u16(c.val[1]), vreinterpretq_u32_u16(d.val[1]));
    auto store = [&](size_t row, uint32x4_t low, uint32x4_t high) {
        const auto x = vreinterpretq_u64_u32(low), y = vreinterpretq_u64_u32(high);
        vst1q_u16(destination + row * destination_stride, vreinterpretq_u16_u64(vtrn1q_u64(x, y)));
        vst1q_u16(destination + (row + 4) * destination_stride, vreinterpretq_u16_u64(vtrn2q_u64(x, y)));
    };
    store(0, e.val[0], g.val[0]);
    store(1, f.val[0], h.val[0]);
    store(2, e.val[1], g.val[1]);
    store(3, f.val[1], h.val[1]);
}
#endif

// Copy validated Core ML FP16 storage into a separate row-major MLX buffer.
// Preserve all 16 bits, including NaN payloads; this is not a numeric conversion.
inline void copy_coreml_fp16(uint16_t *destination, const uint16_t *source,
                            size_t rows, size_t channels, size_t row_stride,
                            size_t channel_stride, bool optimized) {
    if (optimized && channel_stride == 1) {
        if (row_stride == channels) {
            std::memcpy(destination, source, rows * channels * sizeof(uint16_t));
        } else {
            for (size_t row = 0; row < rows; ++row)
                std::memcpy(destination + row * channels, source + row * row_stride,
                            channels * sizeof(uint16_t));
        }
        return;
    }
    if (optimized && row_stride == 1 && channel_stride >= rows) {
        // Core ML may return contiguous rows within each channel. Blocking the
        // transpose avoids reading one cache line per scalar across all channels.
        constexpr size_t tile = 32;
        for (size_t row = 0; row < rows; row += tile) {
            const auto row_end = std::min(row + tile, rows);
            for (size_t channel = 0; channel < channels; channel += tile) {
                const auto channel_end = std::min(channel + tile, channels);
                for (size_t c = channel; c < channel_end; c += 8) {
                    for (size_t r = row; r < row_end; r += 8) {
#if defined(__aarch64__)
                        if (c + 8 <= channel_end && r + 8 <= row_end) {
                            transpose_coreml_fp16_8x8(destination + r * channels + c,
                                source + c * channel_stride + r, channels, channel_stride);
                            continue;
                        }
#endif
                        for (size_t cc = c; cc < std::min(c + 8, channel_end); ++cc)
                            for (size_t rr = r; rr < std::min(r + 8, row_end); ++rr)
                                destination[rr * channels + cc] = source[cc * channel_stride + rr];
                    }
                }
            }
        }
        return;
    }
    for (size_t row = 0; row < rows; ++row)
        for (size_t c = 0; c < channels; ++c)
            destination[row * channels + c] = source[row * row_stride + c * channel_stride];
}
} // namespace tc
