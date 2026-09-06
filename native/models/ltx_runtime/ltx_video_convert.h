#ifndef LTX_VIDEO_CONVERT_H
#define LTX_VIDEO_CONVERT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact planar BCFHW BF16 [-1, 1] to frame-major interleaved RGB8 conversion.
 * Rows are parallelized, but each element keeps the scalar clamp/lround
 * arithmetic used by the reference media finalizer. */
int ltx_video_bf16_planar_to_rgb24(
    uint8_t *output, size_t output_bytes,
    const uint16_t *input, size_t input_elements,
    uint32_t frames, uint32_t height, uint32_t width,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
