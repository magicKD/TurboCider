#ifndef LTX_ANE_V2A_H
#define LTX_ANE_V2A_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_ane_v2a ltx_ane_v2a;

typedef struct {
    uint32_t block_index;
    uint32_t audio_rows;
    uint32_t audio_dim;
    uint32_t video_rows;
    uint32_t video_dim;
    uint32_t heads;
    uint32_t head_dim;
} ltx_ane_v2a_shape;

typedef struct {
    double pack_ms;
    double ane_ms;
    double unpack_ms;
    double total_ms;
    int output_backing_used;
} ltx_ane_v2a_timing;

ltx_ane_v2a *ltx_ane_v2a_create(ltx_gpu *gpu,
                                const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size);
void ltx_ane_v2a_free(ltx_ane_v2a *attention);
const ltx_ane_v2a_shape *ltx_ane_v2a_get_shape(
    const ltx_ane_v2a *attention);

int ltx_ane_v2a_start(ltx_ane_v2a *attention, ltx_gpu *gpu,
                      const ltx_gpu_buffer *audio_input,
                      const ltx_gpu_buffer *video_input,
                      char *error, size_t error_size);
int ltx_ane_v2a_wait(ltx_ane_v2a *attention, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     ltx_ane_v2a_timing *timing,
                     char *error, size_t error_size);

#endif
