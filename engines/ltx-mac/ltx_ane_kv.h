#ifndef LTX_ANE_KV_H
#define LTX_ANE_KV_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_ane_kv ltx_ane_kv;

typedef struct {
    uint32_t block_index;
    uint32_t text_rows;
    uint32_t hidden;
    uint32_t heads;
    uint32_t head_dim;
} ltx_ane_kv_shape;

typedef struct {
    double pack_ms;
    double ane_ms;
    double unpack_ms;
    double total_ms;
    int key_output_backing_used;
    int value_output_backing_used;
} ltx_ane_kv_timing;

ltx_ane_kv *ltx_ane_kv_create(ltx_gpu *gpu,
                              const char *manifest_path,
                              const char *variant,
                              char *error, size_t error_size);
void ltx_ane_kv_free(ltx_ane_kv *kv);
const ltx_ane_kv_shape *ltx_ane_kv_get_shape(const ltx_ane_kv *kv);

/* Start fixed-shape text K/V projection on Core ML. The input is the BF16
 * affine-scaled text context in logical [text_rows, hidden] row-major order. */
int ltx_ane_kv_start(ltx_ane_kv *kv, ltx_gpu *gpu,
                     const ltx_gpu_buffer *scaled_text,
                     char *error, size_t error_size);

/* Wait for ANE completion and return BF16 head-major
 * [1, heads, text_rows, head_dim] K/V tensors. */
int ltx_ane_kv_wait(ltx_ane_kv *kv, ltx_gpu *gpu,
                    ltx_gpu_buffer *key_output,
                    ltx_gpu_buffer *value_output,
                    ltx_ane_kv_timing *timing,
                    char *error, size_t error_size);

#endif
