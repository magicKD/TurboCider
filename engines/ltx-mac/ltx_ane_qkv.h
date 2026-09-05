#ifndef LTX_ANE_QKV_H
#define LTX_ANE_QKV_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_ane_qkv ltx_ane_qkv;

typedef struct {
    uint32_t block_index;
    uint32_t rows;
    uint32_t hidden;
} ltx_ane_qkv_shape;

typedef struct {
    double pack_ms;
    double ane_ms;
    double unpack_ms;
    double total_ms;
    int query_output_backing_used;
    int key_output_backing_used;
    int value_output_backing_used;
} ltx_ane_qkv_timing;

ltx_ane_qkv *ltx_ane_qkv_create(ltx_gpu *gpu,
                                const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size);
void ltx_ane_qkv_free(ltx_ane_qkv *qkv);
const ltx_ane_qkv_shape *ltx_ane_qkv_get_shape(const ltx_ane_qkv *qkv);

/* Pack a contiguous suffix of logical BF16 [input_rows, hidden] into the
 * fixed-shape FP16 Core ML input, then start ANE prediction asynchronously. */
int ltx_ane_qkv_start(ltx_ane_qkv *qkv, ltx_gpu *gpu,
                      const ltx_gpu_buffer *input,
                      uint32_t input_rows, uint32_t start_row,
                      char *error, size_t error_size);

/* Join ANE prediction and concatenate GPU BF16 prefix rows with ANE FP16
 * suffix rows into full logical BF16 Q/K/V tensors. */
int ltx_ane_qkv_wait(ltx_ane_qkv *qkv, ltx_gpu *gpu,
                     ltx_gpu_buffer *query_output,
                     ltx_gpu_buffer *key_output,
                     ltx_gpu_buffer *value_output,
                     const ltx_gpu_buffer *query_prefix,
                     const ltx_gpu_buffer *key_prefix,
                     const ltx_gpu_buffer *value_prefix,
                     uint32_t prefix_rows,
                     ltx_ane_qkv_timing *timing,
                     char *error, size_t error_size);

#endif
