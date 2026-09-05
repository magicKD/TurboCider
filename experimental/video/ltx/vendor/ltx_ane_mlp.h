#ifndef LTX_ANE_MLP_H
#define LTX_ANE_MLP_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_ane_mlp ltx_ane_mlp;

typedef struct {
    uint32_t block_index;
    uint32_t rows;
    uint32_t supported_rows[2];
    uint32_t supported_row_count;
    uint32_t hidden;
    uint32_t full_intermediate;
    uint32_t gpu_intermediate;
    uint32_t ane_intermediate;
} ltx_ane_mlp_shape;

typedef struct {
    double total_ms;
    double pack_ms;
    double overlap_ms;
    double gpu_ms;
    double ane_ms;
    double join_ms;
    int ane_output_backing_used;
} ltx_ane_mlp_timing;

/* Experimental real-checkpoint LTX video FFN split. The GPU suffix keeps
 * checkpoint INT8 ConvRot weights while the ANE prefix uses an FP16 or INT8
 * Core ML artifact whose ConvRot transforms were folded into the weights. */
ltx_ane_mlp *ltx_ane_mlp_create(ltx_gpu *gpu, const char *manifest_path,
                                const char *variant,
                                char *error, size_t error_size);
void ltx_ane_mlp_free(ltx_ane_mlp *mlp);
const ltx_ane_mlp_shape *ltx_ane_mlp_get_shape(const ltx_ane_mlp *mlp);
int ltx_ane_mlp_supports_rows(const ltx_ane_mlp *mlp, uint32_t rows);
int ltx_ane_mlp_set_rows(ltx_ane_mlp *mlp, ltx_gpu *gpu, uint32_t rows,
                         char *error, size_t error_size);

int ltx_ane_mlp_eval(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *input,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size);
int ltx_ane_mlp_eval_adaln(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     ltx_gpu_buffer *normalized,
                     const ltx_gpu_buffer *input,
                     const ltx_gpu_buffer *scale,
                     const ltx_gpu_buffer *shift,
                     uint32_t rows, uint32_t columns,
                     uint32_t parameter_rows, float epsilon,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size);
int ltx_ane_mlp_eval_residual(ltx_ane_mlp *mlp, ltx_gpu *gpu,
                     ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *residual,
                     const ltx_gpu_buffer *input,
                     const ltx_gpu_buffer *gate,
                     uint32_t rows, uint32_t columns,
                     uint32_t gate_rows,
                     ltx_ane_mlp_timing *timing,
                     char *error, size_t error_size);

#endif
