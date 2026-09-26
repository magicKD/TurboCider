#ifndef LTX_GEMMA_ANE_MLP_H
#define LTX_GEMMA_ANE_MLP_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_gemma_ane_mlp ltx_gemma_ane_mlp;

typedef struct {
    uint32_t layer;
    uint32_t rows;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t ane_intermediate;
    uint32_t gpu_intermediate;
    uint32_t minimum_profitable_rows;
} ltx_gemma_ane_mlp_shape;

typedef struct {
    double total_ms;
    double pack_ms;
    double overlap_ms;
    double gpu_ms;
    double ane_ms;
    double join_ms;
    int ane_output_backing_used;
} ltx_gemma_ane_mlp_timing;

/* Read only the manifest's row policy. Missing crossover entries default to
 * exact-bucket-only execution by returning minimum_profitable_rows == rows. */
int ltx_gemma_ane_mlp_plan(
    const char *manifest_path, uint32_t requested_rows,
    uint32_t *execution_rows, uint32_t *minimum_profitable_rows,
    char *error, size_t error_size);

/* Qualification-only Gemma gated-MLP channel split. The artifact must be a
 * compiled, checkpoint-bound ltx-gemma-ane-mlp-v1 manifest. */
ltx_gemma_ane_mlp *ltx_gemma_ane_mlp_create(
    ltx_gpu *gpu, const char *manifest_path, const char *checkpoint,
    uint32_t layer, uint32_t rows, char *error, size_t error_size);
void ltx_gemma_ane_mlp_free(ltx_gemma_ane_mlp *mlp);
const ltx_gemma_ane_mlp_shape *ltx_gemma_ane_mlp_get_shape(
    const ltx_gemma_ane_mlp *mlp);
uint64_t ltx_gemma_ane_mlp_resident_weight_bytes(
    const ltx_gemma_ane_mlp *mlp);
uint64_t ltx_gemma_ane_mlp_workspace_bytes(
    const ltx_gemma_ane_mlp *mlp);
int ltx_gemma_ane_mlp_eval(
    ltx_gemma_ane_mlp *mlp, ltx_gpu *gpu, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, ltx_gemma_ane_mlp_timing *timing,
    char *error, size_t error_size);

#endif
