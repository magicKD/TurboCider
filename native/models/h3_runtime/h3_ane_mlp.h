#ifndef H3_ANE_MLP_H
#define H3_ANE_MLP_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Historical private-ANE MLP interface retained by the shared DiT source.
 * Shipping builds link h3_ane_disabled.c, so this API always fails closed.
 * The research-only implementation lives under experimental/video/h3/vendor
 * and is never compiled or packaged by tools/native/build.sh. */
typedef struct h3_ane_mlp_io h3_ane_mlp_io;
typedef struct h3_ane_mlp h3_ane_mlp;

typedef struct {
    uint32_t block_index;
    uint32_t rows;
    uint32_t hidden;
    uint32_t ane_intermediate;
    uint32_t gpu_intermediate;
    uint32_t full_intermediate;
    float up_scale;
    float output_scale;
} h3_ane_mlp_plan;

int h3_ane_mlp_available(void);

/* One channel-major F32 IOSurface pair can be shared by every sequential
 * block model with the same rows/hidden geometry. */
h3_ane_mlp_io *h3_ane_mlp_io_create(h3_gpu *gpu, uint32_t rows,
                                    uint32_t hidden,
                                    char *error, size_t error_size);
void h3_ane_mlp_io_free(h3_ane_mlp_io *io);
uint32_t h3_ane_mlp_io_rows(const h3_ane_mlp_io *io);
uint32_t h3_ane_mlp_io_plane_rows(const h3_ane_mlp_io *io);
uint32_t h3_ane_mlp_io_hidden(const h3_ane_mlp_io *io);
h3_gpu_tensor *h3_ane_mlp_io_input(h3_ane_mlp_io *io);
h3_gpu_tensor *h3_ane_mlp_io_output(h3_ane_mlp_io *io);

/* fc1 is BF16 [2*intermediate, hidden], fc2 is BF16
 * [hidden, intermediate]. up_scale/output_scale match the existing Core ML
 * artifacts: the up half is divided by up_scale and FC2 is multiplied by
 * up_scale/output_scale, then callers restore output_scale at the join. */
h3_ane_mlp *h3_ane_mlp_create_bf16_files(
                                    const char *name,
                                    const char *fc1_path,
                                    const char *fc2_path,
                                    uint32_t intermediate,
                                    float up_scale, float output_scale,
                                    h3_ane_mlp_io *io,
                                    char *error, size_t error_size);
/* Experimental per-output-channel symmetric INT8 weights. The source files
 * are still the official BF16 shards; this path does not assume ConvRot or a
 * pre-quantized checkpoint layout. */
h3_ane_mlp *h3_ane_mlp_create_int8_files(
                                    const char *name,
                                    const char *fc1_path,
                                    const char *fc2_path,
                                    uint32_t intermediate,
                                    float up_scale, float output_scale,
                                    h3_ane_mlp_io *io,
                                    char *error, size_t error_size);
h3_ane_mlp *h3_ane_mlp_create_int8_plan(
                                    const char *directory,
                                    uint32_t expected_block,
                                    uint32_t expected_rows,
                                    uint32_t expected_hidden,
                                    h3_ane_mlp_io *io,
                                    h3_ane_mlp_plan *plan,
                                    char *error, size_t error_size);
h3_ane_mlp *h3_ane_mlp_create_bf16_plan(
                                    const char *directory,
                                    uint32_t expected_block,
                                    uint32_t expected_rows,
                                    uint32_t expected_hidden,
                                    h3_ane_mlp_io *io,
                                    h3_ane_mlp_plan *plan,
                                    char *error, size_t error_size);
void h3_ane_mlp_free(h3_ane_mlp *mlp);

int h3_ane_mlp_eval(h3_ane_mlp *mlp, char *error, size_t error_size);
int h3_ane_mlp_start(h3_ane_mlp *mlp, char *error, size_t error_size);
int h3_ane_mlp_wait(h3_ane_mlp *mlp, char *error, size_t error_size);
int h3_ane_mlp_inflight(const h3_ane_mlp *mlp);
int h3_ane_mlp_unload(h3_ane_mlp *mlp, char *error, size_t error_size);
int h3_ane_mlp_reload(h3_ane_mlp *mlp, char *error, size_t error_size);
int h3_ane_mlp_is_loaded(const h3_ane_mlp *mlp);

double h3_ane_mlp_compile_seconds(const h3_ane_mlp *mlp);
int h3_ane_mlp_cache_hit(const h3_ane_mlp *mlp);
int h3_ane_mlp_blob_cache_hit(const h3_ane_mlp *mlp);
uint64_t h3_ane_mlp_weight_bytes(const h3_ane_mlp *mlp);
uint32_t h3_ane_mlp_intermediate(const h3_ane_mlp *mlp);
uint32_t h3_ane_mlp_fc2_chunks(const h3_ane_mlp *mlp);
float h3_ane_mlp_output_scale(const h3_ane_mlp *mlp);
/* Runtime power-of-two guard around the SwiGLU up branch and FC2. The MIL
 * graph divides the up activation by this value and the join restores it,
 * so changing it only shifts FP16 exponents and does not require reloading
 * block weights. */
float h3_ane_mlp_runtime_scale(const h3_ane_mlp *mlp);
float h3_ane_mlp_effective_output_scale(const h3_ane_mlp *mlp);
int h3_ane_mlp_set_runtime_scale(h3_ane_mlp *mlp, float scale,
                                 char *error, size_t error_size);

/* Pack consumes and submits the current Metal command buffer. Unpack leaves
 * a conversion command encoded in the currently open buffer. */
int h3_ane_mlp_pack(h3_ane_mlp *mlp, h3_gpu *gpu,
                    const h3_gpu_tensor *input,
                    char *error, size_t error_size);
int h3_ane_mlp_pack_rows(h3_ane_mlp *mlp, h3_gpu *gpu,
                         const h3_gpu_tensor *input,
                         uint32_t input_row_offset,
                         char *error, size_t error_size);
int h3_ane_mlp_unpack(h3_ane_mlp *mlp, h3_gpu *gpu,
                      h3_gpu_tensor *output,
                      char *error, size_t error_size);
int h3_ane_mlp_unpack_rows_checked(h3_ane_mlp *mlp, h3_gpu *gpu,
                                   h3_gpu_tensor *output,
                                   h3_gpu_tensor *nonfinite_flag,
                                   uint32_t output_row_offset,
                                   uint32_t block_index,
                                   char *error, size_t error_size);
int h3_ane_mlp_unpack_rows_checked_range(h3_ane_mlp *mlp, h3_gpu *gpu,
                                   h3_gpu_tensor *output,
                                   h3_gpu_tensor *range_stats,
                                   uint32_t output_row_offset,
                                   uint32_t block_index,
                                   char *error, size_t error_size);

#endif
