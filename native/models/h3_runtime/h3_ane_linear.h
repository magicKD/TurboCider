#ifndef H3_ANE_LINEAR_H
#define H3_ANE_LINEAR_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Historical private-ANE projection interface retained by the shared DiT
 * source. Shipping builds link h3_ane_disabled.c, so this API always fails
 * closed. The implementation is isolated under experimental/video/h3/vendor
 * and is never compiled or packaged by tools/native/build.sh. */
typedef struct h3_ane_linear_io h3_ane_linear_io;
typedef struct h3_ane_linear h3_ane_linear;

/* A retained checkpoint range lets callers discard a transient ANE model and
 * recreate it later without keeping another copy of the weight tensor.
 * Zero-initialize the recipe before its first set. */
typedef struct {
    char *weight_path;
    uint64_t file_offset;
} h3_ane_linear_recipe;

int h3_ane_linear_recipe_set(h3_ane_linear_recipe *recipe,
                             const char *weight_path, uint64_t file_offset,
                             char *error, size_t error_size);
int h3_ane_linear_recipe_present(const h3_ane_linear_recipe *recipe);
void h3_ane_linear_recipe_clear(h3_ane_linear_recipe *recipe);
int h3_ane_linear_transient_requested(const char *value);

int h3_ane_linear_available(void);

h3_ane_linear_io *h3_ane_linear_io_create(h3_gpu *gpu, uint32_t rows,
                                          uint32_t input_width,
                                          uint32_t output_width,
                                          char *error, size_t error_size);
void h3_ane_linear_io_free(h3_ane_linear_io *io);
uint32_t h3_ane_linear_io_rows(const h3_ane_linear_io *io);
uint32_t h3_ane_linear_io_plane_rows(const h3_ane_linear_io *io);
uint32_t h3_ane_linear_io_input_width(const h3_ane_linear_io *io);
uint32_t h3_ane_linear_io_output_width(const h3_ane_linear_io *io);
h3_gpu_tensor *h3_ane_linear_io_output(h3_ane_linear_io *io);

h3_ane_linear *h3_ane_linear_create_bf16_file(
                                          const char *name,
                                          const char *weight_path,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size);
h3_ane_linear *h3_ane_linear_create_bf16_file_range(
                                          const char *name,
                                          const char *weight_path,
                                          uint64_t file_offset,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size);
h3_ane_linear *h3_ane_linear_create_int8_file(
                                          const char *name,
                                          const char *weight_path,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size);
h3_ane_linear *h3_ane_linear_create_int8_file_range(
                                          const char *name,
                                          const char *weight_path,
                                          uint64_t file_offset,
                                          h3_ane_linear_io *io,
                                          char *error, size_t error_size);
void h3_ane_linear_free(h3_ane_linear *linear);

int h3_ane_linear_start(h3_ane_linear *linear,
                        char *error, size_t error_size);
int h3_ane_linear_wait(h3_ane_linear *linear,
                       char *error, size_t error_size);
int h3_ane_linear_inflight(const h3_ane_linear *linear);
int h3_ane_linear_unload(h3_ane_linear *linear,
                         char *error, size_t error_size);
int h3_ane_linear_reload(h3_ane_linear *linear,
                         char *error, size_t error_size);
int h3_ane_linear_is_loaded(const h3_ane_linear *linear);

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear);
int h3_ane_linear_cache_hit(const h3_ane_linear *linear);
int h3_ane_linear_blob_cache_hit(const h3_ane_linear *linear);
uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear);

/* Pack consumes and submits the current Metal command buffer. */
int h3_ane_linear_pack(h3_ane_linear *linear, h3_gpu *gpu,
                       const h3_gpu_tensor *input,
                       char *error, size_t error_size);

#endif
