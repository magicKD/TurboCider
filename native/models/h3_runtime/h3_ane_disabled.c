#include "h3_ane_linear.h"
#include "h3_ane_mlp.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * The shipping H3 runtime retains the historical call sites so the public
 * Core ML path can share one upstream-derived DiT implementation, but it must
 * never link Apple's private ANE classes.  The research implementation lives
 * under experimental/video/h3/vendor; these stubs make every private route
 * fail closed in the product build.
 */

static void private_ane_disabled(char *error, size_t error_size) {
    if (error && error_size)
        snprintf(error, error_size,
                 "private ANE support is excluded from TurboCider shipping builds");
}

int h3_ane_mlp_available(void) { return 0; }

h3_ane_mlp_io *h3_ane_mlp_io_create(h3_gpu *gpu, uint32_t rows,
                                    uint32_t hidden,
                                    char *error, size_t error_size) {
    (void)gpu; (void)rows; (void)hidden;
    private_ane_disabled(error, error_size);
    return NULL;
}

void h3_ane_mlp_io_free(h3_ane_mlp_io *io) { (void)io; }
uint32_t h3_ane_mlp_io_rows(const h3_ane_mlp_io *io) { (void)io; return 0; }
uint32_t h3_ane_mlp_io_plane_rows(const h3_ane_mlp_io *io) { (void)io; return 0; }
uint32_t h3_ane_mlp_io_hidden(const h3_ane_mlp_io *io) { (void)io; return 0; }
h3_gpu_tensor *h3_ane_mlp_io_input(h3_ane_mlp_io *io) { (void)io; return NULL; }
h3_gpu_tensor *h3_ane_mlp_io_output(h3_ane_mlp_io *io) { (void)io; return NULL; }

static h3_ane_mlp *disabled_mlp(char *error, size_t error_size) {
    private_ane_disabled(error, error_size);
    return NULL;
}

h3_ane_mlp *h3_ane_mlp_create_bf16_files(
    const char *name, const char *fc1_path, const char *fc2_path,
    uint32_t intermediate, float up_scale, float output_scale,
    h3_ane_mlp_io *io, char *error, size_t error_size) {
    (void)name; (void)fc1_path; (void)fc2_path; (void)intermediate;
    (void)up_scale; (void)output_scale; (void)io;
    return disabled_mlp(error, error_size);
}

h3_ane_mlp *h3_ane_mlp_create_int8_files(
    const char *name, const char *fc1_path, const char *fc2_path,
    uint32_t intermediate, float up_scale, float output_scale,
    h3_ane_mlp_io *io, char *error, size_t error_size) {
    (void)name; (void)fc1_path; (void)fc2_path; (void)intermediate;
    (void)up_scale; (void)output_scale; (void)io;
    return disabled_mlp(error, error_size);
}

h3_ane_mlp *h3_ane_mlp_create_int8_plan(
    const char *directory, uint32_t expected_block, uint32_t expected_rows,
    uint32_t expected_hidden, h3_ane_mlp_io *io, h3_ane_mlp_plan *plan,
    char *error, size_t error_size) {
    (void)directory; (void)expected_block; (void)expected_rows;
    (void)expected_hidden; (void)io; (void)plan;
    return disabled_mlp(error, error_size);
}

h3_ane_mlp *h3_ane_mlp_create_bf16_plan(
    const char *directory, uint32_t expected_block, uint32_t expected_rows,
    uint32_t expected_hidden, h3_ane_mlp_io *io, h3_ane_mlp_plan *plan,
    char *error, size_t error_size) {
    (void)directory; (void)expected_block; (void)expected_rows;
    (void)expected_hidden; (void)io; (void)plan;
    return disabled_mlp(error, error_size);
}

void h3_ane_mlp_free(h3_ane_mlp *mlp) { (void)mlp; }

static int disabled_mlp_operation(char *error, size_t error_size) {
    private_ane_disabled(error, error_size);
    return 0;
}

int h3_ane_mlp_eval(h3_ane_mlp *mlp, char *error, size_t error_size) {
    (void)mlp; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_start(h3_ane_mlp *mlp, char *error, size_t error_size) {
    (void)mlp; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_wait(h3_ane_mlp *mlp, char *error, size_t error_size) {
    (void)mlp; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_inflight(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
int h3_ane_mlp_unload(h3_ane_mlp *mlp, char *error, size_t error_size) {
    (void)mlp; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_reload(h3_ane_mlp *mlp, char *error, size_t error_size) {
    (void)mlp; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_is_loaded(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
double h3_ane_mlp_compile_seconds(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
int h3_ane_mlp_cache_hit(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
int h3_ane_mlp_blob_cache_hit(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
uint64_t h3_ane_mlp_weight_bytes(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
uint32_t h3_ane_mlp_intermediate(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
uint32_t h3_ane_mlp_fc2_chunks(const h3_ane_mlp *mlp) { (void)mlp; return 0; }
float h3_ane_mlp_output_scale(const h3_ane_mlp *mlp) { (void)mlp; return 1.0f; }
float h3_ane_mlp_runtime_scale(const h3_ane_mlp *mlp) { (void)mlp; return 1.0f; }
float h3_ane_mlp_effective_output_scale(const h3_ane_mlp *mlp) {
    (void)mlp; return 1.0f;
}
int h3_ane_mlp_set_runtime_scale(h3_ane_mlp *mlp, float scale,
                                 char *error, size_t error_size) {
    (void)mlp; (void)scale; return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_pack(h3_ane_mlp *mlp, h3_gpu *gpu,
                    const h3_gpu_tensor *input,
                    char *error, size_t error_size) {
    (void)mlp; (void)gpu; (void)input;
    return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_pack_rows(h3_ane_mlp *mlp, h3_gpu *gpu,
                         const h3_gpu_tensor *input, uint32_t input_row_offset,
                         char *error, size_t error_size) {
    (void)mlp; (void)gpu; (void)input; (void)input_row_offset;
    return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_unpack(h3_ane_mlp *mlp, h3_gpu *gpu,
                      h3_gpu_tensor *output,
                      char *error, size_t error_size) {
    (void)mlp; (void)gpu; (void)output;
    return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_unpack_rows_checked(
    h3_ane_mlp *mlp, h3_gpu *gpu, h3_gpu_tensor *output,
    h3_gpu_tensor *nonfinite_flag, uint32_t output_row_offset,
    uint32_t block_index, char *error, size_t error_size) {
    (void)mlp; (void)gpu; (void)output; (void)nonfinite_flag;
    (void)output_row_offset; (void)block_index;
    return disabled_mlp_operation(error, error_size);
}
int h3_ane_mlp_unpack_rows_checked_range(
    h3_ane_mlp *mlp, h3_gpu *gpu, h3_gpu_tensor *output,
    h3_gpu_tensor *range_stats, uint32_t output_row_offset,
    uint32_t block_index, char *error, size_t error_size) {
    (void)mlp; (void)gpu; (void)output; (void)range_stats;
    (void)output_row_offset; (void)block_index;
    return disabled_mlp_operation(error, error_size);
}

int h3_ane_linear_recipe_set(h3_ane_linear_recipe *recipe,
                             const char *weight_path, uint64_t file_offset,
                             char *error, size_t error_size) {
    (void)recipe; (void)weight_path; (void)file_offset;
    private_ane_disabled(error, error_size);
    return 0;
}
int h3_ane_linear_recipe_present(const h3_ane_linear_recipe *recipe) {
    return recipe && recipe->weight_path;
}
void h3_ane_linear_recipe_clear(h3_ane_linear_recipe *recipe) {
    if (!recipe) return;
    free(recipe->weight_path);
    recipe->weight_path = NULL;
    recipe->file_offset = 0;
}
int h3_ane_linear_transient_requested(const char *value) {
    (void)value; return 0;
}
int h3_ane_linear_available(void) { return 0; }

h3_ane_linear_io *h3_ane_linear_io_create(
    h3_gpu *gpu, uint32_t rows, uint32_t input_width, uint32_t output_width,
    char *error, size_t error_size) {
    (void)gpu; (void)rows; (void)input_width; (void)output_width;
    private_ane_disabled(error, error_size);
    return NULL;
}
void h3_ane_linear_io_free(h3_ane_linear_io *io) { (void)io; }
uint32_t h3_ane_linear_io_rows(const h3_ane_linear_io *io) { (void)io; return 0; }
uint32_t h3_ane_linear_io_plane_rows(const h3_ane_linear_io *io) { (void)io; return 0; }
uint32_t h3_ane_linear_io_input_width(const h3_ane_linear_io *io) { (void)io; return 0; }
uint32_t h3_ane_linear_io_output_width(const h3_ane_linear_io *io) { (void)io; return 0; }
h3_gpu_tensor *h3_ane_linear_io_output(h3_ane_linear_io *io) { (void)io; return NULL; }

static h3_ane_linear *disabled_linear(char *error, size_t error_size) {
    private_ane_disabled(error, error_size);
    return NULL;
}
h3_ane_linear *h3_ane_linear_create_bf16_file(
    const char *name, const char *weight_path, h3_ane_linear_io *io,
    char *error, size_t error_size) {
    (void)name; (void)weight_path; (void)io;
    return disabled_linear(error, error_size);
}
h3_ane_linear *h3_ane_linear_create_bf16_file_range(
    const char *name, const char *weight_path, uint64_t file_offset,
    h3_ane_linear_io *io, char *error, size_t error_size) {
    (void)name; (void)weight_path; (void)file_offset; (void)io;
    return disabled_linear(error, error_size);
}
h3_ane_linear *h3_ane_linear_create_int8_file(
    const char *name, const char *weight_path, h3_ane_linear_io *io,
    char *error, size_t error_size) {
    (void)name; (void)weight_path; (void)io;
    return disabled_linear(error, error_size);
}
h3_ane_linear *h3_ane_linear_create_int8_file_range(
    const char *name, const char *weight_path, uint64_t file_offset,
    h3_ane_linear_io *io, char *error, size_t error_size) {
    (void)name; (void)weight_path; (void)file_offset; (void)io;
    return disabled_linear(error, error_size);
}
void h3_ane_linear_free(h3_ane_linear *linear) { (void)linear; }

static int disabled_linear_operation(char *error, size_t error_size) {
    private_ane_disabled(error, error_size);
    return 0;
}
int h3_ane_linear_start(h3_ane_linear *linear,
                        char *error, size_t error_size) {
    (void)linear; return disabled_linear_operation(error, error_size);
}
int h3_ane_linear_wait(h3_ane_linear *linear,
                       char *error, size_t error_size) {
    (void)linear; return disabled_linear_operation(error, error_size);
}
int h3_ane_linear_inflight(const h3_ane_linear *linear) { (void)linear; return 0; }
int h3_ane_linear_unload(h3_ane_linear *linear,
                         char *error, size_t error_size) {
    (void)linear; return disabled_linear_operation(error, error_size);
}
int h3_ane_linear_reload(h3_ane_linear *linear,
                         char *error, size_t error_size) {
    (void)linear; return disabled_linear_operation(error, error_size);
}
int h3_ane_linear_is_loaded(const h3_ane_linear *linear) { (void)linear; return 0; }
double h3_ane_linear_compile_seconds(const h3_ane_linear *linear) {
    (void)linear; return 0;
}
int h3_ane_linear_cache_hit(const h3_ane_linear *linear) { (void)linear; return 0; }
int h3_ane_linear_blob_cache_hit(const h3_ane_linear *linear) { (void)linear; return 0; }
uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear) { (void)linear; return 0; }
int h3_ane_linear_pack(h3_ane_linear *linear, h3_gpu *gpu,
                       const h3_gpu_tensor *input,
                       char *error, size_t error_size) {
    (void)linear; (void)gpu; (void)input;
    return disabled_linear_operation(error, error_size);
}
