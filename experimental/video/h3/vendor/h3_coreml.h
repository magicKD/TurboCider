#ifndef H3_COREML_H
#define H3_COREML_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_coreml_mlp h3_coreml_mlp;

enum { H3_COREML_SHA256_HEX_SIZE = 65 };

typedef struct {
    uint32_t block_index;
    uint32_t rows;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t full_intermediate;
    float output_scale;
    uint64_t checkpoint_identity;
} h3_coreml_mlp_manifest_expectation;

typedef struct {
    int enabled;
    int forced;
    int latch;
} h3_coreml_mlp_fallback_options;

/* Parse the three fallback environment values. Values are either absent/0 or
 * exactly 1; force and latch require the explicit GPU fallback opt-in. */
int h3_coreml_mlp_configure_fallback(
    const char *enabled_value, const char *forced_value,
    const char *latch_value, h3_coreml_mlp_fallback_options *options,
    char *error, size_t error_size);

/* Validate a generated MLP package manifest before Core ML is allowed to
 * load. The manifest binds the block/shape/split/checkpoint identity to the
 * exact GPU complement and ANE-prefix fallback bytes and to a SHA-256 stored
 * in the Core ML model's creator-defined metadata. Supplied weight pointers
 * are already loaded Metal shared buffers, avoiding a second SSD read solely
 * for hashing. Passing pointers as NULL performs schema/size validation before
 * those buffers are loaded. Runtime ANE fallback preflight and lazy mapped-byte
 * verification use validate_fallback_shards below. ANE fallback sizes may both
 * be zero when exact GPU fallback is disabled; their entries are then ignored. */
int h3_coreml_mlp_validate_manifest(
    const char *manifest_path, const char *model_path,
    const char *checkpoint_index_path,
    const void *gpu_fc1, size_t gpu_fc1_bytes,
    const void *gpu_fc2, size_t gpu_fc2_bytes,
    const void *ane_fc1, size_t ane_fc1_bytes,
    const void *ane_fc2, size_t ane_fc2_bytes,
    const h3_coreml_mlp_manifest_expectation *expected,
    char manifest_sha256[H3_COREML_SHA256_HEX_SIZE],
    char *error, size_t error_size);

/* Validate only the exact GPU fallback artifacts after the main manifest has
 * already been accepted. Both shard paths must name the fixed ANE shard files
 * beside the manifest. NULL data pointers perform a cheap preflight that
 * checks the manifest identity, entries, file existence, and exact sizes.
 * Supplying both pointers additionally hashes the actual mapped bytes. */
int h3_coreml_mlp_validate_fallback_shards(
    const char *manifest_path, const char *expected_manifest_sha256,
    const char *ane_fc1_path, const void *ane_fc1, size_t ane_fc1_bytes,
    const char *ane_fc2_path, const void *ane_fc2, size_t ane_fc2_bytes,
    char *error, size_t error_size);

/* Count FP16 infinities and NaNs without converting them. The fallback helper
 * also makes forced-fallback policy deterministic and independently testable. */
size_t h3_coreml_f16_nonfinite_count(const uint16_t *values, size_t elements);
int h3_coreml_mlp_fallback_required(const uint16_t *values, size_t elements,
                                    int forced,
                                    size_t *nonfinite_count);

/* Load a fixed-shape channel-major [1,H,1,S] FP16 MLP model on the public
 * CPU+NeuralEngine Core ML path. Input/output storage is owned by Metal and
 * exposed to Core ML as external MLMultiArrays. */
h3_coreml_mlp *h3_coreml_mlp_create(h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    char *error, size_t error_size);
/* Begin model compilation/loading asynchronously. Shared Metal tensors are
 * allocated before this returns, so callers may overlap Core ML loading with
 * subsequent transformer weight I/O. finish_loading is idempotent. */
h3_coreml_mlp *h3_coreml_mlp_create_async(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    char *error, size_t error_size);
/* Variant for sequential model sets that reuse one Metal-owned input/output
 * pair. The caller owns both tensors and must keep them alive until every
 * model using them has been freed. */
h3_coreml_mlp *h3_coreml_mlp_create_async_shared(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    char *error, size_t error_size);
/* Generic fixed-shape projection variant. It retains the same resident Core
 * ML session and shared-buffer behavior, but permits unequal channel counts
 * for output-segmented QKV experiments. */
h3_coreml_mlp *h3_coreml_mlp_create_async_shared_io(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t input_width,
                                    uint32_t output_width,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    char *error, size_t error_size);
/* Strict variant used by H3 inference. Loading fails unless the Core ML
 * package carries the exact SHA-256 returned by validate_manifest. */
h3_coreml_mlp *h3_coreml_mlp_create_async_shared_verified(
                                    h3_gpu *gpu, const char *model_path,
                                    uint32_t rows, uint32_t hidden,
                                    h3_gpu_tensor *shared_input,
                                    h3_gpu_tensor *shared_output,
                                    const char *manifest_sha256,
                                    char *error, size_t error_size);
int h3_coreml_mlp_finish_loading(h3_coreml_mlp *mlp,
                                 char *error, size_t error_size);
int h3_coreml_mlp_warmup(h3_coreml_mlp *mlp, unsigned iterations,
                         char *error, size_t error_size);
void h3_coreml_mlp_free(h3_coreml_mlp *mlp);
h3_gpu_tensor *h3_coreml_mlp_input(h3_coreml_mlp *mlp);
const h3_gpu_tensor *h3_coreml_mlp_output(const h3_coreml_mlp *mlp);
int h3_coreml_mlp_prediction_inflight(const h3_coreml_mlp *mlp);

/* Start an asynchronous ANE prediction and wait for its caller-provided
 * output backing. Only one prediction may be in flight per model. */
int h3_coreml_mlp_start(h3_coreml_mlp *mlp,
                        char *error, size_t error_size);
int h3_coreml_mlp_wait(h3_coreml_mlp *mlp,
                       char *error, size_t error_size);

#endif
