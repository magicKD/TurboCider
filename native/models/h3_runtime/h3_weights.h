#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
/* Build a store directly from request-scoped sources. Each header retains its
 * own duplicate descriptor, and no directory enumeration/path reopen occurs. */
h3_weight_store *h3_weight_store_open_sources(
    const h3_weight_source_v1 *sources, size_t source_count,
    char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);
/* Borrowed header view in the same sorted order used by the store. */
const h3_st_header *h3_weight_store_header(const h3_weight_store *store,
                                           size_t index);
/* Stable for an unchanged local shard set. The identity covers every sorted
 * canonical shard path plus its file metadata, so relative/absolute aliases
 * agree while derived caches fail closed when a checkpoint is replaced. */
int h3_weight_store_identity(const h3_weight_store *store, uint64_t *identity,
                             char *error, size_t error_size);
/* Bind duplicate descriptors from a request-scoped SourceLease to every
 * matching shard. The store retains its own dup() and closes it on free. */
int h3_weight_store_bind_sources(
    h3_weight_store *store, const h3_weight_source_v1 *sources,
    size_t source_count, char *error, size_t error_size);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
