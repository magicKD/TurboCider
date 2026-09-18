#include "h3_weights.h"

#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

static int compare_source_paths(const void *left, const void *right) {
    const h3_weight_source_v1 *const *a = left;
    const h3_weight_source_v1 *const *b = right;
    return strcmp((*a)->path, (*b)->path);
}

static h3_weight_store *allocate_store(size_t count,
                                       char *error, size_t error_size) {
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++)
        store->headers[index].descriptor = -1;
    return store;
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = allocate_store(count, error, error_size);
    if (!store) {
        free_paths(paths, count);
        return NULL;
    }
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

h3_weight_store *h3_weight_store_open_sources(
        const h3_weight_source_v1 *sources, size_t source_count,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!sources || !source_count) {
        fail(error, error_size, "H3 source lease is empty");
        return NULL;
    }
    const h3_weight_source_v1 **ordered =
        calloc(source_count, sizeof(*ordered));
    if (!ordered) {
        fail(error, error_size, "out of memory sorting H3 source lease");
        return NULL;
    }
    for (size_t index = 0; index < source_count; index++) {
        if (!sources[index].path || !*sources[index].path ||
            sources[index].descriptor < 0) {
            fail(error, error_size, "invalid H3 source lease entry");
            free(ordered);
            return NULL;
        }
        ordered[index] = &sources[index];
    }
    qsort(ordered, source_count, sizeof(*ordered), compare_source_paths);
    for (size_t index = 1; index < source_count; index++) {
        if (!strcmp(ordered[index - 1]->path, ordered[index]->path)) {
            fail(error, error_size, "duplicate H3 source lease path: %s",
                 ordered[index]->path);
            free(ordered);
            return NULL;
        }
    }
    h3_weight_store *store = allocate_store(
        source_count, error, error_size);
    if (!store) {
        free(ordered);
        return NULL;
    }
    for (size_t index = 0; index < source_count; index++) {
        char detail[384] = {};
        if (!h3_st_read_header_fd(
                ordered[index]->path, ordered[index]->descriptor,
                &store->headers[index], detail, sizeof(detail))) {
            fail(error, error_size, "%s",
                 detail[0] ? detail : "cannot read H3 leased header");
            free(ordered);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free(ordered);
    return store;
}

int h3_weight_store_bind_sources(
        h3_weight_store *store, const h3_weight_source_v1 *sources,
        size_t source_count, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !store->count || !sources || !source_count) {
        fail(error, error_size, "invalid H3 source lease binding");
        return 0;
    }
    int *duplicates = malloc(store->count * sizeof(*duplicates));
    if (!duplicates) {
        fail(error, error_size,
             "out of memory binding H3 source lease descriptors");
        return 0;
    }
    for (size_t index = 0; index < store->count; index++)
        duplicates[index] = -1;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_header *header = &store->headers[index];
        char *canonical = header->path ? realpath(header->path, NULL) : NULL;
        int matched = -1;
        if (!canonical) {
            fail(error, error_size, "cannot resolve H3 shard %s",
                 header->path ? header->path : "(null)");
            goto failed;
        }
        for (size_t source = 0; source < source_count; source++) {
            if (!sources[source].path || sources[source].descriptor < 0)
                continue;
            char *source_canonical = realpath(sources[source].path, NULL);
            const int equal = source_canonical &&
                !strcmp(canonical, source_canonical);
            free(source_canonical);
            if (equal) {
                matched = (int)source;
                break;
            }
        }
        free(canonical);
        if (matched < 0) {
            fail(error, error_size,
                 "H3 source lease is missing transformer shard %s",
                 header->path ? header->path : "(null)");
            goto failed;
        }
        int duplicate = dup(sources[matched].descriptor);
        if (duplicate < 0) {
            fail(error, error_size, "cannot duplicate H3 source lease fd: %s",
                 strerror(errno));
            goto failed;
        }
        struct stat status;
        if (fstat(duplicate, &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_size < 0 ||
            (uint64_t)status.st_size != header->file_size) {
            fail(error, error_size,
                 "H3 source lease fd differs from transformer shard %s",
                 header->path ? header->path : "(null)");
            close(duplicate);
            goto failed;
        }
        duplicates[index] = duplicate;
    }
    for (size_t index = 0; index < store->count; index++) {
        if (store->headers[index].descriptor >= 0)
            close(store->headers[index].descriptor);
        store->headers[index].descriptor = duplicates[index];
        duplicates[index] = -1;
    }
    free(duplicates);
    return 1;

failed:
    for (size_t index = 0; index < store->count; index++)
        if (duplicates[index] >= 0) close(duplicates[index]);
    free(duplicates);
    return 0;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_header *h3_weight_store_header(const h3_weight_store *store,
                                           size_t index) {
    return store && index < store->count ? &store->headers[index] : NULL;
}

static void identity_bytes(uint64_t *hash, const void *data, size_t bytes) {
    const uint8_t *octets = data;
    for (size_t index = 0; index < bytes; index++) {
        *hash ^= octets[index];
        *hash *= UINT64_C(1099511628211);
    }
}

int h3_weight_store_identity(const h3_weight_store *store, uint64_t *identity,
                             char *error, size_t error_size) {
    if (!store || !identity || !store->count) {
        fail(error, error_size, "invalid weight-store identity request");
        return 0;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    uint64_t count = store->count;
    identity_bytes(&hash, &count, sizeof(count));
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_header *header = &store->headers[index];
        char *canonical_path = header->path ? realpath(header->path, NULL) : NULL;
        struct stat status;
        const int stat_ok = header->descriptor >= 0 ?
            fstat(header->descriptor, &status) :
            (canonical_path ? stat(canonical_path, &status) : -1);
        if (!canonical_path || stat_ok != 0) {
            fail(error, error_size, "cannot resolve weight shard %s",
                 header->path ? header->path : "(null)");
            free(canonical_path);
            return 0;
        }
        size_t path_length = strlen(canonical_path);
        uint64_t metadata[] = {
            path_length,
            header->file_size,
            header->header_size,
            header->tensor_count,
            (uint64_t)status.st_dev,
            (uint64_t)status.st_ino,
            (uint64_t)status.st_size,
            (uint64_t)status.st_mtimespec.tv_sec,
            (uint64_t)status.st_mtimespec.tv_nsec,
            (uint64_t)status.st_ctimespec.tv_sec,
            (uint64_t)status.st_ctimespec.tv_nsec
        };
        identity_bytes(&hash, metadata, sizeof(metadata));
        identity_bytes(&hash, canonical_path, path_length);
        free(canonical_path);
    }
    *identity = hash;
    return 1;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}
