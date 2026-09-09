#include "h3_quant_cache.h"

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    h3_dtype dtype;
    uint64_t rows;
    uint64_t columns;
    int rank;
    h3_quant_cache_field field;
} h3_quant_expected;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int checked_elements(uint64_t rows, uint64_t columns,
                            size_t *elements) {
    if (!elements || (columns && rows > UINT64_MAX / columns)) return 0;
    uint64_t value = rows * columns;
    if (value > SIZE_MAX) return 0;
    *elements = (size_t)value;
    return 1;
}

static int validate_metadata(const h3_st_header *header,
                             uint64_t source_identity, unsigned block,
                             unsigned blocks, uint32_t hidden,
                             uint32_t inner, uint32_t ffn,
                             char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(
        header, H3_QUANT_CACHE_METADATA);
    if (!tensor || tensor->dtype != H3_DTYPE_U64 || tensor->ndim != 1 ||
        tensor->shape[0] != H3_QUANT_CACHE_METADATA_VALUES) {
        fail(error, error_size,
             "H3 quantized cache block %u has invalid metadata schema",
             block);
        return 0;
    }
    uint64_t values[H3_QUANT_CACHE_METADATA_VALUES];
    if (!h3_st_read_data(header, tensor, values, sizeof(values),
                         error, error_size)) return 0;
    if (values[0] != H3_QUANT_CACHE_MAGIC ||
        values[1] != H3_QUANT_CACHE_SCHEMA ||
        values[2] != source_identity || values[3] != block ||
        values[4] != blocks || values[5] != hidden ||
        values[6] != inner || values[7] != ffn) {
        fail(error, error_size,
             "H3 quantized cache block %u provenance/schema mismatch "
             "(source=%016" PRIx64 ")", block, source_identity);
        return 0;
    }
    return 1;
}

static int validate_source(const h3_st_header *header,
                           const h3_quant_expected *expected,
                           h3_quant_cache_source *source,
                           char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, expected->name);
    if (!tensor || tensor->dtype != expected->dtype ||
        tensor->ndim != expected->rank ||
        tensor->shape[0] != expected->rows ||
        (expected->rank == 2 && tensor->shape[1] != expected->columns)) {
        fail(error, error_size,
             "H3 quantized cache tensor %s has the wrong dtype/shape",
             expected->name);
        return 0;
    }
    size_t elements = 0;
    if (!checked_elements(expected->rows, expected->columns, &elements)) {
        fail(error, error_size,
             "H3 quantized cache tensor %s shape overflows",
             expected->name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = elements;
    source->dtype = tensor->dtype;
    source->field = expected->field;
    return 1;
}

void h3_quant_cache_close(h3_quant_cache *cache) {
    if (!cache) return;
    if (cache->layers) {
        for (unsigned block = 0; block < cache->blocks; block++)
            h3_st_free_header(&cache->layers[block].header);
    }
    free(cache->layers);
    memset(cache, 0, sizeof(*cache));
}

int h3_quant_cache_open(h3_quant_cache *cache, const char *directory,
                        uint64_t source_identity, unsigned blocks,
                        uint32_t hidden, uint32_t inner, uint32_t ffn,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache || !directory || !*directory || !source_identity || !blocks ||
        !hidden || !inner || !ffn) {
        fail(error, error_size, "invalid H3 quantized cache request");
        return 0;
    }
    memset(cache, 0, sizeof(*cache));
    cache->source_identity = source_identity;
    cache->blocks = blocks;
    cache->hidden = hidden;
    cache->inner = inner;
    cache->ffn = ffn;
    cache->layers = calloc(blocks, sizeof(*cache->layers));
    if (!cache->layers) {
        fail(error, error_size, "out of memory opening H3 quantized cache");
        return 0;
    }
    const h3_quant_expected expected[H3_QUANT_CACHE_SOURCES] = {
        {"qkv.weight", H3_DTYPE_I8, (uint64_t)inner * 3u, hidden, 2,
         H3_QUANT_QKV_WEIGHT},
        {"qkv.scales", H3_DTYPE_F32, (uint64_t)inner * 3u, 1, 1,
         H3_QUANT_QKV_SCALES},
        {"out.weight", H3_DTYPE_I8, hidden, inner, 2,
         H3_QUANT_OUT_WEIGHT},
        {"out.scales", H3_DTYPE_F32, hidden, 1, 1,
         H3_QUANT_OUT_SCALES},
        {"fc1.weight", H3_DTYPE_I8, (uint64_t)ffn * 2u, hidden, 2,
         H3_QUANT_FC1_WEIGHT},
        {"fc1.scales", H3_DTYPE_F32, (uint64_t)ffn * 2u, 1, 1,
         H3_QUANT_FC1_SCALES},
        {"fc2.weight", H3_DTYPE_I8, hidden, ffn, 2,
         H3_QUANT_FC2_WEIGHT},
        {"fc2.scales", H3_DTYPE_F32, hidden, 1, 1,
         H3_QUANT_FC2_SCALES},
    };
    uint64_t first_block_bytes = 0;
    for (unsigned block = 0; block < blocks; block++) {
        char path[PATH_MAX];
        int count = snprintf(path, sizeof(path), "%s/block-%02u.safetensors",
                             directory, block);
        if (count < 0 || (size_t)count >= sizeof(path)) {
            fail(error, error_size, "H3 quantized cache path is too long");
            goto failed;
        }
        h3_quant_cache_layer *layer = &cache->layers[block];
        if (!h3_st_read_header(path, &layer->header, error, error_size) ||
            !validate_metadata(&layer->header, source_identity, block, blocks,
                               hidden, inner, ffn, error, error_size))
            goto failed;
        uint64_t block_bytes = 0;
        for (unsigned index = 0; index < H3_QUANT_CACHE_SOURCES; index++) {
            if (!validate_source(&layer->header, &expected[index],
                                 &layer->sources[index],
                                 error, error_size)) goto failed;
            size_t item_size = h3_dtype_size(expected[index].dtype);
            size_t elements = layer->sources[index].elements;
            if (!item_size || elements > UINT64_MAX / item_size) {
                fail(error, error_size,
                     "H3 quantized cache tensor %s byte size overflows",
                     expected[index].name);
                goto failed;
            }
            block_bytes += (uint64_t)elements * item_size;
        }
        if (!block) first_block_bytes = block_bytes;
        else if (block_bytes != first_block_bytes) {
            fail(error, error_size,
                 "H3 quantized cache block %u byte size differs", block);
            goto failed;
        }
    }
    cache->block_bytes = first_block_bytes;
    return 1;

failed:
    h3_quant_cache_close(cache);
    return 0;
}

const h3_quant_cache_layer *h3_quant_cache_layer_at(
                        const h3_quant_cache *cache, unsigned block) {
    if (!cache || !cache->layers || block >= cache->blocks) return NULL;
    return &cache->layers[block];
}
