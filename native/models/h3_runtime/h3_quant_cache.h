#ifndef H3_QUANT_CACHE_H
#define H3_QUANT_CACHE_H

#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

enum {
    H3_QUANT_CACHE_SCHEMA = 1,
    H3_QUANT_CACHE_METADATA_VALUES = 8,
    H3_QUANT_CACHE_SOURCES = 8
};

#define H3_QUANT_CACHE_MAGIC UINT64_C(0x4833515354524d31)
#define H3_QUANT_CACHE_METADATA "__turbocider_h3_quant_cache_v1__"

typedef enum {
    H3_QUANT_QKV_WEIGHT = 0,
    H3_QUANT_QKV_SCALES,
    H3_QUANT_OUT_WEIGHT,
    H3_QUANT_OUT_SCALES,
    H3_QUANT_FC1_WEIGHT,
    H3_QUANT_FC1_SCALES,
    H3_QUANT_FC2_WEIGHT,
    H3_QUANT_FC2_SCALES
} h3_quant_cache_field;

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    h3_dtype dtype;
    h3_quant_cache_field field;
} h3_quant_cache_source;

typedef struct {
    h3_st_header header;
    h3_quant_cache_source sources[H3_QUANT_CACHE_SOURCES];
} h3_quant_cache_layer;

typedef struct {
    uint64_t source_identity;
    uint64_t block_bytes;
    unsigned blocks;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    h3_quant_cache_layer *layers;
} h3_quant_cache;

/* Open a complete directory of block-NN.safetensors files. Every block is
 * provenance-bound to the exact source weight-store identity and validates
 * all I8 weight/F32 row-scale dtypes and shapes before inference begins. */
int h3_quant_cache_open(h3_quant_cache *cache, const char *directory,
                        uint64_t source_identity, unsigned blocks,
                        uint32_t hidden, uint32_t inner, uint32_t ffn,
                        char *error, size_t error_size);
void h3_quant_cache_close(h3_quant_cache *cache);

const h3_quant_cache_layer *h3_quant_cache_layer_at(
                        const h3_quant_cache *cache, unsigned block);

#endif
