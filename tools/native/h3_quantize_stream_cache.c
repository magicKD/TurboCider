#include "h3_gpu.h"
#include "h3_quant_cache.h"
#include "h3_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    HIDDEN = 5376,
    INNER = 7168,
    FFN = 14336,
    BLOCKS = 50
};

typedef struct {
    const char *cache_name;
    const char *source_suffix;
    uint32_t rows;
    uint32_t columns;
} matrix_spec;

static const matrix_spec matrices[] = {
    {"qkv", "attn.qkv_proj.weight", INNER * 3, HIDDEN},
    {"out", "attn.out_proj.weight", HIDDEN, INNER},
    {"fc1", "mlp.fc1.weight", FFN * 2, HIDDEN},
    {"fc2", "mlp.fc2.weight", HIDDEN, FFN},
};

static int write_all(FILE *stream, const void *data, size_t bytes) {
    return !bytes || fwrite(data, 1, bytes, stream) == bytes;
}

static int write_u64_le(FILE *stream, uint64_t value) {
    unsigned char encoded[8];
    for (unsigned index = 0; index < 8; index++)
        encoded[index] = (unsigned char)(value >> (index * 8));
    return write_all(stream, encoded, sizeof(encoded));
}

static int append(char *buffer, size_t capacity, size_t *length,
                  const char *format, ...) {
    if (!buffer || !length || *length >= capacity) return 0;
    va_list arguments;
    va_start(arguments, format);
    int count = vsnprintf(buffer + *length, capacity - *length,
                          format, arguments);
    va_end(arguments);
    if (count < 0 || (size_t)count >= capacity - *length) return 0;
    *length += (size_t)count;
    return 1;
}

static int write_header(FILE *stream, uint64_t source_identity,
                        unsigned block, uint64_t *data_bytes) {
    char json[8192];
    size_t length = 0;
    uint64_t offset = H3_QUANT_CACHE_METADATA_VALUES * sizeof(uint64_t);
    if (!append(json, sizeof(json), &length,
                "{\"%s\":{\"dtype\":\"U64\",\"shape\":[%u],"
                "\"data_offsets\":[0,%llu]}",
                H3_QUANT_CACHE_METADATA, H3_QUANT_CACHE_METADATA_VALUES,
                (unsigned long long)offset)) return 0;
    for (size_t index = 0; index < sizeof(matrices) / sizeof(*matrices);
         index++) {
        const matrix_spec *matrix = &matrices[index];
        uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
        uint64_t weight_end = offset + elements;
        uint64_t scale_end = weight_end + (uint64_t)matrix->rows * sizeof(float);
        if (!append(json, sizeof(json), &length,
                    ",\"%s.weight\":{\"dtype\":\"I8\","
                    "\"shape\":[%u,%u],\"data_offsets\":[%llu,%llu]},"
                    "\"%s.scales\":{\"dtype\":\"F32\","
                    "\"shape\":[%u],\"data_offsets\":[%llu,%llu]}",
                    matrix->cache_name, matrix->rows, matrix->columns,
                    (unsigned long long)offset,
                    (unsigned long long)weight_end,
                    matrix->cache_name, matrix->rows,
                    (unsigned long long)weight_end,
                    (unsigned long long)scale_end)) return 0;
        offset = scale_end;
    }
    if (!append(json, sizeof(json), &length, "}")) return 0;
    while (length % 8) {
        if (length >= sizeof(json)) return 0;
        json[length++] = ' ';
    }
    if (!write_u64_le(stream, length) || !write_all(stream, json, length))
        return 0;
    const uint64_t metadata[H3_QUANT_CACHE_METADATA_VALUES] = {
        H3_QUANT_CACHE_MAGIC, H3_QUANT_CACHE_SCHEMA, source_identity, block,
        BLOCKS, HIDDEN, INNER, FFN
    };
    for (unsigned index = 0; index < H3_QUANT_CACHE_METADATA_VALUES; index++)
        if (!write_u64_le(stream, metadata[index])) return 0;
    *data_bytes = offset;
    return 1;
}

static int quantize_matrix(h3_weight_store *weights, h3_gpu *gpu,
                           FILE *stream, unsigned block,
                           const matrix_spec *matrix,
                           char *error, size_t error_size) {
    char name[160];
    int count = snprintf(name, sizeof(name), "blocks.%u.%s", block,
                         matrix->source_suffix);
    if (count < 0 || (size_t)count >= sizeof(name)) return 0;
    uint64_t shape[] = {matrix->rows, matrix->columns};
    size_t elements = (size_t)matrix->rows * matrix->columns;
    h3_gpu_tensor *source = h3_weight_load_bf16(
        weights, gpu, name, 2, shape, error, error_size);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, matrix->rows);
    int ok = source && quantized && scales && h3_gpu_begin(gpu) &&
        h3_gpu_quantize_weight_int8(gpu, quantized, scales, source,
                                    matrix->rows, matrix->columns) &&
        h3_gpu_submit(gpu);
    if (!ok) {
        if (error && error_size && !error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", name,
                     h3_gpu_error(gpu));
    } else if (!write_all(stream, h3_gpu_tensor_host_pointer(quantized),
                          elements) ||
               !write_all(stream, h3_gpu_tensor_host_pointer(scales),
                          (size_t)matrix->rows * sizeof(float))) {
        if (error && error_size)
            snprintf(error, error_size, "cannot write quantized %s: %s",
                     name, strerror(errno));
        ok = 0;
    }
    h3_gpu_tensor_free(source);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(scales);
    return ok;
}

static int output_directory(const char *path, char *error,
                            size_t error_size) {
    struct stat status;
    if (stat(path, &status) == 0) {
        if (S_ISDIR(status.st_mode)) return 1;
        snprintf(error, error_size, "output exists and is not a directory: %s",
                 path);
        return 0;
    }
    if (errno != ENOENT || mkdir(path, 0755) != 0) {
        snprintf(error, error_size, "cannot create output directory %s: %s",
                 path, strerror(errno));
        return 0;
    }
    return 1;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --transformer DIR --shader PATH --output DIR [--force]\n",
            program);
}

int main(int argc, char **argv) {
    const char *transformer = NULL;
    const char *shader = NULL;
    const char *output = NULL;
    int force = 0;
    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--force")) force = 1;
        else if (index + 1 < argc && !strcmp(argv[index], "--transformer"))
            transformer = argv[++index];
        else if (index + 1 < argc && !strcmp(argv[index], "--shader"))
            shader = argv[++index];
        else if (index + 1 < argc && !strcmp(argv[index], "--output"))
            output = argv[++index];
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!transformer || !shader || !output) {
        usage(argv[0]);
        return 2;
    }
    char error[512] = {0};
    if (!output_directory(output, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    h3_weight_store *weights = h3_weight_store_open(
        transformer, error, sizeof(error));
    if (!weights) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    uint64_t source_identity = 0;
    h3_gpu *gpu = NULL;
    int result = 1;
    if (!h3_weight_store_identity(weights, &source_identity,
                                  error, sizeof(error))) goto done;
    gpu = h3_gpu_create(shader, error, sizeof(error));
    if (!gpu) goto done;
    if (!h3_gpu_has_int8_mlp(gpu)) {
        snprintf(error, sizeof(error),
                 "H3 INT8 cache generation requires an M5-class Metal 4 GPU");
        goto done;
    }
    uint64_t total_data_bytes = 0;
    for (unsigned block = 0; block < BLOCKS; block++) {
        char final_path[PATH_MAX];
        char temporary_path[PATH_MAX];
        int final_count = snprintf(final_path, sizeof(final_path),
                                   "%s/block-%02u.safetensors", output, block);
        int temporary_count = snprintf(temporary_path, sizeof(temporary_path),
                                       "%s/.block-%02u.tmp.%ld", output,
                                       block, (long)getpid());
        if (final_count < 0 || (size_t)final_count >= sizeof(final_path) ||
            temporary_count < 0 ||
            (size_t)temporary_count >= sizeof(temporary_path)) {
            snprintf(error, sizeof(error), "quantized cache path is too long");
            goto done;
        }
        if (!force && access(final_path, F_OK) == 0) {
            snprintf(error, sizeof(error),
                     "%s already exists; use --force to replace derived shards",
                     final_path);
            goto done;
        }
        FILE *stream = fopen(temporary_path, "wb");
        if (!stream) {
            snprintf(error, sizeof(error), "cannot create %s: %s",
                     temporary_path, strerror(errno));
            goto done;
        }
        uint64_t block_data_bytes = 0;
        int ok = write_header(stream, source_identity, block,
                              &block_data_bytes);
        for (size_t index = 0; ok &&
             index < sizeof(matrices) / sizeof(*matrices); index++) {
            fprintf(stderr, "h3 quant cache: block %u/%u %s\n",
                    block + 1, BLOCKS, matrices[index].cache_name);
            ok = quantize_matrix(weights, gpu, stream, block,
                                 &matrices[index], error, sizeof(error));
        }
        if (ok && fflush(stream) != 0) ok = 0;
        if (ok && fsync(fileno(stream)) != 0) ok = 0;
        if (fclose(stream) != 0) ok = 0;
        if (!ok) {
            if (!error[0])
                snprintf(error, sizeof(error), "cannot publish block %u: %s",
                         block, strerror(errno));
            unlink(temporary_path);
            goto done;
        }
        if (rename(temporary_path, final_path) != 0) {
            snprintf(error, sizeof(error), "cannot publish %s: %s",
                     final_path, strerror(errno));
            unlink(temporary_path);
            goto done;
        }
        total_data_bytes += block_data_bytes;
    }
    h3_quant_cache cache;
    if (!h3_quant_cache_open(&cache, output, source_identity, BLOCKS,
                             HIDDEN, INNER, FFN, error, sizeof(error)))
        goto done;
    uint64_t block_bytes = cache.block_bytes;
    h3_quant_cache_close(&cache);
    printf("{\"schema\":\"h3-quantized-stream-cache-v1\","
           "\"source_identity\":\"%016" PRIx64 "\","
           "\"blocks\":%u,\"block_bytes\":%llu,"
           "\"payload_bytes\":%llu,\"output\":\"%s\"}\n",
           source_identity, BLOCKS, (unsigned long long)block_bytes,
           (unsigned long long)total_data_bytes, output);
    result = 0;

done:
    if (result && error[0]) fprintf(stderr, "%s\n", error);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return result;
}
