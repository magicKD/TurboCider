#include "ltx_connector.h"
#include "ltx_gpu.h"
#include "ltx_safetensors.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s CHECKPOINT RAW_DIR OUTPUT_DIR [WARMUP]\n",
            program);
}

static int checked_bytes(uint64_t elements, size_t element_size,
                         size_t *bytes) {
    if (!bytes || (element_size && elements > SIZE_MAX / element_size))
        return 0;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int make_path(char *path, size_t path_size,
                     const char *directory, const char *name,
                     char *error, size_t error_size) {
    int length = snprintf(path, path_size, "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= path_size) {
        snprintf(error, error_size, "path is too long: %s/%s",
                 directory, name);
        return 0;
    }
    return 1;
}

static int file_size(const char *path, size_t *bytes,
                     char *error, size_t error_size) {
    struct stat info;
    if (stat(path, &info) || info.st_size < 0 ||
        (uint64_t)info.st_size > SIZE_MAX) {
        snprintf(error, error_size, "cannot stat %s: %s", path,
                 strerror(errno));
        return 0;
    }
    *bytes = (size_t)info.st_size;
    return 1;
}

static int read_file(const char *path, void *data, size_t bytes,
                     char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "cannot open %s: %s", path,
                 strerror(errno));
        return 0;
    }
    int ok = fread(data, 1u, bytes, file) == bytes && !ferror(file);
    if (fclose(file) && ok) ok = 0;
    if (!ok)
        snprintf(error, error_size, "cannot read exactly %zu bytes from %s",
                 bytes, path);
    return ok;
}

static int write_file(const char *path, const void *data, size_t bytes,
                      char *error, size_t error_size) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        snprintf(error, error_size, "cannot create %s: %s", path,
                 strerror(errno));
        return 0;
    }
    int ok = fwrite(data, 1u, bytes, file) == bytes && !ferror(file);
    if (fclose(file) && ok) ok = 0;
    if (!ok)
        snprintf(error, error_size, "cannot write exactly %zu bytes to %s",
                 bytes, path);
    return ok;
}

static int ensure_directory(const char *path,
                            char *error, size_t error_size) {
    if (!mkdir(path, 0755)) return 1;
    if (errno == EEXIST) {
        struct stat info;
        if (!stat(path, &info) && S_ISDIR(info.st_mode)) return 1;
    }
    snprintf(error, error_size, "cannot create output directory %s: %s",
             path, strerror(errno));
    return 0;
}

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint32_t parse_warmup(const char *text) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)value;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void tensor_stats(const uint16_t *values, uint64_t elements,
                         uint64_t *non_finite, double *rms) {
    uint64_t invalid = 0;
    long double sum_square = 0.0L;
    for (uint64_t index = 0; index < elements; index++) {
        float value = bf16_to_f32(values[index]);
        if (!isfinite(value)) {
            invalid++;
        } else {
            sum_square += (long double)value * value;
        }
    }
    *non_finite = invalid;
    *rms = elements ?
        (double)sqrtl(sum_square / (long double)elements) : 0.0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint = argv[1];
    const char *raw_directory = argv[2];
    const char *output_directory = argv[3];
    uint32_t warmup = argc == 5 ? parse_warmup(argv[4]) : 0u;
    if (warmup == UINT32_MAX) {
        fprintf(stderr, "run_connector: invalid warmup count: %s\n", argv[4]);
        return 2;
    }

    int result = 1;
    char error[1024] = {0};
    char video_path[4096];
    char audio_path[4096];
    char output_path[4096];
    ltx_st_header header = {0};
    ltx_st_mapping mapping = {0};
    ltx_gpu *gpu = NULL;
    ltx_connector *connector = NULL;
    ltx_gpu_buffer *video_input = NULL;
    ltx_gpu_buffer *audio_input = NULL;
    ltx_gpu_buffer *video_output = NULL;
    ltx_gpu_buffer *audio_output = NULL;
    uint16_t *video_host = NULL;
    uint16_t *audio_host = NULL;
    uint16_t *mask_host = NULL;

    if (!make_path(video_path, sizeof(video_path), raw_directory,
                   "raw_video_context.bf16", error, sizeof(error)) ||
        !make_path(audio_path, sizeof(audio_path), raw_directory,
                   "raw_audio_context.bf16", error, sizeof(error)) ||
        !ensure_directory(output_directory, error, sizeof(error)) ||
        !ltx_st_read_header(checkpoint, &header, error, sizeof(error)) ||
        !ltx_st_map_open(&header, &mapping, error, sizeof(error)))
        goto cleanup;

    gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    if (!gpu) goto cleanup;
    connector = ltx_connector_load(
        &header, &mapping, gpu, "model.diffusion_model",
        error, sizeof(error));
    if (!connector) goto cleanup;

    uint32_t video_dim = ltx_connector_video_dim(connector);
    uint32_t audio_dim = ltx_connector_audio_dim(connector);
    size_t video_file_bytes = 0;
    size_t audio_file_bytes = 0;
    size_t video_row_bytes = 0;
    size_t audio_row_bytes = 0;
    if (!checked_bytes(video_dim, sizeof(uint16_t), &video_row_bytes) ||
        !checked_bytes(audio_dim, sizeof(uint16_t), &audio_row_bytes) ||
        !file_size(video_path, &video_file_bytes, error, sizeof(error)) ||
        !file_size(audio_path, &audio_file_bytes, error, sizeof(error)) ||
        !video_file_bytes || video_file_bytes % video_row_bytes ||
        !audio_file_bytes || audio_file_bytes % audio_row_bytes ||
        video_file_bytes / video_row_bytes !=
            audio_file_bytes / audio_row_bytes ||
        video_file_bytes / video_row_bytes > UINT32_MAX) {
        if (!error[0])
            snprintf(error, sizeof(error),
                     "raw video/audio context geometry does not match");
        goto cleanup;
    }
    uint32_t input_rows = (uint32_t)(video_file_bytes / video_row_bytes);
    uint32_t output_rows = ltx_connector_output_rows(connector, input_rows);
    size_t video_output_bytes = 0;
    size_t audio_output_bytes = 0;
    size_t mask_bytes = 0;
    if (!output_rows ||
        !checked_bytes((uint64_t)output_rows * video_dim,
                       sizeof(uint16_t), &video_output_bytes) ||
        !checked_bytes((uint64_t)output_rows * audio_dim,
                       sizeof(uint16_t), &audio_output_bytes) ||
        !checked_bytes(output_rows, sizeof(uint16_t), &mask_bytes)) {
        snprintf(error, sizeof(error), "connector output geometry overflow");
        goto cleanup;
    }

    video_host = malloc(video_output_bytes);
    audio_host = malloc(audio_output_bytes);
    mask_host = calloc(output_rows, sizeof(uint16_t));
    uint16_t *video_raw = malloc(video_file_bytes);
    uint16_t *audio_raw = malloc(audio_file_bytes);
    if (!video_host || !audio_host || !mask_host || !video_raw || !audio_raw) {
        free(video_raw);
        free(audio_raw);
        snprintf(error, sizeof(error), "out of host memory");
        goto cleanup;
    }
    if (!read_file(video_path, video_raw, video_file_bytes,
                   error, sizeof(error)) ||
        !read_file(audio_path, audio_raw, audio_file_bytes,
                   error, sizeof(error))) {
        free(video_raw);
        free(audio_raw);
        goto cleanup;
    }
    video_input = ltx_gpu_buffer_new_copy(
        gpu, video_raw, video_file_bytes, error, sizeof(error));
    audio_input = ltx_gpu_buffer_new_copy(
        gpu, audio_raw, audio_file_bytes, error, sizeof(error));
    free(video_raw);
    free(audio_raw);
    video_output = ltx_gpu_buffer_new(
        gpu, video_output_bytes, error, sizeof(error));
    audio_output = ltx_gpu_buffer_new(
        gpu, audio_output_bytes, error, sizeof(error));
    if (!video_input || !audio_input || !video_output || !audio_output)
        goto cleanup;

    for (uint32_t index = 0; index < warmup; index++)
        if (!ltx_connector_run_bf16(
                connector, video_output, audio_output,
                video_input, audio_input, input_rows,
                error, sizeof(error))) goto cleanup;
    double started = now_seconds();
    if (!ltx_connector_run_bf16(
            connector, video_output, audio_output,
            video_input, audio_input, input_rows,
            error, sizeof(error))) goto cleanup;
    double connector_seconds = now_seconds() - started;
    if (!ltx_gpu_buffer_read(video_output, video_host, video_output_bytes,
                             error, sizeof(error)) ||
        !ltx_gpu_buffer_read(audio_output, audio_host, audio_output_bytes,
                             error, sizeof(error))) goto cleanup;

    uint64_t video_non_finite = 0;
    uint64_t audio_non_finite = 0;
    double video_rms = 0.0;
    double audio_rms = 0.0;
    tensor_stats(video_host, (uint64_t)output_rows * video_dim,
                 &video_non_finite, &video_rms);
    tensor_stats(audio_host, (uint64_t)output_rows * audio_dim,
                 &audio_non_finite, &audio_rms);
    if (video_non_finite || audio_non_finite) {
        snprintf(error, sizeof(error),
                 "connector output has non-finite values");
        goto cleanup;
    }

    if (!make_path(output_path, sizeof(output_path), output_directory,
                   "video_context.bf16", error, sizeof(error)) ||
        !write_file(output_path, video_host, video_output_bytes,
                    error, sizeof(error)) ||
        !make_path(output_path, sizeof(output_path), output_directory,
                   "audio_context.bf16", error, sizeof(error)) ||
        !write_file(output_path, audio_host, audio_output_bytes,
                    error, sizeof(error)) ||
        !make_path(output_path, sizeof(output_path), output_directory,
                   "text_mask.bf16", error, sizeof(error)) ||
        !write_file(output_path, mask_host, mask_bytes,
                    error, sizeof(error)) ||
        !make_path(output_path, sizeof(output_path), output_directory,
                   "conditioning.json", error, sizeof(error))) goto cleanup;

    char manifest[8192];
    int manifest_length = snprintf(
        manifest, sizeof(manifest),
        "{\n"
        "  \"format\": \"ltx-mac-conditioning-v1\",\n"
        "  \"checkpoint\": \"%s\",\n"
        "  \"raw_directory\": \"%s\",\n"
        "  \"input_rows\": %u,\n"
        "  \"output_rows\": %u,\n"
        "  \"register_rows\": %u,\n"
        "  \"video_dim\": %u,\n"
        "  \"audio_dim\": %u,\n"
        "  \"warmup\": %u,\n"
        "  \"connector_seconds\": %.9f,\n"
        "  \"video_rms\": %.9g,\n"
        "  \"audio_rms\": %.9g,\n"
        "  \"video_non_finite\": %llu,\n"
        "  \"audio_non_finite\": %llu\n"
        "}\n",
        checkpoint, raw_directory, input_rows, output_rows,
        ltx_connector_register_rows(connector), video_dim, audio_dim,
        warmup, connector_seconds, video_rms, audio_rms,
        (unsigned long long)video_non_finite,
        (unsigned long long)audio_non_finite);
    if (manifest_length < 0 || (size_t)manifest_length >= sizeof(manifest) ||
        !write_file(output_path, manifest, (size_t)manifest_length,
                    error, sizeof(error))) goto cleanup;

    printf("connector input_rows=%u output_rows=%u warmup=%u time=%.3f s "
           "video_rms=%.6g audio_rms=%.6g\n",
           input_rows, output_rows, warmup, connector_seconds,
           video_rms, audio_rms);
    result = 0;

cleanup:
    if (result)
        fprintf(stderr, "run_connector: %s\n",
                error[0] ? error : "unknown failure");
    free(video_host);
    free(audio_host);
    free(mask_host);
    ltx_gpu_buffer_free(video_input);
    ltx_gpu_buffer_free(audio_input);
    ltx_gpu_buffer_free(video_output);
    ltx_gpu_buffer_free(audio_output);
    ltx_connector_free(connector);
    ltx_gpu_free(gpu);
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
