#include "ltx.h"
#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
} bf16_linear;

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !value || value > UINT32_MAX) {
        fprintf(stderr, "bench_adaln: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static float parse_f32(const char *text, const char *label) {
    char *end = NULL;
    float value = strtof(text, &end);
    if (!text[0] || !end || *end || !isfinite(value)) {
        fprintf(stderr, "bench_adaln: invalid %s: %s\n", label, text);
        exit(2);
    }
    return value;
}

static int make_name(char *name, size_t name_size,
                     const char *prefix, const char *suffix,
                     char *error, size_t error_size) {
    int length = snprintf(name, name_size, "%s.%s", prefix, suffix);
    if (length < 0 || (size_t)length >= name_size) {
        snprintf(error, error_size, "tensor name is too long");
        return 0;
    }
    return 1;
}

static void free_linear(bf16_linear *linear) {
    ltx_gpu_buffer_free(linear->weight);
    ltx_gpu_buffer_free(linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int load_linear(const ltx_st_header *header,
                       const ltx_st_mapping *mapping,
                       ltx_gpu *gpu, const char *prefix,
                       bf16_linear *linear,
                       char *error, size_t error_size) {
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(
            header, mapping, prefix, &info, error, error_size)) return 0;
    if (info.quantized_int8 || info.weight->dtype != LTX_DTYPE_BF16 ||
        !info.bias || info.bias->dtype != LTX_DTYPE_BF16) {
        snprintf(error, error_size,
                 "%s is not a BF16 linear with BF16 bias", prefix);
        return 0;
    }
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    const void *weight = ltx_st_map_tensor(
        mapping, info.weight, &weight_bytes, error, error_size);
    const void *bias = ltx_st_map_tensor(
        mapping, info.bias, &bias_bytes, error, error_size);
    if (!weight || !bias) return 0;
    linear->weight = ltx_gpu_buffer_new_copy(
        gpu, weight, weight_bytes, error, error_size);
    linear->bias = ltx_gpu_buffer_new_copy(
        gpu, bias, bias_bytes, error, error_size);
    linear->input_dim = info.input_dim;
    linear->output_dim = info.output_dim;
    if (!linear->weight || !linear->bias) {
        free_linear(linear);
        return 0;
    }
    return 1;
}

static int read_exact(const char *path, void *data, size_t bytes,
                      char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "cannot open %s", path);
        return 0;
    }
    size_t count = fread(data, 1u, bytes, file);
    int extra = fgetc(file);
    int close_result = fclose(file);
    if (count != bytes || extra != EOF || close_result != 0) {
        snprintf(error, error_size, "%s does not have %zu bytes", path, bytes);
        return 0;
    }
    return 1;
}

static void print_metrics(const char *label,
                          const uint16_t *actual,
                          const uint16_t *reference,
                          uint64_t elements) {
    double difference2 = 0.0;
    double reference2 = 0.0;
    double actual2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite = 0;
    for (uint64_t index = 0; index < elements; index++) {
        double expected = bf16_to_f32(reference[index]);
        double candidate = bf16_to_f32(actual[index]);
        if (!isfinite(candidate)) {
            nonfinite++;
            continue;
        }
        double difference = candidate - expected;
        difference2 += difference * difference;
        reference2 += expected * expected;
        actual2 += candidate * candidate;
        dot += expected * candidate;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_l2 = reference2 > 0.0 ?
        sqrt(difference2 / reference2) : 0.0;
    double cosine = reference2 > 0.0 && actual2 > 0.0 ?
        dot / sqrt(reference2 * actual2) : 0.0;
    printf("%s rel_l2=%.9g cosine=%.9g max_abs=%.9g nonfinite=%llu\n",
           label, rel_l2, cosine, max_abs,
           (unsigned long long)nonfinite);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT PREFIX PARAM_COUNT TIMESTEP "
        "[PARAM_REF EMBED_REF [warmup [iterations]]]\n", program);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc == 6 || argc > 9) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint = argv[1];
    const char *prefix = argv[2];
    uint32_t parameter_count = parse_u32(argv[3], "parameter count");
    float timestep = parse_f32(argv[4], "timestep");
    const char *parameter_reference_path = argc >= 7 ? argv[5] : NULL;
    const char *embedded_reference_path = argc >= 7 ? argv[6] : NULL;
    uint32_t warmup = argc >= 8 ? parse_u32(argv[7], "warmup") : 1u;
    uint32_t iterations = argc >= 9 ? parse_u32(argv[8], "iterations") : 5u;
    char error[1024] = {0};
    int result = 1;
    ltx_st_header header;
    ltx_st_mapping mapping;
    bf16_linear linear1 = {0};
    bf16_linear linear2 = {0};
    bf16_linear parameter = {0};
    ltx_gpu *gpu = NULL;
    ltx_gpu_buffer *input = NULL;
    ltx_gpu_buffer *parameters = NULL;
    ltx_gpu_buffer *embedded = NULL;
    uint16_t *host_input = NULL;
    uint16_t *actual_parameters = NULL;
    uint16_t *actual_embedded = NULL;
    uint16_t *reference_parameters = NULL;
    uint16_t *reference_embedded = NULL;

    if (!ltx_st_read_header(checkpoint, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_adaln: %s\n", error);
        return 1;
    }
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_adaln: %s\n", error);
        ltx_st_free_header(&header);
        return 1;
    }
    gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    char name[4096];
    if (!gpu ||
        !make_name(name, sizeof(name), prefix,
                   "emb.timestep_embedder.linear_1", error, sizeof(error)) ||
        !load_linear(&header, &mapping, gpu, name, &linear1,
                     error, sizeof(error)) ||
        !make_name(name, sizeof(name), prefix,
                   "emb.timestep_embedder.linear_2", error, sizeof(error)) ||
        !load_linear(&header, &mapping, gpu, name, &linear2,
                     error, sizeof(error)) ||
        !make_name(name, sizeof(name), prefix, "linear",
                   error, sizeof(error)) ||
        !load_linear(&header, &mapping, gpu, name, &parameter,
                     error, sizeof(error))) {
        fprintf(stderr, "bench_adaln: load: %s\n", error);
        goto cleanup;
    }
    uint32_t timestep_dim = linear1.input_dim;
    uint32_t hidden_dim = linear1.output_dim;
    uint64_t parameter_dim = (uint64_t)hidden_dim * parameter_count;
    if (linear2.input_dim != hidden_dim ||
        linear2.output_dim != hidden_dim ||
        parameter.input_dim != hidden_dim ||
        parameter.output_dim != parameter_dim) {
        fputs("bench_adaln: incompatible AdaLN geometry\n", stderr);
        goto cleanup;
    }
    size_t input_bytes = (size_t)timestep_dim * sizeof(uint16_t);
    size_t embedded_bytes = (size_t)hidden_dim * sizeof(uint16_t);
    size_t parameter_bytes = (size_t)parameter_dim * sizeof(uint16_t);
    host_input = malloc(input_bytes);
    actual_embedded = malloc(embedded_bytes);
    actual_parameters = malloc(parameter_bytes);
    if (!host_input || !actual_embedded || !actual_parameters ||
        !ltx_compute_timestep_embedding_bf16(
            host_input, timestep_dim, &timestep, 1u, timestep_dim,
            1, 0.0f, 1.0f, 10000.0f, error, sizeof(error))) {
        fprintf(stderr, "bench_adaln: input: %s\n", error);
        goto cleanup;
    }
    input = ltx_gpu_buffer_new_copy(
        gpu, host_input, input_bytes, error, sizeof(error));
    embedded = ltx_gpu_buffer_new(
        gpu, embedded_bytes, error, sizeof(error));
    parameters = ltx_gpu_buffer_new(
        gpu, parameter_bytes, error, sizeof(error));
    if (!input || !embedded || !parameters) {
        fprintf(stderr, "bench_adaln: buffers: %s\n", error);
        goto cleanup;
    }
#define LTX_RUN_ADALN() ltx_gpu_adaln_single_mps_bf16( \
    gpu, parameters, embedded, input, \
    linear1.weight, linear1.bias, linear2.weight, linear2.bias, \
    parameter.weight, parameter.bias, 1u, timestep_dim, hidden_dim, \
    parameter_count, error, sizeof(error))
    for (uint32_t index = 0; index < warmup; index++)
        if (!LTX_RUN_ADALN()) {
            fprintf(stderr, "bench_adaln: warmup: %s\n", error);
            goto cleanup;
        }
    double start = now_seconds();
    for (uint32_t index = 0; index < iterations; index++)
        if (!LTX_RUN_ADALN()) {
            fprintf(stderr, "bench_adaln: run: %s\n", error);
            goto cleanup;
        }
    double milliseconds =
        (now_seconds() - start) * 1000.0 / (double)iterations;
#undef LTX_RUN_ADALN
    if (!ltx_gpu_buffer_read(parameters, actual_parameters,
                             parameter_bytes, error, sizeof(error)) ||
        !ltx_gpu_buffer_read(embedded, actual_embedded,
                             embedded_bytes, error, sizeof(error))) {
        fprintf(stderr, "bench_adaln: read: %s\n", error);
        goto cleanup;
    }
    printf("checkpoint=%s\nprefix=%s timestep=%.9g hidden_dim=%u "
           "parameters=%u mean_ms=%.3f\n",
           checkpoint, prefix, timestep, hidden_dim,
           parameter_count, milliseconds);
    if (parameter_reference_path) {
        reference_parameters = malloc(parameter_bytes);
        reference_embedded = malloc(embedded_bytes);
        if (!reference_parameters || !reference_embedded ||
            !read_exact(parameter_reference_path, reference_parameters,
                        parameter_bytes, error, sizeof(error)) ||
            !read_exact(embedded_reference_path, reference_embedded,
                        embedded_bytes, error, sizeof(error))) {
            fprintf(stderr, "bench_adaln: reference: %s\n", error);
            goto cleanup;
        }
        print_metrics("parameters", actual_parameters,
                      reference_parameters, parameter_dim);
        print_metrics("embedded", actual_embedded,
                      reference_embedded, hidden_dim);
    }
    result = 0;

cleanup:
    free(reference_parameters);
    free(reference_embedded);
    free(actual_parameters);
    free(actual_embedded);
    free(host_input);
    ltx_gpu_buffer_free(parameters);
    ltx_gpu_buffer_free(embedded);
    ltx_gpu_buffer_free(input);
    free_linear(&parameter);
    free_linear(&linear2);
    free_linear(&linear1);
    ltx_gpu_free(gpu);
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
