#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !value || value > UINT32_MAX) {
        fprintf(stderr, "bench_linear: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void convrot_256_row(const uint16_t *input, uint16_t *output,
                            uint32_t columns) {
    for (uint32_t group = 0; group < columns / 256u; group++) {
        float current[256];
        float next[256];
        uint32_t base = group * 256u;
        for (uint32_t lane = 0; lane < 256u; lane++)
            current[lane] = bf16_to_f32(input[base + lane]);
        for (uint32_t stride = 1u; stride < 256u; stride *= 4u) {
            for (uint32_t lane = 0; lane < 256u; lane++) {
                uint32_t digit = (lane / stride) & 3u;
                uint32_t butterfly_base = lane - digit * stride;
                float a = current[butterfly_base];
                float b = current[butterfly_base + stride];
                float c = current[butterfly_base + 2u * stride];
                float d = current[butterfly_base + 3u * stride];
                switch (digit) {
                    case 0u: next[lane] = a + b + c - d; break;
                    case 1u: next[lane] = a + b - c + d; break;
                    case 2u: next[lane] = a - b + c + d; break;
                    default: next[lane] = -a + b + c + d; break;
                }
            }
            memcpy(current, next, sizeof(current));
        }
        for (uint32_t lane = 0; lane < 256u; lane++)
            output[base + lane] = f32_to_bf16(current[lane] * 0.0625f);
    }
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT LINEAR_PREFIX [rows] [warmup] [iterations]\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint_path = argv[1];
    const char *prefix = argv[2];
    uint32_t rows = argc > 3 ? parse_u32(argv[3], "rows") : 1u;
    uint32_t warmup = argc > 4 ? parse_u32(argv[4], "warmup") : 3u;
    uint32_t iterations = argc > 5 ? parse_u32(argv[5], "iterations") : 9u;
    const char *backend_value = getenv("LTX_LINEAR_BACKEND");
    enum { BACKEND_TILE, BACKEND_MPS } backend = BACKEND_TILE;
    if (backend_value && !strcmp(backend_value, "mps"))
        backend = BACKEND_MPS;
    else if (backend_value && strcmp(backend_value, "tile")) {
        fprintf(stderr, "bench_linear: invalid LTX_LINEAR_BACKEND: %s\n",
                backend_value);
        return 2;
    }
    char error[1024] = {0};
    int result = 1;

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint_path, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_linear: %s\n", error);
        return 1;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_linear: %s\n", error);
        ltx_st_free_header(&header);
        return 1;
    }
    ltx_linear_weight_info linear;
    if (!ltx_linear_weight_resolve(&header, &mapping, prefix, &linear,
                                   error, sizeof(error)) ||
        !linear.quantized_int8 || !linear.convrot ||
        linear.convrot_group_size != 256u ||
        (linear.bias && linear.bias->dtype != LTX_DTYPE_BF16)) {
        fprintf(stderr, "bench_linear: unsupported weight: %s\n",
                error[0] ? error : "requires ConvRot INT8 + optional BF16 bias");
        ltx_st_map_close(&mapping);
        ltx_st_free_header(&header);
        return 1;
    }

    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    size_t bias_bytes = 0;
    const void *weight_data = ltx_st_map_tensor(
        &mapping, linear.weight, &weight_bytes, error, sizeof(error));
    const void *scale_data = ltx_st_map_tensor(
        &mapping, linear.weight_scale, &scale_bytes, error, sizeof(error));
    const void *bias_data = linear.bias ? ltx_st_map_tensor(
        &mapping, linear.bias, &bias_bytes, error, sizeof(error)) : NULL;
    uint64_t input_elements = (uint64_t)rows * linear.input_dim;
    uint64_t output_elements = (uint64_t)rows * linear.output_dim;
    if (!weight_data || !scale_data || (linear.bias && !bias_data) ||
        input_elements > SIZE_MAX / sizeof(uint16_t) ||
        output_elements > SIZE_MAX / sizeof(uint16_t)) {
        fprintf(stderr, "bench_linear: %s\n",
                error[0] ? error : "host allocation size overflow");
        ltx_st_map_close(&mapping);
        ltx_st_free_header(&header);
        return 1;
    }
    size_t input_bytes = (size_t)input_elements * sizeof(uint16_t);
    size_t output_bytes = (size_t)output_elements * sizeof(uint16_t);
    uint16_t *host_input = malloc(input_bytes);
    uint16_t *host_output = malloc(output_bytes);
    uint16_t *oracle_rotated = malloc(
        (size_t)linear.input_dim * sizeof(uint16_t));
    double *timings = calloc(iterations, sizeof(double));
    if (!host_input || !host_output || !oracle_rotated || !timings) {
        fputs("bench_linear: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (uint64_t index = 0; index < input_elements; index++) {
        float value = sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f;
        host_input[index] = f32_to_bf16(value);
    }
    convrot_256_row(host_input, oracle_rotated, linear.input_dim);

    double upload_start = now_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    ltx_gpu_buffer *input = NULL;
    ltx_gpu_buffer *rotated = NULL;
    ltx_gpu_buffer *weight = NULL;
    ltx_gpu_buffer *scale = NULL;
    ltx_gpu_buffer *bias = NULL;
    ltx_gpu_buffer *output = NULL;
    if (gpu) {
        input = ltx_gpu_buffer_new_copy(gpu, host_input, input_bytes,
                                        error, sizeof(error));
        rotated = ltx_gpu_buffer_new(gpu, input_bytes, error, sizeof(error));
        weight = ltx_gpu_buffer_new_copy(gpu, weight_data, weight_bytes,
                                         error, sizeof(error));
        scale = ltx_gpu_buffer_new_copy(gpu, scale_data, scale_bytes,
                                        error, sizeof(error));
        if (bias_data)
            bias = ltx_gpu_buffer_new_copy(gpu, bias_data, bias_bytes,
                                           error, sizeof(error));
        output = ltx_gpu_buffer_new(gpu, output_bytes, error, sizeof(error));
    }
    double upload_seconds = now_seconds() - upload_start;
    if (!gpu || !input || !rotated || !weight || !scale || !output ||
        (bias_data && !bias)) {
        fprintf(stderr, "bench_linear: GPU setup failed: %s\n", error);
        goto cleanup_gpu;
    }

    for (uint32_t index = 0; index < warmup; index++)
        if (!ltx_gpu_convrot_bf16(gpu, rotated, input, rows,
                                  linear.input_dim, 256u,
                                  error, sizeof(error)) ||
            !(backend == BACKEND_MPS ?
              ltx_gpu_linear_int8_weight_mps_bf16(
                  gpu, output, rotated, weight, scale, bias, rows,
                  linear.input_dim, linear.output_dim, error, sizeof(error)) :
              ltx_gpu_linear_int8_weight_bf16(
                  gpu, output, rotated, weight, scale, bias, rows,
                  linear.input_dim, linear.output_dim, error, sizeof(error)))) {
            fprintf(stderr, "bench_linear: warmup failed: %s\n", error);
            goto cleanup_gpu;
        }
    for (uint32_t index = 0; index < iterations; index++) {
        double start = now_seconds();
        if (!ltx_gpu_convrot_bf16(gpu, rotated, input, rows,
                                  linear.input_dim, 256u,
                                  error, sizeof(error)) ||
            !(backend == BACKEND_MPS ?
              ltx_gpu_linear_int8_weight_mps_bf16(
                  gpu, output, rotated, weight, scale, bias, rows,
                  linear.input_dim, linear.output_dim, error, sizeof(error)) :
              ltx_gpu_linear_int8_weight_bf16(
                  gpu, output, rotated, weight, scale, bias, rows,
                  linear.input_dim, linear.output_dim, error, sizeof(error)))) {
            fprintf(stderr, "bench_linear: iteration failed: %s\n", error);
            goto cleanup_gpu;
        }
        timings[index] = now_seconds() - start;
    }
    if (!ltx_gpu_buffer_read(output, host_output, output_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_linear: output read failed: %s\n", error);
        goto cleanup_gpu;
    }

    const int8_t *weight_i8 = weight_data;
    const float *weight_scale = scale_data;
    const uint16_t *bias_bf16 = bias_data;
    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite = 0;
    for (uint32_t column = 0; column < linear.output_dim; column++) {
        float sum = 0.0f;
        const int8_t *weight_row = weight_i8 +
            (uint64_t)column * linear.input_dim;
        for (uint32_t k = 0; k < linear.input_dim; k++)
            sum = fmaf(bf16_to_f32(oracle_rotated[k]),
                       (float)weight_row[k], sum);
        sum *= weight_scale[column];
        if (bias_bf16) sum += bf16_to_f32(bias_bf16[column]);
        float reference = bf16_to_f32(f32_to_bf16(sum));
        float candidate = bf16_to_f32(host_output[column]);
        if (!isfinite(candidate)) nonfinite++;
        double difference = (double)candidate - reference;
        diff2 += difference * difference;
        reference2 += (double)reference * reference;
        candidate2 += (double)candidate * candidate;
        dot += (double)reference * candidate;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    qsort(timings, iterations, sizeof(*timings), compare_double);
    size_t p50_index = (size_t)(iterations - 1u) / 2u;
    size_t p95_index = (size_t)ceil(0.95 * (double)iterations) - 1u;
    if (p95_index >= iterations) p95_index = iterations - 1u;
    double p50 = timings[p50_index];
    double p95 = timings[p95_index];
    double operations = 2.0 * rows * linear.input_dim * linear.output_dim;
    double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
    double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
        dot / sqrt(reference2 * candidate2) : 0.0;
    printf("checkpoint=%s\n", checkpoint_path);
    const char *backend_name = backend == BACKEND_MPS ?
        "mpsgraph" : "metal-tile";
    printf("linear=%s shape=%ux%u rows=%u convrot=%u backend=%s\n", prefix,
           linear.output_dim, linear.input_dim, rows,
           linear.convrot_group_size, backend_name);
    printf("checkpoint_weight_bytes=%zu resident_weight_bytes=%zu "
           "upload_and_pipeline_seconds=%.6f\n",
           weight_bytes, weight_bytes, upload_seconds);
    printf("warmup=%u iterations=%u p50_ms=%.3f p95_ms=%.3f "
           "effective_gflops=%.2f\n",
           warmup, iterations, p50 * 1000.0, p95 * 1000.0,
           operations / p50 / 1e9);
    printf("row0_rel_l2=%.9g row0_cosine=%.9g max_abs=%.9g "
           "nonfinite=%llu\n",
           rel_l2, cosine, max_abs, (unsigned long long)nonfinite);
    result = nonfinite || cosine <
        (backend == BACKEND_TILE ? 0.99999 : 0.999) ? 1 : 0;

cleanup_gpu:
    ltx_gpu_buffer_free(input);
    ltx_gpu_buffer_free(rotated);
    ltx_gpu_buffer_free(weight);
    ltx_gpu_buffer_free(scale);
    ltx_gpu_buffer_free(bias);
    ltx_gpu_buffer_free(output);
    ltx_gpu_free(gpu);
cleanup_host:
    free(host_input);
    free(host_output);
    free(oracle_rotated);
    free(timings);
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
