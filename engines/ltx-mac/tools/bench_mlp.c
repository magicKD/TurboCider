#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    MLP_BACKEND_FUSED = 0,
    MLP_BACKEND_STAGED,
} mlp_backend;

typedef struct {
    const void *weight;
    size_t weight_bytes;
    const void *scale;
    size_t scale_bytes;
    const void *bias;
    size_t bias_bytes;
} mapped_linear;

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
        fprintf(stderr, "bench_mlp: invalid %s: %s\n", label, text);
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

static void linear_int8_dequant_bf16_row(
        const uint16_t *input, const int8_t *weight,
        const float *scale, const uint16_t *bias,
        uint32_t input_dim, uint32_t output_dim,
        uint16_t *output) {
    for (uint32_t column = 0; column < output_dim; column++) {
        const int8_t *weight_row = weight + (uint64_t)column * input_dim;
        float row_scale = scale[column];
        float sum = 0.0f;
        for (uint32_t k = 0; k < input_dim; k++) {
            float dequantized = bf16_to_f32(f32_to_bf16(
                (float)weight_row[k] * row_scale));
            sum = fmaf(bf16_to_f32(input[k]), dequantized, sum);
        }
        if (bias) sum += bf16_to_f32(bias[column]);
        output[column] = f32_to_bf16(sum);
    }
}

static void gelu_tanh_bf16_row(uint16_t *values, uint32_t elements) {
    for (uint32_t index = 0; index < elements; index++) {
        float x = bf16_to_f32(values[index]);
        float inner = 0.7978845608028654f *
            (x + 0.044715f * x * x * x);
        values[index] = f32_to_bf16(
            0.5f * x * (1.0f + tanhf(inner)));
    }
}

static int map_linear(const ltx_st_mapping *mapping,
                      const ltx_linear_weight_info *linear,
                      mapped_linear *mapped,
                      char *error, size_t error_size) {
    memset(mapped, 0, sizeof(*mapped));
    mapped->weight = ltx_st_map_tensor(mapping, linear->weight,
                                       &mapped->weight_bytes,
                                       error, error_size);
    mapped->scale = ltx_st_map_tensor(mapping, linear->weight_scale,
                                      &mapped->scale_bytes,
                                      error, error_size);
    if (linear->bias)
        mapped->bias = ltx_st_map_tensor(mapping, linear->bias,
                                         &mapped->bias_bytes,
                                         error, error_size);
    return mapped->weight && mapped->scale &&
        (!linear->bias || mapped->bias);
}

static int resolve_mlp(const ltx_st_header *header,
                       const ltx_st_mapping *mapping,
                       const char *prefix,
                       ltx_linear_weight_info *fc1,
                       ltx_linear_weight_info *fc2,
                       char *error, size_t error_size) {
    char fc1_prefix[1024];
    char fc2_prefix[1024];
    int first_length = snprintf(fc1_prefix, sizeof(fc1_prefix),
                                "%s.net.0.proj", prefix);
    int second_length = snprintf(fc2_prefix, sizeof(fc2_prefix),
                                 "%s.net.2", prefix);
    if (first_length < 0 || (size_t)first_length >= sizeof(fc1_prefix) ||
        second_length < 0 || (size_t)second_length >= sizeof(fc2_prefix)) {
        snprintf(error, error_size, "MLP prefix is too long");
        return 0;
    }
    if (!ltx_linear_weight_resolve(header, mapping, fc1_prefix, fc1,
                                   error, error_size) ||
        !ltx_linear_weight_resolve(header, mapping, fc2_prefix, fc2,
                                   error, error_size))
        return 0;
    if (!fc1->quantized_int8 || !fc2->quantized_int8 ||
        !fc1->convrot || !fc2->convrot ||
        fc1->convrot_group_size != 256u ||
        fc2->convrot_group_size != 256u ||
        fc1->output_dim != fc2->input_dim ||
        fc1->input_dim != fc2->output_dim ||
        (fc1->bias && fc1->bias->dtype != LTX_DTYPE_BF16) ||
        (fc2->bias && fc2->bias->dtype != LTX_DTYPE_BF16)) {
        snprintf(error, error_size,
                 "requires matching ConvRot-256 INT8 MLP weights and "
                 "optional BF16 biases");
        return 0;
    }
    return 1;
}

static int run_mlp(
        mlp_backend backend, ltx_gpu *gpu,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        const ltx_gpu_buffer *fc1_weight,
        const ltx_gpu_buffer *fc1_scale,
        const ltx_gpu_buffer *fc1_bias,
        const ltx_gpu_buffer *fc2_weight,
        const ltx_gpu_buffer *fc2_scale,
        const ltx_gpu_buffer *fc2_bias,
        ltx_gpu_buffer *rotated_input,
        ltx_gpu_buffer *hidden,
        ltx_gpu_buffer *activated,
        ltx_gpu_buffer *rotated_hidden,
        uint32_t rows, uint32_t input_dim,
        uint32_t hidden_dim, uint32_t output_dim,
        char *error, size_t error_size) {
    if (backend == MLP_BACKEND_FUSED)
        return ltx_gpu_mlp_int8_convrot_mps_bf16(
            gpu, output, input,
            fc1_weight, fc1_scale, fc1_bias,
            fc2_weight, fc2_scale, fc2_bias,
            rows, input_dim, hidden_dim, output_dim, 256u,
            error, error_size);
    return ltx_gpu_convrot_bf16(
               gpu, rotated_input, input, rows, input_dim, 256u,
               error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
               gpu, hidden, rotated_input, fc1_weight, fc1_scale, fc1_bias,
               rows, input_dim, hidden_dim, error, error_size) &&
        ltx_gpu_gelu_tanh_bf16(
               gpu, activated, hidden, rows * hidden_dim,
               error, error_size) &&
        ltx_gpu_convrot_bf16(
               gpu, rotated_hidden, activated, rows, hidden_dim, 256u,
               error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
               gpu, output, rotated_hidden, fc2_weight, fc2_scale, fc2_bias,
               rows, hidden_dim, output_dim, error, error_size);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT MLP_PREFIX [rows] [warmup] [iterations]\n"
        "Set LTX_MLP_BACKEND=staged to compare the unfused path.\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint_path = argv[1];
    const char *prefix = argv[2];
    uint32_t rows = argc > 3 ? parse_u32(argv[3], "rows") : 1001u;
    uint32_t warmup = argc > 4 ? parse_u32(argv[4], "warmup") : 2u;
    uint32_t iterations = argc > 5 ? parse_u32(argv[5], "iterations") : 5u;
    const char *backend_value = getenv("LTX_MLP_BACKEND");
    mlp_backend backend = backend_value && !strcmp(backend_value, "staged") ?
        MLP_BACKEND_STAGED : MLP_BACKEND_FUSED;
    char error[1024] = {0};
    int result = 1;

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint_path, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_mlp: %s\n", error);
        return 1;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_mlp: %s\n", error);
        ltx_st_free_header(&header);
        return 1;
    }

    ltx_linear_weight_info fc1;
    ltx_linear_weight_info fc2;
    if (!resolve_mlp(&header, &mapping, prefix, &fc1, &fc2,
                     error, sizeof(error))) {
        fprintf(stderr, "bench_mlp: unsupported MLP: %s\n", error);
        ltx_st_map_close(&mapping);
        ltx_st_free_header(&header);
        return 1;
    }
    mapped_linear mapped_fc1;
    mapped_linear mapped_fc2;
    if (!map_linear(&mapping, &fc1, &mapped_fc1, error, sizeof(error)) ||
        !map_linear(&mapping, &fc2, &mapped_fc2, error, sizeof(error))) {
        fprintf(stderr, "bench_mlp: map weights failed: %s\n", error);
        ltx_st_map_close(&mapping);
        ltx_st_free_header(&header);
        return 1;
    }

    uint32_t input_dim = fc1.input_dim;
    uint32_t hidden_dim = fc1.output_dim;
    uint32_t output_dim = fc2.output_dim;
    uint64_t input_elements = (uint64_t)rows * input_dim;
    uint64_t hidden_elements = (uint64_t)rows * hidden_dim;
    uint64_t output_elements = (uint64_t)rows * output_dim;
    if (input_elements > SIZE_MAX / sizeof(uint16_t) ||
        hidden_elements > SIZE_MAX / sizeof(uint16_t) ||
        output_elements > SIZE_MAX / sizeof(uint16_t)) {
        fputs("bench_mlp: allocation size overflow\n", stderr);
        goto cleanup_mapping;
    }
    size_t input_bytes = (size_t)input_elements * sizeof(uint16_t);
    size_t hidden_bytes = (size_t)hidden_elements * sizeof(uint16_t);
    size_t output_bytes = (size_t)output_elements * sizeof(uint16_t);
    size_t output_row_bytes = (size_t)output_dim * sizeof(uint16_t);

    uint16_t *host_input = malloc(input_bytes);
    uint16_t *host_output = malloc(output_row_bytes);
    uint16_t *oracle_rotated_input = malloc(
        (size_t)input_dim * sizeof(uint16_t));
    uint16_t *oracle_hidden = malloc(
        (size_t)hidden_dim * sizeof(uint16_t));
    uint16_t *oracle_rotated_hidden = malloc(
        (size_t)hidden_dim * sizeof(uint16_t));
    uint16_t *oracle_output = malloc(output_row_bytes);
    double *timings = calloc(iterations, sizeof(double));
    if (!host_input || !host_output || !oracle_rotated_input ||
        !oracle_hidden || !oracle_rotated_hidden || !oracle_output ||
        !timings) {
        fputs("bench_mlp: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (uint64_t index = 0; index < input_elements; index++) {
        float value = sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f;
        host_input[index] = f32_to_bf16(value);
    }

    double oracle_start = now_seconds();
    convrot_256_row(host_input, oracle_rotated_input, input_dim);
    linear_int8_dequant_bf16_row(
        oracle_rotated_input, mapped_fc1.weight, mapped_fc1.scale,
        mapped_fc1.bias, input_dim, hidden_dim, oracle_hidden);
    gelu_tanh_bf16_row(oracle_hidden, hidden_dim);
    convrot_256_row(oracle_hidden, oracle_rotated_hidden, hidden_dim);
    linear_int8_dequant_bf16_row(
        oracle_rotated_hidden, mapped_fc2.weight, mapped_fc2.scale,
        mapped_fc2.bias, hidden_dim, output_dim, oracle_output);
    double oracle_seconds = now_seconds() - oracle_start;

    double setup_start = now_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    ltx_gpu_buffer *input = NULL;
    ltx_gpu_buffer *fc1_weight = NULL;
    ltx_gpu_buffer *fc1_scale = NULL;
    ltx_gpu_buffer *fc1_bias = NULL;
    ltx_gpu_buffer *fc2_weight = NULL;
    ltx_gpu_buffer *fc2_scale = NULL;
    ltx_gpu_buffer *fc2_bias = NULL;
    ltx_gpu_buffer *output = NULL;
    ltx_gpu_buffer *rotated_input = NULL;
    ltx_gpu_buffer *hidden = NULL;
    ltx_gpu_buffer *activated = NULL;
    ltx_gpu_buffer *rotated_hidden = NULL;
    if (gpu) {
        input = ltx_gpu_buffer_new_copy(gpu, host_input, input_bytes,
                                        error, sizeof(error));
        fc1_weight = ltx_gpu_buffer_new_copy(
            gpu, mapped_fc1.weight, mapped_fc1.weight_bytes,
            error, sizeof(error));
        fc1_scale = ltx_gpu_buffer_new_copy(
            gpu, mapped_fc1.scale, mapped_fc1.scale_bytes,
            error, sizeof(error));
        if (mapped_fc1.bias)
            fc1_bias = ltx_gpu_buffer_new_copy(
                gpu, mapped_fc1.bias, mapped_fc1.bias_bytes,
                error, sizeof(error));
        fc2_weight = ltx_gpu_buffer_new_copy(
            gpu, mapped_fc2.weight, mapped_fc2.weight_bytes,
            error, sizeof(error));
        fc2_scale = ltx_gpu_buffer_new_copy(
            gpu, mapped_fc2.scale, mapped_fc2.scale_bytes,
            error, sizeof(error));
        if (mapped_fc2.bias)
            fc2_bias = ltx_gpu_buffer_new_copy(
                gpu, mapped_fc2.bias, mapped_fc2.bias_bytes,
                error, sizeof(error));
        output = ltx_gpu_buffer_new(gpu, output_bytes, error, sizeof(error));
        if (backend == MLP_BACKEND_STAGED) {
            rotated_input = ltx_gpu_buffer_new(
                gpu, input_bytes, error, sizeof(error));
            hidden = ltx_gpu_buffer_new(gpu, hidden_bytes,
                                         error, sizeof(error));
            activated = ltx_gpu_buffer_new(gpu, hidden_bytes,
                                            error, sizeof(error));
            rotated_hidden = ltx_gpu_buffer_new(
                gpu, hidden_bytes, error, sizeof(error));
        }
    }
    double setup_seconds = now_seconds() - setup_start;
    if (!gpu || !input || !fc1_weight || !fc1_scale ||
        !fc2_weight || !fc2_scale || !output ||
        (mapped_fc1.bias && !fc1_bias) ||
        (mapped_fc2.bias && !fc2_bias) ||
        (backend == MLP_BACKEND_STAGED &&
         (!rotated_input || !hidden || !activated || !rotated_hidden))) {
        fprintf(stderr, "bench_mlp: GPU setup failed: %s\n", error);
        goto cleanup_gpu;
    }

    for (uint32_t index = 0; index < warmup; index++)
        if (!run_mlp(backend, gpu, output, input,
                     fc1_weight, fc1_scale, fc1_bias,
                     fc2_weight, fc2_scale, fc2_bias,
                     rotated_input, hidden, activated, rotated_hidden,
                     rows, input_dim, hidden_dim, output_dim,
                     error, sizeof(error))) {
            fprintf(stderr, "bench_mlp: warmup failed: %s\n", error);
            goto cleanup_gpu;
        }
    for (uint32_t index = 0; index < iterations; index++) {
        double start = now_seconds();
        if (!run_mlp(backend, gpu, output, input,
                     fc1_weight, fc1_scale, fc1_bias,
                     fc2_weight, fc2_scale, fc2_bias,
                     rotated_input, hidden, activated, rotated_hidden,
                     rows, input_dim, hidden_dim, output_dim,
                     error, sizeof(error))) {
            fprintf(stderr, "bench_mlp: iteration failed: %s\n", error);
            goto cleanup_gpu;
        }
        timings[index] = now_seconds() - start;
    }
    if (!ltx_gpu_buffer_read(output, host_output, output_row_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_mlp: output read failed: %s\n", error);
        goto cleanup_gpu;
    }

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite = 0;
    for (uint32_t column = 0; column < output_dim; column++) {
        double reference = bf16_to_f32(oracle_output[column]);
        double candidate = bf16_to_f32(host_output[column]);
        if (!isfinite(candidate)) nonfinite++;
        double difference = candidate - reference;
        diff2 += difference * difference;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }

    qsort(timings, iterations, sizeof(*timings), compare_double);
    size_t p50_index = (size_t)(iterations - 1u) / 2u;
    size_t p95_index = (size_t)ceil(0.95 * (double)iterations) - 1u;
    if (p95_index >= iterations) p95_index = iterations - 1u;
    double p50 = timings[p50_index];
    double p95 = timings[p95_index];
    double operations = 2.0 * (double)rows * (double)hidden_dim *
        ((double)input_dim + (double)output_dim);
    double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
    double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
        dot / sqrt(reference2 * candidate2) : 0.0;
    printf("checkpoint=%s\n", checkpoint_path);
    printf("mlp=%s shape=%u->%u->%u rows=%u backend=%s\n",
           prefix, input_dim, hidden_dim, output_dim, rows,
           backend == MLP_BACKEND_FUSED ? "fused-mpsgraph" :
                                          "staged-mpsgraph");
    printf("weight_bytes=%zu setup_seconds=%.6f oracle_seconds=%.6f\n",
           mapped_fc1.weight_bytes + mapped_fc2.weight_bytes,
           setup_seconds, oracle_seconds);
    printf("warmup=%u iterations=%u p50_ms=%.3f p95_ms=%.3f "
           "effective_gflops=%.2f\n",
           warmup, iterations, p50 * 1000.0, p95 * 1000.0,
           operations / p50 / 1e9);
    printf("row0_rel_l2=%.9g row0_cosine=%.9g max_abs=%.9g "
           "nonfinite=%llu\n",
           rel_l2, cosine, max_abs, (unsigned long long)nonfinite);
    result = nonfinite || cosine < 0.99 ? 1 : 0;

cleanup_gpu:
    ltx_gpu_buffer_free(input);
    ltx_gpu_buffer_free(fc1_weight);
    ltx_gpu_buffer_free(fc1_scale);
    ltx_gpu_buffer_free(fc1_bias);
    ltx_gpu_buffer_free(fc2_weight);
    ltx_gpu_buffer_free(fc2_scale);
    ltx_gpu_buffer_free(fc2_bias);
    ltx_gpu_buffer_free(output);
    ltx_gpu_buffer_free(rotated_input);
    ltx_gpu_buffer_free(hidden);
    ltx_gpu_buffer_free(activated);
    ltx_gpu_buffer_free(rotated_hidden);
    ltx_gpu_free(gpu);
cleanup_host:
    free(host_input);
    free(host_output);
    free(oracle_rotated_input);
    free(oracle_hidden);
    free(oracle_rotated_hidden);
    free(oracle_output);
    free(timings);
cleanup_mapping:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
