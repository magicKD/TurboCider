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
    QKV_BACKEND_FUSED = 0,
    QKV_BACKEND_STAGED,
} qkv_backend;

typedef struct {
    const void *weight;
    size_t weight_bytes;
    const void *scale;
    size_t scale_bytes;
    const void *bias;
    size_t bias_bytes;
} mapped_linear;

typedef struct {
    double diff2;
    double reference2;
    double candidate2;
    double dot;
    double max_abs;
    uint64_t nonfinite;
} parity_metrics;

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
        fprintf(stderr, "bench_qkv: invalid %s: %s\n", label, text);
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

static void rms_norm_weighted_bf16_row(uint16_t *values,
                                        const uint16_t *weight,
                                        uint32_t columns, float epsilon) {
    double sum = 0.0;
    for (uint32_t column = 0; column < columns; column++) {
        double value = bf16_to_f32(values[column]);
        sum += value * value;
    }
    float inverse_rms = 1.0f /
        sqrtf((float)(sum / (double)columns) + epsilon);
    for (uint32_t column = 0; column < columns; column++)
        values[column] = f32_to_bf16(
            bf16_to_f32(values[column]) * inverse_rms *
            bf16_to_f32(weight[column]));
}

static void update_metrics(parity_metrics *metrics,
                           const uint16_t *reference,
                           const uint16_t *candidate,
                           uint32_t elements) {
    for (uint32_t index = 0; index < elements; index++) {
        double expected = bf16_to_f32(reference[index]);
        double actual = bf16_to_f32(candidate[index]);
        if (!isfinite(actual)) metrics->nonfinite++;
        double difference = actual - expected;
        metrics->diff2 += difference * difference;
        metrics->reference2 += expected * expected;
        metrics->candidate2 += actual * actual;
        metrics->dot += expected * actual;
        if (fabs(difference) > metrics->max_abs)
            metrics->max_abs = fabs(difference);
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

static int resolve_projection(const ltx_st_header *header,
                              const ltx_st_mapping *mapping,
                              const char *attention_prefix,
                              const char *suffix,
                              ltx_linear_weight_info *linear,
                              char *error, size_t error_size) {
    char prefix[1024];
    int length = snprintf(prefix, sizeof(prefix), "%s.%s",
                          attention_prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        snprintf(error, error_size, "attention prefix is too long");
        return 0;
    }
    return ltx_linear_weight_resolve(header, mapping, prefix, linear,
                                     error, error_size);
}

static const ltx_st_tensor *resolve_norm(const ltx_st_header *header,
                                         const char *attention_prefix,
                                         const char *suffix,
                                         uint32_t inner_dim,
                                         char *error, size_t error_size) {
    char name[1024];
    int length = snprintf(name, sizeof(name), "%s.%s.weight",
                          attention_prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        snprintf(error, error_size, "attention prefix is too long");
        return NULL;
    }
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor || tensor->dtype != LTX_DTYPE_BF16 || tensor->ndim != 1u ||
        tensor->shape[0] != inner_dim) {
        snprintf(error, error_size, "invalid or missing %s", name);
        return NULL;
    }
    return tensor;
}

static int validate_qkv(const ltx_linear_weight_info *query,
                        const ltx_linear_weight_info *key,
                        const ltx_linear_weight_info *value,
                        char *error, size_t error_size) {
    if (!query->quantized_int8 || !key->quantized_int8 ||
        !value->quantized_int8 || !query->convrot || !key->convrot ||
        !value->convrot || query->convrot_group_size != 256u ||
        key->convrot_group_size != 256u ||
        value->convrot_group_size != 256u ||
        query->input_dim != key->input_dim ||
        query->input_dim != value->input_dim ||
        query->output_dim != key->output_dim ||
        query->output_dim != value->output_dim ||
        (query->bias && query->bias->dtype != LTX_DTYPE_BF16) ||
        (key->bias && key->bias->dtype != LTX_DTYPE_BF16) ||
        (value->bias && value->bias->dtype != LTX_DTYPE_BF16)) {
        snprintf(error, error_size,
                 "requires matching ConvRot-256 INT8 Q/K/V weights and "
                 "optional BF16 biases");
        return 0;
    }
    return 1;
}

static int run_qkv(
        qkv_backend backend, ltx_gpu *gpu,
        ltx_gpu_buffer *query_output,
        ltx_gpu_buffer *key_output,
        ltx_gpu_buffer *value_output,
        const ltx_gpu_buffer *input,
        const ltx_gpu_buffer *query_weight,
        const ltx_gpu_buffer *query_scale,
        const ltx_gpu_buffer *query_bias,
        const ltx_gpu_buffer *key_weight,
        const ltx_gpu_buffer *key_scale,
        const ltx_gpu_buffer *key_bias,
        const ltx_gpu_buffer *value_weight,
        const ltx_gpu_buffer *value_scale,
        const ltx_gpu_buffer *value_bias,
        const ltx_gpu_buffer *query_norm_weight,
        const ltx_gpu_buffer *key_norm_weight,
        ltx_gpu_buffer *rotated,
        ltx_gpu_buffer *query_raw,
        ltx_gpu_buffer *key_raw,
        uint32_t rows, uint32_t input_dim, uint32_t inner_dim,
        char *error, size_t error_size) {
    if (backend == QKV_BACKEND_FUSED)
        return ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu, query_output, key_output, value_output, input,
            query_weight, query_scale, query_bias,
            key_weight, key_scale, key_bias,
            value_weight, value_scale, value_bias,
            query_norm_weight, key_norm_weight,
            rows, input_dim, inner_dim, 256u, 1e-6f,
            error, error_size);
    return ltx_gpu_convrot_bf16(
               gpu, rotated, input, rows, input_dim, 256u,
               error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
               gpu, query_raw, rotated, query_weight, query_scale, query_bias,
               rows, input_dim, inner_dim, error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
               gpu, key_raw, rotated, key_weight, key_scale, key_bias,
               rows, input_dim, inner_dim, error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
               gpu, value_output, rotated, value_weight, value_scale,
               value_bias, rows, input_dim, inner_dim, error, error_size) &&
        ltx_gpu_rms_norm_weighted_bf16(
               gpu, query_output, query_raw, query_norm_weight,
               rows, inner_dim, 1e-6f, error, error_size) &&
        ltx_gpu_rms_norm_weighted_bf16(
               gpu, key_output, key_raw, key_norm_weight,
               rows, inner_dim, 1e-6f, error, error_size);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT ATTENTION_PREFIX [rows] [warmup] [iterations]\n"
        "Set LTX_QKV_BACKEND=staged to compare the unfused path.\n",
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
    const char *backend_value = getenv("LTX_QKV_BACKEND");
    qkv_backend backend = backend_value && !strcmp(backend_value, "staged") ?
        QKV_BACKEND_STAGED : QKV_BACKEND_FUSED;
    char error[1024] = {0};
    int result = 1;

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint_path, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_qkv: %s\n", error);
        return 1;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_qkv: %s\n", error);
        ltx_st_free_header(&header);
        return 1;
    }

    ltx_linear_weight_info query;
    ltx_linear_weight_info key;
    ltx_linear_weight_info value;
    if (!resolve_projection(&header, &mapping, prefix, "to_q", &query,
                            error, sizeof(error)) ||
        !resolve_projection(&header, &mapping, prefix, "to_k", &key,
                            error, sizeof(error)) ||
        !resolve_projection(&header, &mapping, prefix, "to_v", &value,
                            error, sizeof(error)) ||
        !validate_qkv(&query, &key, &value, error, sizeof(error))) {
        fprintf(stderr, "bench_qkv: unsupported attention: %s\n", error);
        goto cleanup_mapping;
    }
    uint32_t input_dim = query.input_dim;
    uint32_t inner_dim = query.output_dim;
    const ltx_st_tensor *query_norm_tensor = resolve_norm(
        &header, prefix, "q_norm", inner_dim, error, sizeof(error));
    const ltx_st_tensor *key_norm_tensor = resolve_norm(
        &header, prefix, "k_norm", inner_dim, error, sizeof(error));
    if (!query_norm_tensor || !key_norm_tensor) {
        fprintf(stderr, "bench_qkv: %s\n", error);
        goto cleanup_mapping;
    }

    mapped_linear mapped_query;
    mapped_linear mapped_key;
    mapped_linear mapped_value;
    if (!map_linear(&mapping, &query, &mapped_query, error, sizeof(error)) ||
        !map_linear(&mapping, &key, &mapped_key, error, sizeof(error)) ||
        !map_linear(&mapping, &value, &mapped_value, error, sizeof(error))) {
        fprintf(stderr, "bench_qkv: map projections failed: %s\n", error);
        goto cleanup_mapping;
    }
    size_t query_norm_bytes = 0;
    size_t key_norm_bytes = 0;
    const void *query_norm_data = ltx_st_map_tensor(
        &mapping, query_norm_tensor, &query_norm_bytes, error, sizeof(error));
    const void *key_norm_data = ltx_st_map_tensor(
        &mapping, key_norm_tensor, &key_norm_bytes, error, sizeof(error));
    if (!query_norm_data || !key_norm_data) {
        fprintf(stderr, "bench_qkv: map norm weights failed: %s\n", error);
        goto cleanup_mapping;
    }

    uint64_t input_elements = (uint64_t)rows * input_dim;
    uint64_t output_elements = (uint64_t)rows * inner_dim;
    if (input_elements > SIZE_MAX / sizeof(uint16_t) ||
        output_elements > SIZE_MAX / sizeof(uint16_t)) {
        fputs("bench_qkv: allocation size overflow\n", stderr);
        goto cleanup_mapping;
    }
    size_t input_bytes = (size_t)input_elements * sizeof(uint16_t);
    size_t output_bytes = (size_t)output_elements * sizeof(uint16_t);
    size_t output_row_bytes = (size_t)inner_dim * sizeof(uint16_t);
    uint16_t *host_input = malloc(input_bytes);
    uint16_t *host_query = malloc(output_row_bytes);
    uint16_t *host_key = malloc(output_row_bytes);
    uint16_t *host_value = malloc(output_row_bytes);
    uint16_t *oracle_rotated = malloc(
        (size_t)input_dim * sizeof(uint16_t));
    uint16_t *oracle_query = malloc(output_row_bytes);
    uint16_t *oracle_key = malloc(output_row_bytes);
    uint16_t *oracle_value = malloc(output_row_bytes);
    double *timings = calloc(iterations, sizeof(double));
    if (!host_input || !host_query || !host_key || !host_value ||
        !oracle_rotated || !oracle_query || !oracle_key || !oracle_value ||
        !timings) {
        fputs("bench_qkv: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (uint64_t index = 0; index < input_elements; index++) {
        float input_value =
            sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f;
        host_input[index] = f32_to_bf16(input_value);
    }

    double oracle_start = now_seconds();
    convrot_256_row(host_input, oracle_rotated, input_dim);
    linear_int8_dequant_bf16_row(
        oracle_rotated, mapped_query.weight, mapped_query.scale,
        mapped_query.bias, input_dim, inner_dim, oracle_query);
    linear_int8_dequant_bf16_row(
        oracle_rotated, mapped_key.weight, mapped_key.scale,
        mapped_key.bias, input_dim, inner_dim, oracle_key);
    linear_int8_dequant_bf16_row(
        oracle_rotated, mapped_value.weight, mapped_value.scale,
        mapped_value.bias, input_dim, inner_dim, oracle_value);
    rms_norm_weighted_bf16_row(
        oracle_query, query_norm_data, inner_dim, 1e-6f);
    rms_norm_weighted_bf16_row(
        oracle_key, key_norm_data, inner_dim, 1e-6f);
    double oracle_seconds = now_seconds() - oracle_start;

    double setup_start = now_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    ltx_gpu_buffer *input = NULL;
    ltx_gpu_buffer *query_weight = NULL;
    ltx_gpu_buffer *query_scale = NULL;
    ltx_gpu_buffer *query_bias = NULL;
    ltx_gpu_buffer *key_weight = NULL;
    ltx_gpu_buffer *key_scale = NULL;
    ltx_gpu_buffer *key_bias = NULL;
    ltx_gpu_buffer *value_weight = NULL;
    ltx_gpu_buffer *value_scale = NULL;
    ltx_gpu_buffer *value_bias = NULL;
    ltx_gpu_buffer *query_norm = NULL;
    ltx_gpu_buffer *key_norm = NULL;
    ltx_gpu_buffer *query_output = NULL;
    ltx_gpu_buffer *key_output = NULL;
    ltx_gpu_buffer *value_output = NULL;
    ltx_gpu_buffer *rotated = NULL;
    ltx_gpu_buffer *query_raw = NULL;
    ltx_gpu_buffer *key_raw = NULL;
#define LTX_UPLOAD_LINEAR(NAME, MAPPED) \
        NAME##_weight = ltx_gpu_buffer_new_copy( \
            gpu, (MAPPED).weight, (MAPPED).weight_bytes, error, sizeof(error)); \
        NAME##_scale = ltx_gpu_buffer_new_copy( \
            gpu, (MAPPED).scale, (MAPPED).scale_bytes, error, sizeof(error)); \
        if ((MAPPED).bias) \
            NAME##_bias = ltx_gpu_buffer_new_copy( \
                gpu, (MAPPED).bias, (MAPPED).bias_bytes, error, sizeof(error))
    if (gpu) {
        input = ltx_gpu_buffer_new_copy(gpu, host_input, input_bytes,
                                        error, sizeof(error));
        LTX_UPLOAD_LINEAR(query, mapped_query);
        LTX_UPLOAD_LINEAR(key, mapped_key);
        LTX_UPLOAD_LINEAR(value, mapped_value);
        query_norm = ltx_gpu_buffer_new_copy(
            gpu, query_norm_data, query_norm_bytes, error, sizeof(error));
        key_norm = ltx_gpu_buffer_new_copy(
            gpu, key_norm_data, key_norm_bytes, error, sizeof(error));
        query_output = ltx_gpu_buffer_new(
            gpu, output_bytes, error, sizeof(error));
        key_output = ltx_gpu_buffer_new(
            gpu, output_bytes, error, sizeof(error));
        value_output = ltx_gpu_buffer_new(
            gpu, output_bytes, error, sizeof(error));
        if (backend == QKV_BACKEND_STAGED) {
            rotated = ltx_gpu_buffer_new(gpu, input_bytes,
                                          error, sizeof(error));
            query_raw = ltx_gpu_buffer_new(gpu, output_bytes,
                                            error, sizeof(error));
            key_raw = ltx_gpu_buffer_new(gpu, output_bytes,
                                          error, sizeof(error));
        }
    }
#undef LTX_UPLOAD_LINEAR
    double setup_seconds = now_seconds() - setup_start;
    if (!gpu || !input || !query_weight || !query_scale ||
        !key_weight || !key_scale || !value_weight || !value_scale ||
        !query_norm || !key_norm || !query_output || !key_output ||
        !value_output || (mapped_query.bias && !query_bias) ||
        (mapped_key.bias && !key_bias) ||
        (mapped_value.bias && !value_bias) ||
        (backend == QKV_BACKEND_STAGED &&
         (!rotated || !query_raw || !key_raw))) {
        fprintf(stderr, "bench_qkv: GPU setup failed: %s\n", error);
        goto cleanup_gpu;
    }

    for (uint32_t index = 0; index < warmup; index++)
        if (!run_qkv(
                backend, gpu, query_output, key_output, value_output, input,
                query_weight, query_scale, query_bias,
                key_weight, key_scale, key_bias,
                value_weight, value_scale, value_bias,
                query_norm, key_norm, rotated, query_raw, key_raw,
                rows, input_dim, inner_dim, error, sizeof(error))) {
            fprintf(stderr, "bench_qkv: warmup failed: %s\n", error);
            goto cleanup_gpu;
        }
    for (uint32_t index = 0; index < iterations; index++) {
        double start = now_seconds();
        if (!run_qkv(
                backend, gpu, query_output, key_output, value_output, input,
                query_weight, query_scale, query_bias,
                key_weight, key_scale, key_bias,
                value_weight, value_scale, value_bias,
                query_norm, key_norm, rotated, query_raw, key_raw,
                rows, input_dim, inner_dim, error, sizeof(error))) {
            fprintf(stderr, "bench_qkv: iteration failed: %s\n", error);
            goto cleanup_gpu;
        }
        timings[index] = now_seconds() - start;
    }
    if (!ltx_gpu_buffer_read(query_output, host_query, output_row_bytes,
                             error, sizeof(error)) ||
        !ltx_gpu_buffer_read(key_output, host_key, output_row_bytes,
                             error, sizeof(error)) ||
        !ltx_gpu_buffer_read(value_output, host_value, output_row_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_qkv: output read failed: %s\n", error);
        goto cleanup_gpu;
    }

    parity_metrics metrics = {0};
    update_metrics(&metrics, oracle_query, host_query, inner_dim);
    update_metrics(&metrics, oracle_key, host_key, inner_dim);
    update_metrics(&metrics, oracle_value, host_value, inner_dim);
    qsort(timings, iterations, sizeof(*timings), compare_double);
    size_t p50_index = (size_t)(iterations - 1u) / 2u;
    size_t p95_index = (size_t)ceil(0.95 * (double)iterations) - 1u;
    if (p95_index >= iterations) p95_index = iterations - 1u;
    double p50 = timings[p50_index];
    double p95 = timings[p95_index];
    double operations = 6.0 * (double)rows *
        (double)input_dim * (double)inner_dim;
    double rel_l2 = metrics.reference2 > 0.0 ?
        sqrt(metrics.diff2 / metrics.reference2) : 0.0;
    double cosine = metrics.reference2 > 0.0 && metrics.candidate2 > 0.0 ?
        metrics.dot / sqrt(metrics.reference2 * metrics.candidate2) : 0.0;
    printf("checkpoint=%s\n", checkpoint_path);
    printf("attention=%s shape=%u->3x%u rows=%u backend=%s\n",
           prefix, input_dim, inner_dim, rows,
           backend == QKV_BACKEND_FUSED ? "fused-mpsgraph" :
                                          "staged-mpsgraph");
    printf("weight_bytes=%zu setup_seconds=%.6f oracle_seconds=%.6f\n",
           mapped_query.weight_bytes + mapped_key.weight_bytes +
               mapped_value.weight_bytes,
           setup_seconds, oracle_seconds);
    printf("warmup=%u iterations=%u p50_ms=%.3f p95_ms=%.3f "
           "effective_gflops=%.2f\n",
           warmup, iterations, p50 * 1000.0, p95 * 1000.0,
           operations / p50 / 1e9);
    printf("row0_qkv_rel_l2=%.9g row0_qkv_cosine=%.9g max_abs=%.9g "
           "nonfinite=%llu\n",
           rel_l2, cosine, metrics.max_abs,
           (unsigned long long)metrics.nonfinite);
    result = metrics.nonfinite || cosine < 0.99 ? 1 : 0;

cleanup_gpu:
    ltx_gpu_buffer_free(input);
    ltx_gpu_buffer_free(query_weight);
    ltx_gpu_buffer_free(query_scale);
    ltx_gpu_buffer_free(query_bias);
    ltx_gpu_buffer_free(key_weight);
    ltx_gpu_buffer_free(key_scale);
    ltx_gpu_buffer_free(key_bias);
    ltx_gpu_buffer_free(value_weight);
    ltx_gpu_buffer_free(value_scale);
    ltx_gpu_buffer_free(value_bias);
    ltx_gpu_buffer_free(query_norm);
    ltx_gpu_buffer_free(key_norm);
    ltx_gpu_buffer_free(query_output);
    ltx_gpu_buffer_free(key_output);
    ltx_gpu_buffer_free(value_output);
    ltx_gpu_buffer_free(rotated);
    ltx_gpu_buffer_free(query_raw);
    ltx_gpu_buffer_free(key_raw);
    ltx_gpu_free(gpu);
cleanup_host:
    free(host_input);
    free(host_query);
    free(host_key);
    free(host_value);
    free(oracle_rotated);
    free(oracle_query);
    free(oracle_key);
    free(oracle_value);
    free(timings);
cleanup_mapping:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
