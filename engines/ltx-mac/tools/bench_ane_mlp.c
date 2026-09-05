#include "ltx_ane_mlp.h"
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
    const void *weight;
    size_t weight_bytes;
    const void *scale;
    size_t scale_bytes;
    const void *bias;
    size_t bias_bytes;
} mapped_linear;

static double now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return ((double)value.tv_sec + (double)value.tv_nsec * 1e-9) * 1000.0;
}

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

static unsigned parse_unsigned(const char *text, const char *label,
                               int allow_zero) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || (!allow_zero && !value) ||
        value > UINT32_MAX) {
        fprintf(stderr, "bench_ane_mlp: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (unsigned)value;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *values, unsigned count,
                         double quantile) {
    double *copy = malloc((size_t)count * sizeof(*copy));
    if (!copy) return NAN;
    memcpy(copy, values, (size_t)count * sizeof(*copy));
    qsort(copy, count, sizeof(*copy), compare_double);
    double index = quantile * (double)(count - 1u);
    unsigned low = (unsigned)floor(index);
    unsigned high = (unsigned)ceil(index);
    double result = copy[low] + (copy[high] - copy[low]) *
        (index - (double)low);
    free(copy);
    return result;
}

static int map_linear(const ltx_st_mapping *mapping,
                      const ltx_linear_weight_info *linear,
                      mapped_linear *mapped,
                      char *error, size_t error_size) {
    memset(mapped, 0, sizeof(*mapped));
    mapped->weight = ltx_st_map_tensor(
        mapping, linear->weight, &mapped->weight_bytes, error, error_size);
    mapped->scale = ltx_st_map_tensor(
        mapping, linear->weight_scale, &mapped->scale_bytes,
        error, error_size);
    if (linear->bias)
        mapped->bias = ltx_st_map_tensor(
            mapping, linear->bias, &mapped->bias_bytes, error, error_size);
    return mapped->weight && mapped->scale &&
        (!linear->bias || mapped->bias);
}

static int resolve_mlp(const ltx_st_header *header,
                       const ltx_st_mapping *mapping,
                       uint32_t block,
                       ltx_linear_weight_info *fc1,
                       ltx_linear_weight_info *fc2,
                       char *error, size_t error_size) {
    char first[1024];
    char second[1024];
    int first_length = snprintf(
        first, sizeof(first),
        "model.diffusion_model.transformer_blocks.%u.ff.net.0.proj", block);
    int second_length = snprintf(
        second, sizeof(second),
        "model.diffusion_model.transformer_blocks.%u.ff.net.2", block);
    if (first_length < 0 || (size_t)first_length >= sizeof(first) ||
        second_length < 0 || (size_t)second_length >= sizeof(second)) {
        snprintf(error, error_size, "MLP tensor name is too long");
        return 0;
    }
    if (!ltx_linear_weight_resolve(
            header, mapping, first, fc1, error, error_size) ||
        !ltx_linear_weight_resolve(
            header, mapping, second, fc2, error, error_size)) return 0;
    if (!fc1->quantized_int8 || !fc2->quantized_int8 ||
        !fc1->convrot || !fc2->convrot ||
        fc1->convrot_group_size != 256u ||
        fc2->convrot_group_size != 256u ||
        fc1->input_dim != 4096u || fc1->output_dim != 16384u ||
        fc2->input_dim != 16384u || fc2->output_dim != 4096u ||
        fc1->bias || fc2->bias) {
        snprintf(error, error_size,
                 "block %u is not the expected bias-free video FFN", block);
        return 0;
    }
    return 1;
}

static int run_gpu_reference(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input,
        const ltx_gpu_buffer *fc1_weight,
        const ltx_gpu_buffer *fc1_scale,
        const ltx_gpu_buffer *fc2_weight,
        const ltx_gpu_buffer *fc2_scale,
        uint32_t rows, char *error, size_t error_size) {
    return ltx_gpu_mlp_int8_convrot_mps_bf16(
        gpu, output, input, fc1_weight, fc1_scale, NULL,
        fc2_weight, fc2_scale, NULL, rows, 4096u, 16384u, 4096u,
        256u, error, error_size);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT MANIFEST VARIANT [warmup] [iterations] "
        "[rows]\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint = argv[1];
    const char *manifest = argv[2];
    const char *variant = argv[3];
    unsigned warmup = argc > 4 ?
        parse_unsigned(argv[4], "warmup", 1) : 3u;
    unsigned iterations = argc > 5 ?
        parse_unsigned(argv[5], "iterations", 0) : 9u;
    unsigned requested_rows = argc > 6 ?
        parse_unsigned(argv[6], "rows", 0) : 0u;
    char error[2048] = {0};
    int status = 1;

    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "bench_ane_mlp: %s\n", error);
        return 1;
    }
    double setup_started = now_ms();
    ltx_ane_mlp *ane = ltx_ane_mlp_create(
        gpu, manifest, variant, error, sizeof(error));
    if (!ane) {
        fprintf(stderr, "bench_ane_mlp: create ANE runtime: %s\n", error);
        ltx_gpu_free(gpu);
        return 1;
    }
    if (requested_rows && !ltx_ane_mlp_set_rows(
            ane, gpu, requested_rows, error, sizeof(error))) {
        fprintf(stderr, "bench_ane_mlp: select rows: %s\n", error);
        ltx_ane_mlp_free(ane);
        ltx_gpu_free(gpu);
        return 1;
    }
    const ltx_ane_mlp_shape *shape = ltx_ane_mlp_get_shape(ane);

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_ane_mlp: %s\n", error);
        goto cleanup_ane;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_ane_mlp: %s\n", error);
        goto cleanup_header;
    }
    ltx_linear_weight_info fc1;
    ltx_linear_weight_info fc2;
    mapped_linear mapped_fc1;
    mapped_linear mapped_fc2;
    if (!resolve_mlp(&header, &mapping, shape->block_index,
                     &fc1, &fc2, error, sizeof(error)) ||
        !map_linear(&mapping, &fc1, &mapped_fc1, error, sizeof(error)) ||
        !map_linear(&mapping, &fc2, &mapped_fc2, error, sizeof(error))) {
        fprintf(stderr, "bench_ane_mlp: resolve weights: %s\n", error);
        goto cleanup_mapping;
    }

    size_t input_elements = (size_t)shape->rows * shape->hidden;
    size_t input_bytes = input_elements * sizeof(uint16_t);
    uint16_t *host_input = malloc(input_bytes);
    uint16_t *reference_values = malloc(input_bytes);
    uint16_t *candidate_values = malloc(input_bytes);
    double *gpu_times = calloc(iterations, sizeof(double));
    double *hetero_times = calloc(iterations, sizeof(double));
    double *pack_times = calloc(iterations, sizeof(double));
    double *overlap_times = calloc(iterations, sizeof(double));
    double *gpu_branch_times = calloc(iterations, sizeof(double));
    double *ane_branch_times = calloc(iterations, sizeof(double));
    double *join_times = calloc(iterations, sizeof(double));
    int output_backing_used = 0;
    if (!host_input || !reference_values || !candidate_values ||
        !gpu_times || !hetero_times || !pack_times || !overlap_times ||
        !gpu_branch_times || !ane_branch_times || !join_times) {
        fputs("bench_ane_mlp: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (size_t index = 0; index < input_elements; index++) {
        float value = sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f;
        host_input[index] = f32_to_bf16(value);
    }

    ltx_gpu_buffer *input = ltx_gpu_buffer_new_copy(
        gpu, host_input, input_bytes, error, sizeof(error));
    ltx_gpu_buffer *fc1_weight = ltx_gpu_buffer_new_copy(
        gpu, mapped_fc1.weight, mapped_fc1.weight_bytes,
        error, sizeof(error));
    ltx_gpu_buffer *fc1_scale = ltx_gpu_buffer_new_copy(
        gpu, mapped_fc1.scale, mapped_fc1.scale_bytes,
        error, sizeof(error));
    ltx_gpu_buffer *fc2_weight = ltx_gpu_buffer_new_copy(
        gpu, mapped_fc2.weight, mapped_fc2.weight_bytes,
        error, sizeof(error));
    ltx_gpu_buffer *fc2_scale = ltx_gpu_buffer_new_copy(
        gpu, mapped_fc2.scale, mapped_fc2.scale_bytes,
        error, sizeof(error));
    ltx_gpu_buffer *reference = ltx_gpu_buffer_new(
        gpu, input_bytes, error, sizeof(error));
    ltx_gpu_buffer *candidate = ltx_gpu_buffer_new(
        gpu, input_bytes, error, sizeof(error));
    if (!input || !fc1_weight || !fc1_scale || !fc2_weight ||
        !fc2_scale || !reference || !candidate) {
        fprintf(stderr, "bench_ane_mlp: GPU allocation: %s\n", error);
        goto cleanup_gpu;
    }
    double setup_ms = now_ms() - setup_started;

    for (unsigned index = 0; index < warmup + 1u; index++) {
        ltx_ane_mlp_timing timing;
        if (!run_gpu_reference(
                gpu, reference, input, fc1_weight, fc1_scale,
                fc2_weight, fc2_scale, shape->rows,
                error, sizeof(error)) ||
            !ltx_ane_mlp_eval(
                ane, gpu, candidate, input, &timing,
                error, sizeof(error))) {
            fprintf(stderr, "bench_ane_mlp: warmup: %s\n", error);
            goto cleanup_gpu;
        }
    }

    for (unsigned index = 0; index < iterations; index++) {
        ltx_ane_mlp_timing timing;
        double started;
        if (index & 1u) {
            if (!ltx_ane_mlp_eval(
                    ane, gpu, candidate, input, &timing,
                    error, sizeof(error))) {
                fprintf(stderr, "bench_ane_mlp: candidate: %s\n", error);
                goto cleanup_gpu;
            }
            started = now_ms();
            if (!run_gpu_reference(
                    gpu, reference, input, fc1_weight, fc1_scale,
                    fc2_weight, fc2_scale, shape->rows,
                    error, sizeof(error))) {
                fprintf(stderr, "bench_ane_mlp: reference: %s\n", error);
                goto cleanup_gpu;
            }
            gpu_times[index] = now_ms() - started;
        } else {
            started = now_ms();
            if (!run_gpu_reference(
                    gpu, reference, input, fc1_weight, fc1_scale,
                    fc2_weight, fc2_scale, shape->rows,
                    error, sizeof(error))) {
                fprintf(stderr, "bench_ane_mlp: reference: %s\n", error);
                goto cleanup_gpu;
            }
            gpu_times[index] = now_ms() - started;
            if (!ltx_ane_mlp_eval(
                    ane, gpu, candidate, input, &timing,
                    error, sizeof(error))) {
                fprintf(stderr, "bench_ane_mlp: candidate: %s\n", error);
                goto cleanup_gpu;
            }
        }
        hetero_times[index] = timing.total_ms;
        pack_times[index] = timing.pack_ms;
        overlap_times[index] = timing.overlap_ms;
        gpu_branch_times[index] = timing.gpu_ms;
        ane_branch_times[index] = timing.ane_ms;
        join_times[index] = timing.join_ms;
        output_backing_used = timing.ane_output_backing_used;
    }
    if (!ltx_gpu_buffer_read(
            reference, reference_values, input_bytes,
            error, sizeof(error)) ||
        !ltx_gpu_buffer_read(
            candidate, candidate_values, input_bytes,
            error, sizeof(error))) {
        fprintf(stderr, "bench_ane_mlp: output read: %s\n", error);
        goto cleanup_gpu;
    }

    long double difference2 = 0.0L;
    long double reference2 = 0.0L;
    long double candidate2 = 0.0L;
    long double dot = 0.0L;
    double max_abs = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < input_elements; index++) {
        double a = bf16_to_f32(reference_values[index]);
        double b = bf16_to_f32(candidate_values[index]);
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = b - a;
        difference2 += difference * difference;
        reference2 += a * a;
        candidate2 += b * b;
        dot += a * b;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double gpu_p50 = percentile(gpu_times, iterations, 0.5);
    double gpu_p95 = percentile(gpu_times, iterations, 0.95);
    double hetero_p50 = percentile(hetero_times, iterations, 0.5);
    double hetero_p95 = percentile(hetero_times, iterations, 0.95);
    double rel_l2 = reference2 > 0.0L ?
        sqrt((double)(difference2 / reference2)) : INFINITY;
    double cosine = reference2 > 0.0L && candidate2 > 0.0L ?
        (double)(dot / sqrtl(reference2 * candidate2)) : 0.0;
    double rms_ratio = reference2 > 0.0L ?
        sqrt((double)(candidate2 / reference2)) : INFINITY;
    printf(
        "block=%u rows=%u hidden=%u full_F=%u GPU_F=%u ANE_F=%u "
        "variant=%s warmup=%u iterations=%u\n",
        shape->block_index, shape->rows, shape->hidden,
        shape->full_intermediate, shape->gpu_intermediate,
        shape->ane_intermediate, variant, warmup, iterations);
    printf(
        "setup_ms=%.3f gpu_p50_ms=%.3f gpu_p95_ms=%.3f "
        "hetero_p50_ms=%.3f hetero_p95_ms=%.3f speedup=%.6fx\n",
        setup_ms, gpu_p50, gpu_p95, hetero_p50, hetero_p95,
        gpu_p50 / hetero_p50);
    printf(
        "pack_p50_ms=%.3f overlap_p50_ms=%.3f gpu_branch_p50_ms=%.3f "
        "ane_branch_p50_ms=%.3f join_p50_ms=%.3f\n",
        percentile(pack_times, iterations, 0.5),
        percentile(overlap_times, iterations, 0.5),
        percentile(gpu_branch_times, iterations, 0.5),
        percentile(ane_branch_times, iterations, 0.5),
        percentile(join_times, iterations, 0.5));
    printf(
        "rel_l2=%.9g cosine=%.9g rms_ratio=%.9g max_abs=%.9g "
        "nonfinite=%zu backing=%d\n",
        rel_l2, cosine, rms_ratio, max_abs, nonfinite,
        output_backing_used);
    status = 0;

cleanup_gpu:
    ltx_gpu_buffer_free(candidate);
    ltx_gpu_buffer_free(reference);
    ltx_gpu_buffer_free(fc2_scale);
    ltx_gpu_buffer_free(fc2_weight);
    ltx_gpu_buffer_free(fc1_scale);
    ltx_gpu_buffer_free(fc1_weight);
    ltx_gpu_buffer_free(input);
cleanup_host:
    free(join_times);
    free(ane_branch_times);
    free(gpu_branch_times);
    free(overlap_times);
    free(pack_times);
    free(hetero_times);
    free(gpu_times);
    free(candidate_values);
    free(reference_values);
    free(host_input);
cleanup_mapping:
    ltx_st_map_close(&mapping);
cleanup_header:
    ltx_st_free_header(&header);
cleanup_ane:
    ltx_ane_mlp_free(ane);
    ltx_gpu_free(gpu);
    return status;
}
