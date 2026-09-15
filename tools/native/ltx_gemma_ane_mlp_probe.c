#include "ltx_gemma_ane_mlp.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    GEMMA_HIDDEN = 3840,
    GEMMA_INTERMEDIATE = 15360,
    MAX_ROWS = 1024,
    MAX_RUNS = 50
};

static uint32_t random_state = 42u;

static uint32_t random_bits(void) {
    random_state ^= random_state << 13u;
    random_state ^= random_state >> 17u;
    random_state ^= random_state << 5u;
    return random_state;
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

static int parse_u32(const char *text, uint32_t maximum, int allow_zero,
                     uint32_t *value) {
    if (!text || !value || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || end == text || *end || (!allow_zero && !parsed) ||
        parsed > maximum)
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int write_exact(const char *directory, const char *name,
                       const void *data, size_t bytes) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= sizeof(path)) return 0;
    FILE *stream = fopen(path, "wb");
    if (!stream) return 0;
    size_t written = fwrite(data, 1u, bytes, stream);
    int close_status = fclose(stream);
    return written == bytes && close_status == 0;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(const double *values, uint32_t count) {
    double sorted[MAX_RUNS];
    memcpy(sorted, values, (size_t)count * sizeof(*values));
    qsort(sorted, count, sizeof(*sorted), compare_double);
    return count % 2u ? sorted[count / 2u] :
        (sorted[count / 2u - 1u] + sorted[count / 2u]) * 0.5;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *values = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    uint32_t layer = 0, rows = 0, runs = 0;
    if (argc == 4 && strcmp(argv[1], "--plan") == 0) {
        uint32_t execution_rows = 0;
        uint32_t minimum_profitable_rows = 0;
        char error[2048] = {};
        if (!parse_u32(argv[3], MAX_ROWS, 0, &rows) ||
            !ltx_gemma_ane_mlp_plan(
                argv[2], rows, &execution_rows, &minimum_profitable_rows,
                error, sizeof(error))) {
            fprintf(stderr, "%s\n", error[0] ? error :
                    "invalid Gemma ANE row plan");
            return 1;
        }
        printf("{\"requested_rows\":%u,\"execution_rows\":%u,"
               "\"minimum_profitable_rows\":%u}\n",
               rows, execution_rows, minimum_profitable_rows);
        return 0;
    }
    if (argc != 8 || !parse_u32(argv[4], 47u, 1, &layer) ||
        !parse_u32(argv[5], MAX_ROWS, 0, &rows) ||
        !parse_u32(argv[6], MAX_RUNS, 0, &runs)) {
        fprintf(stderr,
                "usage: %s --plan MANIFEST ROWS\n"
                "       %s MANIFEST CHECKPOINT SHADER LAYER ROWS RUNS "
                "OUTPUT_DIR\n", argv[0], argv[0]);
        return 2;
    }
    struct stat output_status;
    if (lstat(argv[7], &output_status) == 0 || errno != ENOENT) {
        fprintf(stderr, "output path must not already exist: %s\n", argv[7]);
        return 1;
    }

    const size_t elements = (size_t)rows * GEMMA_HIDDEN;
    const size_t bytes = elements * sizeof(uint16_t);
    uint16_t *input = malloc(bytes);
    uint16_t *output = malloc(bytes);
    if (!input || !output) {
        fprintf(stderr, "out of memory for Gemma ANE probe tensors\n");
        free(input);
        free(output);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        float uniform = (float)(random_bits() >> 8u) / 16777216.0f;
        input[index] = f32_to_bf16((uniform - 0.5f) * 2.0f);
    }

    char error[2048] = {};
    ltx_gpu *gpu = ltx_gpu_create(argv[3], error, sizeof(error));
    ltx_gemma_ane_mlp *mlp = gpu ? ltx_gemma_ane_mlp_create(
        gpu, argv[1], argv[2], layer, rows, error, sizeof(error)) : NULL;
    ltx_gpu_buffer *input_buffer = gpu ? ltx_gpu_buffer_new(
        gpu, bytes, error, sizeof(error)) : NULL;
    ltx_gpu_buffer *output_buffer = gpu ? ltx_gpu_buffer_new(
        gpu, bytes, error, sizeof(error)) : NULL;
    if (!gpu || !mlp || !input_buffer || !output_buffer ||
        !ltx_gpu_buffer_write(input_buffer, input, bytes, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error[0] ? error : "Gemma ANE probe setup failed");
        ltx_gpu_buffer_free(input_buffer);
        ltx_gpu_buffer_free(output_buffer);
        ltx_gemma_ane_mlp_free(mlp);
        ltx_gpu_free(gpu);
        free(input);
        free(output);
        return 1;
    }

    double warm_seconds[MAX_RUNS] = {};
    double warm_gpu_ms[MAX_RUNS] = {};
    double warm_ane_ms[MAX_RUNS] = {};
    double warm_pack_ms[MAX_RUNS] = {};
    double warm_join_ms[MAX_RUNS] = {};
    ltx_gemma_ane_mlp_timing timing = {};
    if (!ltx_gemma_ane_mlp_eval(mlp, gpu, output_buffer, input_buffer,
                                &timing, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error[0] ? error :
                "Gemma ANE probe first evaluation failed");
        ltx_gpu_buffer_free(input_buffer);
        ltx_gpu_buffer_free(output_buffer);
        ltx_gemma_ane_mlp_free(mlp);
        ltx_gpu_free(gpu);
        free(input);
        free(output);
        return 1;
    }
    double first_seconds = timing.total_ms / 1000.0;
    ltx_gemma_ane_mlp_timing first_timing = timing;
    for (uint32_t run = 0; run < runs; run++) {
        if (!ltx_gemma_ane_mlp_eval(mlp, gpu, output_buffer, input_buffer,
                                    &timing, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error[0] ? error :
                    "Gemma ANE probe evaluation failed");
            ltx_gpu_buffer_free(input_buffer);
            ltx_gpu_buffer_free(output_buffer);
            ltx_gemma_ane_mlp_free(mlp);
            ltx_gpu_free(gpu);
            free(input);
            free(output);
            return 1;
        }
        warm_seconds[run] = timing.total_ms / 1000.0;
        warm_gpu_ms[run] = timing.gpu_ms;
        warm_ane_ms[run] = timing.ane_ms;
        warm_pack_ms[run] = timing.pack_ms;
        warm_join_ms[run] = timing.join_ms;
    }
    if (!ltx_gpu_buffer_read(output_buffer, output, bytes, error, sizeof(error)) ||
        mkdir(argv[7], 0755) != 0) {
        fprintf(stderr, "%s\n", error[0] ? error :
                "cannot write Gemma ANE probe output");
        ltx_gpu_buffer_free(input_buffer);
        ltx_gpu_buffer_free(output_buffer);
        ltx_gemma_ane_mlp_free(mlp);
        ltx_gpu_free(gpu);
        free(input);
        free(output);
        return 1;
    }
    double sum_squares = 0.0;
    double maximum = 0.0;
    int all_finite = 1;
    for (size_t index = 0; index < elements; index++) {
        double value = bf16_to_f32(output[index]);
        if (!isfinite(value)) {
            all_finite = 0;
            continue;
        }
        sum_squares += value * value;
        if (fabs(value) > maximum) maximum = fabs(value);
    }
    if (!all_finite || !write_exact(argv[7], "input.bf16", input, bytes) ||
        !write_exact(argv[7], "output.bf16", output, bytes)) {
        fprintf(stderr, "Gemma ANE probe produced invalid or unwritable output\n");
        ltx_gpu_buffer_free(input_buffer);
        ltx_gpu_buffer_free(output_buffer);
        ltx_gemma_ane_mlp_free(mlp);
        ltx_gpu_free(gpu);
        free(input);
        free(output);
        return 1;
    }
    const ltx_gemma_ane_mlp_shape *shape = ltx_gemma_ane_mlp_get_shape(mlp);
    printf("{\"format\":\"turbocider-ltx-gemma-ane-mlp-probe-v1\","
           "\"layer\":%u,\"rows\":%u,\"hidden\":%u,"
           "\"intermediate\":%u,\"ane_intermediate\":%u,"
           "\"gpu_intermediate\":%u,\"warm_runs\":%u,"
           "\"first_seconds\":%.9g,\"warm_median_seconds\":%.9g,"
           "\"warm_seconds\":[",
           shape->layer, shape->rows, shape->hidden, shape->intermediate,
           shape->ane_intermediate, shape->gpu_intermediate, runs,
           first_seconds, median(warm_seconds, runs));
    for (uint32_t run = 0; run < runs; run++)
        printf("%s%.9g", run ? "," : "", warm_seconds[run]);
    printf("],\"first_timing_ms\":{\"total\":%.9g,\"pack\":%.9g,"
           "\"overlap\":%.9g,\"gpu\":%.9g,\"ane\":%.9g,\"join\":%.9g},"
           "\"warm_timing_median_ms\":{\"total\":%.9g,\"gpu\":%.9g,"
           "\"ane\":%.9g,\"pack\":%.9g,\"join\":%.9g},"
           "\"resident_weight_bytes\":%llu,\"workspace_bytes\":%llu,"
           "\"ane_output_backing_used\":%d,\"output_elements\":%zu,"
           "\"output_all_finite\":true,\"output_rms\":%.9g,"
           "\"output_max_abs\":%.9g,\"output_fnv1a64\":\"%016llx\","
           "\"artifacts\":{\"input\":\"input.bf16\","
           "\"output\":\"output.bf16\"}}\n",
           first_timing.total_ms, first_timing.pack_ms,
           first_timing.overlap_ms, first_timing.gpu_ms,
           first_timing.ane_ms, first_timing.join_ms,
           median(warm_seconds, runs) * 1000.0,
           median(warm_gpu_ms, runs), median(warm_ane_ms, runs),
           median(warm_pack_ms, runs), median(warm_join_ms, runs),
           (unsigned long long)ltx_gemma_ane_mlp_resident_weight_bytes(mlp),
           (unsigned long long)ltx_gemma_ane_mlp_workspace_bytes(mlp),
           timing.ane_output_backing_used, elements,
           sqrt(sum_squares / (double)elements), maximum,
           (unsigned long long)fnv1a64(output, bytes));
    ltx_gpu_buffer_free(input_buffer);
    ltx_gpu_buffer_free(output_buffer);
    ltx_gemma_ane_mlp_free(mlp);
    ltx_gpu_free(gpu);
    free(input);
    free(output);
    return 0;
}
