#include "ltx_gemma_encoder.h"

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
    MAX_ROWS = 1024
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

static int parse_positive(const char *text, uint32_t maximum,
                          uint32_t *value) {
    if (!text || !value || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || end == text || *end || !parsed || parsed > maximum)
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
    double sorted[LTX_GEMMA_MLP_PROBE_MAX_RUNS];
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
    uint32_t rows = 0;
    uint32_t runs = 0;
    if (argc != 6 || !parse_positive(argv[3], MAX_ROWS, &rows) ||
        !parse_positive(argv[4], LTX_GEMMA_MLP_PROBE_MAX_RUNS, &runs)) {
        fprintf(stderr,
                "usage: %s CHECKPOINT SHADER ROWS WARM_RUNS OUTPUT_DIR\n",
                argv[0]);
        return 2;
    }
    struct stat output_status;
    if (lstat(argv[5], &output_status) == 0 || errno != ENOENT) {
        fprintf(stderr, "output path must not already exist: %s\n", argv[5]);
        return 1;
    }

    const size_t elements = (size_t)rows * GEMMA_HIDDEN;
    const size_t bytes = elements * sizeof(uint16_t);
    uint16_t *input = malloc(bytes);
    uint16_t *output = malloc(bytes);
    if (!input || !output) {
        fprintf(stderr, "out of memory for Gemma MLP probe tensors\n");
        free(input);
        free(output);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        float uniform = (float)(random_bits() >> 8u) / 16777216.0f;
        input[index] = f32_to_bf16((uniform - 0.5f) * 2.0f);
    }

    char error[2048] = {};
    ltx_gemma_mlp_probe_result result = {};
    int ok = ltx_gemma_mlp_gpu_probe(
        argv[1], argv[2], 0u, rows, runs, input, elements, output, elements,
        &result, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "%s\n", error[0] ? error : "Gemma MLP probe failed");
        free(input);
        free(output);
        return 1;
    }

    if (mkdir(argv[5], 0755) != 0) {
        fprintf(stderr, "cannot create output directory: %s\n", argv[5]);
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
        maximum = fmax(maximum, fabs(value));
    }
    if (!all_finite ||
        !write_exact(argv[5], "input.bf16", input, bytes) ||
        !write_exact(argv[5], "output.bf16", output, bytes)) {
        fprintf(stderr, "%s\n", all_finite ?
                "cannot write Gemma MLP probe tensors" :
                "Gemma MLP probe produced non-finite output");
        free(input);
        free(output);
        return 1;
    }

    const double setup_seconds =
        result.checkpoint_validation_seconds + result.checkpoint_open_seconds +
        result.gpu_setup_seconds + result.weight_load_seconds +
        result.workspace_setup_seconds;
    printf("{\"format\":\"turbocider-ltx-gemma-mlp-gpu-probe-v1\","
           "\"layer\":%u,\"rows\":%u,\"hidden\":%u,"
           "\"intermediate\":%u,\"warm_runs\":%u,"
           "\"synthetic_input\":true,\"input_seed\":42,"
           "\"input_distribution\":\"uniform[-1,1) BF16\","
           "\"batch_commands\":true,\"setup_seconds\":%.9g,"
           "\"checkpoint_validation_seconds\":%.9g,"
           "\"checkpoint_open_seconds\":%.9g,"
           "\"gpu_setup_seconds\":%.9g,\"weight_load_seconds\":%.9g,"
           "\"workspace_setup_seconds\":%.9g,\"first_seconds\":%.9g,"
           "\"warm_median_seconds\":%.9g,\"warm_seconds\":[",
           result.layer, result.rows, result.hidden, result.intermediate,
           result.warm_runs, setup_seconds,
           result.checkpoint_validation_seconds,
           result.checkpoint_open_seconds, result.gpu_setup_seconds,
           result.weight_load_seconds, result.workspace_setup_seconds,
           result.first_seconds, median(result.warm_seconds, runs));
    for (uint32_t run = 0; run < runs; run++)
        printf("%s%.9g", run ? "," : "", result.warm_seconds[run]);
    printf("],\"resident_weight_bytes\":%llu,\"workspace_bytes\":%llu,"
           "\"output_elements\":%zu,\"output_all_finite\":true,"
           "\"output_rms\":%.9g,\"output_max_abs\":%.9g,"
           "\"output_fnv1a64\":\"%016llx\","
           "\"artifacts\":{\"input\":\"input.bf16\","
           "\"output\":\"output.bf16\"}}\n",
           (unsigned long long)result.resident_weight_bytes,
           (unsigned long long)result.workspace_bytes, elements,
           sqrt(sum_squares / (double)elements), maximum,
           (unsigned long long)fnv1a64(output, bytes));
    free(input);
    free(output);
    return 0;
}
