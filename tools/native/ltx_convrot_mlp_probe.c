/* Resident real-weight FFN probe; synthetic inputs are not video quality. */
#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char error[2048];
static uint32_t random_state = 42;
static uint16_t bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, 4);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static ltx_gpu_buffer *load(ltx_gpu *gpu, const ltx_st_header *header,
                            const char *name, int optional) {
    const ltx_st_tensor *t = ltx_st_find(header, name);
    if (!t) {
        if (!optional) snprintf(error, sizeof(error), "missing %s", name);
        return NULL;
    }
    size_t bytes = (size_t)(t->data_end - t->data_begin);
    void *data = malloc(bytes);
    if (!data) return NULL;
    int ok = ltx_st_read_data(header, t, data, bytes, error, sizeof(error));
    ltx_gpu_buffer *result = ok ? ltx_gpu_buffer_new_copy(gpu, data, bytes, error, sizeof(error)) : NULL;
    free(data);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s COMPONENT.safetensors SHADER ROWS RUNS mlp|gated OUTPUT.bf16\n", argv[0]);
        return 2;
    }
    uint32_t rows = (uint32_t)strtoul(argv[3], NULL, 10);
    uint32_t runs = (uint32_t)strtoul(argv[4], NULL, 10);
    int gated = !strcmp(argv[5], "gated");
    if (!rows || rows > 4096 || !runs || runs > 100 || (!gated && strcmp(argv[5], "mlp"))) return 2;
    ltx_st_header header = {0};
    ltx_gpu *gpu = NULL;
    ltx_gpu_buffer *w1 = NULL, *s1 = NULL, *b1 = NULL, *w2 = NULL, *s2 = NULL, *b2 = NULL;
    ltx_gpu_buffer *wu = NULL, *su = NULL, *x = NULL, *y = NULL;
    uint16_t *input = NULL, *output = NULL;
    int result = 1;
    if (!ltx_st_read_header(argv[1], &header, error, sizeof(error))) goto cleanup;
    const ltx_st_tensor *first = ltx_st_find(&header, "fc1.weight");
    const ltx_st_tensor *second = ltx_st_find(&header, "fc2.weight");
    if (!first || !second || first->dtype != LTX_DTYPE_I8 || second->dtype != LTX_DTYPE_I8 ||
        first->ndim != 2 || second->ndim != 2 || first->shape[0] != second->shape[1] ||
        first->shape[1] != second->shape[0] || first->shape[0] > 65536 || first->shape[1] > 65536) {
        snprintf(error, sizeof(error), "invalid component FFN geometry"); goto cleanup;
    }
    uint32_t hidden = (uint32_t)first->shape[0], dim = (uint32_t)first->shape[1];
    size_t elements = (size_t)rows * dim, bytes = elements * 2u;
    gpu = ltx_gpu_create(argv[2], error, sizeof(error));
    if (!gpu) goto cleanup;
    w1 = load(gpu, &header, "fc1.weight", 0); s1 = load(gpu, &header, "fc1.weight_scale", 0);
    w2 = load(gpu, &header, "fc2.weight", 0); s2 = load(gpu, &header, "fc2.weight_scale", 0);
    b1 = load(gpu, &header, "fc1.bias", 1); b2 = load(gpu, &header, "fc2.bias", 1);
    if (!w1 || !s1 || !w2 || !s2) goto cleanup;
    if (gated) {
        wu = load(gpu, &header, "up.weight", 0); su = load(gpu, &header, "up.weight_scale", 0);
        if (!wu || !su || b1 || b2) goto cleanup;
    }
    input = malloc(bytes); output = malloc(bytes);
    if (!input || !output) goto cleanup;
    for (size_t i = 0; i < elements; ++i) {
        random_state ^= random_state << 13; random_state ^= random_state >> 17; random_state ^= random_state << 5;
        input[i] = bf16(((float)(random_state >> 8) / 16777216.f - 0.5f) * 2.f);
    }
    x = ltx_gpu_buffer_new_copy(gpu, input, bytes, error, sizeof(error));
    y = ltx_gpu_buffer_new(gpu, bytes, error, sizeof(error));
    if (!x || !y) goto cleanup;
    double samples[100];
    for (uint32_t iteration = 0; iteration < runs + 2; ++iteration) {
        double start = now();
        int ok = gated ? ltx_gpu_gated_mlp_int8_convrot_mps_bf16(gpu, y, x,
            w1, s1, wu, su, w2, s2, rows, dim, hidden, dim, 256u, error, sizeof(error)) :
            ltx_gpu_mlp_int8_convrot_mps_bf16(gpu, y, x,
                w1, s1, b1, w2, s2, b2, rows, dim, hidden, dim, 256u, error, sizeof(error));
        if (!ok) goto cleanup;
        if (iteration >= 2) samples[iteration - 2] = now() - start;
    }
    if (!ltx_gpu_buffer_read(y, output, bytes, error, sizeof(error))) goto cleanup;
    for (size_t i = 0; i < elements; ++i) {
        if ((output[i] & 0x7f80u) == 0x7f80u) {
            snprintf(error, sizeof(error), "nonfinite FFN output"); goto cleanup;
        }
    }
    FILE *stream = fopen(argv[6], "wbx");
    if (!stream) { snprintf(error, sizeof(error), "fresh output path required"); goto cleanup; }
    size_t wrote = fwrite(output, 1, bytes, stream);
    int closed = fclose(stream);
    if (wrote != bytes || closed) goto cleanup;
    qsort(samples, runs, sizeof(double), order);
    double median = runs % 2 ? samples[runs / 2] : (samples[runs / 2 - 1] + samples[runs / 2]) * 0.5;
    uint64_t projections = 0, quantizations = 0;
    int w8a8 = ltx_gpu_convrot_w8a8_info(gpu, &projections, &quantizations);
    printf("{\"scope\":\"real weights, synthetic input, isolated FFN\",\"rows\":%u,\"dim\":%u,"
        "\"hidden\":%u,\"gated\":%s,\"runs\":%u,\"median_seconds\":%.9g,\"w8a8\":%s,\"convrot_simd\":%s,"
        "\"projections\":%llu,\"activation_quantizations\":%llu,\"gelu_convrot_dispatches\":%llu,\"rot_quant_dispatches\":%llu,\"gelu_output_dispatches\":%llu}\n",
        rows, dim, hidden, gated ? "true" : "false", runs, median, w8a8 ? "true" : "false",
        ltx_gpu_convrot_simd_enabled(gpu) ? "true" : "false",
        (unsigned long long)projections, (unsigned long long)quantizations,
        (unsigned long long)ltx_gpu_gelu_convrot_dispatches(gpu),
        (unsigned long long)ltx_gpu_rot_quant_dispatches(gpu),
        (unsigned long long)ltx_gpu_gelu_output_dispatches(gpu));
    result = 0;
cleanup:
    if (result) fprintf(stderr, "%s\n", error[0] ? error : "FFN probe failed");
    ltx_gpu_buffer_free(y); ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(su); ltx_gpu_buffer_free(wu);
    ltx_gpu_buffer_free(b2); ltx_gpu_buffer_free(s2); ltx_gpu_buffer_free(w2);
    ltx_gpu_buffer_free(b1); ltx_gpu_buffer_free(s1); ltx_gpu_buffer_free(w1);
    free(output); free(input); ltx_gpu_free(gpu); ltx_st_free_header(&header);
    return result;
}
