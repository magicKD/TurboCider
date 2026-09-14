/* Isolate packed QKV projection: synthetic inputs, not generated-video quality. */
#include "../../native/models/ltx_runtime/ltx_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t state = 42;
static uint32_t random_bits(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static uint16_t bf16(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static float f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16; float y; memcpy(&y, &bits, 4); return y;
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *a, unsigned n) {
    qsort(a, n, sizeof(*a), order);
    return n % 2 ? a[n/2] : (a[n/2-1] + a[n/2]) * .5;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s SHADER_PATH ROWS RUNS\n", argv[0]); return 2;
    }
    unsigned rows = (unsigned)strtoul(argv[2], NULL, 10);
    unsigned runs = (unsigned)strtoul(argv[3], NULL, 10);
    if (!rows || rows > 16384 || !runs || runs > 100) return 2;
    const unsigned dim = 4096;
    const size_t elements = (size_t)rows * dim, weights = (size_t)dim * dim;
    char error[2048] = {0};
    ltx_gpu *gpu = ltx_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    ltx_gpu_buffer *b[20] = {0};
    /* input, Wq/k/v, Sq/k/v, Bq/k/v, Nq/k, packed W/S/B, 3 outputs */
    size_t sizes[] = {elements*2, weights, weights, weights,
        dim*4, dim*4, dim*4, dim*2, dim*2, dim*2, dim*2, dim*2,
        weights*3, dim*12, dim*6, elements*2, elements*2, elements*2};
    int result = 1;
    uint16_t *reference = NULL;
    for (unsigned i = 0; i < 18; ++i) {
        b[i] = ltx_gpu_buffer_new(gpu, sizes[i], error, sizeof(error));
        if (!b[i]) goto done;
    }
    uint16_t *input = ltx_gpu_buffer_contents(b[0]);
    for (size_t i = 0; i < elements; ++i)
        input[i] = bf16(((float)(random_bits() >> 8) / 16777216.0f - .5f) * 2);
    for (unsigned p = 0; p < 3; ++p) {
        int8_t *w = ltx_gpu_buffer_contents(b[1+p]);
        float *s = ltx_gpu_buffer_contents(b[4+p]);
        uint16_t *bias = ltx_gpu_buffer_contents(b[7+p]);
        for (size_t i = 0; i < weights; ++i) w[i] = (int8_t)(random_bits() % 255 - 127);
        for (unsigned i = 0; i < dim; ++i) { s[i] = 0.0002f; bias[i] = bf16(.01f); }
        memcpy((char *)ltx_gpu_buffer_contents(b[12]) + p*weights, w, weights);
        memcpy((char *)ltx_gpu_buffer_contents(b[13]) + p*dim*4, s, dim*4);
        memcpy((char *)ltx_gpu_buffer_contents(b[14]) + p*dim*2, bias, dim*2);
    }
    for (unsigned p = 10; p < 12; ++p) {
        uint16_t *norm = ltx_gpu_buffer_contents(b[p]);
        for (unsigned i = 0; i < dim; ++i) norm[i] = bf16(1.0f);
    }
    double times[2][100] = {{0}};
    double error2 = 0, norm2 = 0, maximum = 0;
    reference = malloc(elements * 3 * sizeof(*reference));
    if (!reference) { snprintf(error, sizeof(error), "reference allocation failed"); goto done; }
    for (unsigned iteration = 0; iteration <= runs; ++iteration) {
        for (unsigned slot = 0; slot < 2; ++slot) {
            unsigned packed = iteration == 0 ? slot : (slot + iteration) % 2;
            double start = now();
            int ok = packed ? ltx_gpu_qkv_packed_int8_convrot_mps_bf16(
                gpu, b[15], b[16], b[17], b[0], b[12], b[13], b[14], b[10], b[11],
                rows, dim, dim, 256, 1e-6f, error, sizeof(error)) :
                ltx_gpu_qkv_int8_convrot_mps_bf16(
                gpu, b[15], b[16], b[17], b[0], b[1], b[4], b[7], b[2], b[5], b[8],
                b[3], b[6], b[9], b[10], b[11], rows, dim, dim, 256, 1e-6f, error, sizeof(error));
            double elapsed = now() - start;
            if (!ok) goto done;
            if (iteration) times[packed][iteration-1] = elapsed;
            if (iteration == 0) {
                for (unsigned p = 0; p < 3; ++p) {
                    uint16_t *output = ltx_gpu_buffer_contents(b[15+p]);
                    if (!packed) memcpy(reference + p*elements, output, elements*2);
                    else for (size_t i = 0; i < elements; ++i) {
                        double expected = f32(reference[p*elements+i]), actual = f32(output[i]);
                        if (!isfinite(expected) || !isfinite(actual)) {
                            snprintf(error, sizeof(error), "nonfinite projection output"); goto done;
                        }
                        double d = actual - expected;
                        error2 += d*d; norm2 += expected*expected;
                        maximum = fmax(maximum, fabs(d));
                    }
                }
            }
        }
    }
    double separate = median(times[0], runs), packed = median(times[1], runs);
    printf("{\"rows\":%u,\"dim\":%u,\"runs\":%u,\"separate_seconds\":%.9g,"
           "\"packed_seconds\":%.9g,\"speedup\":%.9g,\"relative_l2\":%.9g,"
           "\"max_abs\":%.9g,\"packing_included\":false,\"synthetic_input\":true}\n",
           rows, dim, runs, separate, packed, separate/packed, sqrt(error2/fmax(norm2,1e-30)), maximum);
    result = 0;
done:
    if (result) fprintf(stderr, "%s\n", error);
    free(reference);
    for (unsigned i = 0; i < 18; ++i) ltx_gpu_buffer_free(b[i]);
    ltx_gpu_free(gpu);
    return result;
}
