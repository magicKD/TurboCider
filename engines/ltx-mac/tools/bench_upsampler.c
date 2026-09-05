#include "ltx_gpu.h"
#include "ltx_upsampler.h"

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

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !value || value > UINT32_MAX) {
        fprintf(stderr, "bench_upsampler: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 6) {
        fprintf(stderr,
                "Usage: %s CHECKPOINT [frames height width iterations]\n",
                argv[0]);
        return 2;
    }
    uint32_t frames = argc > 2 ? parse_u32(argv[2], "frames") : 13u;
    uint32_t height = argc > 3 ? parse_u32(argv[3], "height") : 7u;
    uint32_t width = argc > 4 ? parse_u32(argv[4], "width") : 11u;
    uint32_t iterations = argc > 5 ?
        parse_u32(argv[5], "iterations") : 3u;
    const uint32_t channels = 128u;
    size_t input_count =
        (size_t)channels * frames * height * width;
    size_t output_count =
        (size_t)channels * frames * height * 2u * width * 2u;
    uint16_t *input = malloc(input_count * sizeof(*input));
    if (!input) {
        fputs("bench_upsampler: host allocation failed\n", stderr);
        return 1;
    }
    for (size_t index = 0; index < input_count; index++) {
        float value = (float)((int)(index % 257u) - 128) / 64.0f;
        input[index] = f32_to_bf16(value);
    }

    char error[2048] = {0};
    double load_start = monotonic_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal",
                                  error, sizeof(error));
    ltx_upsampler *upsampler = gpu ?
        ltx_upsampler_create(gpu, argv[1], error, sizeof(error)) : NULL;
    double load_seconds = monotonic_seconds() - load_start;
    ltx_gpu_buffer *x = upsampler ?
        ltx_gpu_buffer_new_copy(gpu, input,
            input_count * sizeof(*input), error, sizeof(error)) : NULL;
    ltx_gpu_buffer *y = x ?
        ltx_gpu_buffer_new(gpu, output_count * sizeof(uint16_t),
                           error, sizeof(error)) : NULL;
    int ok = y != NULL;
    double first_seconds = 0.0;
    if (ok) {
        double start = monotonic_seconds();
        ok = ltx_upsampler_run_bf16(
            upsampler, y, x, 1u, frames, height, width,
            error, sizeof(error));
        first_seconds = monotonic_seconds() - start;
    }
    double warm_start = monotonic_seconds();
    for (uint32_t iteration = 0; ok && iteration < iterations; iteration++)
        ok = ltx_upsampler_run_bf16(
            upsampler, y, x, 1u, frames, height, width,
            error, sizeof(error));
    double warm_seconds = monotonic_seconds() - warm_start;

    ltx_upsampler_info info;
    memset(&info, 0, sizeof(info));
    if (upsampler) ltx_upsampler_get_info(upsampler, &info);
    if (ok) {
        printf("Upsampler: C=%u hidden=%u blocks=%u weights=%u "
               "(%.3f GiB)\n",
               info.input_channels, info.hidden_channels,
               info.residual_blocks_per_stage, info.weight_tensors,
               (double)info.weight_bytes /
                   (1024.0 * 1024.0 * 1024.0));
        printf("Shape: [1,128,%u,%u,%u] -> [1,128,%u,%u,%u]\n",
               frames, height, width, frames, height * 2u, width * 2u);
        printf("Load: %.3f s | first: %.3f s | warm mean: %.3f s "
               "(%u iterations)\n",
               load_seconds, first_seconds,
               warm_seconds / (double)iterations, iterations);
    } else {
        fprintf(stderr, "bench_upsampler: %s\n", error);
    }

    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(x);
    ltx_upsampler_free(upsampler);
    ltx_gpu_free(gpu);
    free(input);
    return ok ? 0 : 1;
}
