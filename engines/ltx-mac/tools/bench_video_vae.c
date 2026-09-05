#include "ltx_gpu.h"
#include "ltx_video_vae.h"

#include <errno.h>
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
        fprintf(stderr, "bench_video_vae: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static int read_exact(const char *path, void *data, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    int ok = fread(data, 1u, bytes, file) == bytes;
    ok = ok && fgetc(file) == EOF;
    fclose(file);
    return ok;
}

static int write_exact(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(data, 1u, bytes, file) == bytes;
    ok = ok && fclose(file) == 0;
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 8) {
        fprintf(stderr,
                "Usage: %s CHECKPOINT [frames height width iterations "
                "[token_input.bf16 output.bf16]]\n",
                argv[0]);
        return 2;
    }
    uint32_t frames = argc > 2 ? parse_u32(argv[2], "frames") : 2u;
    uint32_t height = argc > 3 ? parse_u32(argv[3], "height") : 2u;
    uint32_t width = argc > 4 ? parse_u32(argv[4], "width") : 3u;
    uint32_t iterations = argc > 5 ? parse_u32(argv[5], "iterations") : 1u;
    const char *input_path = argc > 6 ? argv[6] : NULL;
    const char *output_path = argc > 7 ? argv[7] : NULL;
    if ((input_path == NULL) != (output_path == NULL)) {
        fputs("bench_video_vae: input and output paths must be paired\n",
              stderr);
        return 2;
    }

    uint32_t output_frames = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    if (!ltx_video_vae_output_shape(frames, height, width,
                                    &output_frames, &output_height,
                                    &output_width)) {
        fputs("bench_video_vae: invalid shape\n", stderr);
        return 2;
    }
    const size_t input_count =
        (size_t)frames * height * width * 128u;
    const size_t output_count =
        (size_t)3u * output_frames * output_height * output_width;
    uint16_t *input = malloc(input_count * sizeof(*input));
    uint16_t *output = output_path ?
        malloc(output_count * sizeof(*output)) : NULL;
    if (!input || (output_path && !output)) {
        fputs("bench_video_vae: host allocation failed\n", stderr);
        free(input);
        free(output);
        return 1;
    }
    if (input_path) {
        if (!read_exact(input_path, input,
                        input_count * sizeof(*input))) {
            fprintf(stderr, "bench_video_vae: cannot read %s: %s\n",
                    input_path, strerror(errno));
            free(input);
            free(output);
            return 1;
        }
    } else {
        for (size_t index = 0; index < input_count; index++) {
            float value = (float)((int)(index % 257u) - 128) / 256.0f;
            input[index] = f32_to_bf16(value);
        }
    }

    char error[2048] = {0};
    double load_start = monotonic_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal",
                                  error, sizeof(error));
    ltx_video_vae *vae = gpu ?
        ltx_video_vae_create(gpu, argv[1], error, sizeof(error)) : NULL;
    double load_seconds = monotonic_seconds() - load_start;
    ltx_gpu_buffer *x = vae ?
        ltx_gpu_buffer_new_copy(gpu, input,
            input_count * sizeof(*input), error, sizeof(error)) : NULL;
    ltx_gpu_buffer *y = x ?
        ltx_gpu_buffer_new(gpu, output_count * sizeof(uint16_t),
                           error, sizeof(error)) : NULL;
    int ok = y != NULL;
    double first_seconds = 0.0;
    if (ok) {
        double start = monotonic_seconds();
        ok = ltx_video_vae_decode_tokens_bf16(
            vae, y, x, 1u, frames, height, width,
            error, sizeof(error));
        first_seconds = monotonic_seconds() - start;
    }
    double warm_start = monotonic_seconds();
    for (uint32_t iteration = 0; ok && iteration < iterations; iteration++) {
        ok = ltx_video_vae_decode_tokens_bf16(
            vae, y, x, 1u, frames, height, width,
            error, sizeof(error));
    }
    double warm_seconds = monotonic_seconds() - warm_start;

    ltx_video_vae_info info;
    memset(&info, 0, sizeof(info));
    if (vae) ltx_video_vae_get_info(vae, &info);
    if (ok && output_path) {
        ok = ltx_gpu_buffer_read(
                 y, output, output_count * sizeof(*output),
                 error, sizeof(error)) &&
             write_exact(output_path, output,
                         output_count * sizeof(*output));
        if (!ok && !error[0]) {
            snprintf(error, sizeof(error), "write %s failed: %s",
                     output_path, strerror(errno));
        }
    }
    if (ok) {
        printf("Video VAE: latent=%u rgb=%u stages=%u weights=%u "
               "(%.3f GiB)\n",
               info.latent_channels, info.output_channels,
               info.graph_stages, info.weight_tensors,
               (double)info.weight_bytes /
                   (1024.0 * 1024.0 * 1024.0));
        printf("Shape: [1,%u,%u,%u,128] -> [1,3,%u,%u,%u]\n",
               frames, height, width,
               output_frames, output_height, output_width);
        printf("Load: %.3f s | first: %.3f s | warm mean: %.3f s "
               "(%u iterations)\n",
               load_seconds, first_seconds,
               warm_seconds / (double)iterations, iterations);
        if (output_path) printf("Output: %s\n", output_path);
    } else {
        fprintf(stderr, "bench_video_vae: %s\n", error);
    }

    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(x);
    ltx_video_vae_free(vae);
    ltx_gpu_free(gpu);
    free(input);
    free(output);
    return ok ? 0 : 1;
}
