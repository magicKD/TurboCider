#include "ltx_mlx_video_vae.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

static void fail(const char *message) {
    fprintf(stderr, "ltx-video-vae-decode: %s\n", message);
    exit(1);
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !text[0] || !end || *end || value == 0 ||
        value > UINT32_MAX) {
        fprintf(stderr, "ltx-video-vae-decode: invalid %s: %s\n",
                label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static int checkpoint_fd_from_environment(void) {
    const char *text = getenv("TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD");
    if (!text || !text[0]) return -1;
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > INT_MAX)
        fail("invalid TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD");
    return (int)value;
}

static void *read_exact(const char *path, size_t bytes) {
    FILE *stream = fopen(path, "rb");
    if (!stream) fail(strerror(errno));
    void *data = malloc(bytes);
    if (!data) fail("out of memory reading latent");
    if (fread(data, 1, bytes, stream) != bytes || ferror(stream)) {
        fclose(stream);
        free(data);
        fail("cannot read latent input");
    }
    if (fgetc(stream) != EOF) {
        fclose(stream);
        free(data);
        fail("latent input has trailing bytes");
    }
    fclose(stream);
    return data;
}

static void write_exact(const char *path, const void *data, size_t bytes) {
    FILE *stream = fopen(path, "wb");
    if (!stream) fail(strerror(errno));
    if (fwrite(data, 1, bytes, stream) != bytes || fclose(stream) != 0)
        fail("cannot write decoded pixels");
}

static double elapsed_seconds(struct timespec started, struct timespec ended) {
    return (double)(ended.tv_sec - started.tv_sec) +
        (double)(ended.tv_nsec - started.tv_nsec) / 1000000000.0;
}

static void write_timings(const char *path,
                          double input_read,
                          double weight_load,
                          double decode,
                          double output_write,
                          double process_wall) {
    if (!path || !path[0]) return;
    FILE *stream = fopen(path, "w");
    if (!stream) fail("cannot create Video VAE timing report");
    int count = fprintf(
        stream,
        "{\"input_read_seconds\":%.9f,\"weight_load_seconds\":%.9f,"
        "\"decode_compute_seconds\":%.9f,\"output_write_seconds\":%.9f,"
        "\"child_wall_seconds\":%.9f}\n",
        input_read, weight_load, decode, output_write, process_wall);
    if (count < 0 || fclose(stream) != 0)
        fail("cannot write Video VAE timing report");
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s CHECKPOINT INPUT_BF16 LATENT_FRAMES "
                "LATENT_HEIGHT LATENT_WIDTH OUTPUT_BF16 [TIMINGS_JSON]\n",
                argv[0]);
        return 2;
    }
    struct timespec process_started = {0}, input_read_finished = {0};
    struct timespec weight_load_finished = {0}, decode_finished = {0};
    struct timespec output_write_finished = {0};
    clock_gettime(CLOCK_MONOTONIC, &process_started);
    const uint32_t frames = parse_u32(argv[3], "latent frames");
    const uint32_t height = parse_u32(argv[4], "latent height");
    const uint32_t width = parse_u32(argv[5], "latent width");
    const uint64_t output_frames64 = (uint64_t)frames * 8u - 7u;
    const uint64_t output_height64 = (uint64_t)height * 32u;
    const uint64_t output_width64 = (uint64_t)width * 32u;
    if ((uint64_t)frames * height * width > SIZE_MAX / 128u / sizeof(uint16_t) ||
        output_frames64 > UINT32_MAX || output_height64 > UINT32_MAX ||
        output_width64 > UINT32_MAX ||
        output_frames64 * output_height64 * output_width64 >
            SIZE_MAX / 3u / sizeof(uint16_t)) {
        fail("latent geometry overflows host size");
    }
    const size_t input_elements =
        (size_t)frames * height * width * 128u;
    const uint32_t output_frames = (uint32_t)output_frames64;
    const uint32_t output_height = (uint32_t)output_height64;
    const uint32_t output_width = (uint32_t)output_width64;
    const size_t output_elements =
        (size_t)3u * output_frames * output_height * output_width;
    uint16_t *input = read_exact(argv[2], input_elements * sizeof(uint16_t));
    clock_gettime(CLOCK_MONOTONIC, &input_read_finished);
    uint16_t *output = calloc(output_elements, sizeof(uint16_t));
    if (!output) fail("out of memory allocating decoded pixels");

    char error[2048] = {0};
    const int checkpoint_fd = checkpoint_fd_from_environment();
    ltx_mlx_video_vae *vae = checkpoint_fd >= 0 ?
        ltx_mlx_video_vae_create_fd(checkpoint_fd, argv[1], error,
                                    sizeof(error)) :
        ltx_mlx_video_vae_create(argv[1], error, sizeof(error));
    if (!vae) fail(error[0] ? error : "cannot load Video VAE");
    clock_gettime(CLOCK_MONOTONIC, &weight_load_finished);
    int ok = ltx_mlx_video_vae_decode_tokens_bf16(
        vae, output, output_elements, input, input_elements,
        1u, frames, height, width, error, sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &decode_finished);
    if (ok) {
        write_exact(argv[6], output, output_elements * sizeof(uint16_t));
        clock_gettime(CLOCK_MONOTONIC, &output_write_finished);
        write_timings(
            argc == 8 ? argv[7] : NULL,
            elapsed_seconds(process_started, input_read_finished),
            elapsed_seconds(input_read_finished, weight_load_finished),
            elapsed_seconds(weight_load_finished, decode_finished),
            elapsed_seconds(decode_finished, output_write_finished),
            elapsed_seconds(process_started, output_write_finished));
    }
    else fprintf(stderr, "ltx-video-vae-decode: %s\n",
                 error[0] ? error : "Video VAE decode failed");
    ltx_mlx_video_vae_free(vae);
    free(output);
    free(input);
    return ok ? 0 : 1;
}
