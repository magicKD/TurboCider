#include "ltx_mlx_video_vae.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s CHECKPOINT INPUT_BF16 LATENT_FRAMES "
                "LATENT_HEIGHT LATENT_WIDTH OUTPUT_BF16\n",
                argv[0]);
        return 2;
    }
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
    uint16_t *output = calloc(output_elements, sizeof(uint16_t));
    if (!output) fail("out of memory allocating decoded pixels");

    char error[2048] = {0};
    ltx_mlx_video_vae *vae = ltx_mlx_video_vae_create(
        argv[1], error, sizeof(error));
    if (!vae) fail(error[0] ? error : "cannot load Video VAE");
    int ok = ltx_mlx_video_vae_decode_tokens_bf16(
        vae, output, output_elements, input, input_elements,
        1u, frames, height, width, error, sizeof(error));
    if (ok) write_exact(argv[6], output, output_elements * sizeof(uint16_t));
    else fprintf(stderr, "ltx-video-vae-decode: %s\n",
                 error[0] ? error : "Video VAE decode failed");
    ltx_mlx_video_vae_free(vae);
    free(output);
    free(input);
    return ok ? 0 : 1;
}
