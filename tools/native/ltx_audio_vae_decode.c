#include "ltx_mlx_audio_vae.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(1);
}

static void *read_file(const char *path, size_t expected_bytes) {
    FILE *stream = fopen(path, "rb");
    if (!stream) fail(strerror(errno));
    if (fseek(stream, 0, SEEK_END) != 0) fail("cannot seek input");
    long length = ftell(stream);
    if (length < 0 || (size_t)length != expected_bytes) {
        fail("input byte count does not match BLC shape");
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data) fail("out of memory reading audio latent");
    if (fread(data, 1, expected_bytes, stream) != expected_bytes) {
        fail("cannot read audio latent");
    }
    fclose(stream);
    return data;
}

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *stream = fopen(path, "wb");
    if (!stream) fail(strerror(errno));
    if (fwrite(data, 1, bytes, stream) != bytes || fclose(stream) != 0) {
        fail("cannot write mel output");
    }
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s CHECKPOINT INPUT_BF16 BATCH TOKENS OUTPUT_BF16 [DUMP_DIR]\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long batch_value = strtoul(argv[3], &end, 10);
    if (!end || *end || batch_value == 0 || batch_value > UINT32_MAX) {
        fail("invalid batch");
    }
    unsigned long tokens_value = strtoul(argv[4], &end, 10);
    if (!end || *end || tokens_value == 0 || tokens_value > UINT32_MAX / 4u) {
        fail("invalid token count");
    }
    uint32_t batch = (uint32_t)batch_value;
    uint32_t tokens = (uint32_t)tokens_value;
    size_t input_elements = (size_t)batch * tokens * 128u;
    size_t output_elements = (size_t)batch * 2u *
        (tokens * 4u - 3u) * 64u;
    uint16_t *input = read_file(argv[2], input_elements * sizeof(uint16_t));
    uint16_t *output = calloc(output_elements, sizeof(uint16_t));
    if (!output) fail("out of memory allocating mel output");
    char error[1024] = {0};
    ltx_mlx_audio_vae *vae = ltx_mlx_audio_vae_create(
        argv[1], error, sizeof(error));
    if (!vae) fail(error);
    ltx_mlx_audio_vae_info info = {0};
    if (!ltx_mlx_audio_vae_get_info(vae, &info)) fail("cannot inspect audio VAE");
    int decoded = argc == 7 ? ltx_mlx_audio_vae_decode_bf16_debug(
        vae, output, output_elements, input, input_elements,
        batch, tokens, argv[6], error, sizeof(error)) :
        ltx_mlx_audio_vae_decode_bf16(
        vae, output, output_elements, input, input_elements,
        batch, tokens, error, sizeof(error));
    if (!decoded) {
        fail(error);
    }
    write_file(argv[5], output, output_elements * sizeof(uint16_t));
    printf("{\"batch\":%u,\"tokens\":%u,\"mel_time\":%u,"
           "\"mel_bins\":64,\"channels\":2,\"weight_tensors\":%u,"
           "\"weight_bytes\":%llu}\n",
           batch, tokens, tokens * 4u - 3u, info.weight_tensors,
           (unsigned long long)info.weight_bytes);
    ltx_mlx_audio_vae_free(vae);
    free(output);
    free(input);
    return 0;
}
