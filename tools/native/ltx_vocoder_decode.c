#include "ltx_mlx_vocoder.h"

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
        fail("input byte count does not match BCTF mel shape");
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data) fail("out of memory reading mel input");
    if (fread(data, 1, expected_bytes, stream) != expected_bytes) {
        fail("cannot read mel input");
    }
    fclose(stream);
    return data;
}

static void write_file(const char *path, const void *data, size_t bytes) {
    FILE *stream = fopen(path, "wb");
    if (!stream) fail(strerror(errno));
    if (fwrite(data, 1, bytes, stream) != bytes || fclose(stream) != 0) {
        fail("cannot write waveform output");
    }
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s CHECKPOINT MEL_BF16 BATCH MEL_TIME OUTPUT_F32 [DUMP_DIR]\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long batch_value = strtoul(argv[3], &end, 10);
    if (!end || *end || batch_value == 0 || batch_value > UINT32_MAX) {
        fail("invalid batch");
    }
    unsigned long time_value = strtoul(argv[4], &end, 10);
    if (!end || *end || time_value == 0 || time_value > UINT32_MAX / 160u) {
        fail("invalid mel time");
    }
    uint32_t batch = (uint32_t)batch_value;
    uint32_t mel_time = (uint32_t)time_value;
    size_t input_elements = (size_t)batch * 2u * mel_time * 64u;
    size_t output_elements = (size_t)batch * mel_time * 160u * 2u;
    uint16_t *input = read_file(argv[2], input_elements * sizeof(uint16_t));
    float *output = calloc(output_elements, sizeof(float));
    if (!output) fail("out of memory allocating waveform output");
    char error[1024] = {0};
    ltx_mlx_vocoder *vocoder = ltx_mlx_vocoder_create_base(
        argv[1], error, sizeof(error));
    if (!vocoder) fail(error);
    ltx_mlx_vocoder_info info = {0};
    if (!ltx_mlx_vocoder_get_info(vocoder, &info)) fail("cannot inspect vocoder");
    int decoded = argc == 7 ? ltx_mlx_vocoder_decode_base_bf16_debug(
        vocoder, output, output_elements, input, input_elements,
        batch, mel_time, argv[6], error, sizeof(error)) :
        ltx_mlx_vocoder_decode_base_bf16(
        vocoder, output, output_elements, input, input_elements,
        batch, mel_time, error, sizeof(error));
    if (!decoded) fail(error);
    write_file(argv[5], output, output_elements * sizeof(float));
    printf("{\"batch\":%u,\"mel_time\":%u,\"samples\":%u,"
           "\"channels\":2,\"sample_rate\":16000,\"weight_tensors\":%u,"
           "\"source_weight_bytes\":%llu,\"resident_weight_bytes\":%llu}\n",
           batch, mel_time, mel_time * 160u, info.weight_tensors,
           (unsigned long long)info.source_weight_bytes,
           (unsigned long long)info.resident_weight_bytes);
    ltx_mlx_vocoder_free(vocoder);
    free(output);
    free(input);
    return 0;
}
