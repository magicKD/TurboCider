#include "ltx_mlx_bwe.h"

#include "ltx_audio_decode_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s CHECKPOINT WAVE16_F32 BATCH SAMPLES OUTPUT48_F32 [DUMP_DIR]\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    unsigned long batch_value = strtoul(argv[3], &end, 10);
    if (!end || *end || batch_value == 0 || batch_value > UINT32_MAX) {
        fail("invalid batch");
    }
    unsigned long samples_value = strtoul(argv[4], &end, 10);
    if (!end || *end || samples_value == 0 || samples_value > UINT32_MAX / 3u) {
        fail("invalid sample count");
    }
    uint32_t batch = (uint32_t)batch_value;
    uint32_t samples = (uint32_t)samples_value;
    size_t input_elements = (size_t)batch * samples * 2u;
    size_t output_elements = (size_t)batch * samples * 3u * 2u;
    float *input = read_file(
        argv[2], input_elements * sizeof(float),
        "input byte count does not match BTC waveform shape",
        "out of memory reading waveform input", "cannot read waveform input");
    float *output = calloc(output_elements, sizeof(float));
    if (!output) fail("out of memory allocating BWE output");
    char error[1024] = {0};
    ltx_mlx_bwe *bwe = ltx_mlx_bwe_create(argv[1], error, sizeof(error));
    if (!bwe) fail(error);
    ltx_mlx_bwe_info info = {0};
    if (!ltx_mlx_bwe_get_info(bwe, &info)) fail("cannot inspect BWE");
    int decoded = argc == 7 ? ltx_mlx_bwe_extend_f32_debug(
        bwe, output, output_elements, input, input_elements,
        batch, samples, argv[6], error, sizeof(error)) :
        ltx_mlx_bwe_extend_f32(
        bwe, output, output_elements, input, input_elements,
        batch, samples, error, sizeof(error));
    if (!decoded) fail(error);
    write_file(argv[5], output, output_elements * sizeof(float),
               "cannot write BWE waveform output");
    printf("{\"batch\":%u,\"input_samples\":%u,\"output_samples\":%u,"
           "\"channels\":2,\"sample_rate\":48000,\"weight_tensors\":%u,"
           "\"source_weight_bytes\":%llu,\"resident_weight_bytes\":%llu}\n",
           batch, samples, samples * 3u, info.weight_tensors,
           (unsigned long long)info.source_weight_bytes,
           (unsigned long long)info.resident_weight_bytes);
    ltx_mlx_bwe_free(bwe);
    free(output);
    free(input);
    return 0;
}
