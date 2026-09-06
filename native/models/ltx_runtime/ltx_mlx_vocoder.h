#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_vocoder ltx_mlx_vocoder;

typedef struct {
    uint32_t weight_tensors;
    uint64_t source_weight_bytes;
    uint64_t resident_weight_bytes;
} ltx_mlx_vocoder_info;

ltx_mlx_vocoder *ltx_mlx_vocoder_create_base(
    const char *checkpoint_path, char *error, size_t error_size);
void ltx_mlx_vocoder_free(ltx_mlx_vocoder *vocoder);
int ltx_mlx_vocoder_get_info(
    const ltx_mlx_vocoder *vocoder, ltx_mlx_vocoder_info *info);

/* Decode BF16 BCTF stereo mel into FP32 BTC stereo PCM at 16 kHz. */
int ltx_mlx_vocoder_decode_base_bf16(
    ltx_mlx_vocoder *vocoder,
    float *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    uint32_t batch, uint32_t mel_time,
    char *error, size_t error_size);

int ltx_mlx_vocoder_decode_base_bf16_debug(
    ltx_mlx_vocoder *vocoder,
    float *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    uint32_t batch, uint32_t mel_time, const char *dump_directory,
    char *error, size_t error_size);


void ltx_mlx_vocoder_clear_cache(void);

#ifdef __cplusplus
}
#endif
