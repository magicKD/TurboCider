#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_bwe ltx_mlx_bwe;

typedef struct {
    uint32_t weight_tensors;
    uint64_t source_weight_bytes;
    uint64_t resident_weight_bytes;
} ltx_mlx_bwe_info;

ltx_mlx_bwe *ltx_mlx_bwe_create(
    const char *checkpoint_path, char *error, size_t error_size);
void ltx_mlx_bwe_free(ltx_mlx_bwe *bwe);
int ltx_mlx_bwe_get_info(const ltx_mlx_bwe *bwe, ltx_mlx_bwe_info *info);

/* Extend FP32 BTC stereo PCM from 16 kHz to 48 kHz. */
int ltx_mlx_bwe_extend_f32(
    ltx_mlx_bwe *bwe,
    float *output, size_t output_elements,
    const float *input, size_t input_elements,
    uint32_t batch, uint32_t samples,
    char *error, size_t error_size);

int ltx_mlx_bwe_extend_f32_debug(
    ltx_mlx_bwe *bwe,
    float *output, size_t output_elements,
    const float *input, size_t input_elements,
    uint32_t batch, uint32_t samples, const char *dump_directory,
    char *error, size_t error_size);

void ltx_mlx_bwe_clear_cache(void);

#ifdef __cplusplus
}
#endif
