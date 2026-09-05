#ifndef LTX_MLX_UPSAMPLER_H
#define LTX_MLX_UPSAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_upsampler ltx_mlx_upsampler;

typedef struct {
    uint32_t input_channels;
    uint32_t hidden_channels;
    uint32_t weight_tensors;
    uint64_t weight_bytes;
} ltx_mlx_upsampler_info;

ltx_mlx_upsampler *ltx_mlx_upsampler_create(
    const char *checkpoint_path, char *error, size_t error_size);
void ltx_mlx_upsampler_free(ltx_mlx_upsampler *upsampler);
int ltx_mlx_upsampler_get_info(
    const ltx_mlx_upsampler *upsampler, ltx_mlx_upsampler_info *info);

/* Spatial x2 upsampling with BFHWC/token-major BF16 input and output. */
int ltx_mlx_upsampler_run_tokens_bf16(
    ltx_mlx_upsampler *upsampler,
    uint16_t *output,
    size_t output_elements,
    const uint16_t *input,
    size_t input_elements,
    uint32_t batch,
    uint32_t frames,
    uint32_t height,
    uint32_t width,
    char *error,
    size_t error_size);

/* Shared MLX allocator controls used by the integrated media pipeline. */
uint64_t ltx_mlx_active_memory_bytes(void);
uint64_t ltx_mlx_cache_memory_bytes(void);
uint64_t ltx_mlx_peak_memory_bytes(void);
void ltx_mlx_clear_cache(void);

#ifdef __cplusplus
}
#endif

#endif
