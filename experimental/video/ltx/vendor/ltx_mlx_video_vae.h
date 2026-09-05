#ifndef LTX_MLX_VIDEO_VAE_H
#define LTX_MLX_VIDEO_VAE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_video_vae ltx_mlx_video_vae;

typedef struct {
    uint32_t weight_tensors;
    uint64_t weight_bytes;
} ltx_mlx_video_vae_info;

ltx_mlx_video_vae *ltx_mlx_video_vae_create(
    const char *checkpoint_path, char *error, size_t error_size);
/* Create an encoder-only instance. Keeping encoder and decoder weights in
 * separate short-lived processes avoids holding both halves of the large VAE
 * checkpoint in unified memory during generation. */
ltx_mlx_video_vae *ltx_mlx_video_vae_create_encoder(
    const char *checkpoint_path, char *error, size_t error_size);
void ltx_mlx_video_vae_free(ltx_mlx_video_vae *vae);
int ltx_mlx_video_vae_get_info(
    const ltx_mlx_video_vae *vae, ltx_mlx_video_vae_info *info);

/*
 * Decode normalized BF16 tokens in BFHWC layout to BF16 pixels in BCFHW
 * layout. The output shape is [batch, 3, 8 * frames - 7,
 * 32 * height, 32 * width].
 */
int ltx_mlx_video_vae_decode_tokens_bf16(
    ltx_mlx_video_vae *vae,
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

/* Encode BF16 pixels in BCFHW layout and [-1, 1] range to normalized BF16
 * latent tokens in BFHWC layout. Input height and width must be divisible by
 * 32. The output shape is [batch, ceil(frames / 8), height / 32,
 * width / 32, 128]. */
int ltx_mlx_video_vae_encode_pixels_bf16(
    ltx_mlx_video_vae *vae,
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

/* Release temporary MLX allocations while preserving loaded VAE weights and
 * compiled operation state in a long-lived worker process. */
void ltx_mlx_video_vae_clear_cache(void);

#ifdef __cplusplus
}
#endif

#endif
