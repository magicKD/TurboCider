#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_audio_vae ltx_mlx_audio_vae;

typedef struct {
    uint32_t weight_tensors;
    uint64_t weight_bytes;
} ltx_mlx_audio_vae_info;

ltx_mlx_audio_vae *ltx_mlx_audio_vae_create(
    const char *checkpoint_path, char *error, size_t error_size);
void ltx_mlx_audio_vae_free(ltx_mlx_audio_vae *vae);
int ltx_mlx_audio_vae_get_info(
    const ltx_mlx_audio_vae *vae, ltx_mlx_audio_vae_info *info);

/* Decode BLC-token-major BF16 audio latents into contiguous BCTF BF16 mel.
 * The native Audio VAE expects 128 latent channels and returns stereo 64-bin
 * mel with time length 4*tokens-3. */
int ltx_mlx_audio_vae_decode_bf16(
    ltx_mlx_audio_vae *vae,
    uint16_t *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    uint32_t batch, uint32_t tokens,
    char *error, size_t error_size);

/* Development-only parity hook used by the standalone tool. */
int ltx_mlx_audio_vae_decode_bf16_debug(
    ltx_mlx_audio_vae *vae,
    uint16_t *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    uint32_t batch, uint32_t tokens, const char *dump_directory,
    char *error, size_t error_size);

void ltx_mlx_audio_vae_clear_cache(void);

#ifdef __cplusplus
}
#endif
