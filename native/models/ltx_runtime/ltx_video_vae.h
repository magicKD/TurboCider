#ifndef LTX_VIDEO_VAE_H
#define LTX_VIDEO_VAE_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_video_vae ltx_video_vae;

typedef struct {
    uint32_t latent_channels;
    uint32_t output_channels;
    uint32_t temporal_scale;
    uint32_t temporal_offset;
    uint32_t spatial_scale;
    uint32_t graph_stages;
    uint32_t weight_tensors;
    uint64_t weight_bytes;
} ltx_video_vae_info;

ltx_video_vae *ltx_video_vae_create(ltx_gpu *gpu,
                                    const char *checkpoint_path,
                                    char *error, size_t error_size);
void ltx_video_vae_free(ltx_video_vae *vae);
int ltx_video_vae_get_info(const ltx_video_vae *vae,
                           ltx_video_vae_info *info);

/*
 * Returns the decoded dimensions for an LTX-2.5 conv VAE latent.
 * Frames expand as 8 * frames - 7; height and width expand by 32.
 */
int ltx_video_vae_output_shape(uint32_t latent_frames,
                               uint32_t latent_height,
                               uint32_t latent_width,
                               uint32_t *frames,
                               uint32_t *height,
                               uint32_t *width);

/*
 * Decode a normalized BF16 latent in BCFHW layout to BF16 RGB pixels in
 * BCFHW layout. Pixel values are the raw decoder output, normally [-1, 1].
 */
int ltx_video_vae_decode_bf16(ltx_video_vae *vae,
                              ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              uint32_t batch,
                              uint32_t frames,
                              uint32_t height,
                              uint32_t width,
                              char *error, size_t error_size);

/*
 * Transformer integration variant. Input is token-major BFHWC while output
 * remains BCFHW so it can be streamed or encoded without another GPU copy.
 */
int ltx_video_vae_decode_tokens_bf16(ltx_video_vae *vae,
                                     ltx_gpu_buffer *output,
                                     const ltx_gpu_buffer *input,
                                     uint32_t batch,
                                     uint32_t frames,
                                     uint32_t height,
                                     uint32_t width,
                                     char *error, size_t error_size);

#endif
