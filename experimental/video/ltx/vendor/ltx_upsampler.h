#ifndef LTX_UPSAMPLER_H
#define LTX_UPSAMPLER_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_upsampler ltx_upsampler;

typedef struct {
    uint32_t input_channels;
    uint32_t hidden_channels;
    uint32_t residual_blocks_per_stage;
    uint64_t weight_bytes;
    uint32_t weight_tensors;
} ltx_upsampler_info;

ltx_upsampler *ltx_upsampler_create(ltx_gpu *gpu,
                                    const char *checkpoint_path,
                                    char *error, size_t error_size);
void ltx_upsampler_free(ltx_upsampler *upsampler);
int ltx_upsampler_get_info(const ltx_upsampler *upsampler,
                           ltx_upsampler_info *info);

/*
 * Spatial x2 latent upsampling in BCFHW layout.
 * Input:  [batch, 128, frames, height, width] BF16.
 * Output: [batch, 128, frames, 2*height, 2*width] BF16.
 */
int ltx_upsampler_run_bf16(ltx_upsampler *upsampler,
                           ltx_gpu_buffer *output,
                           const ltx_gpu_buffer *input,
                           uint32_t batch, uint32_t frames,
                           uint32_t height, uint32_t width,
                           char *error, size_t error_size);

/*
 * Token-major variant used by the Transformer pipeline.
 * Input/output memory is flattened BFHWC, with channels contiguous.
 */
int ltx_upsampler_run_tokens_bf16(ltx_upsampler *upsampler,
                                  ltx_gpu_buffer *output,
                                  const ltx_gpu_buffer *input,
                                  uint32_t batch, uint32_t frames,
                                  uint32_t height, uint32_t width,
                                  char *error, size_t error_size);

#endif
