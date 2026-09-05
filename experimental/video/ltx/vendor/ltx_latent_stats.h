#ifndef LTX_LATENT_STATS_H
#define LTX_LATENT_STATS_H

#include "ltx_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_latent_stats ltx_latent_stats;

ltx_latent_stats *ltx_latent_stats_load(ltx_gpu *gpu,
                                        const char *video_vae_path,
                                        char *error, size_t error_size);
void ltx_latent_stats_free(ltx_latent_stats *stats);
uint32_t ltx_latent_stats_channels(const ltx_latent_stats *stats);

int ltx_latent_denormalize_tokens_bf16(
        const ltx_latent_stats *stats,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, char *error, size_t error_size);
int ltx_latent_normalize_tokens_bf16(
        const ltx_latent_stats *stats,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, char *error, size_t error_size);

#endif
