#ifndef H3_TAEH3_H
#define H3_TAEH3_H

#include "h3_video_vae.h"
#include "h3_gpu.h"

#include <stddef.h>

typedef struct h3_taeh3_decoder h3_taeh3_decoder;

h3_taeh3_decoder *h3_taeh3_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        char *error, size_t error_size);

/* Input is normalized channel-major F32 [24,T,H,W], matching the DiT latent.
 * Output is frame-major interleaved F32 RGB in [0,1]. */
int h3_taeh3_decoder_decode(h3_taeh3_decoder *decoder,
                            const float *normalized_latent,
                            int latent_time, h3_video_frames *output,
                            char *error, size_t error_size);

int h3_taeh3_decoder_get_gpu_stats(const h3_taeh3_decoder *decoder,
                                   h3_gpu_stats *stats);

void h3_taeh3_decoder_free(h3_taeh3_decoder *decoder);

#endif
