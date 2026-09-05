#ifndef LTX_H
#define LTX_H

#include <stddef.h>
#include <stdint.h>

#define LTX_VERSION "0.1.0-dev"
#define LTX_VIDEO_CHANNELS 128u
#define LTX_AUDIO_CHANNELS 128u
#define LTX_VIDEO_TEMPORAL_COMPRESSION 8u
#define LTX_VIDEO_SPATIAL_COMPRESSION 32u
#define LTX_AUDIO_LATENTS_PER_SECOND 25.0
#define LTX_DISTILLED_STAGE1_SIGMA_COUNT 9u
#define LTX_DISTILLED_STAGE2_SIGMA_COUNT 4u

typedef struct {
    uint32_t output_width;
    uint32_t output_height;
    uint32_t frames;
    uint32_t fps;
    uint32_t stage1_width;
    uint32_t stage1_height;
    uint32_t latent_frames;
    uint32_t stage1_latent_height;
    uint32_t stage1_latent_width;
    uint32_t stage2_latent_height;
    uint32_t stage2_latent_width;
    uint64_t stage1_video_tokens;
    uint64_t stage2_video_tokens;
    uint32_t audio_tokens;
} ltx_workload;

uint32_t ltx_snap_dimension(uint32_t value, int two_stage);
uint32_t ltx_snap_frames(uint32_t frames);
int ltx_workload_init(ltx_workload *workload, uint32_t width,
                      uint32_t height, uint32_t frames, uint32_t fps,
                      char *error, size_t error_size);
int ltx_compute_video_positions(float *positions, uint64_t position_values,
                                uint32_t latent_frames,
                                uint32_t latent_height,
                                uint32_t latent_width, float frame_rate,
                                char *error, size_t error_size);
int ltx_compute_audio_positions(float *positions, uint64_t position_values,
                                uint32_t tokens,
                                char *error, size_t error_size);
int ltx_compute_timestep_embedding_bf16(
                                uint16_t *embedding,
                                uint64_t embedding_values,
                                const float *timesteps,
                                uint32_t rows, uint32_t embedding_dim,
                                int flip_sin_to_cos,
                                float downscale_frequency_shift,
                                float scale, float max_period,
                                char *error, size_t error_size);
int ltx_compute_rope_split_bf16(
                                uint16_t *cosine, uint16_t *sine,
                                uint64_t frequency_values,
                                const float *positions,
                                uint32_t rows, uint32_t position_axes,
                                uint32_t heads, uint32_t head_dim,
                                double theta, const float *max_positions,
                                int double_precision_grid,
                                char *error, size_t error_size);
const float *ltx_distilled_stage1_sigmas(size_t *count);
const float *ltx_distilled_stage2_sigmas(size_t *count);
int ltx_velocity_to_denoised_f32(float *output, const float *sample,
                                 const float *velocity, uint64_t elements,
                                 float sigma,
                                 char *error, size_t error_size);
int ltx_euler_step_f32(float *output, const float *sample,
                       const float *denoised, uint64_t elements,
                       float sigma, float sigma_next,
                       char *error, size_t error_size);
int ltx_euler_ancestral_step_f32(
                       float *output, const float *sample,
                       const float *denoised, const float *noise,
                       uint64_t elements, float sigma, float sigma_next,
                       float eta, float s_noise,
                       char *error, size_t error_size);
int ltx_renoise_f32(float *output, const float *clean,
                    const float *noise, uint64_t elements, float sigma,
                    char *error, size_t error_size);

#endif
