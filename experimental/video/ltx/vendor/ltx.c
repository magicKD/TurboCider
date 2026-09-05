#include "ltx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t ltx_f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static int ltx_fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static const float ltx_stage1_sigmas[LTX_DISTILLED_STAGE1_SIGMA_COUNT] = {
    1.0f,
    0.99375f,
    0.9875f,
    0.98125f,
    0.975f,
    0.909375f,
    0.725f,
    0.421875f,
    0.0f,
};

static const float ltx_stage2_sigmas[LTX_DISTILLED_STAGE2_SIGMA_COUNT] = {
    0.909375f,
    0.725f,
    0.421875f,
    0.0f,
};

const float *ltx_distilled_stage1_sigmas(size_t *count) {
    if (count) *count = LTX_DISTILLED_STAGE1_SIGMA_COUNT;
    return ltx_stage1_sigmas;
}

const float *ltx_distilled_stage2_sigmas(size_t *count) {
    if (count) *count = LTX_DISTILLED_STAGE2_SIGMA_COUNT;
    return ltx_stage2_sigmas;
}

static int ltx_valid_diffusion_step(float sigma, float sigma_next) {
    return isfinite(sigma) && isfinite(sigma_next) && sigma > 0.0f &&
        sigma <= 1.0f && sigma_next >= 0.0f && sigma_next <= sigma;
}

int ltx_velocity_to_denoised_f32(float *output, const float *sample,
                                 const float *velocity, uint64_t elements,
                                 float sigma,
                                 char *error, size_t error_size) {
    if (!output || !sample || !velocity || !elements ||
        !isfinite(sigma) || sigma < 0.0f || sigma > 1.0f)
        return ltx_fail(error, error_size,
                        "invalid velocity-to-denoised arguments");
    for (uint64_t index = 0; index < elements; index++)
        output[index] = fmaf(-sigma, velocity[index], sample[index]);
    return 1;
}

int ltx_euler_step_f32(float *output, const float *sample,
                       const float *denoised, uint64_t elements,
                       float sigma, float sigma_next,
                       char *error, size_t error_size) {
    if (!output || !sample || !denoised || !elements ||
        !ltx_valid_diffusion_step(sigma, sigma_next))
        return ltx_fail(error, error_size,
                        "invalid Euler step arguments");
    float delta = sigma_next - sigma;
    for (uint64_t index = 0; index < elements; index++) {
        float velocity = (sample[index] - denoised[index]) / sigma;
        output[index] = fmaf(velocity, delta, sample[index]);
    }
    return 1;
}

int ltx_euler_ancestral_step_f32(
                       float *output, const float *sample,
                       const float *denoised, const float *noise,
                       uint64_t elements, float sigma, float sigma_next,
                       float eta, float s_noise,
                       char *error, size_t error_size) {
    if (!output || !sample || !denoised || !elements ||
        !ltx_valid_diffusion_step(sigma, sigma_next) ||
        !isfinite(eta) || eta < 0.0f || eta > 1.0f ||
        !isfinite(s_noise))
        return ltx_fail(error, error_size,
                        "invalid ancestral Euler step arguments");
    if (sigma_next == 0.0f) {
        if (output != denoised)
            memcpy(output, denoised, (size_t)elements * sizeof(*output));
        return 1;
    }
    if (eta > 0.0f && !noise)
        return ltx_fail(error, error_size,
                        "ancestral Euler noise is required when eta > 0");

    double sigma_d = sigma;
    double sigma_next_d = sigma_next;
    double downstep_ratio =
        1.0 + (sigma_next_d / sigma_d - 1.0) * (double)eta;
    double sigma_down = sigma_next_d * downstep_ratio;
    double ratio = sigma_down / sigma_d;
    double signal_scale = 1.0;
    double noise_scale = 0.0;
    if (eta > 0.0f) {
        double alpha_next = 1.0 - sigma_next_d;
        double alpha_down = 1.0 - sigma_down;
        if (!(alpha_down > 0.0))
            return ltx_fail(error, error_size,
                            "ancestral Euler alpha_down is not positive");
        double variance = sigma_next_d * sigma_next_d -
            sigma_down * sigma_down * alpha_next * alpha_next /
                (alpha_down * alpha_down);
        signal_scale = alpha_next / alpha_down;
        noise_scale = (double)s_noise * sqrt(fmax(variance, 0.0));
    }

    float ratio_f = (float)ratio;
    float inverse_ratio_f = (float)(1.0 - ratio);
    float signal_scale_f = (float)signal_scale;
    float noise_scale_f = (float)noise_scale;
    for (uint64_t index = 0; index < elements; index++) {
        float deterministic = fmaf(ratio_f, sample[index],
                                   inverse_ratio_f * denoised[index]);
        output[index] = eta > 0.0f ?
            fmaf(noise[index], noise_scale_f,
                 signal_scale_f * deterministic) : deterministic;
    }
    return 1;
}

int ltx_renoise_f32(float *output, const float *clean,
                    const float *noise, uint64_t elements, float sigma,
                    char *error, size_t error_size) {
    if (!output || !clean || !noise || !elements || !isfinite(sigma) ||
        sigma < 0.0f || sigma > 1.0f)
        return ltx_fail(error, error_size,
                        "invalid latent renoise arguments");
    float clean_scale = 1.0f - sigma;
    for (uint64_t index = 0; index < elements; index++)
        output[index] = fmaf(noise[index], sigma,
                             clean[index] * clean_scale);
    return 1;
}

uint32_t ltx_snap_dimension(uint32_t value, int two_stage) {
    uint32_t modulus = LTX_VIDEO_SPATIAL_COMPRESSION;
    if (two_stage) modulus *= 2u;
    if (value < modulus) return modulus;
    return (value / modulus) * modulus;
}

uint32_t ltx_snap_frames(uint32_t frames) {
    if (frames <= 1u) return 1u;
    return ((frames - 1u) / LTX_VIDEO_TEMPORAL_COMPRESSION) *
           LTX_VIDEO_TEMPORAL_COMPRESSION + 1u;
}

int ltx_workload_init(ltx_workload *workload, uint32_t width,
                      uint32_t height, uint32_t frames, uint32_t fps,
                      char *error, size_t error_size) {
    if (!workload) {
        if (error && error_size) snprintf(error, error_size, "missing workload");
        return 0;
    }
    memset(workload, 0, sizeof(*workload));
    if (!width || !height || !frames || !fps) {
        if (error && error_size)
            snprintf(error, error_size, "width, height, frames, and fps must be nonzero");
        return 0;
    }

    uint32_t snapped_width = ltx_snap_dimension(width, 1);
    uint32_t snapped_height = ltx_snap_dimension(height, 1);
    uint32_t snapped_frames = ltx_snap_frames(frames);
    uint32_t latent_frames = (snapped_frames +
        LTX_VIDEO_TEMPORAL_COMPRESSION - 1u) /
        LTX_VIDEO_TEMPORAL_COMPRESSION;
    uint32_t stage1_width = snapped_width / 2u;
    uint32_t stage1_height = snapped_height / 2u;
    uint32_t stage1_lh = stage1_height / LTX_VIDEO_SPATIAL_COMPRESSION;
    uint32_t stage1_lw = stage1_width / LTX_VIDEO_SPATIAL_COMPRESSION;
    uint32_t stage2_lh = snapped_height / LTX_VIDEO_SPATIAL_COMPRESSION;
    uint32_t stage2_lw = snapped_width / LTX_VIDEO_SPATIAL_COMPRESSION;

    uint64_t stage1_tokens = (uint64_t)latent_frames * stage1_lh * stage1_lw;
    uint64_t stage2_tokens = (uint64_t)latent_frames * stage2_lh * stage2_lw;
    double duration = (double)snapped_frames / (double)fps;
    long audio_tokens = lround(duration * LTX_AUDIO_LATENTS_PER_SECOND);
    if (audio_tokens < 1 || (unsigned long)audio_tokens > UINT32_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "audio token count is out of range");
        return 0;
    }

    workload->output_width = snapped_width;
    workload->output_height = snapped_height;
    workload->frames = snapped_frames;
    workload->fps = fps;
    workload->stage1_width = stage1_width;
    workload->stage1_height = stage1_height;
    workload->latent_frames = latent_frames;
    workload->stage1_latent_height = stage1_lh;
    workload->stage1_latent_width = stage1_lw;
    workload->stage2_latent_height = stage2_lh;
    workload->stage2_latent_width = stage2_lw;
    workload->stage1_video_tokens = stage1_tokens;
    workload->stage2_video_tokens = stage2_tokens;
    workload->audio_tokens = (uint32_t)audio_tokens;
    return 1;
}

int ltx_compute_video_positions(float *positions, uint64_t position_values,
                                uint32_t latent_frames,
                                uint32_t latent_height,
                                uint32_t latent_width, float frame_rate,
                                char *error, size_t error_size) {
    uint64_t tokens = (uint64_t)latent_frames * latent_height * latent_width;
    if (!positions || !latent_frames || !latent_height || !latent_width ||
        !(frame_rate > 0.0f) || !isfinite(frame_rate) ||
        tokens > UINT64_MAX / 3u || position_values < tokens * 3u)
        return ltx_fail(error, error_size,
                        "invalid video position buffer/geometry");
    uint64_t index = 0;
    for (uint32_t frame = 0; frame < latent_frames; frame++) {
        float frame_start = fmaxf(
            (float)frame * (float)LTX_VIDEO_TEMPORAL_COMPRESSION + 1.0f -
                (float)LTX_VIDEO_TEMPORAL_COMPRESSION,
            0.0f);
        float frame_end = fmaxf(
            (float)(frame + 1u) *
                (float)LTX_VIDEO_TEMPORAL_COMPRESSION + 1.0f -
                (float)LTX_VIDEO_TEMPORAL_COMPRESSION,
            0.0f);
        float time = (frame_start + frame_end) * 0.5f / frame_rate;
        for (uint32_t row = 0; row < latent_height; row++) {
            float y = (float)row * (float)LTX_VIDEO_SPATIAL_COMPRESSION +
                (float)LTX_VIDEO_SPATIAL_COMPRESSION * 0.5f;
            for (uint32_t column = 0; column < latent_width; column++) {
                float x =
                    (float)column * (float)LTX_VIDEO_SPATIAL_COMPRESSION +
                    (float)LTX_VIDEO_SPATIAL_COMPRESSION * 0.5f;
                positions[index++] = time;
                positions[index++] = y;
                positions[index++] = x;
            }
        }
    }
    return 1;
}

int ltx_compute_audio_positions(float *positions, uint64_t position_values,
                                uint32_t tokens,
                                char *error, size_t error_size) {
    const float seconds_per_mel_frame = 160.0f / 16000.0f;
    const float downsample = 4.0f;
    if (!positions || !tokens || position_values < tokens)
        return ltx_fail(error, error_size,
                        "invalid audio position buffer/geometry");
    for (uint32_t token = 0; token < tokens; token++) {
        float start_frame = fmaxf((float)token * downsample + 1.0f -
                                  downsample, 0.0f);
        float end_frame = fmaxf((float)(token + 1u) * downsample + 1.0f -
                                downsample, 0.0f);
        positions[token] =
            (start_frame + end_frame) * 0.5f * seconds_per_mel_frame;
    }
    return 1;
}

int ltx_compute_timestep_embedding_bf16(
                                uint16_t *embedding,
                                uint64_t embedding_values,
                                const float *timesteps,
                                uint32_t rows, uint32_t embedding_dim,
                                int flip_sin_to_cos,
                                float downscale_frequency_shift,
                                float scale, float max_period,
                                char *error, size_t error_size) {
    uint32_t half = embedding_dim / 2u;
    uint64_t required = (uint64_t)rows * embedding_dim;
    float denominator = (float)half - downscale_frequency_shift;
    if (!embedding || !timesteps || !rows || !embedding_dim || !half ||
        embedding_values < required || !(denominator > 0.0f) ||
        !(scale > 0.0f) || !isfinite(scale) ||
        !(max_period > 0.0f) || !isfinite(max_period) ||
        !isfinite(downscale_frequency_shift))
        return ltx_fail(error, error_size,
                        "invalid timestep embedding arguments");
    double log_period = log((double)max_period);
    for (uint32_t row = 0; row < rows; row++) {
        if (!isfinite(timesteps[row]))
            return ltx_fail(error, error_size,
                            "timestep values must be finite");
        for (uint32_t index = 0; index < half; index++) {
            double exponent = -log_period * (double)index /
                (double)denominator;
            float angle = timesteps[row] * (float)exp(exponent) * scale;
            float sine = sinf(angle);
            float cosine = cosf(angle);
            uint64_t base = (uint64_t)row * embedding_dim;
            if (flip_sin_to_cos) {
                embedding[base + index] = ltx_f32_to_bf16(cosine);
                embedding[base + half + index] = ltx_f32_to_bf16(sine);
            } else {
                embedding[base + index] = ltx_f32_to_bf16(sine);
                embedding[base + half + index] = ltx_f32_to_bf16(cosine);
            }
        }
        if (embedding_dim % 2u)
            embedding[(uint64_t)(row + 1u) * embedding_dim - 1u] = 0u;
    }
    return 1;
}

int ltx_compute_rope_split_bf16(
                                uint16_t *cosine, uint16_t *sine,
                                uint64_t frequency_values,
                                const float *positions,
                                uint32_t rows, uint32_t position_axes,
                                uint32_t heads, uint32_t head_dim,
                                double theta, const float *max_positions,
                                int double_precision_grid,
                                char *error, size_t error_size) {
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    uint64_t half_inner = inner_dim / 2u;
    uint64_t required = (uint64_t)heads * rows * (head_dim / 2u);
    if (!cosine || !sine || !positions || !max_positions ||
        !rows || !position_axes || !heads || !head_dim || head_dim % 2u ||
        inner_dim % 2u || !(theta > 0.0) || !isfinite(theta) ||
        frequency_values < required)
        return ltx_fail(error, error_size,
                        "invalid split-RoPE buffer/geometry");
    uint64_t divisor = 2u * (uint64_t)position_axes;
    uint64_t frequency_count = inner_dim / divisor;
    uint64_t used = frequency_count * position_axes;
    if (!frequency_count || used > half_inner)
        return ltx_fail(error, error_size,
                        "split-RoPE dimensions cannot be partitioned");
    for (uint32_t axis = 0; axis < position_axes; axis++)
        if (!(max_positions[axis] > 0.0f) ||
            !isfinite(max_positions[axis]))
            return ltx_fail(error, error_size,
                            "split-RoPE max positions must be positive");

    float *frequency_grid = malloc(
        (size_t)frequency_count * sizeof(*frequency_grid));
    if (!frequency_grid)
        return ltx_fail(error, error_size,
                        "out of memory creating split-RoPE grid");
    const double pi_over_two = 1.57079632679489661923;
    for (uint64_t index = 0; index < frequency_count; index++) {
        double fraction = frequency_count > 1u ?
            (double)index / (double)(frequency_count - 1u) : 0.0;
        if (double_precision_grid)
            frequency_grid[index] =
                (float)(pow(theta, fraction) * pi_over_two);
        else
            frequency_grid[index] = powf(
                (float)theta, (float)fraction) * (float)pi_over_two;
    }

    uint64_t padding = half_inner - used;
    uint32_t half_head = head_dim / 2u;
    for (uint32_t head = 0; head < heads; head++)
        for (uint32_t row = 0; row < rows; row++)
            for (uint32_t pair = 0; pair < half_head; pair++) {
                uint64_t flattened = (uint64_t)head * half_head + pair;
                float angle = 0.0f;
                if (flattened >= padding) {
                    uint64_t unpadded = flattened - padding;
                    uint64_t frequency = unpadded / position_axes;
                    uint32_t axis = (uint32_t)(unpadded % position_axes);
                    float fraction = positions[(uint64_t)row * position_axes +
                                               axis] / max_positions[axis];
                    angle = frequency_grid[frequency] *
                        (fraction * 2.0f - 1.0f);
                }
                uint64_t output_index =
                    ((uint64_t)head * rows + row) * half_head + pair;
                cosine[output_index] = ltx_f32_to_bf16(cosf(angle));
                sine[output_index] = ltx_f32_to_bf16(sinf(angle));
            }
    free(frequency_grid);
    return 1;
}
