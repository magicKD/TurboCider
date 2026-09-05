#include "ltx.h"
#include "ltx_rng.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "test_ltx: check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(void) {
    CHECK(ltx_snap_dimension(480u, 1) == 448u);
    CHECK(ltx_snap_dimension(704u, 1) == 704u);
    CHECK(ltx_snap_dimension(480u, 0) == 480u);
    CHECK(ltx_snap_frames(100u) == 97u);
    CHECK(ltx_snap_frames(97u) == 97u);

    ltx_workload workload;
    char error[256];
    CHECK(ltx_workload_init(&workload, 704u, 480u, 97u, 24u,
                            error, sizeof(error)));
    CHECK(workload.output_width == 704u);
    CHECK(workload.output_height == 448u);
    CHECK(workload.latent_frames == 13u);
    CHECK(workload.stage1_latent_height == 7u);
    CHECK(workload.stage1_latent_width == 11u);
    CHECK(workload.stage2_latent_height == 14u);
    CHECK(workload.stage2_latent_width == 22u);
    CHECK(workload.stage1_video_tokens == 1001u);
    CHECK(workload.stage2_video_tokens == 4004u);
    CHECK(workload.audio_tokens == 101u);

    float video_positions[2u * 2u * 2u * 3u];
    CHECK(ltx_compute_video_positions(
        video_positions, sizeof(video_positions) / sizeof(video_positions[0]),
        2u, 2u, 2u, 24.0f, error, sizeof(error)));
    CHECK(fabsf(video_positions[0] - (0.5f / 24.0f)) < 1e-7f);
    CHECK(video_positions[1] == 16.0f);
    CHECK(video_positions[2] == 16.0f);
    CHECK(video_positions[5] == 48.0f);
    CHECK(fabsf(video_positions[12] - (5.0f / 24.0f)) < 1e-7f);

    float audio_positions[3];
    CHECK(ltx_compute_audio_positions(
        audio_positions, 3u, 3u, error, sizeof(error)));
    CHECK(fabsf(audio_positions[0] - 0.005f) < 1e-7f);
    CHECK(fabsf(audio_positions[1] - 0.03f) < 1e-7f);
    CHECK(fabsf(audio_positions[2] - 0.07f) < 1e-7f);

    uint16_t timestep_embedding[2u * 8u];
    float timesteps[2] = {0.0f, 1.0f};
    CHECK(ltx_compute_timestep_embedding_bf16(
        timestep_embedding,
        sizeof(timestep_embedding) / sizeof(timestep_embedding[0]),
        timesteps, 2u, 8u, 1, 0.0f, 1.0f, 10000.0f,
        error, sizeof(error)));
    for (unsigned index = 0; index < 4u; index++) {
        CHECK(bf16_to_f32(timestep_embedding[index]) == 1.0f);
        CHECK(bf16_to_f32(timestep_embedding[4u + index]) == 0.0f);
    }
    for (unsigned index = 0; index < 4u; index++) {
        float frequency = expf(-logf(10000.0f) * (float)index / 4.0f);
        float expected_cosine = cosf(frequency);
        float expected_sine = sinf(frequency);
        CHECK(fabsf(bf16_to_f32(timestep_embedding[8u + index]) -
                    expected_cosine) < 0.004f);
        CHECK(fabsf(bf16_to_f32(timestep_embedding[12u + index]) -
                    expected_sine) < 0.004f);
    }

    enum { rope_rows = 2, rope_axes = 3, rope_heads = 2, rope_head_dim = 8 };
    float rope_positions[rope_rows * rope_axes] = {
        0.5f, 16.0f, 16.0f,
        5.0f, 48.0f, 80.0f,
    };
    float rope_max[rope_axes] = {20.0f, 2048.0f, 2048.0f};
    uint16_t rope_cos[rope_heads * rope_rows * (rope_head_dim / 2)];
    uint16_t rope_sin[rope_heads * rope_rows * (rope_head_dim / 2)];
    CHECK(ltx_compute_rope_split_bf16(
        rope_cos, rope_sin,
        sizeof(rope_cos) / sizeof(rope_cos[0]), rope_positions,
        rope_rows, rope_axes, rope_heads, rope_head_dim,
        10000.0, rope_max, 1, error, sizeof(error)));
    CHECK(bf16_to_f32(rope_cos[0]) == 1.0f);
    CHECK(bf16_to_f32(rope_sin[0]) == 0.0f);
    CHECK(bf16_to_f32(rope_cos[1]) == 1.0f);
    CHECK(bf16_to_f32(rope_sin[1]) == 0.0f);
    CHECK(isfinite(bf16_to_f32(rope_cos[2])));
    CHECK(isfinite(bf16_to_f32(rope_sin[2])));

    size_t stage1_count = 0;
    size_t stage2_count = 0;
    const float *stage1 = ltx_distilled_stage1_sigmas(&stage1_count);
    const float *stage2 = ltx_distilled_stage2_sigmas(&stage2_count);
    CHECK(stage1_count == LTX_DISTILLED_STAGE1_SIGMA_COUNT);
    CHECK(stage2_count == LTX_DISTILLED_STAGE2_SIGMA_COUNT);
    CHECK(stage1[0] == 1.0f);
    CHECK(stage1[4] == 0.975f);
    CHECK(stage1[5] == 0.909375f);
    CHECK(stage1[8] == 0.0f);
    for (size_t index = 0; index < stage2_count; index++)
        CHECK(stage2[index] == stage1[stage1_count - stage2_count + index]);

    float sample[4] = {0.8f, -1.2f, 0.3f, 2.0f};
    float velocity[4] = {0.25f, -0.5f, 1.0f, -1.5f};
    float denoised[4];
    CHECK(ltx_velocity_to_denoised_f32(
        denoised, sample, velocity, 4u, 0.6f, error, sizeof(error)));
    for (size_t index = 0; index < 4u; index++)
        CHECK(fabsf(denoised[index] -
                    (sample[index] - 0.6f * velocity[index])) < 1e-7f);

    float euler[4];
    CHECK(ltx_euler_step_f32(euler, sample, denoised, 4u,
                             0.6f, 0.2f, error, sizeof(error)));
    for (size_t index = 0; index < 4u; index++) {
        float recovered_velocity =
            (sample[index] - denoised[index]) / 0.6f;
        float expected = fmaf(recovered_velocity, 0.2f - 0.6f,
                              sample[index]);
        CHECK(euler[index] == expected);
    }

    float x0[4] = {0.1f, -0.4f, 0.05f, 1.1f};
    float noise[4] = {0.5f, -0.5f, 1.0f, -1.0f};
    float ancestral[4];
    const float sigma = 0.9f;
    const float sigma_next = 0.6f;
    CHECK(ltx_euler_ancestral_step_f32(
        ancestral, sample, x0, noise, 4u, sigma, sigma_next,
        1.0f, 1.0f, error, sizeof(error)));
    double downstep_ratio =
        1.0 + ((double)sigma_next / sigma - 1.0);
    double sigma_down = sigma_next * downstep_ratio;
    double ratio = sigma_down / sigma;
    double alpha_next = 1.0 - sigma_next;
    double alpha_down = 1.0 - sigma_down;
    double renoise = sqrt(fmax(
        (double)sigma_next * sigma_next -
        sigma_down * sigma_down * alpha_next * alpha_next /
            (alpha_down * alpha_down),
        0.0));
    for (size_t index = 0; index < 4u; index++) {
        double deterministic = ratio * sample[index] +
            (1.0 - ratio) * x0[index];
        double expected = alpha_next / alpha_down * deterministic +
            noise[index] * renoise;
        CHECK(fabs((double)ancestral[index] - expected) < 2e-6);
    }

    float eta_zero[4];
    CHECK(ltx_euler_ancestral_step_f32(
        eta_zero, sample, x0, NULL, 4u, sigma, sigma_next,
        0.0f, 1.0f, error, sizeof(error)));
    CHECK(ltx_euler_step_f32(euler, sample, x0, 4u,
                             sigma, sigma_next, error, sizeof(error)));
    for (size_t index = 0; index < 4u; index++)
        CHECK(fabsf(eta_zero[index] - euler[index]) < 2e-6f);

    CHECK(ltx_euler_ancestral_step_f32(
        ancestral, sample, x0, NULL, 4u, 0.421875f, 0.0f,
        1.0f, 1.0f, error, sizeof(error)));
    for (size_t index = 0; index < 4u; index++)
        CHECK(ancestral[index] == x0[index]);
    CHECK(!ltx_euler_ancestral_step_f32(
        ancestral, sample, x0, NULL, 4u, sigma, sigma_next,
        1.0f, 1.0f, error, sizeof(error)));

    float renoised[4];
    CHECK(ltx_renoise_f32(renoised, x0, noise, 4u, 0.25f,
                          error, sizeof(error)));
    for (size_t index = 0; index < 4u; index++)
        CHECK(fabsf(renoised[index] -
                    (0.75f * x0[index] + 0.25f * noise[index])) < 1e-7f);

    enum { random_count = 100000 };
    float *random_values = malloc(
        random_count * sizeof(*random_values));
    CHECK(random_values != NULL);
    ltx_rng first_rng;
    ltx_rng second_rng;
    ltx_rng other_stream;
    ltx_rng_seed(&first_rng, 42u, 0u);
    ltx_rng_seed(&second_rng, 42u, 0u);
    ltx_rng_seed(&other_stream, 42u, 1u);
    for (unsigned index = 0; index < 32u; index++) {
        float first = ltx_rng_next_normal_f32(&first_rng);
        float second = ltx_rng_next_normal_f32(&second_rng);
        float other = ltx_rng_next_normal_f32(&other_stream);
        CHECK(first == second);
        if (index == 0u) CHECK(first != other);
    }
    ltx_rng_fill_normal_f32(
        &first_rng, random_values, random_count);
    double random_sum = 0.0;
    double random_square_sum = 0.0;
    for (size_t index = 0; index < random_count; index++) {
        CHECK(isfinite(random_values[index]));
        random_sum += random_values[index];
        random_square_sum +=
            (double)random_values[index] * random_values[index];
    }
    double random_mean = random_sum / (double)random_count;
    double random_variance =
        random_square_sum / (double)random_count -
        random_mean * random_mean;
    CHECK(fabs(random_mean) < 0.015);
    CHECK(fabs(sqrt(random_variance) - 1.0) < 0.015);
    free(random_values);
    puts("test_ltx: PASS");
    return 0;
}
