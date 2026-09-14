#pragma once

#include "../ltx_runtime/ltx_native.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_mlx_denoiser ltx_mlx_denoiser;

typedef struct {
    const char *checkpoint;
    uint32_t width;
    uint32_t height;
    uint32_t frames;
    uint32_t fps;
    uint32_t block_cache_capacity;
    uint32_t convrot_group_size;
    int force_eval_each_block;
    float av_ca_timestep_scale_multiplier;
} ltx_mlx_options;

typedef struct {
    uint32_t block_count;
    uint32_t block_cache_capacity;
    uint32_t pinned_block_count;
    uint32_t refill_slot_count;
    uint64_t top_weight_bytes;
    uint64_t cache_loads;
    uint64_t cache_hits;
    uint64_t cache_evictions;
    uint64_t cache_load_bytes;
    uint64_t slot_allocations;
    uint64_t slot_refills;
    double cache_load_seconds;
    uint64_t resident_bytes;
    uint64_t peak_resident_bytes;
    double last_run_seconds;
} ltx_mlx_info;

ltx_mlx_denoiser *ltx_mlx_create(const ltx_mlx_options *,
                                 ltx_native_progress, void *, char *, size_t);
void ltx_mlx_free(ltx_mlx_denoiser *);
int ltx_mlx_get_info(const ltx_mlx_denoiser *, ltx_mlx_info *);

int ltx_mlx_run(
    ltx_mlx_denoiser *, int stage, uint64_t seed,
    uint16_t *video, size_t video_elements,
    uint16_t *audio, size_t audio_elements,
    const uint16_t *video_text, const uint16_t *audio_text,
    const uint16_t *mask, uint32_t text_rows,
    const uint16_t *first_frame, float strength,
    ltx_native_progress, void *, char *, size_t);

int ltx_mlx_upsample_stage2(
    uint16_t *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    const char *upsampler_checkpoint, const char *video_vae_checkpoint,
    uint32_t frames, uint32_t height, uint32_t width,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
