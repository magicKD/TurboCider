#ifndef H3_DIT_H
#define H3_DIT_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_quant_cache.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_dit h3_dit;

typedef enum {
    H3_DIT_COREML_BENCHMARK_DEFAULT = 0,
    H3_DIT_COREML_BENCHMARK_FULL_GPU,
    H3_DIT_COREML_BENCHMARK_EXACT_SPLIT,
    H3_DIT_COREML_BENCHMARK_CANDIDATE
} h3_dit_coreml_benchmark_route;

typedef struct {
    int enabled;
    unsigned active_blocks;
    unsigned pinned_blocks;
    unsigned streamed_blocks;
    uint64_t memory_budget_bytes;
    int quantized;
    uint64_t block_bytes;
    uint64_t activation_reserve_bytes;
    uint64_t bytes_read;
    double read_seconds;
    double wait_seconds;
} h3_dit_streaming_info;

typedef void (*h3_dit_progress)(const char *phase, int completed, int total,
                                void *opaque);

typedef int (*h3_dit_preview)(int completed_steps, int total_steps,
                              const float *video_latent,
                              size_t video_elements, void *opaque);

/* Load a text-only FL2VA transformer. Text refinement and AdaLN precomputation
 * happen before the persistent core is loaded. SSD streaming retains only the
 * small block norms and two alternating BF16 matrix slots. */
h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         int ssd_pinned_prefix,
                         uint64_t ssd_memory_budget_bytes,
                         const char *ssd_quantized_cache_directory,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load only the request-compatible resident FL2VA transformer core. The text
 * and layout are validated and select sequence/modality-dependent backends,
 * but prompt/layout tensors are deliberately deferred. The returned object
 * must pass h3_dit_reprepare() before forward or denoise. This is intended for
 * speculative cold loading while the real text encoder is still running. */
h3_dit *h3_dit_load_t2va_core(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         int ssd_pinned_prefix,
                         uint64_t ssd_memory_budget_bytes,
                         const char *ssd_quantized_cache_directory,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load FL2VA/Ref2VA packing. Condition inputs are already patchified F32 row
 * sources: visual rows have width 96 and audio rows width 32. Their element
 * counts must exactly match layout.img_cond_rows/audio_cond_rows. */
h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         int ssd_pinned_prefix,
                         uint64_t ssd_memory_budget_bytes,
                         const char *ssd_quantized_cache_directory,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);
void h3_dit_free(h3_dit *dit);

/* Reset mutable sampler state and replace seed-dependent condition rows before
 * reusing an otherwise identical prepared transformer. */
int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size);

/* Opt-in benchmark/test hook. H3_BENCH_COREML_SAME_LOADED_AB must be set
 * before load; normal runtimes neither retain the full selected MLPs nor enter
 * these route overrides. The override applies to every selected Core ML block. */
int h3_dit_set_coreml_benchmark_route(
                     h3_dit *dit, h3_dit_coreml_benchmark_route route,
                     char *error, size_t error_size);

/* Read one selected Core ML block's BF16 MLP input and ungated branch output
 * from the most recent successful forward. The first selected block is used
 * unless H3_BENCH_COREML_CAPTURE_BLOCK names another selected block. These
 * captures exist only for the same-loaded benchmark mode above, and both
 * destination counts must exactly match
 * h3_dit_coreml_benchmark_capture_elements(). */
size_t h3_dit_coreml_benchmark_capture_elements(const h3_dit *dit);
int h3_dit_read_coreml_benchmark_capture(
                     const h3_dit *dit,
                     uint16_t *modulated_mlp_input, size_t input_elements,
                     uint16_t *mlp_branch_output, size_t output_elements,
                     char *error, size_t error_size);

/* Rebuild prompt/layout-sized state while retaining the compatible resident
 * transformer core, quantized weights, AdaLN schedule, and Metal runtime.
 * The sigma schedule and conditioning modalities must match the resident
 * model. On failure the object is no longer runnable and must be freed before
 * falling back to a full load. */
int h3_dit_reprepare(h3_dit *dit,
                     const h3_text_embedding *text,
                     const h3_layout *layout,
                     const h3_sigma_schedule *sigmas,
                     float spatial_rope_scale,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     h3_dit_progress progress, void *progress_opaque,
                     char *error, size_t error_size);

/* Drop prompt/layout-sized tensors before another component runs while
 * retaining the resident model core. A later h3_dit_reprepare() is required
 * before the transformer can run again. */
void h3_dit_release_request(h3_dit *dit);

size_t h3_dit_video_elements(const h3_dit *dit);
size_t h3_dit_audio_elements(const h3_dit *dit);

/* One raw data-ward velocity evaluation. Input/output video layout is
 * [24,T,H,W], audio is [32,2,T], all F32 on the host boundary. */
int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size);

/* Run every configured RES multistep update in place. Solver state remains
 * F32 between transformer evaluations, matching the released MLX path. */
int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size);

/* Current serving sampler: independent video/audio shifted Euler grids. */
int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Preview variant. The callback runs after every Euler transition with the
 * current channel-major F32 video latent. It is deliberately opt-in because it
 * introduces a synchronization point after each step. */
int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size);

/* Build the velocity-evaluation mask used by the serving sampler. Returns the
 * evaluation count, or -1 for invalid arguments. Public internally so the
 * quality-tuned aggressive schedule remains pinned by cheap host tests. */
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count);

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats);
int h3_dit_get_streaming_info(const h3_dit *dit,
                              h3_dit_streaming_info *info);
/* True after an opt-in final-pass progressive eviction consumed resident
 * block weights. Such a DiT is intentionally one-shot and cannot be cached or
 * reprepared for another request. */
int h3_dit_final_evicted(const h3_dit *dit);

/* Exact row-order conversions, public internally so they can be pinned by
 * cheap tests independently of the 62 GiB checkpoint. */
int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements);
int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements);
int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements);
int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements);

#endif
