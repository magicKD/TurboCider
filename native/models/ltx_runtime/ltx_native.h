#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ltx_native_denoiser ltx_native_denoiser;
typedef int (*ltx_native_progress)(const char *phase,int current,int total,void *opaque);
typedef struct {
    const char *checkpoint;
    const char *shader_source;
    uint32_t width,height,frames,fps;
    int parallel_av;
    /* Low-memory execution keeps a budget-selected prefix resident and
     * refills the remaining Transformer blocks through one or two reusable
     * slots. A zero budget selects the default 12-GiB-plus-geometry denoiser
     * target; resident execution ignores both fields. */
    int stream_blocks;
    uint64_t memory_budget_bytes;
    /* Lifecycle options are explicit at the library boundary.  The embedded
     * runtime deliberately does not inherit the benchmark CLI's environment
     * variables. */
    int preload_ane_stage2;
    int release_full_gpu_mlp;
    int detach_ane_stage1;
    int detach_ane_stage2;
    int release_blocks_final_step;
    int ane_mlp_fused_residual;
    int ane_mlp_fused_adaln_pack;
    /* Restrict ANE MLP execution to a contiguous Transformer block window.
     * A zero count preserves the legacy all-48-block ABI default. Partial
     * windows retain the original GPU MLP weights outside the window and
     * cannot request release_full_gpu_mlp. */
    uint32_t ane_mlp_first_block;
    uint32_t ane_mlp_block_count;
    /* Stage masks use bit zero for Stage 1 and bit one for Stage 2. */
    uint32_t ane_mlp_stage_mask;
    uint32_t ane_kv_stage_mask;
    const char *mlp_directories[2];
    const char *v2a_directories[2];
    const char *kv_directory;
    const char *qkv_directories[2];
} ltx_native_options;
typedef struct {
    int enabled;
    uint32_t pinned_blocks;
    uint32_t streamed_blocks;
    uint32_t refill_slots;
    uint64_t memory_budget_bytes;
    uint64_t activation_reserve_bytes;
    uint64_t block_bytes;
    uint64_t estimated_working_set_bytes;
    uint64_t bytes_loaded;
    uint64_t slot_allocations;
    uint64_t slot_refills;
    double load_seconds;
    double wait_seconds;
} ltx_native_streaming_info;
ltx_native_denoiser *ltx_native_create(const ltx_native_options*,ltx_native_progress,void*,char*,size_t);
void ltx_native_free(ltx_native_denoiser*);
int ltx_native_get_streaming_info(
    const ltx_native_denoiser*, ltx_native_streaming_info*);
/* All inputs are BF16. Stage 1 receives seeded noise, stage 2 receives the
 * normalized upsampled stage-1 latent. Each stage updates video/audio in place
 * only after successful completion; cancelled runs leave caller buffers intact. */
int ltx_native_run(ltx_native_denoiser*,int stage,uint64_t seed,
    uint16_t *video,size_t video_elements,uint16_t *audio,size_t audio_elements,
    const uint16_t *video_text,const uint16_t *audio_text,const uint16_t *mask,
    uint32_t text_rows,const uint16_t *first_frame,float strength,
    ltx_native_progress,void*,char*,size_t);
/* Exact Stage-1 -> Stage-2 transition: video-latent denormalization on the
 * native Metal context, MLX spatial x2 upsampling, then renormalization. */
int ltx_native_upsample_stage2(
    ltx_native_denoiser*, const char *upsampler_checkpoint,
    const char *video_vae_checkpoint,
    uint16_t *output, size_t output_elements,
    const uint16_t *input, size_t input_elements,
    char *error, size_t error_size);
/* Run the checkpoint's native embedding connectors over precomputed Gemma
 * video/audio projections. The output row count must match the connector's
 * padded register geometry (normally 1024 for text-only prompts). */
int ltx_native_connect_conditioning(
    ltx_native_denoiser *,
    uint16_t *video_output, size_t video_output_elements,
    uint16_t *audio_output, size_t audio_output_elements,
    uint16_t *mask_output, size_t mask_output_elements,
    uint32_t output_rows,
    const uint16_t *video_input, size_t video_input_elements,
    const uint16_t *audio_input, size_t audio_input_elements,
    const uint16_t *mask_input, size_t mask_input_elements,
    uint32_t input_rows,
    char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
