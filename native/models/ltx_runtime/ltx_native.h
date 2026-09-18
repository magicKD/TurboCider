#pragma once
#include "ltx_gpu.h"
#include "../../core/memory_schedule_c.h"
#include "../../core/stream_slot_c.h"
#include "ltx_safetensors.h"
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
    /* Exact Video self/text attention command batching. It is opt-in because
     * complete-request gains vary with device scheduling. */
    int video_attention_batch;
    /* Queue the dependent Audio self/text attention and FFN command buffers
     * before waiting. This removes per-operation host fences while preserving
     * the same command order and BF16 kernels. */
    int batch_audio_commands;
    /* Low-memory execution keeps a budget-selected prefix resident and
     * refills the remaining Transformer blocks through one or two reusable
     * slots. A zero budget selects the default 12-GiB-plus-geometry denoiser
     * target; resident execution ignores both fields. */
    int stream_blocks;
    uint64_t memory_budget_bytes;
    /* Zero preserves the legacy maximum. Constrained callers pass 1..3 and
     * the runtime must not allocate or prefetch more refill slots. */
    uint32_t max_refill_slots;
    /* Optional per-buffer admission installed before any Transformer Metal
     * buffer is created. Constrained callers must supply all hook callbacks;
     * default callers leave this NULL and retain the legacy fast path. */
    const ltx_gpu_memory_hooks *memory_hooks;
    uint64_t memory_allocator_domain;
    uint64_t memory_generation;
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
    /* Optional approximate Video self-attention backend.  Dense remains the
     * default; callers must opt in with allow_approximation at the request
     * layer because Sol changes the diffusion trajectory. */
    int sol_stage1;
    int sol_stage2;
    int sol_batch_commands;
    uint32_t sol_dense_edge_blocks;
    uint32_t sol_dense_edge_steps;
    float sol_tau;
    uint32_t sparse_mode;
    uint32_t sparse_radius;
    uint32_t sparse_anchor_stride;
    uint32_t sparse_tokens_per_frame;
    uint32_t sparse_keep_blocks;
    /* Restrict ANE MLP execution to a contiguous Transformer block window.
     * A zero count preserves the legacy all-48-block ABI default. Partial
     * windows retain the original GPU MLP weights outside the window and
     * cannot request release_full_gpu_mlp. */
    uint32_t ane_mlp_first_block;
    uint32_t ane_mlp_block_count;
    /* Stage masks use bit zero for Stage 1 and bit one for Stage 2. */
    uint32_t ane_mlp_stage_mask;
    uint32_t ane_kv_stage_mask;
    /* Core ML artifact variant for ANE MLPs (normally int8_pc; fp16 is
     * retained as a quality/parity diagnostic). */
    const char *ane_variant;
    const char *mlp_directories[2];
    const char *v2a_directories[2];
    const char *kv_directory;
    const char *qkv_directories[2];
    /* Optional owner-thread semantic schedule callback. The native runtime
     * copies this POD during create; workers and Metal callbacks never emit. */
    const tc_memory_schedule_hooks_v1 *schedule_hooks;
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
/* Internal experimental layout-only entry. Public TurboCider eligibility is
 * checked separately; this does not certify a layout or a memory upper.
 * Exact G=1/P>=1/single-class plan, GPU-only, no legacy budget authority.
 * Plan arrays are copied at create. On failure, a non-null *out is quarantined
 * and must be retained until streaming_destroy succeeds. */
typedef struct {
    uint32_t struct_size, version, resident_prefix_blocks;
    const tc_stream_stage_plan_v1 *plan;
} ltx_native_streaming_options_v1;
int ltx_native_create_streamed_v1(const ltx_native_options *,
    const ltx_native_streaming_options_v1 *, ltx_native_denoiser **out,
    ltx_native_progress, void *, char *, size_t);
int ltx_native_streaming_destroy(ltx_native_denoiser **, char *, size_t);
int ltx_native_streaming_counters(ltx_native_denoiser *, tc_stream_counters_v1 *, char *, size_t);
/* Public adapters may enable and copy the common executor's real receipt
 * after exact creation and before the first stage.  The V2 getter follows the
 * common size-query/two-call ABI from stream_slot_c.h. */
int ltx_native_streaming_enable_receipt(
    ltx_native_denoiser *, const tc_stream_receipt_config_v1 *,
    char *, size_t);
int ltx_native_streaming_receipt_v2(
    ltx_native_denoiser *, tc_stream_receipt_v2 *, char *, size_t);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
/* Private deterministic fault injection for lifecycle tests. Release builds
 * do not contain this symbol and no request/environment setting can reach it. */
int ltx_native_streaming_test_set_destroy_failures(
    ltx_native_denoiser *, uint32_t, char *, size_t);
int ltx_native_streaming_test_cancel_first_fill(
    ltx_native_denoiser *, char *, size_t);
#endif
/* v2 exact entry borrows a header parsed by ltx_st_read_header_fd and its fd.
 * Identity/change checks run before construction and at stage boundaries;
 * these are NOT a content hash or protection against concurrent modification.
 * All request artifacts must be immutable during execution.
 * The caller owns both objects and must keep them alive until destroy returns.
 * The native context never closes/free's the borrowed descriptor or header. */
typedef struct {
    uint32_t struct_size, version;
    ltx_native_streaming_options_v1 base;
    const ltx_st_header *metadata_header;
    const ltx_st_mapping *metadata_mapping;
} ltx_native_streaming_options_v2;
int ltx_native_create_streamed_v2(const ltx_native_options *,
    const ltx_native_streaming_options_v2 *, ltx_native_denoiser **out,
    ltx_native_progress, void *, char *, size_t);
void ltx_native_free(ltx_native_denoiser*);
/* Owner-thread completion barrier for native GPU and auxiliary GPU queues. */
int ltx_native_drain(ltx_native_denoiser*, char*, size_t);
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
/* Request-lease variant. Both descriptors are borrowed for the call and
 * duplicated by the component loaders; diagnostic paths are never reopened. */
int ltx_native_upsample_stage2_fd(
    ltx_native_denoiser*, int upsampler_fd,
    const char *upsampler_diagnostic_path, int video_vae_fd,
    const char *video_vae_diagnostic_path,
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
/* Connector-only entry point for the C++/MLX Transformer path. It creates a
 * short-lived Metal context and loads only the video/audio connector weights;
 * no C Transformer blocks are materialized. */
int ltx_native_connect_conditioning_file(
    const char *checkpoint, const char *shader_source,
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
