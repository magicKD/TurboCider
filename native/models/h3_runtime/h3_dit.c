#include "h3_runtime_config.h"
#include "h3_dit.h"

#include "h3_coreml.h"
#include "h3_ane_linear.h"
#include "h3_ane_mlp.h"
#include "h3_dit_schedule.h"
#include "h3_streaming_policy.h"
#include "h3_weights.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
    h3_coreml_mlp *coreml_mlp;
    h3_ane_mlp *ane_mlp;
    h3_ane_linear *ane_qkv;
    h3_ane_linear *ane_attention_out;
    h3_ane_linear_recipe ane_attention_out_recipe;
    int ane_attention_out_transient;
    h3_coreml_mlp *coreml_qkv;
    h3_gpu_tensor *coreml_qkv_gpu_weight;
    h3_gpu_tensor *ane_qkv_gpu_scales;
    h3_gpu_tensor *ane_attention_out_gpu_weight;
    h3_gpu_tensor *coreml_full_fc1;
    h3_gpu_tensor *coreml_full_fc2;
    h3_gpu_tensor *coreml_fallback_fc1;
    h3_gpu_tensor *coreml_fallback_fc2;
    char *coreml_manifest_path;
    char *coreml_fallback_fc1_path;
    char *coreml_fallback_fc2_path;
    char coreml_manifest_sha256[H3_COREML_SHA256_HEX_SIZE];
    uint32_t coreml_intermediate;
    float coreml_output_scale;
    uint32_t ane_gpu_intermediate;
    int ane_int8_weights;
    unsigned ane_predictions;
    unsigned ane_prediction_failures;
    unsigned ane_nonfinite_failures;
    unsigned ane_range_retries;
    unsigned ane_range_headroom_retries;
    unsigned ane_range_recoveries;
    float ane_range_peak_max;
    double ane_range_retry_seconds;
    double ane_pack_seconds;
    double ane_overlap_seconds;
    double ane_join_seconds;
    double ane_unload_seconds;
    unsigned coreml_predictions;
    unsigned coreml_fallbacks;
    unsigned coreml_forced_fallbacks;
    unsigned coreml_latched_fallbacks;
    unsigned coreml_prediction_failures;
    uint32_t coreml_qkv_heads;
    unsigned coreml_qkv_predictions;
    unsigned coreml_qkv_prediction_failures;
    uint32_t ane_qkv_heads;
    int ane_qkv_int8_weights;
    int ane_qkv_gpu_int8_weights;
    unsigned ane_qkv_predictions;
    unsigned ane_qkv_prediction_failures;
    unsigned ane_qkv_nonfinite_failures;
    double ane_qkv_pack_seconds;
    double ane_qkv_overlap_seconds;
    double ane_qkv_join_seconds;
    double ane_qkv_unload_seconds;
    unsigned ane_qkv_micro_done;
    uint32_t ane_attention_out_width;
    int ane_attention_out_int8_weights;
    unsigned ane_attention_out_predictions;
    unsigned ane_attention_out_prediction_failures;
    unsigned ane_attention_out_nonfinite_failures;
    double ane_attention_out_pack_seconds;
    double ane_attention_out_overlap_seconds;
    double ane_attention_out_join_seconds;
    double ane_attention_out_unload_seconds;
    unsigned ane_attention_out_micro_done;
    unsigned coreml_fallback_lazy_maps;
    size_t coreml_nonfinite_values;
    double coreml_fallback_first_map_seconds;
    double coreml_fallback_map_seconds;
    int coreml_fallback_latched;
} h3_dit_block;

static int block_has_private_ane_attention_out(
                                      const h3_dit_block *block) {
    return block && (block->ane_attention_out ||
                     h3_ane_linear_recipe_present(
                         &block->ane_attention_out_recipe));
}

static int quantize_block_mlp_width(h3_dit *dit, h3_dit_block *block,
                                    uint32_t intermediate,
                                    int release_bf16,
                                    char *error, size_t error_size);
static int quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size);

enum {
    STREAM_QKV,
    STREAM_OUT,
    STREAM_FC1,
    STREAM_FC2,
    STREAM_QKV_INT8,
    STREAM_QKV_SCALES,
    STREAM_OUT_INT8,
    STREAM_OUT_SCALES,
    STREAM_FC1_INT8,
    STREAM_FC1_SCALES,
    STREAM_FC2_INT8,
    STREAM_FC2_SCALES,
    STREAM_FIELDS
};

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
    h3_gpu_dtype dtype;
} h3_dit_stream_source;

typedef struct {
    h3_dit_stream_source sources[H3_QUANT_CACHE_SOURCES];
    unsigned source_count;
} h3_dit_stream_layer;

struct h3_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    char *weight_directory;
    h3_dit_schedule *schedule;
    int fused_mlp;
    int nax_mlp;
    int int8_mlp;
    int ane_gpu_int8_mlp;
    uint32_t ane_mlp_row_split_rows;
    float ane_mlp_range_headroom;
    unsigned ane_mlp_range_retry_limit;
    int ane_mlp_range_guard_ready;
    int int8_qkv;
    int int8_attention_out;
    int keep_bf16_qkv;
    int keep_bf16_attention_out;
    int use_slower_row_major_attention_output;
    int use_slower_unfused_int8_inputs;
    int use_slower_unfused_qkv_rope;
    int use_slower_scalar_qkv_rms;
    int use_slower_uncached_int8_scales;
    int use_slower_dynamic_fc1_k;
    int use_slower_grouped_quantizer;
    int use_int8_row_fc2;
    int ssd_streaming;
    int ssd_quantized;
    int ssd_pinned_prefix;
    uint64_t ssd_memory_budget_bytes;
    h3_quant_cache quant_cache;
    int keep_bf16_mlp;
    int request_ready;
    int activation_aliases;
    int fused_patch_projection;
    int fused_patch_pack;
    int token_reduction;
    int token_reduction_active;
    unsigned token_reduction_begin;
    unsigned token_reduction_end;
    unsigned token_reduction_early_steps;
    unsigned token_reduction_early_end;
    float token_reduction_scale;
    float spatial_rope_scale;
    int bf16_final;
    unsigned core_reuse_interval;
    unsigned core_forward_count;
    int core_residual_ready;
    int first_block_cache;
    float first_block_cache_threshold;
    unsigned first_block_cache_max_hits;
    unsigned first_block_cache_consecutive_hits;
    unsigned first_block_cache_hit_cap_forced;
    int first_block_cache_previous_ready;
    int first_block_cache_tail_ready;
    uint32_t first_block_cache_partial_count;
    unsigned first_block_cache_calls;
    unsigned first_block_cache_fresh;
    unsigned first_block_cache_reuse;
    double first_block_cache_wait_seconds;
    int first_block_cache_last_reused;
    int tea_cache;
    float tea_cache_threshold;
    float tea_cache_audio_threshold;
    unsigned tea_cache_retain_steps;
    unsigned tea_cache_cooldown_steps;
    unsigned tea_cache_max_hits;
    unsigned tea_cache_consecutive_hits;
    unsigned tea_cache_hit_cap_forced;
    int tea_cache_previous_ready;
    int tea_cache_residual_ready;
    uint32_t tea_cache_partial_count;
    uint32_t tea_cache_audio_partial_count;
    unsigned tea_cache_calls;
    unsigned tea_cache_fresh;
    unsigned tea_cache_reuse;
    double tea_cache_accumulator;
    double tea_cache_audio_accumulator;
    double tea_cache_wait_seconds;
    int tea_cache_last_reused;
    int sol_attention;
    float sol_tau;
    unsigned sol_dense_layers;
    unsigned sol_dense_steps;
    unsigned sol_dense_final_steps;
    unsigned sol_calls;
    unsigned sol_dense_calls;
    size_t sol_route_elements;
    unsigned sol_route_profile_layer;
    int sol_route_profile_pending;
    uint32_t sol_verify_elements;
    uint32_t sol_verify_partial_count;
    unsigned sol_verify_layer;
    int sol_verify_pending;
    uint64_t coreml_checkpoint_identity;
    int coreml_checkpoint_identity_ready;
    int coreml_gpu_fallback;
    int coreml_force_gpu_fallback;
    int coreml_fallback_latch;
    int coreml_same_loaded_ab;
    h3_ane_mlp_io *ane_mlp_io;
    unsigned ane_mlp_blocks;
    h3_ane_linear_io *ane_qkv_io;
    unsigned ane_qkv_blocks;
    h3_ane_linear_io *ane_attention_out_io;
    unsigned ane_attention_out_blocks;
    h3_dit_coreml_benchmark_route coreml_benchmark_route;
    unsigned coreml_benchmark_capture_block;
    size_t coreml_benchmark_capture_elements;
    int coreml_benchmark_capture_pending;
    int coreml_benchmark_capture_ready;
    unsigned active_block_count;
    unsigned final_eviction_groups;
    unsigned final_evicted_blocks;
    uint64_t final_evicted_bytes;
    int explicit_gate_skip;
    int cached_gate_skip;
    int gate_cache_identity_ready;
    uint64_t gate_cache_fingerprint;
    uint8_t block_active[H3_DIT_BLOCKS];
    int step_gate_skip;
    uint8_t step_block_skip[H3_MAX_STEPS][H3_DIT_BLOCKS];
    unsigned step_active_block_count[H3_MAX_STEPS];
    h3_layout layout;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    uint32_t reduced_sequence;
    uint32_t reduced_video_rows;
    uint32_t token_baseline_rows;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *reduced_rope_cos;
    h3_gpu_tensor *reduced_rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **reduced_row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_gpu_tensor *coreml_input;
    h3_gpu_tensor *coreml_output;
    h3_gpu_tensor *coreml_qkv_input;
    h3_gpu_tensor *coreml_qkv_output;
    h3_gpu_tensor *coreml_qkv_gpu_output;
    h3_gpu_tensor *ane_qkv_micro_partials;
    uint32_t ane_qkv_micro_partial_count;
    h3_gpu_tensor *ane_attention_out_gpu_output;
    h3_gpu_tensor *ane_attention_out_micro_reference;
    h3_gpu_tensor *ane_attention_out_micro_partials;
    uint32_t ane_attention_out_micro_partial_count;
    h3_gpu_tensor *coreml_fallback_output;
    h3_gpu_tensor *coreml_nonfinite;
    h3_gpu_tensor *ane_nonfinite;
    h3_gpu_tensor *coreml_benchmark_mlp_input;
    h3_gpu_tensor *coreml_benchmark_mlp_output;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    h3_dit_block stream_slots[2];
    h3_dit_stream_layer stream_layers[H3_DIT_BLOCKS];
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes;
    double stream_read_seconds;
    double stream_wait_seconds;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *video_projection_map;
    h3_gpu_tensor *audio_projection_map;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *core_input;
    h3_gpu_tensor *core_residual;
    h3_gpu_tensor *first_block_cache_previous;
    h3_gpu_tensor *first_block_cache_partials;
    float *first_block_cache_host_partials;
    h3_gpu_tensor *tea_cache_previous;
    h3_gpu_tensor *tea_cache_partials;
    float *tea_cache_host_partials;
    h3_gpu_tensor *sol_query_centroids;
    h3_gpu_tensor *sol_key_centroids;
    h3_gpu_tensor *sol_value_sums;
    h3_gpu_tensor *sol_thresholds;
    h3_gpu_tensor *sol_routes;
    h3_gpu_tensor *sol_verify_dense;
    h3_gpu_tensor *sol_verify_partials;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *token_pool_pairs;
    h3_gpu_tensor *token_baseline_indices;
    h3_gpu_tensor *token_expand_parents;
    h3_gpu_tensor *token_original;
    int token_original_in_qkv;
    size_t token_original_offset;
    size_t token_baseline_offset;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *int8_activation;
    h3_gpu_tensor *int8_activation_scales;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_inverse;
    h3_gpu_tensor *final_video_inverse;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
    h3_gpu_tensor *previous_audio_velocity;
    h3_gpu_tensor *previous_video_velocity;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static unsigned command_block_interval(const h3_dit *dit) {
    const char *value = h3_runtime_getenv("H3_DIT_COMMAND_BLOCKS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        return end != value && !*end && parsed >= 0 &&
               parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
    }
    if (h3_gpu_is_m5(dit->gpu))
        return dit->active_block_count * 3 / 5;
    return dit->active_block_count == H3_DIT_BLOCKS ? 30u : 0u;
}

static int gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static void report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static int copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

static int validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    return 1;
}

static int configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size) {
    const char *enabled = h3_runtime_getenv("H3_TOKEN_REDUCTION");
    if (!requested &&
        (!enabled || !*enabled || !strcmp(enabled, "0"))) return 1;
    unsigned begin = 4, end = 30;
    const char *range = h3_runtime_getenv("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        char *middle = NULL;
        unsigned long parsed_begin = strtoul(range, &middle, 10);
        if (middle == range || *middle != ':') {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END");
            return 0;
        }
        char *tail = NULL;
        unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
        if (tail == middle + 1 || *tail || parsed_begin >= parsed_end ||
            parsed_end > H3_DIT_BLOCKS) {
            fail(error, error_size,
                 "token-reduction block range must satisfy 0 <= BEGIN < END <= 50");
            return 0;
        }
        begin = (unsigned)parsed_begin;
        end = (unsigned)parsed_end;
    }
    /* Coarse structure is tolerant of a deeper reduced stack while the first
     * noisy samples form. Restore earlier once fine detail starts resolving. */
    unsigned early_steps = end < 40 ? 10 : 0;
    unsigned early_end = end < 40 ? 40 : end;
    const char *early = h3_runtime_getenv("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            char *middle = NULL;
            unsigned long parsed_steps = strtoul(early, &middle, 10);
            if (middle == early || *middle != ':') {
                fail(error, error_size,
                     "H3_TOKEN_REDUCTION_EARLY must be STEPS:END");
                return 0;
            }
            char *tail = NULL;
            unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
            if (tail == middle + 1 || *tail || !parsed_steps ||
                parsed_steps > 1000 || parsed_end <= end ||
                parsed_end > H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "early token reduction requires STEPS > 0 and "
                     "base END < END <= 50");
                return 0;
            }
            early_steps = (unsigned)parsed_steps;
            early_end = (unsigned)parsed_end;
        }
    }
    float scale = 1.0f;
    const char *scale_text = h3_runtime_getenv("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        scale = strtof(scale_text, &tail);
        if (tail == scale_text || *tail || !isfinite(scale) ||
            scale < 0.0f || scale > 2.0f) {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]");
            return 0;
        }
    }
    uint32_t spatial_height = (uint32_t)dit->latent_h / 2;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint64_t reduced_video =
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * reduced_width;
    if (!spatial_height || !spatial_width ||
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * spatial_width !=
            dit->video_rows ||
        dit->video_target_start + dit->video_rows != dit->sequence ||
        reduced_video > UINT32_MAX ||
        reduced_video > UINT32_MAX - dit->video_target_start) {
        fail(error, error_size,
             "token reduction requires the target video to end the packed layout");
        return 0;
    }
    dit->token_reduction = 1;
    dit->token_reduction_begin = begin;
    dit->token_reduction_end = end;
    dit->token_reduction_early_steps = early_steps;
    dit->token_reduction_early_end = early_end;
    dit->token_reduction_scale = scale;
    dit->reduced_video_rows = (uint32_t)reduced_video;
    dit->token_baseline_rows = dit->video_rows - dit->reduced_video_rows;
    dit->reduced_sequence = dit->video_target_start +
                            dit->reduced_video_rows;
    return 1;
}

static int configure_first_block_cache(h3_dit *dit, char *error,
                                       size_t error_size) {
    const char *value = h3_runtime_getenv("H3_FBC_THRESHOLD");
    if (!value || !*value) return 1;
    char *tail = NULL;
    float threshold = strtof(value, &tail);
    if (tail == value || *tail || !isfinite(threshold) ||
        threshold < 0.0f || threshold > 1.0f) {
        fail(error, error_size,
             "H3_FBC_THRESHOLD must be a finite value in [0, 1]");
        return 0;
    }
    if (dit->core_reuse_interval > 1 || dit->token_reduction ||
        dit->ssd_streaming) {
        fail(error, error_size,
             "FirstBlockCache cannot be combined with core reuse, "
             "token reduction, or SSD streaming");
        return 0;
    }
    unsigned max_hits = 1;
    const char *max_hits_text = h3_runtime_getenv("H3_FBC_MAX_HITS");
    if (max_hits_text && *max_hits_text) {
        errno = 0;
        char *max_hits_tail = NULL;
        unsigned long parsed = strtoul(max_hits_text, &max_hits_tail, 10);
        if (errno || max_hits_tail == max_hits_text || *max_hits_tail ||
            parsed > 32) {
            fail(error, error_size,
                 "H3_FBC_MAX_HITS must be an integer in [0, 32]");
            return 0;
        }
        max_hits = (unsigned)parsed;
    }
    dit->first_block_cache = 1;
    dit->first_block_cache_threshold = threshold;
    dit->first_block_cache_max_hits = max_hits;
    return 1;
}

static int parse_cache_unsigned(const char *name, unsigned fallback,
                                unsigned maximum, unsigned *result,
                                char *error, size_t error_size) {
    const char *value = h3_runtime_getenv(name);
    if (!value || !*value) {
        *result = fallback;
        return 1;
    }
    errno = 0;
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    if (errno || tail == value || *tail || parsed > maximum) {
        fail(error, error_size, "%s must be an integer in [0, %u]",
             name, maximum);
        return 0;
    }
    *result = (unsigned)parsed;
    return 1;
}

static int configure_tea_cache(h3_dit *dit, char *error,
                               size_t error_size) {
    const char *value = h3_runtime_getenv("H3_TEACACHE_THRESHOLD");
    if (!value || !*value) return 1;
    char *tail = NULL;
    float threshold = strtof(value, &tail);
    if (tail == value || *tail || !isfinite(threshold) ||
        threshold <= 0.0f || threshold > 10.0f) {
        fail(error, error_size,
             "H3_TEACACHE_THRESHOLD must be a finite value in (0, 10]");
        return 0;
    }
    if (dit->core_reuse_interval > 1 || dit->token_reduction ||
        dit->ssd_streaming || dit->first_block_cache) {
        fail(error, error_size,
             "TeaCache cannot be combined with core reuse, token reduction, "
             "SSD streaming, or FirstBlockCache");
        return 0;
    }
    if (!parse_cache_unsigned(
            "H3_TEACACHE_RETAIN_STEPS", 5, H3_MAX_STEPS,
            &dit->tea_cache_retain_steps, error, error_size) ||
        !parse_cache_unsigned(
            "H3_TEACACHE_COOLDOWN_STEPS", 1, H3_MAX_STEPS,
            &dit->tea_cache_cooldown_steps, error, error_size) ||
        !parse_cache_unsigned(
            "H3_TEACACHE_MAX_HITS", 1, 32,
            &dit->tea_cache_max_hits, error, error_size)) return 0;
    const char *audio_value = h3_runtime_getenv("H3_TEACACHE_AUDIO_THRESHOLD");
    if (audio_value && *audio_value) {
        char *audio_tail = NULL;
        float audio_threshold = strtof(audio_value, &audio_tail);
        if (audio_tail == audio_value || *audio_tail ||
            !isfinite(audio_threshold) || audio_threshold <= 0.0f ||
            audio_threshold > 10.0f) {
            fail(error, error_size,
                 "H3_TEACACHE_AUDIO_THRESHOLD must be a finite value "
                 "in (0, 10]");
            return 0;
        }
        dit->tea_cache_audio_threshold = audio_threshold;
    }
    dit->tea_cache = 1;
    dit->tea_cache_threshold = threshold;
    return 1;
}

static int parse_sol_unsigned(const char *name, unsigned fallback,
                              unsigned maximum, unsigned *result,
                              char *error, size_t error_size) {
    const char *value = h3_runtime_getenv(name);
    if (!value || !*value) {
        *result = fallback;
        return 1;
    }
    errno = 0;
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    if (errno || tail == value || *tail || parsed > maximum) {
        fail(error, error_size, "%s must be an integer in [0, %u]",
             name, maximum);
        return 0;
    }
    *result = (unsigned)parsed;
    return 1;
}

static int configure_sol_attention(h3_dit *dit, char *error,
                                   size_t error_size) {
    const char *enabled = h3_runtime_getenv("H3_SOL_ATTN");
    if (!enabled || !*enabled || !strcmp(enabled, "0")) return 1;
    if (!h3_runtime_getenv("H3_SOL_EXPERIMENTAL")) {
        fail(error, error_size,
             "H3_SOL_ATTN selects experimental attention backends; "
             "set H3_SOL_EXPERIMENTAL=1 only for benchmark development");
        return 0;
    }
    if (dit->token_reduction || dit->tea_cache) {
        fail(error, error_size,
             "standalone Sol-Attn does not yet support token reduction or "
             "adaptive cache");
        return 0;
    }
    float tau = 1.0f;
    const char *tau_text = h3_runtime_getenv("H3_SOL_TAU");
    if (tau_text && *tau_text) {
        char *tail = NULL;
        tau = strtof(tau_text, &tail);
        if (tail == tau_text || *tail || !isfinite(tau) ||
            tau < 0.0f || tau > 100.0f) {
            fail(error, error_size,
                 "H3_SOL_TAU must be a finite value in [0, 100]");
            return 0;
        }
    }
    if (!parse_sol_unsigned("H3_SOL_DENSE_LAYERS", 2,
                            H3_DIT_BLOCKS, &dit->sol_dense_layers,
                            error, error_size) ||
        !parse_sol_unsigned("H3_SOL_DENSE_STEPS", 0, H3_MAX_STEPS,
                            &dit->sol_dense_steps, error, error_size) ||
        !parse_sol_unsigned("H3_SOL_DENSE_FINAL_STEPS", 1, H3_MAX_STEPS,
                            &dit->sol_dense_final_steps,
                            error, error_size)) return 0;
    dit->sol_attention = 1;
    dit->sol_tau = tau;
    return 1;
}

static void token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second) {
    if (reduced_row < dit->video_target_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = reduced_row - dit->video_target_start;
    uint32_t source = dit->video_target_start +
        (local / reduced_width) * spatial_width +
        (local % reduced_width) * 2;
    *first = source;
    *second = source + ((source - dit->video_target_start) % spatial_width + 1 <
                        spatial_width ? 1u : 0u);
}

static uint32_t token_reduced_parent(const h3_dit *dit, uint32_t full_row) {
    if (full_row < dit->video_target_start) return full_row;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = full_row - dit->video_target_start;
    return dit->video_target_start + (local / spatial_width) * reduced_width +
           (local % spatial_width) / 2;
}

static int load_block_matrices(h3_dit *dit, h3_dit_block *block,
                               const char *prefix, int load_mlp,
                               char *error, size_t error_size) {
    char name[160];
#define LOAD2(field, suffix, rows, columns) do {                               \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    if (load_mlp) {
        LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
        LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
    }
#undef LOAD2
    return 1;
}

static int load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

static int load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      int load_mlp, char *error, size_t error_size) {
    return load_block_norms(dit, block, prefix, error, error_size) &&
        load_block_matrices(
            dit, block, prefix, load_mlp, error, error_size);
}

static int parse_coreml_block(unsigned *selected,
                              char *error, size_t error_size) {
    const char *model = h3_runtime_getenv("H3_COREML_ANE_MODEL");
    if (!model || !*model) return 0;
    const char *value = h3_runtime_getenv("H3_COREML_ANE_BLOCK");
    if (!value || !*value) {
        *selected = 0;
        return 1;
    }
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    if (tail == value || *tail || parsed >= H3_DIT_BLOCKS) {
        fail(error, error_size,
             "H3_COREML_ANE_BLOCK must be in [0, %u]",
             H3_DIT_BLOCKS - 1);
        return -1;
    }
    *selected = (unsigned)parsed;
    return 1;
}

static int coreml_block_in_list(const char *option, const char *list,
                                unsigned index,
                                char *error, size_t error_size) {
    if (!list || !*list) {
        fail(error, error_size,
             "%s is required with its Core ML directory", option);
        return -1;
    }
    const char *cursor = list;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') cursor++;
        if (!*cursor) break;
        char *tail = NULL;
        unsigned long first = strtoul(cursor, &tail, 10);
        if (tail == cursor || first >= H3_DIT_BLOCKS) {
            fail(error, error_size, "invalid %s: %s", option, list);
            return -1;
        }
        unsigned long last = first;
        cursor = tail;
        if (*cursor == '-') {
            cursor++;
            last = strtoul(cursor, &tail, 10);
            if (tail == cursor || last < first || last >= H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "invalid %s range: %s", option, list);
                return -1;
            }
            cursor = tail;
        }
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor && *cursor != ',') {
            fail(error, error_size, "invalid %s: %s", option, list);
            return -1;
        }
        if (index >= first && index <= last) return 1;
        if (*cursor == ',') cursor++;
    }
    return 0;
}

static int coreml_mlp_enabled(unsigned index,
                              char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_COREML_ANE_DIR");
    const char *single_model = h3_runtime_getenv("H3_COREML_ANE_MODEL");
    if ((!directory || !*directory) && (!single_model || !*single_model))
        return 0;
    if (directory && *directory && single_model && *single_model) {
        fail(error, error_size,
             "set only one of H3_COREML_ANE_DIR or H3_COREML_ANE_MODEL");
        return -1;
    }
    if (directory && *directory) {
        return coreml_block_in_list(
            "H3_COREML_ANE_BLOCKS", h3_runtime_getenv("H3_COREML_ANE_BLOCKS"),
            index, error, error_size);
    }
    unsigned selected = 0;
    int configured = parse_coreml_block(&selected, error, error_size);
    if (configured <= 0) return configured;
    return index == selected;
}

static int private_ane_mlp_enabled(void) {
    const char *directory = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_DIR");
    return directory && *directory;
}

static int configure_private_ane_mlp_row_split(h3_dit *dit,
                                                char *error,
                                                size_t error_size) {
    const char *value = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_ROW_SPLIT_ROWS");
    if (!value || !*value) return 1;
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    if (!tail || *tail || !parsed || parsed >= dit->sequence ||
        parsed > UINT32_MAX || (parsed % 128u)) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_MLP_ROW_SPLIT_ROWS must be a positive "
             "multiple of 128 below sequence rows %u", dit->sequence);
        return 0;
    }
    if (dit->ane_mlp_row_split_rows &&
        dit->ane_mlp_row_split_rows != (uint32_t)parsed) {
        fail(error, error_size,
             "private ANE MLP row split changed while loading blocks");
        return 0;
    }
    dit->ane_mlp_row_split_rows = (uint32_t)parsed;
    return 1;
}

static int configure_private_ane_mlp_range_guard(h3_dit *dit,
                                                 char *error,
                                                 size_t error_size) {
    if (dit->ane_mlp_range_guard_ready) return 1;
    float headroom = 40000.0f;
    const char *headroom_value =
        h3_runtime_getenv("H3_PRIVATE_ANE_MLP_RANGE_HEADROOM");
    if (headroom_value && *headroom_value) {
        char *tail = NULL;
        errno = 0;
        headroom = strtof(headroom_value, &tail);
        if (tail == headroom_value || *tail || errno == ERANGE ||
            !isfinite(headroom) || headroom < 0.0f || headroom > 65504.0f) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_RANGE_HEADROOM must be in "
                 "[0, 65504]: %s", headroom_value);
            return 0;
        }
    }
    unsigned long retry_limit = 4;
    const char *retry_value =
        h3_runtime_getenv("H3_PRIVATE_ANE_MLP_RANGE_RETRIES");
    if (retry_value && *retry_value) {
        char *tail = NULL;
        errno = 0;
        retry_limit = strtoul(retry_value, &tail, 10);
        if (tail == retry_value || *tail || errno == ERANGE ||
            retry_limit > 8) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_RANGE_RETRIES must be in [0, 8]: %s",
                 retry_value);
            return 0;
        }
    }
    dit->ane_mlp_range_headroom = headroom;
    dit->ane_mlp_range_retry_limit = (unsigned)retry_limit;
    dit->ane_mlp_range_guard_ready = 1;
    return 1;
}

static int private_ane_mlp_block_enabled(unsigned index,
                                         char *error, size_t error_size) {
    if (!private_ane_mlp_enabled()) return 0;
    const char *blocks = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_BLOCKS");
    if (!blocks || !*blocks) return 1;
    return coreml_block_in_list(
        "H3_PRIVATE_ANE_MLP_BLOCKS", blocks, index, error, error_size);
}

static int private_ane_mlp_scale_valid(float scale) {
    int exponent = 0;
    return isfinite(scale) && scale >= 1.0f &&
        frexpf(scale, &exponent) == 0.5f;
}

static int private_ane_mlp_runtime_scale_for_block(
        unsigned index, float *scale,
        char *error, size_t error_size) {
    *scale = 1.0f;
    const char *global = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_RUNTIME_SCALE");
    if (global && *global) {
        char *tail = NULL;
        errno = 0;
        float parsed = strtof(global, &tail);
        if (tail == global || *tail || errno == ERANGE ||
            !private_ane_mlp_scale_valid(parsed)) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_RUNTIME_SCALE must be a power of "
                 "two >= 1: %s", global);
            return 0;
        }
        *scale = parsed;
    }

    const char *profile =
        h3_runtime_getenv("H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE");
    if (!profile || !*profile) return 1;
    const char *cursor = profile;
    int matched = 0;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',')
            cursor++;
        if (!*cursor) break;
        char *tail = NULL;
        errno = 0;
        unsigned long first = strtoul(cursor, &tail, 10);
        if (tail == cursor || errno == ERANGE || first >= H3_DIT_BLOCKS) {
            fail(error, error_size,
                 "invalid H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE: %s",
                 profile);
            return 0;
        }
        unsigned long last = first;
        cursor = tail;
        if (*cursor == '-') {
            cursor++;
            errno = 0;
            last = strtoul(cursor, &tail, 10);
            if (tail == cursor || errno == ERANGE || last < first ||
                last >= H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "invalid H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE "
                     "range: %s", profile);
                return 0;
            }
            cursor = tail;
        }
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor != ':') {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE entries must "
                 "use block[-block]:scale: %s", profile);
            return 0;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        errno = 0;
        float parsed = strtof(cursor, &tail);
        if (tail == cursor || errno == ERANGE ||
            !private_ane_mlp_scale_valid(parsed)) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE scales must "
                 "be powers of two >= 1: %s", profile);
            return 0;
        }
        cursor = tail;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor && *cursor != ',') {
            fail(error, error_size,
                 "invalid H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE: %s",
                 profile);
            return 0;
        }
        if (index >= first && index <= last) {
            if (matched) {
                fail(error, error_size,
                     "overlapping block %u in "
                     "H3_PRIVATE_ANE_MLP_RUNTIME_SCALE_PROFILE: %s",
                     index, profile);
                return 0;
            }
            *scale = parsed;
            matched = 1;
        }
        if (*cursor == ',') cursor++;
    }
    return 1;
}

static int configure_private_ane_mlp(h3_dit *dit, h3_dit_block *block,
                                     unsigned index,
                                     char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_DIR");
    if (!directory || !*directory) return 1;
    if (!configure_private_ane_mlp_row_split(
            dit, error, error_size) ||
        !configure_private_ane_mlp_range_guard(
            dit, error, error_size)) return 0;
    uint32_t ane_rows = dit->ane_mlp_row_split_rows ?
        dit->ane_mlp_row_split_rows : dit->sequence;
    int row_split = dit->ane_mlp_row_split_rows != 0;
    if (dit->ssd_streaming || dit->token_reduction || dit->nax_mlp ||
        dit->first_block_cache) {
        fail(error, error_size,
             "experimental private ANE MLP requires resident fixed-row "
             "GPU complement weights without FirstBlockCache");
        return 0;
    }
    if (h3_runtime_getenv("H3_COREML_ANE_DIR") || h3_runtime_getenv("H3_COREML_ANE_MODEL")) {
        fail(error, error_size,
             "private ANE MLP and Core ML MLP cannot be enabled together");
        return 0;
    }
    if (!h3_ane_mlp_available()) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_MLP_DIR is set but the private ANE bridge "
             "is unavailable");
        return 0;
    }
    if (!dit->ane_mlp_io) {
        dit->ane_mlp_io = h3_ane_mlp_io_create(
            dit->gpu, ane_rows, HIDDEN, error, error_size);
        dit->ane_nonfinite = h3_gpu_tensor_new_f32(dit->gpu, 2);
        if (!dit->ane_mlp_io || !dit->ane_nonfinite) {
            if (!error || !error[0])
                fail(error, error_size,
                     "cannot allocate private ANE MLP shared IO: %s",
                     h3_gpu_error(dit->gpu));
            return 0;
        }
        memset(h3_gpu_tensor_host_pointer(dit->ane_nonfinite), 0,
               2u * sizeof(uint32_t));
    }
    char block_directory[PATH_MAX];
    char gpu_fc1_path[PATH_MAX];
    char gpu_fc2_path[PATH_MAX];
    char ane_fc1_path[PATH_MAX];
    char ane_fc2_path[PATH_MAX];
    int block_count = snprintf(block_directory, sizeof(block_directory),
                               "%s/block-%u", directory, index);
    int fc1_count = snprintf(gpu_fc1_path, sizeof(gpu_fc1_path),
                             "%s/gpu_fc1.bf16", block_directory);
    int fc2_count = snprintf(gpu_fc2_path, sizeof(gpu_fc2_path),
                             "%s/gpu_fc2.bf16", block_directory);
    int ane_fc1_count = snprintf(ane_fc1_path, sizeof(ane_fc1_path),
                                 "%s/ane_fc1.bf16", block_directory);
    int ane_fc2_count = snprintf(ane_fc2_path, sizeof(ane_fc2_path),
                                 "%s/ane_fc2.bf16", block_directory);
    if (block_count < 0 || (size_t)block_count >= sizeof(block_directory) ||
        fc1_count < 0 || (size_t)fc1_count >= sizeof(gpu_fc1_path) ||
        fc2_count < 0 || (size_t)fc2_count >= sizeof(gpu_fc2_path) ||
        ane_fc1_count < 0 ||
        (size_t)ane_fc1_count >= sizeof(ane_fc1_path) ||
        ane_fc2_count < 0 ||
        (size_t)ane_fc2_count >= sizeof(ane_fc2_path)) {
        fail(error, error_size, "private ANE MLP artifact path is too long");
        return 0;
    }
    const char *precision = h3_runtime_getenv("H3_PRIVATE_ANE_MLP_PRECISION");
    int int8_weights = precision && *precision && !strcmp(precision, "int8");
    if (precision && *precision && !int8_weights &&
        strcmp(precision, "fp16")) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_MLP_PRECISION must be fp16 or int8");
        return 0;
    }
    h3_ane_mlp_plan plan;
    h3_ane_mlp *model = int8_weights ?
        h3_ane_mlp_create_int8_plan(
            block_directory, index, ane_rows, HIDDEN, dit->ane_mlp_io,
            &plan, error, error_size) :
        h3_ane_mlp_create_bf16_plan(
            block_directory, index, ane_rows, HIDDEN, dit->ane_mlp_io,
            &plan, error, error_size);
    if (!model) return 0;
    float runtime_scale = 1.0f;
    if (!private_ane_mlp_runtime_scale_for_block(
            index, &runtime_scale, error, error_size) ||
        !h3_ane_mlp_set_runtime_scale(
            model, runtime_scale, error, error_size)) {
        h3_ane_mlp_free(model);
        return 0;
    }
    if (plan.full_intermediate != FFN) {
        h3_ane_mlp_free(model);
        fail(error, error_size,
             "private ANE block %u full intermediate is %u, expected %u",
             index, plan.full_intermediate, FFN);
        return 0;
    }
    if (row_split &&
        (plan.ane_intermediate != FFN || plan.gpu_intermediate != 0)) {
        h3_ane_mlp_free(model);
        fail(error, error_size,
             "private ANE row-split block %u requires full-width ANE "
             "weights and an empty GPU complement", index);
        return 0;
    }
    uint32_t gpu_intermediate = row_split ? FFN : plan.gpu_intermediate;
    const char *gpu_fc1_source = row_split ? ane_fc1_path : gpu_fc1_path;
    const char *gpu_fc2_source = row_split ? ane_fc2_path : gpu_fc2_path;
    h3_gpu_tensor *fc1 = h3_gpu_tensor_load_bf16(
        dit->gpu, gpu_fc1_source, 0,
        (size_t)gpu_intermediate * 2u * HIDDEN);
    h3_gpu_tensor *fc2 = h3_gpu_tensor_load_bf16(
        dit->gpu, gpu_fc2_source, 0,
        (size_t)HIDDEN * gpu_intermediate);
    if (!fc1 || !fc2) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(fc2);
        h3_ane_mlp_free(model);
        fail(error, error_size,
             "cannot load private ANE GPU complement block %u: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!h3_ane_mlp_unload(model, error, error_size)) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(fc2);
        h3_ane_mlp_free(model);
        return 0;
    }
    block->ane_mlp = model;
    block->fc1 = fc1;
    block->fc2 = fc2;
    block->ane_gpu_intermediate = gpu_intermediate;
    block->ane_int8_weights = int8_weights;
    if (dit->ane_gpu_int8_mlp) {
        if ((gpu_intermediate % 1024u) ||
            h3_runtime_getenv("H3_INT8_MLP_STAGE")) {
            fail(error, error_size,
                 "private ANE block %u GPU INT8 complement width %u must "
                 "be divisible by 1024 and cannot use H3_INT8_MLP_STAGE",
                 index, gpu_intermediate);
            return 0;
        }
        if (!quantize_block_mlp_width(
                dit, block, gpu_intermediate, 1,
                error, error_size)) return 0;
    }
    dit->ane_mlp_blocks++;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: private ANE GPU+ANE MLP block=%u ANE-%s-F=%u "
                "GPU-%s-F=%u rows=%u/%u%s fc2-chunks=%u "
                "output-scale=%.6g runtime-scale=%.6g cached=%d "
                "blob-cached=%d compile-or-load=%.6fs\n",
                index, int8_weights ? "int8" : "fp16",
                plan.ane_intermediate,
                dit->ane_gpu_int8_mlp ? "INT8" : "BF16",
                gpu_intermediate, ane_rows,
                row_split ? dit->sequence - ane_rows : dit->sequence,
                row_split ? " row-split" : "",
                h3_ane_mlp_fc2_chunks(model), plan.output_scale,
                h3_ane_mlp_runtime_scale(model),
                h3_ane_mlp_cache_hit(model),
                h3_ane_mlp_blob_cache_hit(model),
                h3_ane_mlp_compile_seconds(model));
    return 1;
}

static int coreml_qkv_enabled(unsigned index,
                              char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_COREML_QKV_DIR");
    if (!directory || !*directory) return 0;
    return coreml_block_in_list(
        "H3_COREML_QKV_BLOCKS", h3_runtime_getenv("H3_COREML_QKV_BLOCKS"),
        index, error, error_size);
}

static int private_ane_qkv_enabled(unsigned index,
                                   char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_DIR");
    const char *checkpoint = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_CHECKPOINT");
    if ((!directory || !*directory) &&
        (!checkpoint || !*checkpoint || !strcmp(checkpoint, "0"))) return 0;
    return coreml_block_in_list(
        "H3_PRIVATE_ANE_QKV_BLOCKS", h3_runtime_getenv("H3_PRIVATE_ANE_QKV_BLOCKS"),
        index, error, error_size);
}

static int private_ane_qkv_micro_enabled(void) {
    const char *value = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_MICRO");
    return value && *value && strcmp(value, "0");
}

static int private_ane_qkv_micro_selected(int step,
                                          char *error, size_t error_size) {
    if (!private_ane_qkv_micro_enabled()) return 0;
    const char *value = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_MICRO_STEP");
    if (!value || !*value) return 1;
    char *tail = NULL;
    long parsed = strtol(value, &tail, 10);
    if (tail == value || *tail || parsed < 0 || parsed >= H3_MAX_STEPS) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_QKV_MICRO_STEP must be in [0, %d)",
             H3_MAX_STEPS);
        return -1;
    }
    return step == (int)parsed;
}

static int private_ane_attention_out_enabled(
                                   unsigned index,
                                   char *error, size_t error_size) {
    const char *checkpoint =
        h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_CHECKPOINT");
    if (!checkpoint || !*checkpoint || !strcmp(checkpoint, "0")) return 0;
    return coreml_block_in_list(
        "H3_PRIVATE_ANE_ATTENTION_OUT_BLOCKS",
        h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_BLOCKS"),
        index, error, error_size);
}

static int private_ane_attention_out_micro_enabled(void) {
    const char *value = h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_MICRO");
    return value && *value && strcmp(value, "0");
}

static int private_ane_attention_out_transient_enabled(void) {
    return h3_ane_linear_transient_requested(
        h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_TRANSIENT"));
}

static int configure_private_ane_qkv(h3_dit *dit, h3_dit_block *block,
                                     unsigned index,
                                     char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_DIR");
    const char *checkpoint_value =
        h3_runtime_getenv("H3_PRIVATE_ANE_QKV_CHECKPOINT");
    int checkpoint_source = checkpoint_value && *checkpoint_value &&
        strcmp(checkpoint_value, "0");
    if (((!directory || !*directory) && !checkpoint_source) ||
        dit->ssd_streaming) {
        fail(error, error_size,
             "experimental private ANE QKV requires resident QKV weights");
        return 0;
    }
    if (h3_runtime_getenv("H3_COREML_QKV_DIR")) {
        fail(error, error_size,
             "private ANE QKV and Core ML QKV cannot be enabled together");
        return 0;
    }
    if (!h3_ane_linear_available()) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_QKV_DIR is set but the private ANE bridge "
             "is unavailable");
        return 0;
    }
    unsigned ane_heads = 16;
    const char *heads_value = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_HEADS");
    if (heads_value && *heads_value) {
        char *tail = NULL;
        unsigned long parsed = strtoul(heads_value, &tail, 10);
        if (tail == heads_value || *tail || parsed < 1 || parsed >= HEADS) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_QKV_HEADS must be in [1, %u]",
                 HEADS - 1);
            return 0;
        }
        ane_heads = (unsigned)parsed;
    }
    uint32_t ane_width = ane_heads * 3u * HEAD_DIM;
    uint32_t gpu_width = (HEADS - ane_heads) * 3u * HEAD_DIM;
    int micro = private_ane_qkv_micro_enabled();
    if (!dit->ane_qkv_io) {
        dit->ane_qkv_io = h3_ane_linear_io_create(
            dit->gpu, dit->sequence, HIDDEN, ane_width,
            error, error_size);
        dit->coreml_qkv_gpu_output = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)dit->sequence * gpu_width);
        if (!dit->coreml_nonfinite)
            dit->coreml_nonfinite = h3_gpu_tensor_new_f32(dit->gpu, 1);
        size_t micro_elements = (size_t)dit->sequence * ane_width;
        if (micro && micro_elements <= UINT32_MAX) {
            dit->ane_qkv_micro_partial_count =
                h3_gpu_compare_ane_f32_bf16_partial_count(
                    (uint32_t)micro_elements);
            dit->ane_qkv_micro_partials = h3_gpu_tensor_new_f32(
                dit->gpu,
                (size_t)dit->ane_qkv_micro_partial_count * 8);
        }
        if (!dit->ane_qkv_io || !dit->coreml_qkv_gpu_output ||
            !dit->coreml_nonfinite ||
            (micro && (!dit->ane_qkv_micro_partial_count ||
                       !dit->ane_qkv_micro_partials))) {
            if (!error || !error[0])
                fail(error, error_size,
                     "cannot allocate private ANE QKV shared IO: %s",
                     h3_gpu_error(dit->gpu));
            return 0;
        }
        memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
               sizeof(uint32_t));
    } else if (h3_ane_linear_io_output_width(dit->ane_qkv_io) != ane_width ||
               h3_gpu_tensor_bytes(dit->coreml_qkv_gpu_output) !=
                   (size_t)dit->sequence * gpu_width * sizeof(uint16_t)) {
        fail(error, error_size,
             "all private ANE QKV blocks must use the same head split");
        return 0;
    }

    char ane_path[PATH_MAX] = {0};
    char gpu_path[PATH_MAX] = {0};
    char full_path[PATH_MAX] = {0};
    const char *source_path = NULL;
    uint64_t source_offset = 0;
    const char *source_label = NULL;
    int use_full = 0;
    uint64_t full_bytes =
        (uint64_t)HEADS * 3u * HEAD_DIM * HIDDEN * sizeof(uint16_t);
    if (checkpoint_source) {
        char weight_name[160];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.attn.qkv_proj.weight", index);
        const h3_st_header *header = NULL;
        const h3_st_tensor *tensor = h3_weight_find(
            dit->weights, weight_name, &header);
        if (!header || !tensor || tensor->dtype != H3_DTYPE_BF16 ||
            tensor->ndim != 2 || tensor->shape[0] != INNER * 3 ||
            tensor->shape[1] != HIDDEN) {
            fail(error, error_size,
                 "private ANE QKV checkpoint tensor has the wrong schema: %s",
                 weight_name);
            return 0;
        }
        source_path = header->path;
        source_offset = tensor->file_offset;
        source_label = "checkpoint-range";
        use_full = 1;
    } else {
        int ane_count = snprintf(ane_path, sizeof(ane_path),
                                 "%s/block-%u/ane_linear.bf16",
                                 directory, index);
        int gpu_count = snprintf(gpu_path, sizeof(gpu_path),
                                 "%s/block-%u/gpu_linear.bf16",
                                 directory, index);
        int full_count = snprintf(full_path, sizeof(full_path),
                                  "%s/block-%u/full_linear.bf16",
                                  directory, index);
        if (ane_count < 0 || (size_t)ane_count >= sizeof(ane_path) ||
            gpu_count < 0 || (size_t)gpu_count >= sizeof(gpu_path) ||
            full_count < 0 || (size_t)full_count >= sizeof(full_path)) {
            fail(error, error_size,
                 "private ANE QKV artifact path is too long");
            return 0;
        }
        struct stat full_status;
        use_full = stat(full_path, &full_status) == 0 &&
            full_status.st_size >= 0 &&
            (uint64_t)full_status.st_size == full_bytes;
        source_path = use_full ? full_path : ane_path;
        source_label = use_full ? "artifact-range" : "split";
    }
    const char *precision = h3_runtime_getenv("H3_PRIVATE_ANE_QKV_PRECISION");
    int int8_weights = precision && *precision && !strcmp(precision, "int8");
    if (precision && *precision && !int8_weights &&
        strcmp(precision, "fp16")) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_QKV_PRECISION must be fp16 or int8");
        return 0;
    }
    const char *gpu_precision =
        h3_runtime_getenv("H3_PRIVATE_ANE_QKV_GPU_PRECISION");
    int gpu_int8_weights = gpu_precision && *gpu_precision &&
        !strcmp(gpu_precision, "int8");
    if (gpu_precision && *gpu_precision && !gpu_int8_weights &&
        strcmp(gpu_precision, "bf16")) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_QKV_GPU_PRECISION must be bf16 or int8");
        return 0;
    }
    if (gpu_int8_weights && !h3_gpu_has_int8_mlp(dit->gpu)) {
        fail(error, error_size,
             "private ANE QKV GPU INT8 complement requires Metal "
             "TensorOps support");
        return 0;
    }
    char name[64];
    snprintf(name, sizeof(name), "h3-private-ane-qkv-block-%u", index);
    h3_ane_linear *model = use_full ?
        (int8_weights ?
            h3_ane_linear_create_int8_file_range(
                name, source_path, source_offset, dit->ane_qkv_io,
                error, error_size) :
            h3_ane_linear_create_bf16_file_range(
                name, source_path, source_offset, dit->ane_qkv_io,
                error, error_size)) :
        (int8_weights ?
            h3_ane_linear_create_int8_file(
                name, ane_path, dit->ane_qkv_io, error, error_size) :
            h3_ane_linear_create_bf16_file(
                name, ane_path, dit->ane_qkv_io, error, error_size));
    if (!model) return 0;
    h3_gpu_tensor *gpu_weight_bf16 = h3_gpu_tensor_load_bf16(
        dit->gpu, use_full ? source_path : gpu_path,
        use_full ? source_offset +
            (uint64_t)ane_width * HIDDEN * sizeof(uint16_t) : 0,
        (size_t)gpu_width * HIDDEN);
    if (!gpu_weight_bf16) {
        h3_ane_linear_free(model);
        fail(error, error_size,
             "cannot load private ANE QKV GPU complement: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    h3_gpu_tensor *gpu_weight = gpu_weight_bf16;
    h3_gpu_tensor *gpu_scales = NULL;
    if (gpu_int8_weights) {
        gpu_weight = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)gpu_width * HIDDEN);
        gpu_scales = h3_gpu_tensor_new_f32(dit->gpu, gpu_width);
        int quantized = gpu_weight && gpu_scales &&
            h3_gpu_begin(dit->gpu) &&
            h3_gpu_quantize_weight_int8(
                dit->gpu, gpu_weight, gpu_scales, gpu_weight_bf16,
                gpu_width, HIDDEN) &&
            h3_gpu_submit(dit->gpu);
        h3_gpu_tensor_free(gpu_weight_bf16);
        gpu_weight_bf16 = NULL;
        if (!quantized) {
            h3_gpu_tensor_free(gpu_weight);
            h3_gpu_tensor_free(gpu_scales);
            h3_ane_linear_free(model);
            fail(error, error_size,
                 "cannot quantize private ANE QKV GPU complement: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!h3_ane_linear_unload(model, error, error_size)) {
        h3_gpu_tensor_free(gpu_weight);
        h3_gpu_tensor_free(gpu_scales);
        h3_ane_linear_free(model);
        return 0;
    }
    block->ane_qkv = model;
    block->coreml_qkv_gpu_weight = gpu_weight;
    block->ane_qkv_gpu_scales = gpu_scales;
    block->ane_qkv_heads = ane_heads;
    block->ane_qkv_int8_weights = int8_weights;
    block->ane_qkv_gpu_int8_weights = gpu_int8_weights;
    if (micro && dit->int8_qkv) {
        block->qkv_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)INNER * 3 * HIDDEN);
        block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
        if (!block->qkv_int8 || !block->qkv_scales ||
            !h3_gpu_begin(dit->gpu) ||
            !h3_gpu_quantize_weight_int8(
                dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                INNER * 3, HIDDEN) ||
            !h3_gpu_submit(dit->gpu)) {
            fail(error, error_size,
                 "cannot quantize private ANE QKV micro baseline: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    dit->ane_qkv_blocks++;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: private ANE GPU+ANE QKV block=%u ANE-%s-heads=%u "
                "GPU-%s-heads=%u source=%s cached=%d "
                "blob-cached=%d compile-or-load=%.6fs\n",
                index, int8_weights ? "int8" : "fp16", ane_heads,
                gpu_int8_weights ? "INT8" : "BF16",
                HEADS - ane_heads, source_label,
                h3_ane_linear_cache_hit(model),
                h3_ane_linear_blob_cache_hit(model),
                h3_ane_linear_compile_seconds(model));
    if (checkpoint_source && !h3_runtime_getenv("H3_BENCH_COREML_QKV_AB") && !micro)
        free_tensor(&block->qkv);
    return 1;
}

static h3_ane_linear *create_private_ane_attention_out_model(
                                     h3_dit *dit, h3_dit_block *block,
                                     unsigned index,
                                     char *error, size_t error_size) {
    if (!dit || !block || !dit->ane_attention_out_io ||
        !h3_ane_linear_recipe_present(
            &block->ane_attention_out_recipe)) {
        fail(error, error_size,
             "private ANE attention-output block %u has no model recipe",
             index);
        return NULL;
    }
    char name[72];
    snprintf(name, sizeof(name),
             "h3-private-ane-attention-out-block-%u", index);
    return block->ane_attention_out_int8_weights ?
            h3_ane_linear_create_int8_file_range(
            name, block->ane_attention_out_recipe.weight_path,
            block->ane_attention_out_recipe.file_offset,
            dit->ane_attention_out_io, error, error_size) :
        h3_ane_linear_create_bf16_file_range(
            name, block->ane_attention_out_recipe.weight_path,
            block->ane_attention_out_recipe.file_offset,
            dit->ane_attention_out_io, error, error_size);
}

static int configure_private_ane_attention_out(
                                     h3_dit *dit, h3_dit_block *block,
                                     unsigned index,
                                     char *error, size_t error_size) {
    if (dit->ssd_streaming) {
        fail(error, error_size,
             "experimental private ANE attention output requires resident "
             "BF16 weights");
        return 0;
    }
    if (!h3_ane_linear_available()) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_ATTENTION_OUT_CHECKPOINT is set but the "
             "private ANE bridge is unavailable");
        return 0;
    }
    uint32_t ane_width = 1152;
    const char *width_value = h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_WIDTH");
    if (width_value && *width_value) {
        char *tail = NULL;
        unsigned long parsed = strtoul(width_value, &tail, 10);
        if (tail == width_value || *tail || parsed < 64 ||
            parsed >= HIDDEN || parsed % 64) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_ATTENTION_OUT_WIDTH must be a multiple "
                 "of 64 in [64, %u]", HIDDEN - 64);
            return 0;
        }
        ane_width = (uint32_t)parsed;
    }
    uint32_t gpu_width = HIDDEN - ane_width;
    int micro = private_ane_attention_out_micro_enabled();
    if (!dit->ane_attention_out_io) {
        dit->ane_attention_out_io = h3_ane_linear_io_create(
            dit->gpu, dit->sequence, INNER, ane_width,
            error, error_size);
        dit->ane_attention_out_gpu_output = h3_gpu_tensor_new_bf16(
            dit->gpu, (size_t)dit->sequence * gpu_width);
        if (!dit->coreml_nonfinite)
            dit->coreml_nonfinite = h3_gpu_tensor_new_f32(dit->gpu, 1);
        size_t micro_elements = (size_t)dit->sequence * ane_width;
        if (micro && micro_elements <= UINT32_MAX) {
            dit->ane_attention_out_micro_reference =
                h3_gpu_tensor_new_bf16(
                    dit->gpu, (size_t)dit->sequence * HIDDEN);
            dit->ane_attention_out_micro_partial_count =
                h3_gpu_compare_ane_f32_bf16_partial_count(
                    (uint32_t)micro_elements);
            dit->ane_attention_out_micro_partials = h3_gpu_tensor_new_f32(
                dit->gpu,
                (size_t)dit->ane_attention_out_micro_partial_count * 8);
        }
        if (!dit->ane_attention_out_io ||
            !dit->ane_attention_out_gpu_output ||
            !dit->coreml_nonfinite ||
            (micro && (!dit->ane_attention_out_micro_reference ||
                       !dit->ane_attention_out_micro_partial_count ||
                       !dit->ane_attention_out_micro_partials))) {
            if (!error || !error[0])
                fail(error, error_size,
                     "cannot allocate private ANE attention-output state: %s",
                     h3_gpu_error(dit->gpu));
            return 0;
        }
        memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
               sizeof(uint32_t));
    } else if (h3_ane_linear_io_output_width(
                   dit->ane_attention_out_io) != ane_width ||
               h3_gpu_tensor_bytes(dit->ane_attention_out_gpu_output) !=
                   (size_t)dit->sequence * gpu_width * sizeof(uint16_t)) {
        fail(error, error_size,
             "all private ANE attention-output blocks must use the same "
             "output-channel split");
        return 0;
    }

    char weight_name[160];
    snprintf(weight_name, sizeof(weight_name),
             "blocks.%u.attn.out_proj.weight", index);
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(
        dit->weights, weight_name, &header);
    if (!header || !tensor || tensor->dtype != H3_DTYPE_BF16 ||
        tensor->ndim != 2 || tensor->shape[0] != HIDDEN ||
        tensor->shape[1] != INNER) {
        fail(error, error_size,
             "private ANE attention-output checkpoint tensor has the wrong "
             "schema: %s", weight_name);
        return 0;
    }
    const char *precision =
        h3_runtime_getenv("H3_PRIVATE_ANE_ATTENTION_OUT_PRECISION");
    int int8_weights = precision && *precision && !strcmp(precision, "int8");
    if (precision && *precision && !int8_weights &&
        strcmp(precision, "fp16")) {
        fail(error, error_size,
             "H3_PRIVATE_ANE_ATTENTION_OUT_PRECISION must be fp16 or int8");
        return 0;
    }
    block->ane_attention_out_width = ane_width;
    block->ane_attention_out_int8_weights = int8_weights;
    block->ane_attention_out_transient =
        private_ane_attention_out_transient_enabled();
    if (!h3_ane_linear_recipe_set(
            &block->ane_attention_out_recipe, header->path,
            tensor->file_offset, error, error_size)) {
        return 0;
    }
    h3_ane_linear *model = create_private_ane_attention_out_model(
        dit, block, index, error, error_size);
    if (!model) {
        h3_ane_linear_recipe_clear(&block->ane_attention_out_recipe);
        return 0;
    }
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_load_bf16(
        dit->gpu, header->path,
        tensor->file_offset +
            (uint64_t)ane_width * INNER * sizeof(uint16_t),
        (size_t)gpu_width * INNER);
    if (!gpu_weight) {
        h3_ane_linear_free(model);
        h3_ane_linear_recipe_clear(&block->ane_attention_out_recipe);
        fail(error, error_size,
             "cannot load private ANE attention-output GPU complement: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    int cache_hit = h3_ane_linear_cache_hit(model);
    int blob_cache_hit = h3_ane_linear_blob_cache_hit(model);
    double compile_seconds = h3_ane_linear_compile_seconds(model);
    if (!block->ane_attention_out_transient &&
        !h3_ane_linear_unload(model, error, error_size)) {
        h3_gpu_tensor_free(gpu_weight);
        h3_ane_linear_free(model);
        h3_ane_linear_recipe_clear(&block->ane_attention_out_recipe);
        return 0;
    }
    if (block->ane_attention_out_transient)
        h3_ane_linear_free(model);
    else
        block->ane_attention_out = model;
    block->ane_attention_out_gpu_weight = gpu_weight;
    dit->ane_attention_out_blocks++;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: private ANE GPU+ANE attention output block=%u "
                "ANE-%s-width=%u GPU-BF16-width=%u cached=%d "
                "blob-cached=%d compile-or-load=%.6fs%s\n",
                index, int8_weights ? "int8" : "fp16", ane_width,
                gpu_width, cache_hit, blob_cache_hit, compile_seconds,
                block->ane_attention_out_transient ? " transient" : "");
    return 1;
}

static int configure_coreml_qkv(h3_dit *dit, h3_dit_block *block,
                                unsigned index,
                                char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_COREML_QKV_DIR");
    if (!directory || !*directory || dit->ssd_streaming || dit->int8_qkv) {
        fail(error, error_size,
             "experimental Core ML QKV requires resident BF16 QKV weights");
        return 0;
    }
    unsigned ane_heads = 24;
    const char *heads_value = h3_runtime_getenv("H3_COREML_QKV_HEADS");
    if (heads_value && *heads_value) {
        char *tail = NULL;
        unsigned long parsed = strtoul(heads_value, &tail, 10);
        if (tail == heads_value || *tail || parsed < 4 || parsed >= HEADS ||
            parsed % 4) {
            fail(error, error_size,
                 "H3_COREML_QKV_HEADS must be a multiple of 4 in [4, %u]",
                 HEADS - 1);
            return 0;
        }
        ane_heads = (unsigned)parsed;
    }
    uint32_t ane_width = ane_heads * 3u * HEAD_DIM;
    uint32_t gpu_heads = HEADS - ane_heads;
    uint32_t gpu_width = gpu_heads * 3u * HEAD_DIM;
    char model_path[PATH_MAX];
    char weight_path[PATH_MAX];
    int model_count = snprintf(model_path, sizeof(model_path),
                               "%s/block-%u.mlpackage", directory, index);
    int weight_count = snprintf(weight_path, sizeof(weight_path),
                                "%s/block-%u/gpu_linear.bf16",
                                directory, index);
    if (model_count < 0 || (size_t)model_count >= sizeof(model_path) ||
        weight_count < 0 || (size_t)weight_count >= sizeof(weight_path)) {
        fail(error, error_size, "Core ML QKV artifact path is too long");
        return 0;
    }
    block->coreml_qkv_gpu_weight = h3_gpu_tensor_load_bf16(
        dit->gpu, weight_path, 0, (size_t)gpu_width * HIDDEN);
    if (!block->coreml_qkv_gpu_weight) {
        fail(error, error_size,
             "cannot load Core ML QKV GPU complement: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->coreml_qkv_input) {
        size_t input_elements = (size_t)dit->sequence * HIDDEN;
        size_t ane_elements = (size_t)dit->sequence * ane_width;
        size_t gpu_elements = (size_t)dit->sequence * gpu_width;
        dit->coreml_qkv_input = h3_gpu_tensor_new_f16(
            dit->gpu, input_elements);
        dit->coreml_qkv_output = h3_gpu_tensor_new_f16(
            dit->gpu, ane_elements);
        dit->coreml_qkv_gpu_output = h3_gpu_tensor_new_bf16(
            dit->gpu, gpu_elements);
        if (!dit->coreml_nonfinite)
            dit->coreml_nonfinite = h3_gpu_tensor_new_f32(dit->gpu, 1);
        if (!dit->coreml_qkv_input || !dit->coreml_qkv_output ||
            !dit->coreml_qkv_gpu_output || !dit->coreml_nonfinite) {
            fail(error, error_size,
                 "cannot allocate shared Core ML QKV tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        memset(h3_gpu_tensor_host_pointer(dit->coreml_qkv_input), 0,
               input_elements * sizeof(uint16_t));
        memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
               sizeof(uint32_t));
    } else {
        size_t expected_output =
            (size_t)dit->sequence * ane_width * sizeof(uint16_t);
        size_t expected_gpu =
            (size_t)dit->sequence * gpu_width * sizeof(uint16_t);
        if (h3_gpu_tensor_bytes(dit->coreml_qkv_output) != expected_output ||
            h3_gpu_tensor_bytes(dit->coreml_qkv_gpu_output) != expected_gpu) {
            fail(error, error_size,
                 "all Core ML QKV blocks must use the same head split");
            return 0;
        }
    }
    block->coreml_qkv = h3_coreml_mlp_create_async_shared_io(
        dit->gpu, model_path, dit->sequence, HIDDEN, ane_width,
        dit->coreml_qkv_input, dit->coreml_qkv_output,
        error, error_size);
    if (!block->coreml_qkv) return 0;
    block->coreml_qkv_heads = ane_heads;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: Core ML GPU+ANE QKV block=%u ANE-heads=%u "
                "GPU-heads=%u direct-segment-consumer=1\n",
                index, ane_heads, gpu_heads);
    return 1;
}

static int coreml_manifest_sibling_path(
                                char *output, size_t output_size,
                                const char *manifest, const char *filename,
                                char *error, size_t error_size) {
    const char *slash = strrchr(manifest, '/');
    size_t prefix = slash ? (size_t)(slash - manifest) + 1 : 0;
    size_t name_bytes = strlen(filename) + 1;
    if (prefix > output_size || name_bytes > output_size - prefix) {
        fail(error, error_size, "Core ML fallback weight path is too long");
        return 0;
    }
    if (prefix) memcpy(output, manifest, prefix);
    memcpy(output + prefix, filename, name_bytes);
    return 1;
}

static int configure_coreml_mlp(h3_dit *dit, h3_dit_block *block,
                                unsigned index,
                                char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_COREML_ANE_DIR");
    const char *single_model = h3_runtime_getenv("H3_COREML_ANE_MODEL");
    if (dit->ssd_streaming || dit->int8_mlp || dit->nax_mlp) {
        fail(error, error_size,
             "experimental Core ML MLP requires resident BF16 MLP weights");
        return 0;
    }
    char model_path[PATH_MAX];
    char gpu_fc1_path[PATH_MAX];
    char gpu_fc2_path[PATH_MAX];
    char ane_fc1_path[PATH_MAX];
    char ane_fc2_path[PATH_MAX];
    char manifest_path[PATH_MAX];
    char checkpoint_index_path[PATH_MAX];
    const char *model = single_model;
    const char *gpu_fc1 = h3_runtime_getenv("H3_COREML_ANE_GPU_FC1");
    const char *gpu_fc2 = h3_runtime_getenv("H3_COREML_ANE_GPU_FC2");
    const char *ane_fc1 = NULL;
    const char *ane_fc2 = NULL;
    const char *manifest = h3_runtime_getenv("H3_COREML_ANE_MANIFEST");
    if (directory && *directory) {
        int model_count = snprintf(model_path, sizeof(model_path),
                                   "%s/block-%u.mlpackage", directory, index);
        int fc1_count = snprintf(gpu_fc1_path, sizeof(gpu_fc1_path),
                                 "%s/block-%u/gpu_fc1.bf16",
                                 directory, index);
        int fc2_count = snprintf(gpu_fc2_path, sizeof(gpu_fc2_path),
                                 "%s/block-%u/gpu_fc2.bf16",
                                 directory, index);
        int ane_fc1_count = snprintf(ane_fc1_path, sizeof(ane_fc1_path),
                                     "%s/block-%u/ane_fc1.bf16",
                                     directory, index);
        int ane_fc2_count = snprintf(ane_fc2_path, sizeof(ane_fc2_path),
                                     "%s/block-%u/ane_fc2.bf16",
                                     directory, index);
        int manifest_count = snprintf(manifest_path, sizeof(manifest_path),
                                      "%s/block-%u/manifest.json",
                                      directory, index);
        if (model_count < 0 || (size_t)model_count >= sizeof(model_path) ||
            fc1_count < 0 || (size_t)fc1_count >= sizeof(gpu_fc1_path) ||
            fc2_count < 0 || (size_t)fc2_count >= sizeof(gpu_fc2_path) ||
            ane_fc1_count < 0 ||
            (size_t)ane_fc1_count >= sizeof(ane_fc1_path) ||
            ane_fc2_count < 0 ||
            (size_t)ane_fc2_count >= sizeof(ane_fc2_path)) {
            fail(error, error_size, "Core ML MLP path is too long");
            return 0;
        }
        if (manifest_count < 0 ||
            (size_t)manifest_count >= sizeof(manifest_path)) {
            fail(error, error_size, "Core ML manifest path is too long");
            return 0;
        }
        model = model_path;
        gpu_fc1 = gpu_fc1_path;
        gpu_fc2 = gpu_fc2_path;
        ane_fc1 = ane_fc1_path;
        ane_fc2 = ane_fc2_path;
        manifest = manifest_path;
    }
    if (!gpu_fc1 || !*gpu_fc1 || !gpu_fc2 || !*gpu_fc2) {
        fail(error, error_size,
             "H3_COREML_ANE_GPU_FC1 and H3_COREML_ANE_GPU_FC2 are required");
        return 0;
    }
    if (!manifest || !*manifest) {
        fail(error, error_size,
             "H3_COREML_ANE_MANIFEST is required for a single Core ML model");
        return 0;
    }
    if (dit->coreml_gpu_fallback && !ane_fc1 &&
        (!coreml_manifest_sibling_path(
             ane_fc1_path, sizeof(ane_fc1_path), manifest, "ane_fc1.bf16",
             error, error_size) ||
         !coreml_manifest_sibling_path(
             ane_fc2_path, sizeof(ane_fc2_path), manifest, "ane_fc2.bf16",
             error, error_size))) return 0;
    if (dit->coreml_gpu_fallback && !ane_fc1) {
        ane_fc1 = ane_fc1_path;
        ane_fc2 = ane_fc2_path;
    }
    int checkpoint_index_count = snprintf(
        checkpoint_index_path, sizeof(checkpoint_index_path),
        "%s/model.safetensors.index.json", dit->weight_directory);
    if (checkpoint_index_count < 0 ||
        (size_t)checkpoint_index_count >= sizeof(checkpoint_index_path)) {
        fail(error, error_size, "transformer checkpoint index path is too long");
        return 0;
    }
    uint32_t ane_intermediate = 4096;
    const char *width = h3_runtime_getenv("H3_COREML_ANE_INTERMEDIATE");
    if (width && *width) {
        char *tail = NULL;
        unsigned long parsed = strtoul(width, &tail, 10);
        if (tail == width || *tail || !parsed || parsed >= FFN) {
            fail(error, error_size,
                 "H3_COREML_ANE_INTERMEDIATE must be in [1, %u]", FFN - 1);
            return 0;
        }
        ane_intermediate = (uint32_t)parsed;
    }
    uint32_t gpu_intermediate = FFN - ane_intermediate;
    float output_scale = 1.0f;
    char output_scale_name[64];
    int output_scale_name_count = snprintf(
        output_scale_name, sizeof(output_scale_name),
        "H3_COREML_ANE_OUTPUT_SCALE_BLOCK_%u", index);
    if (output_scale_name_count < 0 ||
        (size_t)output_scale_name_count >= sizeof(output_scale_name)) {
        fail(error, error_size, "Core ML block output-scale name is too long");
        return 0;
    }
    const char *output_scale_value = h3_runtime_getenv(output_scale_name);
    if (!output_scale_value || !*output_scale_value)
        output_scale_value = h3_runtime_getenv("H3_COREML_ANE_OUTPUT_SCALE");
    if (output_scale_value && *output_scale_value) {
        char *tail = NULL;
        output_scale = strtof(output_scale_value, &tail);
        if (tail == output_scale_value || *tail ||
            !isfinite(output_scale) || output_scale <= 0.0f) {
            fail(error, error_size,
                 "%s must be finite and positive",
                 h3_runtime_getenv(output_scale_name) ? output_scale_name :
                     "H3_COREML_ANE_OUTPUT_SCALE");
            return 0;
        }
    }
    if (!dit->coreml_checkpoint_identity_ready) {
        if (!h3_weight_store_identity(
                dit->weights, &dit->coreml_checkpoint_identity,
                error, error_size)) return 0;
        dit->coreml_checkpoint_identity_ready = 1;
    }
    size_t gpu_fc1_bytes =
        (size_t)gpu_intermediate * 2 * HIDDEN * sizeof(uint16_t);
    size_t gpu_fc2_bytes =
        (size_t)HIDDEN * gpu_intermediate * sizeof(uint16_t);
    size_t ane_fc1_bytes = dit->coreml_gpu_fallback ?
        (size_t)ane_intermediate * 2 * HIDDEN * sizeof(uint16_t) : 0;
    size_t ane_fc2_bytes = dit->coreml_gpu_fallback ?
        (size_t)HIDDEN * ane_intermediate * sizeof(uint16_t) : 0;
    h3_coreml_mlp_manifest_expectation expectation = {
        .block_index = index,
        .rows = dit->sequence,
        .hidden = HIDDEN,
        .intermediate = ane_intermediate,
        .full_intermediate = FFN,
        .output_scale = output_scale,
        .checkpoint_identity = dit->coreml_checkpoint_identity
    };
    char manifest_sha256[H3_COREML_SHA256_HEX_SIZE];
    if (!h3_coreml_mlp_validate_manifest(
        manifest, model, checkpoint_index_path,
        NULL, gpu_fc1_bytes, NULL, gpu_fc2_bytes,
        NULL, ane_fc1_bytes, NULL, ane_fc2_bytes,
        &expectation, manifest_sha256, error, error_size)) return 0;
    block->fc1 = h3_gpu_tensor_load_bf16(
        dit->gpu, gpu_fc1, 0,
        (size_t)gpu_intermediate * 2 * HIDDEN);
    block->fc2 = h3_gpu_tensor_load_bf16(
        dit->gpu, gpu_fc2, 0,
        (size_t)HIDDEN * gpu_intermediate);
    if (!block->fc1 || !block->fc2) {
        fail(error, error_size, "cannot load Core ML MLP shard: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!h3_coreml_mlp_validate_manifest(
            manifest, model, checkpoint_index_path,
            h3_gpu_tensor_host_pointer(block->fc1), gpu_fc1_bytes,
            h3_gpu_tensor_host_pointer(block->fc2), gpu_fc2_bytes,
            NULL, ane_fc1_bytes, NULL, ane_fc2_bytes,
            &expectation, manifest_sha256, error, error_size)) return 0;
    if (dit->coreml_gpu_fallback) {
        if (!h3_coreml_mlp_validate_fallback_shards(
                manifest, manifest_sha256,
                ane_fc1, NULL, ane_fc1_bytes,
                ane_fc2, NULL, ane_fc2_bytes,
                error, error_size)) return 0;
        block->coreml_manifest_path = strdup(manifest);
        block->coreml_fallback_fc1_path = strdup(ane_fc1);
        block->coreml_fallback_fc2_path = strdup(ane_fc2);
        if (!block->coreml_manifest_path ||
            !block->coreml_fallback_fc1_path ||
            !block->coreml_fallback_fc2_path) {
            fail(error, error_size,
                 "out of memory recording Core ML fallback artifacts");
            return 0;
        }
        memcpy(block->coreml_manifest_sha256, manifest_sha256,
               sizeof(block->coreml_manifest_sha256));
    }
    if (!dit->coreml_input) {
        size_t elements = (size_t)dit->sequence * HIDDEN;
        dit->coreml_input = h3_gpu_tensor_new_f16(dit->gpu, elements);
        dit->coreml_output = h3_gpu_tensor_new_f16(dit->gpu, elements);
        if (!dit->coreml_nonfinite)
            dit->coreml_nonfinite = h3_gpu_tensor_new_f32(dit->gpu, 1);
        if (!dit->coreml_input || !dit->coreml_output ||
            !dit->coreml_nonfinite) {
            fail(error, error_size,
                 "cannot allocate shared Core ML MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        memset(h3_gpu_tensor_host_pointer(dit->coreml_input), 0,
               elements * sizeof(uint16_t));
        memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
               sizeof(uint32_t));
    }
    block->coreml_mlp = h3_coreml_mlp_create_async_shared_verified(
        dit->gpu, model, dit->sequence, HIDDEN,
        dit->coreml_input, dit->coreml_output, manifest_sha256,
        error, error_size);
    if (!block->coreml_mlp) return 0;
    block->coreml_intermediate = ane_intermediate;
    block->coreml_output_scale = output_scale;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: Core ML GPU+ANE MLP block=%u ANE-F=%u GPU-F=%u "
                "output-scale=%.6g fallback=%s latch=%s\n",
                index, ane_intermediate, gpu_intermediate, output_scale,
                !dit->coreml_gpu_fallback ? "disabled" :
                dit->coreml_force_gpu_fallback ? "forced" : "on-failure",
                dit->coreml_fallback_latch ? "enabled" : "disabled");
    return 1;
}

static int load_coreml_benchmark_full_mlp(
                                h3_dit *dit, h3_dit_block *block,
                                const char *prefix,
                                char *error, size_t error_size) {
    if (!dit->coreml_same_loaded_ab) return 1;
    char name[160];
    snprintf(name, sizeof(name), "%smlp.fc1.weight", prefix);
    block->coreml_full_fc1 = bf2(
        dit, name, FFN * 2, HIDDEN, error, error_size);
    if (!block->coreml_full_fc1) return 0;
    snprintf(name, sizeof(name), "%smlp.fc2.weight", prefix);
    block->coreml_full_fc2 = bf2(
        dit, name, HIDDEN, FFN, error, error_size);
    return block->coreml_full_fc2 != NULL;
}


static void free_block(h3_dit_block *block) {
    h3_ane_mlp_free(block->ane_mlp);
    block->ane_mlp = NULL;
    h3_ane_linear_free(block->ane_qkv);
    block->ane_qkv = NULL;
    h3_ane_linear_free(block->ane_attention_out);
    block->ane_attention_out = NULL;
    h3_ane_linear_recipe_clear(&block->ane_attention_out_recipe);
    h3_coreml_mlp_free(block->coreml_mlp);
    block->coreml_mlp = NULL;
    h3_coreml_mlp_free(block->coreml_qkv);
    block->coreml_qkv = NULL;
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->qkv_int8);
    free_tensor(&block->qkv_scales);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->out_int8);
    free_tensor(&block->out_scales);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
    free_tensor(&block->fc1_int8);
    free_tensor(&block->fc1_scales);
    free_tensor(&block->fc2_int8);
    free_tensor(&block->fc2_scales);
    free_tensor(&block->coreml_qkv_gpu_weight);
    free_tensor(&block->ane_qkv_gpu_scales);
    free_tensor(&block->ane_attention_out_gpu_weight);
    free_tensor(&block->coreml_full_fc1);
    free_tensor(&block->coreml_full_fc2);
    free_tensor(&block->coreml_fallback_fc1);
    free_tensor(&block->coreml_fallback_fc2);
    free(block->coreml_manifest_path);
    block->coreml_manifest_path = NULL;
    free(block->coreml_fallback_fc1_path);
    block->coreml_fallback_fc1_path = NULL;
    free(block->coreml_fallback_fc2_path);
    block->coreml_fallback_fc2_path = NULL;
}

static unsigned final_eviction_interval(void) {
    const char *value = h3_runtime_getenv("H3_FINAL_EVICT_BLOCKS");
    if (!value || !*value || !strcmp(value, "0")) return 0;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    return end != value && !*end && parsed >= 1 &&
           parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
}

static size_t append_block_tensors(const h3_dit_block *block,
                                   h3_gpu_tensor **tensors,
                                   size_t capacity, uint64_t *bytes) {
    size_t count = 0;
#define APPEND(field) do {                                                     \
    if (block->field && count < capacity) {                                    \
        tensors[count++] = block->field;                                       \
        *bytes += h3_gpu_tensor_bytes(block->field);                            \
    }                                                                          \
} while (0)
    APPEND(norm1); APPEND(norm2); APPEND(qkv); APPEND(qkv_int8);
    APPEND(qkv_scales); APPEND(q_norm); APPEND(k_norm); APPEND(out);
    APPEND(out_int8); APPEND(out_scales); APPEND(fc1); APPEND(fc2);
    APPEND(fc1_int8); APPEND(fc1_scales); APPEND(fc2_int8);
    APPEND(fc2_scales);
#undef APPEND
    return count;
}

static int evict_final_block_group(h3_dit *dit, unsigned begin, unsigned end,
                                   char *error, size_t error_size) {
    enum { BLOCK_TENSORS = 16 };
    h3_gpu_tensor *tensors[H3_DIT_BLOCKS * BLOCK_TENSORS];
    size_t tensor_count = 0;
    uint64_t bytes = 0;
    for (unsigned block = begin; block < end; block++) {
        tensor_count += append_block_tensors(
            &dit->blocks[block], tensors + tensor_count,
            sizeof(tensors) / sizeof(tensors[0]) - tensor_count, &bytes);
    }
    if (!h3_gpu_continue_releasing(dit->gpu, tensors, tensor_count)) {
        fail(error, error_size, "continue final DiT eviction: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    for (unsigned block = begin; block < end; block++)
        memset(&dit->blocks[block], 0, sizeof(dit->blocks[block]));
    dit->final_eviction_groups++;
    dit->final_evicted_blocks += end - begin;
    dit->final_evicted_bytes += bytes;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: final-pass DiT eviction scheduled blocks=%u-%u "
                "group=%.3f GiB cumulative=%.3f GiB\n",
                begin, end - 1,
                (double)bytes / (1024.0 * 1024.0 * 1024.0),
                (double)dit->final_evicted_bytes /
                    (1024.0 * 1024.0 * 1024.0));
    return 1;
}

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int prepare_coreml_gpu_fallback(h3_dit *dit, h3_dit_block *block,
                                       char *error, size_t error_size) {
    if (block->coreml_fallback_fc1 && block->coreml_fallback_fc2 &&
        dit->coreml_fallback_output) return 1;
    if (block->coreml_fallback_fc1 || block->coreml_fallback_fc2 ||
        !block->coreml_manifest_path ||
        !block->coreml_fallback_fc1_path ||
        !block->coreml_fallback_fc2_path ||
        !block->coreml_manifest_sha256[0]) {
        fail(error, error_size,
             "Core ML GPU fallback artifacts are not ready");
        return 0;
    }
    size_t ane_fc1_elements =
        (size_t)block->coreml_intermediate * 2 * HIDDEN;
    size_t ane_fc2_elements =
        (size_t)HIDDEN * block->coreml_intermediate;
    double started = stream_now();
    h3_gpu_tensor *fc1 = h3_gpu_tensor_map_bf16(
        dit->gpu, block->coreml_fallback_fc1_path, 0, ane_fc1_elements);
    h3_gpu_tensor *fc2 = h3_gpu_tensor_map_bf16(
        dit->gpu, block->coreml_fallback_fc2_path, 0, ane_fc2_elements);
    if (!fc1 || !fc2) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(fc2);
        fail(error, error_size, "cannot map Core ML fallback shard: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!h3_coreml_mlp_validate_fallback_shards(
            block->coreml_manifest_path,
            block->coreml_manifest_sha256,
            block->coreml_fallback_fc1_path,
            h3_gpu_tensor_host_pointer(fc1),
            ane_fc1_elements * sizeof(uint16_t),
            block->coreml_fallback_fc2_path,
            h3_gpu_tensor_host_pointer(fc2),
            ane_fc2_elements * sizeof(uint16_t),
            error, error_size)) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(fc2);
        return 0;
    }
    double elapsed = stream_now() - started;
    if (!dit->coreml_fallback_output) {
        size_t elements = (size_t)dit->sequence * HIDDEN;
        dit->coreml_fallback_output = h3_gpu_tensor_new_bf16(
            dit->gpu, elements);
        if (!dit->coreml_fallback_output) {
            h3_gpu_tensor_free(fc1);
            h3_gpu_tensor_free(fc2);
            fail(error, error_size,
                 "cannot allocate Core ML fallback output: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    block->coreml_fallback_fc1 = fc1;
    block->coreml_fallback_fc2 = fc2;
    block->coreml_fallback_lazy_maps++;
    if (block->coreml_fallback_lazy_maps == 1)
        block->coreml_fallback_first_map_seconds = elapsed;
    block->coreml_fallback_map_seconds += elapsed;
    return 1;
}

static int profile_steps_enabled(void) {
    const char *value = h3_runtime_getenv("H3_PROFILE_STEPS");
    return value && *value && strcmp(value, "0");
}

static uint64_t profile_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void profile_denoise_step(const h3_dit *dit, int step, int steps,
                                 int evaluated, double wall_start,
                                 const h3_gpu_stats *start) {
    h3_gpu_stats value;
    if (!dit || !start || !h3_gpu_get_stats(dit->gpu, &value)) return;
    fprintf(stderr,
        "h3 step profile: step=%d/%d action=%s wall=%8.3fs "
        "encode=%7.3fs wait=%8.3fs root-gpu=%7.3fs "
        "live=%7.3fGiB peak=%7.3fGiB alloc=%7.3fGiB "
        "submissions=%llu direct=%llu linear=%llu conv=%llu attention=%llu\n",
        step + 1, steps,
        !evaluated ? "denoiser-reuse" :
        dit->tea_cache_last_reused ? "teacache-reuse" :
        dit->first_block_cache_last_reused ? "fbc-reuse" : "fresh",
        stream_now() - wall_start,
        value.command_encode_seconds - start->command_encode_seconds,
        value.command_wait_seconds - start->command_wait_seconds,
        value.gpu_seconds - start->gpu_seconds,
        (double)value.live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)profile_counter_delta(value.allocated_bytes,
                                      start->allocated_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)profile_counter_delta(value.submissions,
                                                  start->submissions),
        (unsigned long long)profile_counter_delta(value.direct_dispatches,
                                                  start->direct_dispatches),
        (unsigned long long)profile_counter_delta(
            value.mps_linear_dispatches, start->mps_linear_dispatches),
        (unsigned long long)profile_counter_delta(
            value.mps_conv_dispatches, start->mps_conv_dispatches),
        (unsigned long long)profile_counter_delta(
            value.mps_sdpa_dispatches, start->mps_sdpa_dispatches));
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_dit_stream_source *a = left;
    const h3_dit_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

static int prepare_stream_source(h3_dit *dit,
                                 h3_dit_stream_source *source,
                                 const char *name, uint64_t rows,
                                 uint64_t columns, unsigned field,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, &header);
    if (!tensor) {
        fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (!header || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns) {
        fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = (size_t)(rows * columns);
    source->field = field;
    source->dtype = H3_GPU_BF16;
    return 1;
}

static int prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    if (dit->ssd_quantized) {
        const h3_quant_cache_layer *cached = h3_quant_cache_layer_at(
            &dit->quant_cache, layer);
        if (!cached) {
            fail(error, error_size,
                 "H3 quantized cache has no block %u", layer);
            return 0;
        }
        h3_dit_stream_layer *stream = &dit->stream_layers[layer];
        stream->source_count = H3_QUANT_CACHE_SOURCES;
        for (unsigned index = 0; index < H3_QUANT_CACHE_SOURCES; index++) {
            const h3_quant_cache_source *source = &cached->sources[index];
            stream->sources[index] = (h3_dit_stream_source){
                .path = source->path,
                .file_offset = source->file_offset,
                .elements = source->elements,
                .field = source->field == H3_QUANT_QKV_WEIGHT ?
                    STREAM_QKV_INT8 : source->field == H3_QUANT_QKV_SCALES ?
                    STREAM_QKV_SCALES : source->field == H3_QUANT_OUT_WEIGHT ?
                    STREAM_OUT_INT8 : source->field == H3_QUANT_OUT_SCALES ?
                    STREAM_OUT_SCALES : source->field == H3_QUANT_FC1_WEIGHT ?
                    STREAM_FC1_INT8 : source->field == H3_QUANT_FC1_SCALES ?
                    STREAM_FC1_SCALES : source->field == H3_QUANT_FC2_WEIGHT ?
                    STREAM_FC2_INT8 : STREAM_FC2_SCALES,
                .dtype = source->dtype == H3_DTYPE_I8 ?
                    H3_GPU_I8 : H3_GPU_F32
            };
        }
        return 1;
    }
    char name[160];
    h3_dit_stream_layer *stream = &dit->stream_layers[layer];
    stream->source_count = 4;
#define SOURCE(index, suffix, rows, columns, field) do {                        \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);              \
    if (!prepare_stream_source(dit, &stream->sources[index], name,             \
                               rows, columns, field, error, error_size))        \
        return 0;                                                               \
} while (0)
    SOURCE(0, "attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
    SOURCE(1, "attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
    SOURCE(2, "mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
    SOURCE(3, "mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
#undef SOURCE
    qsort(stream->sources, stream->source_count, sizeof(stream->sources[0]),
          compare_stream_sources);
    return 1;
}

static int allocate_stream_slot(h3_dit *dit, h3_dit_block *slot,
                                char *error, size_t error_size) {
    if (dit->ssd_quantized) {
        slot->qkv_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)INNER * 3 * HIDDEN);
        slot->qkv_scales = h3_gpu_tensor_new_f32(
            dit->gpu, INNER * 3);
        slot->out_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)HIDDEN * INNER);
        slot->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
        slot->fc1_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)FFN * 2 * HIDDEN);
        slot->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
        slot->fc2_int8 = h3_gpu_tensor_new_i8(
            dit->gpu, (size_t)HIDDEN * FFN);
        slot->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
        if (!slot->qkv_int8 || !slot->qkv_scales || !slot->out_int8 ||
            !slot->out_scales || !slot->fc1_int8 || !slot->fc1_scales ||
            !slot->fc2_int8 || !slot->fc2_scales) {
            fail(error, error_size,
                 "cannot allocate INT8 SSD layer slot: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        return 1;
    }
    slot->qkv = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    slot->out = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * INNER);
    slot->fc1 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    slot->fc2 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * FFN);
    if (!slot->qkv || !slot->out || !slot->fc1 || !slot->fc2) {
        fail(error, error_size, "cannot allocate BF16 SSD layer slot: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int load_quantized_block(h3_dit *dit, h3_dit_block *block,
                                unsigned layer,
                                char *error, size_t error_size) {
    const h3_quant_cache_layer *cached = h3_quant_cache_layer_at(
        &dit->quant_cache, layer);
    if (!cached) {
        fail(error, error_size, "H3 quantized cache has no block %u", layer);
        return 0;
    }
    for (unsigned index = 0; index < H3_QUANT_CACHE_SOURCES; index++) {
        const h3_quant_cache_source *source = &cached->sources[index];
        h3_gpu_tensor **target = NULL;
        switch (source->field) {
        case H3_QUANT_QKV_WEIGHT: target = &block->qkv_int8; break;
        case H3_QUANT_QKV_SCALES: target = &block->qkv_scales; break;
        case H3_QUANT_OUT_WEIGHT: target = &block->out_int8; break;
        case H3_QUANT_OUT_SCALES: target = &block->out_scales; break;
        case H3_QUANT_FC1_WEIGHT: target = &block->fc1_int8; break;
        case H3_QUANT_FC1_SCALES: target = &block->fc1_scales; break;
        case H3_QUANT_FC2_WEIGHT: target = &block->fc2_int8; break;
        case H3_QUANT_FC2_SCALES: target = &block->fc2_scales; break;
        }
        if (!target) {
            fail(error, error_size,
                 "H3 quantized cache block %u has an invalid field", layer);
            return 0;
        }
        *target = source->dtype == H3_DTYPE_I8 ?
            h3_gpu_tensor_load_i8(dit->gpu, source->path,
                                  source->file_offset, source->elements) :
            source->dtype == H3_DTYPE_F32 ?
            h3_gpu_tensor_load_f32(dit->gpu, source->path,
                                   source->file_offset, source->elements) :
            NULL;
        if (!*target) {
            fail(error, error_size,
                 "cannot load H3 quantized cache block %u: %s",
                 layer, h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

static h3_gpu_tensor *stream_slot_target(h3_dit_block *slot,
                                         unsigned field) {
    if (field == STREAM_QKV) return slot->qkv;
    if (field == STREAM_OUT) return slot->out;
    if (field == STREAM_FC1) return slot->fc1;
    if (field == STREAM_FC2) return slot->fc2;
    if (field == STREAM_QKV_INT8) return slot->qkv_int8;
    if (field == STREAM_QKV_SCALES) return slot->qkv_scales;
    if (field == STREAM_OUT_INT8) return slot->out_int8;
    if (field == STREAM_OUT_SCALES) return slot->out_scales;
    if (field == STREAM_FC1_INT8) return slot->fc1_int8;
    if (field == STREAM_FC1_SCALES) return slot->fc1_scales;
    if (field == STREAM_FC2_INT8) return slot->fc2_int8;
    if (field == STREAM_FC2_SCALES) return slot->fc2_scales;
    return NULL;
}

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_stream_job;

static int read_stream_layer(h3_dit_stream_job *job) {
    h3_dit_stream_layer *layer = &job->dit->stream_layers[job->layer];
    h3_dit_block *slot = &job->dit->stream_slots[job->slot];
    double started = stream_now();
    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    for (unsigned index = 0; index < layer->source_count; index++) {
        const h3_dit_stream_source *source = &layer->sources[index];
        h3_gpu_tensor *target = stream_slot_target(slot, source->field);
        if (!target) {
            snprintf(job->error, sizeof(job->error),
                     "invalid typed streaming destination");
            job->ok = 0;
            break;
        }
        int ok = source->dtype == H3_GPU_BF16 ?
            h3_gpu_tensor_stream_file_bf16(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error)) :
            source->dtype == H3_GPU_I8 ? h3_gpu_tensor_stream_file_i8(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error)) :
            source->dtype == H3_GPU_F32 && h3_gpu_tensor_stream_file_f32(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error));
        if (!ok) {
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error),
                         "invalid typed streaming destination");
            job->ok = 0;
            break;
        }
        job->bytes += (uint64_t)source->elements *
            (source->dtype == H3_GPU_BF16 ? sizeof(uint16_t) :
             source->dtype == H3_GPU_I8 ? sizeof(int8_t) : sizeof(float));
    }
    job->seconds = stream_now() - started;
    return job->ok;
}

static void *read_stream_layer_thread(void *opaque) {
    read_stream_layer(opaque);
    return NULL;
}

static int quantize_block_mlp_width(h3_dit *dit, h3_dit_block *block,
                                    uint32_t intermediate,
                                    int release_bf16,
                                    char *error, size_t error_size) {
    block->fc1_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)intermediate * 2u * HIDDEN);
    block->fc1_scales = h3_gpu_tensor_new_f32(
        dit->gpu, intermediate * 2u);
    block->fc2_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * intermediate);
    block->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->fc1_int8 && block->fc1_scales &&
             block->fc2_int8 && block->fc2_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc1_int8, block->fc1_scales, block->fc1,
                 intermediate * 2u, HIDDEN) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc2_int8, block->fc2_scales, block->fc2,
                 HIDDEN, intermediate) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        free_tensor(&block->fc1_int8);
        free_tensor(&block->fc1_scales);
        free_tensor(&block->fc2_int8);
        free_tensor(&block->fc2_scales);
        fail(error, error_size, "cannot quantize DiT MLP weights: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (release_bf16) {
        free_tensor(&block->fc1);
        free_tensor(&block->fc2);
    }
    return 1;
}

static int quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    return quantize_block_mlp_width(
        dit, block, FFN, !dit->keep_bf16_mlp, error, error_size);
}

static int quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->qkv_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    int ok = block->qkv_int8 && block->qkv_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                 INNER * 3, HIDDEN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT QKV weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_qkv) free_tensor(&block->qkv);
    return 1;
}

static int quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size) {
    block->out_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * INNER);
    block->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->out_int8 && block->out_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->out_int8, block->out_scales, block->out,
                 HIDDEN, INNER) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size,
             "cannot quantize DiT attention-output weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_attention_out) free_tensor(&block->out);
    return 1;
}

static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
        load_block(dit, &refiner[0], "token_refiner.blocks.0.", 1,
                   error, error_size) &&
        load_block(dit, &refiner[1], "token_refiner.blocks.1.", 1,
                   error, error_size);
    if (ok) final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        free_tensor(&inverse_tensor);
        if (!error || !*error) fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    free_tensor(&inverse_tensor);
    float spatial_scale = dit->spatial_rope_scale;
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    size_t reduced_count = dit->token_reduction ?
        (size_t)dit->reduced_sequence * ROPE_HALF : 0;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *reduced_cosines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_cosines)) : NULL;
    float *reduced_sines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_sines)) : NULL;
    if (!cosines || !sines ||
        (reduced_count && (!reduced_cosines || !reduced_sines))) {
        free(cosines);
        free(sines);
        free(reduced_cosines);
        free(reduced_sines);
        fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h * spatial_scale,
                        (float)dit->layout.positions[row].w * spatial_scale};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        uint32_t first, second;
        token_pool_sources(dit, row, &first, &second);
        float axes[] = {
            (float)((dit->layout.positions[first].t +
                     dit->layout.positions[second].t) * 0.5),
            (float)((dit->layout.positions[first].h +
                     dit->layout.positions[second].h) * 0.5) * spatial_scale,
            (float)((dit->layout.positions[first].w +
                     dit->layout.positions[second].w) * 0.5) * spatial_scale
        };
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                reduced_cosines[index] = cosf(angle);
                reduced_sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    h3_gpu_tensor *reduced_cos_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_cosines, reduced_count) : NULL;
    h3_gpu_tensor *reduced_sin_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_sines, reduced_count) : NULL;
    free(cosines);
    free(sines);
    free(reduced_cosines);
    free(reduced_sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    if (reduced_count) {
        dit->reduced_rope_cos = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
        dit->reduced_rope_sin = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
    }
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
        (!reduced_count || (reduced_cos_f32 && reduced_sin_f32 &&
                            dit->reduced_rope_cos &&
                            dit->reduced_rope_sin));
    if (ok) {
        ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast");
        if (ok && reduced_count) {
            ok = gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_cos, reduced_cos_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE cosine cast") &&
                 gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_sin, reduced_sin_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE sine cast");
        }
        if (ok) ok =
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    free_tensor(&reduced_cos_f32);
    free_tensor(&reduced_sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    if (dit->token_reduction)
        dit->reduced_row_maps = calloc((size_t)steps,
                                       sizeof(*dit->reduced_row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *reduced = dit->token_reduction ?
        malloc((size_t)dit->reduced_sequence * sizeof(*reduced)) : NULL;
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        (dit->token_reduction && (!dit->reduced_row_maps || !reduced)) ||
        !rows || !audio || !video) {
        fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(reduced); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(dit->schedule, step, &dit->layout,
                                     text->tags, text->tokens, rows,
                                     dit->sequence)) {
            fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
        if (dit->token_reduction) {
            for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
                uint32_t first, second;
                token_pool_sources(dit, row, &first, &second);
                (void)second;
                reduced[row] = rows[first];
            }
            dit->reduced_row_maps[step] = h3_gpu_tensor_from_u32(
                dit->gpu, reduced, dit->reduced_sequence);
        }
        uint32_t audio_row = h3_dit_schedule_audio_row(dit->schedule, step);
        uint32_t video_row = h3_dit_schedule_video_row(dit->schedule, step);
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = audio_row;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = video_row;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] ||
            (dit->token_reduction && !dit->reduced_row_maps[step]) ||
            !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(reduced); free(audio); free(video);
    return 1;
}

static int prepare_projection_maps(h3_dit *dit, char *error,
                                   size_t error_size) {
    unsigned video_segments = 0, audio_segments = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        h3_segment_kind kind = dit->layout.segments[index].kind;
        if (kind == H3_SEG_COND || kind == H3_SEG_REF_IMAGE ||
            kind == H3_SEG_VIDEO)
            video_segments++;
        else if (kind != H3_SEG_TEXT)
            audio_segments++;
    }
    uint32_t *video = video_segments > 1 ?
        malloc((size_t)dit->video_total_rows * sizeof(*video)) : NULL;
    uint32_t *audio = audio_segments > 1 ?
        malloc((size_t)dit->audio_total_rows * sizeof(*audio)) : NULL;
    if ((video_segments > 1 && !video) || (audio_segments > 1 && !audio)) {
        free(video); free(audio);
        fail(error, error_size, "out of memory allocating projection maps");
        return 0;
    }
    size_t video_offset = 0, audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (segment->kind == H3_SEG_COND ||
            segment->kind == H3_SEG_REF_IMAGE ||
            segment->kind == H3_SEG_VIDEO) {
            for (size_t row = 0; video && row < rows; row++)
                video[video_offset + row] = (uint32_t)(segment->start + row);
            video_offset += rows;
        } else if (segment->kind != H3_SEG_TEXT) {
            for (size_t row = 0; audio && row < rows; row++)
                audio[audio_offset + row] = (uint32_t)(segment->start + row);
            audio_offset += rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        free(video); free(audio);
        fail(error, error_size, "projection map rows are inconsistent");
        return 0;
    }
    if (video)
        dit->video_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_total_rows);
    if (audio)
        dit->audio_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_total_rows);
    free(video); free(audio);
    if ((video_segments > 1 && !dit->video_projection_map) ||
        (audio_segments > 1 && !dit->audio_projection_map)) {
        fail(error, error_size, "cannot allocate projection map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int prepare_token_reduction_maps(h3_dit *dit, char *error,
                                        size_t error_size) {
    if (!dit->token_reduction) return 1;
    size_t pair_count = (size_t)dit->reduced_sequence * 2;
    uint32_t *pairs = malloc(pair_count * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(
        (size_t)dit->reduced_sequence * sizeof(*baseline_indices));
    uint32_t *parents = malloc((size_t)dit->sequence * sizeof(*parents));
    if (!pairs || !baseline_indices || !parents) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size,
             "out of memory allocating token-reduction maps");
        return 0;
    }
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        token_pool_sources(dit, row, &pairs[(size_t)row * 2],
                           &pairs[(size_t)row * 2 + 1]);
        baseline_indices[row] =
            row >= dit->video_target_start &&
            pairs[(size_t)row * 2] != pairs[(size_t)row * 2 + 1] ?
                baseline_row++ : UINT32_MAX;
    }
    if (baseline_row != dit->token_baseline_rows) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size, "token-reduction baseline map is inconsistent");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++)
        parents[row] = token_reduced_parent(dit, row);
    dit->token_pool_pairs = h3_gpu_tensor_from_u32(
        dit->gpu, pairs, pair_count);
    dit->token_baseline_indices = h3_gpu_tensor_from_u32(
        dit->gpu, baseline_indices, dit->reduced_sequence);
    dit->token_expand_parents = h3_gpu_tensor_from_u32(
        dit->gpu, parents, dit->sequence);
    free(pairs);
    free(baseline_indices);
    free(parents);
    if (!dit->token_pool_pairs || !dit->token_baseline_indices ||
        !dit->token_expand_parents) {
        fail(error, error_size,
             "cannot allocate token-reduction map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static void configure_active_blocks(h3_dit *dit, unsigned active) {
    memset(dit->block_active, 1, sizeof(dit->block_active));
    dit->active_block_count = active;
    unsigned skipped = H3_DIT_BLOCKS - active;
    for (unsigned index = 0; index < skipped; index++) {
        unsigned block = ((2 * index + 1) * H3_DIT_BLOCKS) / (2 * skipped);
        if (block == 0) block = 1;
        if (block >= H3_DIT_BLOCKS - 1) block = H3_DIT_BLOCKS - 2;
        dit->block_active[block] = 0;
    }
}

static int configure_explicit_gate_skip(h3_dit *dit, char *error,
                                        size_t error_size) {
    const char *value = h3_runtime_getenv("H3_DIT_GATE_SKIP");
    if (!value || !*value) return 1;
    const char *policy = h3_runtime_getenv("H3_DIT_LAYER_POLICY");
    if (policy && !strcmp(policy, "uniform")) {
        fail(error, error_size,
             "H3_DIT_GATE_SKIP cannot be combined with uniform layer policy");
        return 0;
    }
    unsigned expected = H3_DIT_BLOCKS - dit->active_block_count;
    uint8_t seen[H3_DIT_BLOCKS] = {0};
    memset(dit->block_active, 1, sizeof(dit->block_active));
    const char *cursor = value;
    unsigned count = 0;
    while (*cursor) {
        while (isspace((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        errno = 0;
        char *tail = NULL;
        unsigned long parsed = strtoul(cursor, &tail, 10);
        if (errno || tail == cursor || parsed < 2 ||
            parsed >= H3_DIT_BLOCKS - 1 || seen[parsed]) {
            fail(error, error_size,
                 "H3_DIT_GATE_SKIP must contain unique block IDs in [2, 48]");
            return 0;
        }
        seen[parsed] = 1;
        dit->block_active[parsed] = 0;
        count++;
        cursor = tail;
        while (isspace((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        if (*cursor != ',') {
            fail(error, error_size,
                 "H3_DIT_GATE_SKIP must be a comma-separated block list");
            return 0;
        }
        cursor++;
        const char *next = cursor;
        while (isspace((unsigned char)*next)) next++;
        if (!*next) {
            fail(error, error_size,
                 "H3_DIT_GATE_SKIP cannot end with a comma");
            return 0;
        }
        cursor = next;
    }
    if (count != expected) {
        fail(error, error_size,
             "H3_DIT_GATE_SKIP has %u blocks; --layers requires exactly %u",
             count, expected);
        return 0;
    }
    dit->explicit_gate_skip = 1;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: explicit gate skip %s\n", value);
    return 1;
}

static int configure_step_gate_skip(h3_dit *dit,
                                    const h3_sigma_schedule *sigmas,
                                    char *error, size_t error_size) {
    const char *value = h3_runtime_getenv("H3_DIT_STEP_GATE_SKIP");
    if (!value || !*value) return 1;
    if (!sigmas || sigmas->steps < 1 || sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size,
             "H3_DIT_STEP_GATE_SKIP requires a valid sigma schedule");
        return 0;
    }
    if (dit->ssd_streaming || dit->token_reduction || dit->first_block_cache ||
        dit->tea_cache || dit->sol_attention ||
        dit->core_reuse_interval > 1) {
        fail(error, error_size,
             "step gate skip cannot be combined with SSD streaming, token "
             "reduction, cache, Sol-Attn, or core reuse");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++)
        dit->step_active_block_count[step] = dit->active_block_count;
    uint8_t seen_steps[H3_MAX_STEPS] = {0};
    const char *cursor = value;
    while (*cursor) {
        while (isspace((unsigned char)*cursor)) cursor++;
        errno = 0;
        char *tail = NULL;
        unsigned long parsed_step = strtoul(cursor, &tail, 10);
        if (errno || tail == cursor || parsed_step < 1 ||
            parsed_step > (unsigned long)sigmas->steps || *tail != ':') {
            fail(error, error_size,
                 "H3_DIT_STEP_GATE_SKIP must use STEP:BLOCK[,BLOCK][;...] "
                 "with one-based steps");
            return 0;
        }
        unsigned step = (unsigned)parsed_step - 1;
        if (seen_steps[step]) {
            fail(error, error_size,
                 "H3_DIT_STEP_GATE_SKIP repeats step %lu", parsed_step);
            return 0;
        }
        seen_steps[step] = 1;
        cursor = tail + 1;
        unsigned skipped = 0;
        for (;;) {
            while (isspace((unsigned char)*cursor)) cursor++;
            errno = 0;
            unsigned long parsed_block = strtoul(cursor, &tail, 10);
            if (errno || tail == cursor || parsed_block < 2 ||
                parsed_block >= H3_DIT_BLOCKS - 1 ||
                !dit->block_active[parsed_block] ||
                dit->step_block_skip[step][parsed_block]) {
                fail(error, error_size,
                     "step gate skip blocks must be unique resident IDs "
                     "in [2, 48]");
                return 0;
            }
            dit->step_block_skip[step][parsed_block] = 1;
            skipped++;
            cursor = tail;
            while (isspace((unsigned char)*cursor)) cursor++;
            if (*cursor != ',') break;
            cursor++;
        }
        if (skipped > dit->active_block_count - H3_DIT_BLOCKS / 2) {
            fail(error, error_size,
                 "step gate skip must retain at least 25 resident blocks");
            return 0;
        }
        dit->step_active_block_count[step] =
            dit->active_block_count - skipped;
        if (!*cursor) break;
        if (*cursor != ';') {
            fail(error, error_size,
                 "H3_DIT_STEP_GATE_SKIP entries must be semicolon-separated");
            return 0;
        }
        cursor++;
        const char *next = cursor;
        while (isspace((unsigned char)*next)) next++;
        if (!*next) {
            fail(error, error_size,
                 "H3_DIT_STEP_GATE_SKIP cannot end with a semicolon");
            return 0;
        }
        cursor = next;
    }
    dit->step_gate_skip = 1;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: step gate skip %s\n", value);
    return 1;
}

static int block_active_at_step(const h3_dit *dit, unsigned block, int step,
                                int use_step_gate_skip) {
    return dit->block_active[block] &&
        (!use_step_gate_skip || !dit->step_block_skip[step][block]);
}

typedef struct {
    h3_ane_mlp *model;
    int ok;
    char error[512];
} h3_private_ane_reload_job;

typedef struct {
    h3_ane_linear *model;
    int ok;
    char error[512];
} h3_private_ane_linear_reload_job;

typedef struct {
    h3_dit *dit;
    h3_dit_block *block;
    h3_ane_linear *model;
    unsigned index;
    int create;
    int ok;
    char error[512];
} h3_private_ane_attention_out_reload_job;

static void *private_ane_reload_worker(void *opaque) {
    h3_private_ane_reload_job *job = opaque;
    job->error[0] = '\0';
    job->ok = h3_ane_mlp_reload(
        job->model, job->error, sizeof(job->error));
    return NULL;
}

static void *private_ane_linear_reload_worker(void *opaque) {
    h3_private_ane_linear_reload_job *job = opaque;
    job->error[0] = '\0';
    job->ok = h3_ane_linear_reload(
        job->model, job->error, sizeof(job->error));
    return NULL;
}

static void *private_ane_attention_out_reload_worker(void *opaque) {
    h3_private_ane_attention_out_reload_job *job = opaque;
    job->error[0] = '\0';
    if (job->create) {
        job->model = create_private_ane_attention_out_model(
            job->dit, job->block, job->index,
            job->error, sizeof(job->error));
        job->ok = job->model != NULL;
    } else {
        job->ok = h3_ane_linear_reload(
            job->model, job->error, sizeof(job->error));
    }
    return NULL;
}

static unsigned next_private_ane_block(const h3_dit *dit, unsigned current,
                                       int step) {
    int use_step_gate_skip = dit->step_gate_skip &&
        !h3_runtime_getenv("H3_STEP_GATE_RUNTIME_DISABLE");
    for (unsigned offset = 1; offset <= H3_DIT_BLOCKS; offset++) {
        unsigned candidate = (current + offset) % H3_DIT_BLOCKS;
        if (block_active_at_step(
                dit, candidate, step, use_step_gate_skip) &&
            dit->blocks[candidate].ane_mlp)
            return candidate;
    }
    return H3_DIT_BLOCKS;
}

static unsigned next_private_ane_qkv_block(const h3_dit *dit,
                                           unsigned current, int step) {
    int use_step_gate_skip = dit->step_gate_skip &&
        !h3_runtime_getenv("H3_STEP_GATE_RUNTIME_DISABLE");
    for (unsigned offset = 1; offset <= H3_DIT_BLOCKS; offset++) {
        unsigned candidate = (current + offset) % H3_DIT_BLOCKS;
        if (block_active_at_step(
                dit, candidate, step, use_step_gate_skip) &&
            dit->blocks[candidate].ane_qkv)
            return candidate;
    }
    return H3_DIT_BLOCKS;
}

static unsigned next_private_ane_attention_out_block(
                                           const h3_dit *dit,
                                           unsigned current, int step) {
    int use_step_gate_skip = dit->step_gate_skip &&
        !h3_runtime_getenv("H3_STEP_GATE_RUNTIME_DISABLE");
    for (unsigned offset = 1; offset <= H3_DIT_BLOCKS; offset++) {
        unsigned candidate = (current + offset) % H3_DIT_BLOCKS;
        if (block_active_at_step(
                dit, candidate, step, use_step_gate_skip) &&
            block_has_private_ane_attention_out(&dit->blocks[candidate]))
            return candidate;
    }
    return H3_DIT_BLOCKS;
}

static int report_private_ane_qkv_micro(
                               h3_dit *dit, h3_dit_block *weight,
                               uint32_t rows, unsigned index, int step,
                               double full_gpu_seconds,
                               double candidate_seconds,
                               char *error, size_t error_size) {
    uint32_t ane_width = weight->ane_qkv_heads * 3u * HEAD_DIM;
    size_t logical_elements = (size_t)rows * ane_width;
    if (!dit->ane_qkv_micro_partials ||
        logical_elements > UINT32_MAX ||
        dit->ane_qkv_micro_partial_count !=
            h3_gpu_compare_ane_f32_bf16_partial_count(
                (uint32_t)logical_elements)) {
        fail(error, error_size,
             "private ANE QKV block %u has invalid micro scratch", index);
        return 0;
    }
    if (!h3_gpu_begin(dit->gpu) ||
        !h3_gpu_compare_ane_f32_bf16_prefix(
            dit->gpu, dit->ane_qkv_micro_partials,
            h3_ane_linear_io_output(dit->ane_qkv_io), dit->qkv,
            rows, ane_width, INNER * 3u,
            h3_ane_linear_io_plane_rows(dit->ane_qkv_io)) ||
        !h3_gpu_submit(dit->gpu)) {
        fail(error, error_size,
             "private ANE QKV block %u micro comparison failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    const float *partials = h3_gpu_tensor_host_pointer(
        dit->ane_qkv_micro_partials);
    double square_error = 0.0;
    double square_reference = 0.0;
    double square_candidate = 0.0;
    double dot_product = 0.0;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    uint64_t reference_nonfinite = 0;
    uint64_t candidate_nonfinite = 0;
    for (uint32_t group = 0;
         group < dit->ane_qkv_micro_partial_count; group++) {
        const float *sum = partials + (size_t)group * 8;
        square_error += sum[0];
        square_reference += sum[1];
        square_candidate += sum[2];
        dot_product += sum[3];
        if (sum[4] > maximum_error) maximum_error = sum[4];
        if (sum[5] > maximum_reference) maximum_reference = sum[5];
        reference_nonfinite += (uint64_t)llroundf(sum[6]);
        candidate_nonfinite += (uint64_t)llroundf(sum[7]);
    }
    double relative_l2 = square_reference > 0.0 ?
        sqrt(square_error / square_reference) : INFINITY;
    double cosine_denominator = sqrt(
        square_reference * square_candidate);
    double cosine = cosine_denominator > 0.0 ?
        dot_product / cosine_denominator : 0.0;
    double amplitude = square_reference > 0.0 ?
        sqrt(square_candidate / square_reference) : INFINITY;
    double rmse = logical_elements ?
        sqrt(square_error / (double)logical_elements) : INFINITY;
    weight->ane_qkv_micro_done = 1;
    fprintf(stderr,
            "h3: private ANE QKV micro step=%d block=%u precision=%s "
            "GPU-complement=%s GPU-baseline=%s ANE-heads=%u "
            "full-gpu=%.6fs candidate=%.6fs speedup=%.6fx "
            "rel-L2=%.9g rel-L2-percent=%.6f cosine=%.9g "
            "amplitude=%.9g rmse=%.9g max-abs=%.9g "
            "max-reference=%.9g reference-nonfinite=%" PRIu64 " "
            "candidate-nonfinite=%" PRIu64 "\n",
            step, index,
            weight->ane_qkv_int8_weights ? "int8" : "fp16",
            weight->ane_qkv_gpu_int8_weights ? "int8" : "bf16",
            dit->int8_qkv && weight->qkv_int8 ? "int8" : "bf16",
            weight->ane_qkv_heads, full_gpu_seconds, candidate_seconds,
            candidate_seconds > 0.0 ?
                full_gpu_seconds / candidate_seconds : 0.0,
            relative_l2, relative_l2 * 100.0, cosine, amplitude, rmse,
            maximum_error, maximum_reference,
            reference_nonfinite, candidate_nonfinite);
    return 1;
}

static int report_private_ane_attention_out_micro(
                               h3_dit *dit, h3_dit_block *weight,
                               uint32_t rows, unsigned index,
                               double full_gpu_seconds,
                               double candidate_seconds,
                               char *error, size_t error_size) {
    uint32_t ane_width = weight->ane_attention_out_width;
    size_t logical_elements = (size_t)rows * ane_width;
    if (!dit->ane_attention_out_micro_partials ||
        logical_elements > UINT32_MAX ||
        dit->ane_attention_out_micro_partial_count !=
            h3_gpu_compare_ane_f32_bf16_partial_count(
                (uint32_t)logical_elements)) {
        fail(error, error_size,
             "private ANE attention-output block %u has invalid micro "
             "scratch", index);
        return 0;
    }
    if (!h3_gpu_begin(dit->gpu) ||
        !h3_gpu_compare_ane_f32_bf16_prefix(
            dit->gpu, dit->ane_attention_out_micro_partials,
            h3_ane_linear_io_output(dit->ane_attention_out_io),
            dit->ane_attention_out_micro_reference,
            rows, ane_width, HIDDEN,
            h3_ane_linear_io_plane_rows(dit->ane_attention_out_io)) ||
        !h3_gpu_submit(dit->gpu)) {
        fail(error, error_size,
             "private ANE attention-output block %u micro comparison "
             "failed: %s", index, h3_gpu_error(dit->gpu));
        return 0;
    }
    const float *partials = h3_gpu_tensor_host_pointer(
        dit->ane_attention_out_micro_partials);
    double square_error = 0.0;
    double square_reference = 0.0;
    double square_candidate = 0.0;
    double dot_product = 0.0;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    uint64_t reference_nonfinite = 0;
    uint64_t candidate_nonfinite = 0;
    for (uint32_t group = 0;
         group < dit->ane_attention_out_micro_partial_count; group++) {
        const float *sum = partials + (size_t)group * 8;
        square_error += sum[0];
        square_reference += sum[1];
        square_candidate += sum[2];
        dot_product += sum[3];
        if (sum[4] > maximum_error) maximum_error = sum[4];
        if (sum[5] > maximum_reference) maximum_reference = sum[5];
        reference_nonfinite += (uint64_t)llroundf(sum[6]);
        candidate_nonfinite += (uint64_t)llroundf(sum[7]);
    }
    double relative_l2 = square_reference > 0.0 ?
        sqrt(square_error / square_reference) : INFINITY;
    double cosine_denominator = sqrt(
        square_reference * square_candidate);
    double cosine = cosine_denominator > 0.0 ?
        dot_product / cosine_denominator : 0.0;
    double amplitude = square_reference > 0.0 ?
        sqrt(square_candidate / square_reference) : INFINITY;
    double rmse = logical_elements ?
        sqrt(square_error / (double)logical_elements) : INFINITY;
    weight->ane_attention_out_micro_done = 1;
    fprintf(stderr,
            "h3: private ANE attention-output micro block=%u precision=%s "
            "ANE-width=%u full-gpu=%.6fs candidate=%.6fs speedup=%.6fx "
            "rel-L2=%.9g rel-L2-percent=%.6f cosine=%.9g "
            "amplitude=%.9g rmse=%.9g max-abs=%.9g "
            "max-reference=%.9g reference-nonfinite=%" PRIu64 " "
            "candidate-nonfinite=%" PRIu64 "\n",
            index,
            weight->ane_attention_out_int8_weights ? "int8" : "fp16",
            ane_width, full_gpu_seconds, candidate_seconds,
            candidate_seconds > 0.0 ?
                full_gpu_seconds / candidate_seconds : 0.0,
            relative_l2, relative_l2 * 100.0, cosine, amplitude, rmse,
            maximum_error, maximum_reference,
            reference_nonfinite, candidate_nonfinite);
    return 1;
}

static int run_private_ane_mlp(h3_dit *dit, h3_dit_block *weight,
                               h3_gpu_tensor *output,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *modulation,
                               const h3_gpu_tensor *row_map,
                               uint32_t rows, unsigned index, int step,
                               int fuse_next_attention,
                               unsigned next_index,
                               int *fused_epilogue,
                               char *error, size_t error_size) {
    uint32_t ane_rows = dit->ane_mlp_row_split_rows ?
        dit->ane_mlp_row_split_rows : rows;
    uint32_t gpu_rows = dit->ane_mlp_row_split_rows ?
        rows - dit->ane_mlp_row_split_rows : rows;
    int row_split = dit->ane_mlp_row_split_rows != 0;
    if (!weight->ane_mlp || !dit->ane_mlp_io || !dit->ane_nonfinite ||
        rows != dit->sequence ||
        h3_ane_mlp_io_rows(dit->ane_mlp_io) != ane_rows) {
        fail(error, error_size,
             "private ANE MLP block %u has invalid fixed-row runtime state",
             index);
        return 0;
    }
    if (h3_ane_mlp_inflight(weight->ane_mlp) ||
        !h3_ane_mlp_is_loaded(weight->ane_mlp)) {
        fail(error, error_size,
             "private ANE MLP block %u is not ready for evaluation", index);
        weight->ane_prediction_failures++;
        return 0;
    }

    char ane_error[512] = {0};
    double pack_started = stream_now();
    int pack_ok = row_split ?
        h3_ane_mlp_pack_rows(
            weight->ane_mlp, dit->gpu, input, gpu_rows,
            ane_error, sizeof(ane_error)) :
        h3_ane_mlp_pack(
            weight->ane_mlp, dit->gpu, input,
            ane_error, sizeof(ane_error));
    if (!pack_ok) {
        weight->ane_prediction_failures++;
        fail(error, error_size, "private ANE MLP block %u pack failed: %s",
             index, ane_error);
        return 0;
    }
    double packed = stream_now();
    weight->ane_pack_seconds += packed - pack_started;
    if (!h3_ane_mlp_start(weight->ane_mlp,
                          ane_error, sizeof(ane_error))) {
        weight->ane_prediction_failures++;
        fail(error, error_size, "private ANE MLP block %u start failed: %s",
             index, ane_error);
        return 0;
    }
    weight->ane_predictions++;

    unsigned next = next_private_ane_block(dit, index, step);
    h3_private_ane_reload_job reload_job;
    memset(&reload_job, 0, sizeof(reload_job));
    pthread_t reload_thread;
    int reload_needed = next < H3_DIT_BLOCKS && next != index &&
        !h3_ane_mlp_is_loaded(dit->blocks[next].ane_mlp);
    int reload_thread_started = 0;
    if (reload_needed) {
        reload_job.model = dit->blocks[next].ane_mlp;
        reload_thread_started = pthread_create(
            &reload_thread, NULL, private_ane_reload_worker,
            &reload_job) == 0;
    }

    char gpu_error[512] = {0};
    int gpu_ok = h3_gpu_begin(dit->gpu);
    if (gpu_ok && dit->ane_gpu_int8_mlp) {
        gpu_ok = h3_gpu_mlp_int8_bf16(
            dit->gpu, output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, input,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2, gpu_rows, HIDDEN,
            weight->ane_gpu_intermediate, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, 0, 0);
    } else if (gpu_ok) {
        gpu_ok = h3_gpu_mlp_bf16(
            dit->gpu, output, input, weight->fc1, weight->fc2,
            gpu_rows, HIDDEN, weight->ane_gpu_intermediate, HIDDEN);
    }
    if (gpu_ok) gpu_ok = h3_gpu_submit(dit->gpu);
    if (!gpu_ok)
        snprintf(gpu_error, sizeof(gpu_error), "%s", h3_gpu_error(dit->gpu));

    int ane_ok = h3_ane_mlp_wait(
        weight->ane_mlp, ane_error, sizeof(ane_error));
    if (!ane_ok) weight->ane_prediction_failures++;
    if (reload_needed && reload_thread_started) {
        int join_error = pthread_join(reload_thread, NULL);
        if (join_error) {
            reload_job.ok = 0;
            snprintf(reload_job.error, sizeof(reload_job.error),
                     "cannot join reload worker: %s", strerror(join_error));
        }
    } else if (reload_needed) {
        private_ane_reload_worker(&reload_job);
    }
    weight->ane_overlap_seconds += stream_now() - packed;

    if (!gpu_ok || !ane_ok || (reload_needed && !reload_job.ok)) {
        char unload_error[512] = {0};
        (void)h3_ane_mlp_unload(
            weight->ane_mlp, unload_error, sizeof(unload_error));
        if (!gpu_ok)
            fail(error, error_size,
                 "private ANE GPU complement block %u failed: %s",
                 index, gpu_error);
        else if (!ane_ok)
            fail(error, error_size,
                 "private ANE MLP block %u evaluation failed: %s",
                 index, ane_error);
        else {
            dit->blocks[next].ane_prediction_failures++;
            fail(error, error_size,
                 "private ANE MLP block %u reload for block %u failed: %s",
                 index, next, reload_job.error);
        }
        return 0;
    }

    int fuse_int8_qkv_input = dit->int8_qkv &&
        !dit->use_slower_unfused_int8_inputs &&
        !h3_runtime_getenv("H3_DISABLE_INT8_QKV") &&
        !h3_runtime_getenv("H3_DISABLE_FUSED_INT8_QKV_INPUT");
    int enable_fused_int8_epilogue =
        h3_runtime_getenv("H3_ENABLE_PRIVATE_ANE_FUSED_INT8_EPILOGUE") != NULL;
    int use_fused_epilogue = !row_split && fuse_next_attention &&
        next_index < H3_DIT_BLOCKS &&
        (!fuse_int8_qkv_input || enable_fused_int8_epilogue) &&
        !h3_runtime_getenv("H3_DISABLE_PRIVATE_ANE_FUSED_EPILOGUE");
    memset(h3_gpu_tensor_host_pointer(dit->ane_nonfinite), 0,
           2u * sizeof(uint32_t));
    double join_started = stream_now();
    int join_ok = h3_gpu_begin(dit->gpu);
    if (join_ok && row_split) {
        join_ok = h3_ane_mlp_unpack_rows_checked_range(
            weight->ane_mlp, dit->gpu, output, dit->ane_nonfinite,
            gpu_rows, index, ane_error, sizeof(ane_error));
    } else if (join_ok && use_fused_epilogue && fuse_int8_qkv_input) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        uint32_t padded_rows = (rows + 127u) & ~127u;
        join_ok =
            h3_gpu_add_ane_f32_bf16_transpose_gate_adaln_quantize_int8_checked(
                dit->gpu, output, dit->hidden,
                dit->int8_activation, dit->int8_activation_scales,
                output, h3_ane_mlp_io_output(dit->ane_mlp_io),
                dit->ane_nonfinite, dit->hidden, next_weight->norm1,
                modulation, next_modulation, row_map, rows, padded_rows,
                HIDDEN, h3_ane_mlp_io_plane_rows(dit->ane_mlp_io), SLOTS,
                5, 0, 1,
                h3_ane_mlp_effective_output_scale(weight->ane_mlp),
                1e-5f, index);
    } else if (join_ok && use_fused_epilogue) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        join_ok = h3_gpu_add_ane_f32_bf16_transpose_gate_adaln_checked(
            dit->gpu, output, dit->hidden, dit->mod_attention,
            output, h3_ane_mlp_io_output(dit->ane_mlp_io),
            dit->ane_nonfinite, dit->hidden, next_weight->norm1,
            modulation, next_modulation, row_map, rows, HIDDEN,
            h3_ane_mlp_io_plane_rows(dit->ane_mlp_io), SLOTS,
            5, 0, 1,
            h3_ane_mlp_effective_output_scale(weight->ane_mlp),
            1e-5f, index);
    } else if (join_ok) {
        join_ok = h3_gpu_add_ane_f32_bf16_transpose_checked(
            dit->gpu, output, output,
            h3_ane_mlp_io_output(dit->ane_mlp_io),
            dit->ane_nonfinite, rows, HIDDEN,
            h3_ane_mlp_io_plane_rows(dit->ane_mlp_io),
            h3_ane_mlp_effective_output_scale(weight->ane_mlp), index);
    }
    if (join_ok) join_ok = h3_gpu_submit(dit->gpu);
    weight->ane_join_seconds += stream_now() - join_started;
    uint32_t *range_stats = h3_gpu_tensor_host_pointer(dit->ane_nonfinite);
    uint32_t nonfinite = range_stats[0];
    float range_peak = 0.0f;
    memcpy(&range_peak, &range_stats[1], sizeof(range_peak));
    if (range_peak > weight->ane_range_peak_max)
        weight->ane_range_peak_max = range_peak;

    if (!join_ok) {
        char unload_error[512] = {0};
        (void)h3_ane_mlp_unload(
            weight->ane_mlp, unload_error, sizeof(unload_error));
        fail(error, error_size, "private ANE MLP block %u join failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    int headroom_violation = row_split &&
        dit->ane_mlp_range_headroom > 0.0f &&
        range_peak >= dit->ane_mlp_range_headroom;
    int range_violation = nonfinite != 0 || headroom_violation;
    if (range_violation) {
        int nonfinite_counted = nonfinite != 0;
        if (nonfinite_counted) weight->ane_nonfinite_failures++;
        if (!row_split) {
            char unload_error[512] = {0};
            (void)h3_ane_mlp_unload(
                weight->ane_mlp, unload_error, sizeof(unload_error));
            fail(error, error_size,
                 "private ANE MLP block %u produced non-finite output "
                 "flag %u", index, nonfinite);
            return 0;
        }
        int retry_enabled =
            h3_runtime_getenv("H3_DISABLE_PRIVATE_ANE_MLP_RANGE_RETRY") == NULL;
        for (unsigned attempt = 0;
             range_violation && retry_enabled &&
             attempt < dit->ane_mlp_range_retry_limit; attempt++) {
            if (!nonfinite && headroom_violation)
                weight->ane_range_headroom_retries++;
            float scale = h3_ane_mlp_runtime_scale(weight->ane_mlp) * 4.0f;
            double retry_started = stream_now();
            int retry_ok = h3_ane_mlp_set_runtime_scale(
                weight->ane_mlp, scale, ane_error, sizeof(ane_error));
            if (retry_ok)
                retry_ok = h3_ane_mlp_eval(
                    weight->ane_mlp, ane_error, sizeof(ane_error));
            memset(h3_gpu_tensor_host_pointer(dit->ane_nonfinite), 0,
                   2u * sizeof(uint32_t));
            if (retry_ok) retry_ok = h3_gpu_begin(dit->gpu);
            if (retry_ok)
                retry_ok = h3_ane_mlp_unpack_rows_checked_range(
                    weight->ane_mlp, dit->gpu, output,
                    dit->ane_nonfinite, gpu_rows, index,
                    ane_error, sizeof(ane_error));
            if (retry_ok) retry_ok = h3_gpu_submit(dit->gpu);
            weight->ane_range_retry_seconds += stream_now() - retry_started;
            weight->ane_range_retries++;
            if (!retry_ok) {
                char unload_error[512] = {0};
                (void)h3_ane_mlp_unload(
                    weight->ane_mlp, unload_error, sizeof(unload_error));
                weight->ane_prediction_failures++;
                fail(error, error_size,
                     "private ANE row-split block %u range retry %.9g "
                     "failed: %s", index, scale,
                     ane_error[0] ? ane_error : h3_gpu_error(dit->gpu));
                return 0;
            }
            range_stats = h3_gpu_tensor_host_pointer(dit->ane_nonfinite);
            nonfinite = range_stats[0];
            memcpy(&range_peak, &range_stats[1], sizeof(range_peak));
            if (range_peak > weight->ane_range_peak_max)
                weight->ane_range_peak_max = range_peak;
            if (nonfinite && !nonfinite_counted) {
                weight->ane_nonfinite_failures++;
                nonfinite_counted = 1;
            }
            headroom_violation = dit->ane_mlp_range_headroom > 0.0f &&
                range_peak >= dit->ane_mlp_range_headroom;
            range_violation = nonfinite != 0 || headroom_violation;
            if (!range_violation) {
                weight->ane_range_recoveries++;
                if (h3_runtime_getenv("H3_PROFILE"))
                    fprintf(stderr,
                            "h3: private ANE row-split MLP block=%u "
                            "step=%d recovered with runtime-scale=%.9g "
                            "raw-peak=%.9g\n",
                            index, step + 1, scale, range_peak);
            }
        }
        if (range_violation) {
            fprintf(stderr,
                    "h3: private ANE row-split MLP block=%u step=%d "
                    "nonfinite-flag=%u raw-peak=%.9g headroom=%.9g "
                    "runtime-scale=%.9g; recomputing full GPU-%s MLP\n",
                    index, step + 1, nonfinite, range_peak,
                    dit->ane_mlp_range_headroom,
                    h3_ane_mlp_runtime_scale(weight->ane_mlp),
                    dit->ane_gpu_int8_mlp ? "INT8" : "BF16");
            int fallback_ok = h3_gpu_begin(dit->gpu);
            if (fallback_ok && dit->ane_gpu_int8_mlp) {
                fallback_ok = h3_gpu_mlp_int8_bf16(
                    dit->gpu, output, dit->activated, dit->int8_activation,
                    dit->int8_activation_scales, input,
                    weight->fc1_int8, weight->fc1_scales,
                    weight->fc2_int8, weight->fc2_scales,
                    weight->fc1, weight->fc2, rows, HIDDEN,
                    weight->ane_gpu_intermediate, HIDDEN,
                    dit->use_slower_grouped_quantizer,
                    dit->use_slower_dynamic_fc1_k, 0, 0);
            } else if (fallback_ok) {
                fallback_ok = h3_gpu_mlp_bf16(
                    dit->gpu, output, input, weight->fc1, weight->fc2,
                    rows, HIDDEN, weight->ane_gpu_intermediate, HIDDEN);
            }
            if (fallback_ok) fallback_ok = h3_gpu_submit(dit->gpu);
            if (!fallback_ok) {
                char unload_error[512] = {0};
                (void)h3_ane_mlp_unload(
                    weight->ane_mlp, unload_error, sizeof(unload_error));
                fail(error, error_size,
                     "private ANE row-split block %u full GPU fallback "
                     "failed: %s", index, h3_gpu_error(dit->gpu));
                return 0;
            }
        }
    }
    double unload_started = stream_now();
    int unload_ok = h3_ane_mlp_unload(
        weight->ane_mlp, ane_error, sizeof(ane_error));
    if (unload_ok && next == index)
        unload_ok = h3_ane_mlp_reload(
            weight->ane_mlp, ane_error, sizeof(ane_error));
    weight->ane_unload_seconds += stream_now() - unload_started;
    if (!unload_ok) {
        weight->ane_prediction_failures++;
        fail(error, error_size, "private ANE MLP block %u unload failed: %s",
             index, ane_error);
        return 0;
    }
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size,
             "resume after private ANE MLP block %u failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    *fused_epilogue = use_fused_epilogue ?
        (fuse_int8_qkv_input ? 2 : 1) : 0;
    return 1;
}

static int run_private_ane_qkv(h3_dit *dit, h3_dit_block *weight,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *rope_cos,
                               const h3_gpu_tensor *rope_sin,
                               uint32_t rows, unsigned index, int step,
                               int input_is_quantized,
                               char *error, size_t error_size) {
    if (!weight->ane_qkv || !dit->ane_qkv_io ||
        !dit->coreml_qkv_gpu_output || !dit->coreml_nonfinite ||
        rows != dit->sequence ||
        h3_ane_linear_inflight(weight->ane_qkv) ||
        !h3_ane_linear_is_loaded(weight->ane_qkv)) {
        weight->ane_qkv_prediction_failures++;
        fail(error, error_size,
             "private ANE QKV block %u has invalid fixed-row runtime state",
             index);
        return 0;
    }
    int micro_selected = private_ane_qkv_micro_selected(
        step, error, error_size);
    if (micro_selected < 0) return 0;
    int micro = dit->ane_qkv_micro_partials &&
        !weight->ane_qkv_micro_done && micro_selected;
    double full_gpu_seconds = 0.0;
    double candidate_started = 0.0;
    if (micro) {
        if (!weight->qkv) {
            fail(error, error_size,
                 "private ANE QKV block %u micro requires full BF16 weight",
                 index);
            return 0;
        }
        if (!h3_gpu_submit(dit->gpu) || !h3_gpu_begin(dit->gpu)) {
            fail(error, error_size,
                 "private ANE QKV block %u micro pre-submit failed: %s",
                 index, h3_gpu_error(dit->gpu));
            return 0;
        }
        if (!h3_gpu_grouped_qkv_linear_rope_bf16(
                dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
                input, weight->qkv, weight->q_norm, weight->k_norm,
                rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM,
                ROPE_HALF, 1e-5f) ||
            !h3_gpu_submit(dit->gpu)) {
            fail(error, error_size,
                 "private ANE QKV block %u BF16 reference failed: %s",
                 index, h3_gpu_error(dit->gpu));
            return 0;
        }
        if (!h3_gpu_begin(dit->gpu)) {
            fail(error, error_size,
                 "private ANE QKV block %u baseline begin failed: %s",
                 index, h3_gpu_error(dit->gpu));
            return 0;
        }
        double full_gpu_started = stream_now();
        int baseline_ok = dit->int8_qkv && weight->qkv_int8 ?
            h3_gpu_grouped_qkv_linear_rope_int8(
                dit->gpu, dit->query, dit->key, dit->value,
                dit->int8_activation, dit->int8_activation_scales,
                input, weight->qkv_int8, weight->qkv_scales,
                weight->q_norm, weight->k_norm, rope_cos, rope_sin,
                rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f, 0,
                dit->use_slower_unfused_qkv_rope,
                dit->use_slower_scalar_qkv_rms,
                dit->use_slower_uncached_int8_scales) :
            h3_gpu_grouped_qkv_linear_rope_bf16(
                dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
                input, weight->qkv, weight->q_norm, weight->k_norm,
                rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM,
                ROPE_HALF, 1e-5f);
        if (!baseline_ok || !h3_gpu_submit(dit->gpu)) {
            fail(error, error_size,
                 "private ANE QKV block %u timed GPU baseline failed: %s",
                 index, h3_gpu_error(dit->gpu));
            return 0;
        }
        full_gpu_seconds = stream_now() - full_gpu_started;
        if (!h3_gpu_begin(dit->gpu)) {
            fail(error, error_size,
                 "private ANE QKV block %u candidate begin failed: %s",
                 index, h3_gpu_error(dit->gpu));
            return 0;
        }
        candidate_started = stream_now();
    }
    uint32_t ane_heads = weight->ane_qkv_heads;
    uint32_t gpu_heads = HEADS - ane_heads;
    uint32_t gpu_width = gpu_heads * 3u * HEAD_DIM;
    char ane_error[512] = {0};
    double pack_started = stream_now();
    if (!h3_ane_linear_pack(
            weight->ane_qkv, dit->gpu, input,
            ane_error, sizeof(ane_error))) {
        weight->ane_qkv_prediction_failures++;
        fail(error, error_size, "private ANE QKV block %u pack failed: %s",
             index, ane_error);
        return 0;
    }
    double packed = stream_now();
    weight->ane_qkv_pack_seconds += packed - pack_started;
    if (!h3_ane_linear_start(
            weight->ane_qkv, ane_error, sizeof(ane_error))) {
        weight->ane_qkv_prediction_failures++;
        fail(error, error_size, "private ANE QKV block %u start failed: %s",
             index, ane_error);
        return 0;
    }
    weight->ane_qkv_predictions++;

    unsigned next = next_private_ane_qkv_block(dit, index, step);
    h3_private_ane_linear_reload_job reload_job;
    memset(&reload_job, 0, sizeof(reload_job));
    pthread_t reload_thread;
    int reload_needed = next < H3_DIT_BLOCKS && next != index &&
        !h3_ane_linear_is_loaded(dit->blocks[next].ane_qkv);
    int reload_thread_started = 0;
    if (reload_needed) {
        reload_job.model = dit->blocks[next].ane_qkv;
        reload_thread_started = pthread_create(
            &reload_thread, NULL, private_ane_linear_reload_worker,
            &reload_job) == 0;
    }

    char gpu_error[512] = {0};
    int gpu_ok = h3_gpu_begin(dit->gpu);
    if (gpu_ok && weight->ane_qkv_gpu_int8_weights)
        gpu_ok = input_is_quantized ?
            h3_gpu_linear_int8_bf16_prequantized(
                dit->gpu, dit->coreml_qkv_gpu_output,
                dit->int8_activation, dit->int8_activation_scales,
                weight->coreml_qkv_gpu_weight,
                weight->ane_qkv_gpu_scales, rows, HIDDEN, gpu_width,
                dit->use_slower_uncached_int8_scales) :
            h3_gpu_linear_int8_bf16(
                dit->gpu, dit->coreml_qkv_gpu_output,
                dit->int8_activation, dit->int8_activation_scales, input,
                weight->coreml_qkv_gpu_weight,
                weight->ane_qkv_gpu_scales, rows, HIDDEN, gpu_width,
                dit->use_slower_uncached_int8_scales);
    else if (gpu_ok)
        gpu_ok = h3_gpu_linear_bf16(
            dit->gpu, dit->coreml_qkv_gpu_output, input,
            weight->coreml_qkv_gpu_weight, NULL, rows, HIDDEN, gpu_width);
    gpu_ok = gpu_ok &&
        h3_gpu_grouped_qkv_rope_bf16_segment(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->coreml_qkv_gpu_output, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, gpu_heads, HEADS, ane_heads,
            HEAD_DIM, ROPE_HALF, 1e-5f) &&
        h3_gpu_submit(dit->gpu);
    if (!gpu_ok)
        snprintf(gpu_error, sizeof(gpu_error), "%s", h3_gpu_error(dit->gpu));
    int ane_ok = h3_ane_linear_wait(
        weight->ane_qkv, ane_error, sizeof(ane_error));
    if (reload_needed && reload_thread_started) {
        int join_error = pthread_join(reload_thread, NULL);
        if (join_error) {
            reload_job.ok = 0;
            snprintf(reload_job.error, sizeof(reload_job.error),
                     "cannot join QKV reload worker: %s",
                     strerror(join_error));
        }
    } else if (reload_needed) {
        private_ane_linear_reload_worker(&reload_job);
    }
    weight->ane_qkv_overlap_seconds += stream_now() - packed;
    if (!gpu_ok || !ane_ok || (reload_needed && !reload_job.ok)) {
        char unload_error[512] = {0};
        (void)h3_ane_linear_unload(
            weight->ane_qkv, unload_error, sizeof(unload_error));
        weight->ane_qkv_prediction_failures++;
        if (!gpu_ok)
            fail(error, error_size,
                 "private ANE QKV block %u GPU complement failed: %s",
                 index, gpu_error);
        else if (!ane_ok)
            fail(error, error_size,
                 "private ANE QKV block %u evaluation failed: %s",
                 index, ane_error);
        else {
            dit->blocks[next].ane_qkv_prediction_failures++;
            fail(error, error_size,
                 "private ANE QKV block %u reload for block %u failed: %s",
                 index, next, reload_job.error);
        }
        return 0;
    }

    memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
           sizeof(uint32_t));
    double join_started = stream_now();
    int join_ok = h3_gpu_begin(dit->gpu) &&
        h3_gpu_grouped_qkv_rope_ane_f32_segment(
            dit->gpu, dit->query, dit->key, dit->value,
            h3_ane_linear_io_output(dit->ane_qkv_io),
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            dit->coreml_nonfinite, rows,
            h3_ane_linear_io_plane_rows(dit->ane_qkv_io),
            ane_heads, HEADS, 0, HEAD_DIM, ROPE_HALF, 1e-5f, index) &&
        h3_gpu_submit(dit->gpu);
    weight->ane_qkv_join_seconds += stream_now() - join_started;
    double candidate_seconds = micro ?
        stream_now() - candidate_started : 0.0;
    uint32_t nonfinite = *(uint32_t *)
        h3_gpu_tensor_host_pointer(dit->coreml_nonfinite);
    double unload_started = stream_now();
    int unload_ok = h3_ane_linear_unload(
        weight->ane_qkv, ane_error, sizeof(ane_error));
    if (unload_ok && next == index)
        unload_ok = h3_ane_linear_reload(
            weight->ane_qkv, ane_error, sizeof(ane_error));
    weight->ane_qkv_unload_seconds += stream_now() - unload_started;
    if (!join_ok) {
        fail(error, error_size, "private ANE QKV block %u join failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!unload_ok) {
        weight->ane_qkv_prediction_failures++;
        fail(error, error_size,
             "private ANE QKV block %u unload failed: %s",
             index, ane_error);
        return 0;
    }
    if (nonfinite) {
        weight->ane_qkv_nonfinite_failures++;
        fail(error, error_size,
             "private ANE QKV block %u produced non-finite output flag %u",
             index, nonfinite);
        return 0;
    }
    if (micro && !report_private_ane_qkv_micro(
            dit, weight, rows, index, step,
            full_gpu_seconds, candidate_seconds,
            error, error_size)) return 0;
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size,
             "resume after private ANE QKV block %u failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int run_private_ane_attention_out(
                               h3_dit *dit, h3_dit_block *weight,
                               h3_gpu_tensor *output,
                               const h3_gpu_tensor *input,
                               uint32_t rows, unsigned index, int step,
                               char *error, size_t error_size) {
    if (!weight->ane_attention_out &&
        block_has_private_ane_attention_out(weight)) {
        weight->ane_attention_out =
            create_private_ane_attention_out_model(
                dit, weight, index, error, error_size);
        if (!weight->ane_attention_out) {
            weight->ane_attention_out_prediction_failures++;
            return 0;
        }
    }
    if (!weight->ane_attention_out || !dit->ane_attention_out_io ||
        !dit->ane_attention_out_gpu_output || !dit->coreml_nonfinite ||
        rows != dit->sequence ||
        h3_ane_linear_inflight(weight->ane_attention_out) ||
        !h3_ane_linear_is_loaded(weight->ane_attention_out)) {
        weight->ane_attention_out_prediction_failures++;
        fail(error, error_size,
             "private ANE attention-output block %u has invalid fixed-row "
             "runtime state", index);
        return 0;
    }
    int micro = dit->ane_attention_out_micro_partials &&
        !weight->ane_attention_out_micro_done;
    double full_gpu_seconds = 0.0;
    double candidate_started = 0.0;
    if (micro) {
        if (!weight->out || !dit->ane_attention_out_micro_reference) {
            fail(error, error_size,
                 "private ANE attention-output block %u micro requires "
                 "the full BF16 weight and reference tensor", index);
            return 0;
        }
        if (!h3_gpu_submit(dit->gpu) || !h3_gpu_begin(dit->gpu)) {
            fail(error, error_size,
                 "private ANE attention-output block %u micro pre-submit "
                 "failed: %s", index, h3_gpu_error(dit->gpu));
            return 0;
        }
        double full_gpu_started = stream_now();
        if (!h3_gpu_linear_bf16(
                dit->gpu, dit->ane_attention_out_micro_reference,
                input, weight->out, NULL, rows, INNER, HIDDEN) ||
            !h3_gpu_submit(dit->gpu)) {
            fail(error, error_size,
                 "private ANE attention-output block %u full GPU micro "
                 "failed: %s", index, h3_gpu_error(dit->gpu));
            return 0;
        }
        full_gpu_seconds = stream_now() - full_gpu_started;
        if (!h3_gpu_begin(dit->gpu)) {
            fail(error, error_size,
                 "private ANE attention-output block %u candidate begin "
                 "failed: %s", index, h3_gpu_error(dit->gpu));
            return 0;
        }
        candidate_started = stream_now();
    }

    uint32_t ane_width = weight->ane_attention_out_width;
    uint32_t gpu_width = HIDDEN - ane_width;
    char ane_error[512] = {0};
    double pack_started = stream_now();
    if (!h3_ane_linear_pack(
            weight->ane_attention_out, dit->gpu, input,
            ane_error, sizeof(ane_error))) {
        weight->ane_attention_out_prediction_failures++;
        fail(error, error_size,
             "private ANE attention-output block %u pack failed: %s",
             index, ane_error);
        return 0;
    }
    double packed = stream_now();
    weight->ane_attention_out_pack_seconds += packed - pack_started;
    if (!h3_ane_linear_start(
            weight->ane_attention_out, ane_error, sizeof(ane_error))) {
        weight->ane_attention_out_prediction_failures++;
        fail(error, error_size,
             "private ANE attention-output block %u start failed: %s",
             index, ane_error);
        return 0;
    }
    weight->ane_attention_out_predictions++;

    unsigned next = next_private_ane_attention_out_block(
        dit, index, step);
    h3_private_ane_attention_out_reload_job reload_job;
    memset(&reload_job, 0, sizeof(reload_job));
    pthread_t reload_thread;
    h3_dit_block *next_block = next < H3_DIT_BLOCKS ?
        &dit->blocks[next] : NULL;
    int reload_needed = next_block && next != index &&
        (!next_block->ane_attention_out ||
         !h3_ane_linear_is_loaded(next_block->ane_attention_out));
    int reload_thread_started = 0;
    if (reload_needed) {
        reload_job.dit = dit;
        reload_job.block = next_block;
        reload_job.model = next_block->ane_attention_out;
        reload_job.index = next;
        reload_job.create = reload_job.model == NULL;
        reload_thread_started = pthread_create(
            &reload_thread, NULL, private_ane_attention_out_reload_worker,
            &reload_job) == 0;
    }

    char gpu_error[512] = {0};
    int gpu_ok = h3_gpu_begin(dit->gpu) &&
        h3_gpu_linear_bf16(
            dit->gpu, dit->ane_attention_out_gpu_output, input,
            weight->ane_attention_out_gpu_weight, NULL,
            rows, INNER, gpu_width) &&
        h3_gpu_submit(dit->gpu);
    if (!gpu_ok)
        snprintf(gpu_error, sizeof(gpu_error), "%s",
                 h3_gpu_error(dit->gpu));
    int ane_ok = h3_ane_linear_wait(
        weight->ane_attention_out, ane_error, sizeof(ane_error));
    if (reload_needed && reload_thread_started) {
        int join_error = pthread_join(reload_thread, NULL);
        if (join_error) {
            reload_job.ok = 0;
            snprintf(reload_job.error, sizeof(reload_job.error),
                     "cannot join attention-output reload worker: %s",
                     strerror(join_error));
        }
    } else if (reload_needed) {
        private_ane_attention_out_reload_worker(&reload_job);
    }
    if (reload_needed && reload_job.ok && reload_job.create)
        next_block->ane_attention_out = reload_job.model;
    weight->ane_attention_out_overlap_seconds += stream_now() - packed;
    if (!gpu_ok || !ane_ok || (reload_needed && !reload_job.ok)) {
        char unload_error[512] = {0};
        (void)h3_ane_linear_unload(
            weight->ane_attention_out,
            unload_error, sizeof(unload_error));
        weight->ane_attention_out_prediction_failures++;
        if (!gpu_ok)
            fail(error, error_size,
                 "private ANE attention-output block %u GPU complement "
                 "failed: %s", index, gpu_error);
        else if (!ane_ok)
            fail(error, error_size,
                 "private ANE attention-output block %u evaluation failed: "
                 "%s", index, ane_error);
        else {
            next_block->ane_attention_out_prediction_failures++;
            fail(error, error_size,
                 "private ANE attention-output block %u prefetch for block %u "
                 "failed: %s", index, next, reload_job.error);
        }
        return 0;
    }

    memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
           sizeof(uint32_t));
    double join_started = stream_now();
    int join_ok = h3_gpu_begin(dit->gpu) &&
        h3_gpu_join_ane_linear_output_bf16_checked(
            dit->gpu, output,
            h3_ane_linear_io_output(dit->ane_attention_out_io),
            dit->ane_attention_out_gpu_output, dit->coreml_nonfinite,
            rows, ane_width, gpu_width,
            h3_ane_linear_io_plane_rows(dit->ane_attention_out_io),
            index) &&
        h3_gpu_submit(dit->gpu);
    weight->ane_attention_out_join_seconds +=
        stream_now() - join_started;
    double candidate_seconds = micro ?
        stream_now() - candidate_started : 0.0;
    uint32_t nonfinite = *(uint32_t *)
        h3_gpu_tensor_host_pointer(dit->coreml_nonfinite);
    double unload_started = stream_now();
    int unload_ok = 1;
    if (weight->ane_attention_out_transient && next != index) {
        h3_ane_linear_free(weight->ane_attention_out);
        weight->ane_attention_out = NULL;
    } else {
        unload_ok = h3_ane_linear_unload(
            weight->ane_attention_out, ane_error, sizeof(ane_error));
        if (unload_ok && next == index)
            unload_ok = h3_ane_linear_reload(
                weight->ane_attention_out, ane_error, sizeof(ane_error));
    }
    weight->ane_attention_out_unload_seconds +=
        stream_now() - unload_started;
    if (!join_ok) {
        fail(error, error_size,
             "private ANE attention-output block %u join failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!unload_ok) {
        weight->ane_attention_out_prediction_failures++;
        fail(error, error_size,
             "private ANE attention-output block %u unload failed: %s",
             index, ane_error);
        return 0;
    }
    if (nonfinite) {
        weight->ane_attention_out_nonfinite_failures++;
        fail(error, error_size,
             "private ANE attention-output block %u produced non-finite "
             "output flag %u", index, nonfinite);
        return 0;
    }
    if (micro && !report_private_ane_attention_out_micro(
            dit, weight, rows, index, full_gpu_seconds, candidate_seconds,
            error, error_size)) return 0;
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size,
             "resume after private ANE attention-output block %u failed: %s",
             index, h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int parse_cached_gate_skip(const char *value, unsigned expected,
                                  uint8_t *active_mask) {
    if (!value || !active_mask) return 0;
    uint8_t seen[H3_DIT_BLOCKS] = {0};
    uint8_t parsed_mask[H3_DIT_BLOCKS];
    memset(parsed_mask, 1, sizeof(parsed_mask));
    const char *cursor = value;
    unsigned count = 0;
    while (*cursor) {
        while (isspace((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        errno = 0;
        char *tail = NULL;
        unsigned long parsed = strtoul(cursor, &tail, 10);
        if (errno || tail == cursor || parsed < 2 ||
            parsed >= H3_DIT_BLOCKS - 1 || seen[parsed] ||
            count >= expected) return 0;
        seen[parsed] = 1;
        parsed_mask[parsed] = 0;
        count++;
        cursor = tail;
        while (isspace((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        if (*cursor != ',') return 0;
        cursor++;
        const char *next = cursor;
        while (isspace((unsigned char)*next)) next++;
        if (!*next) return 0;
        cursor = next;
    }
    if (count != expected) return 0;
    memcpy(active_mask, parsed_mask, sizeof(parsed_mask));
    return 1;
}

static void gate_cache_hash(uint64_t *hash, const void *data, size_t bytes) {
    const uint8_t *octets = data;
    for (size_t index = 0; index < bytes; index++) {
        *hash ^= octets[index];
        *hash *= UINT64_C(1099511628211);
    }
}

static int prepare_gate_cache_identity(h3_dit *dit,
                                       const h3_sigma_schedule *sigmas) {
    const char *path = h3_runtime_getenv("H3_DIT_GATE_CACHE");
    const char *policy = h3_runtime_getenv("H3_DIT_LAYER_POLICY");
    if (!path || !*path || !strcmp(path, "0") || !sigmas ||
        sigmas->steps < 1 || sigmas->steps > H3_MAX_STEPS ||
        dit->explicit_gate_skip ||
        (policy && !strcmp(policy, "uniform")) ||
        dit->active_block_count == H3_DIT_BLOCKS) return 0;
    uint64_t weight_identity = 0;
    char detail[384];
    if (!h3_weight_store_identity(dit->weights, &weight_identity,
                                  detail, sizeof(detail))) {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr, "h3: gate cache disabled: %s\n", detail);
        return 0;
    }
    static const char policy_identity[] =
        "h3-gate-ranking-v1:first-two-and-final-protected";
    uint64_t hash = UINT64_C(1469598103934665603);
    uint64_t metadata[] = {
        weight_identity,
        H3_DIT_BLOCKS,
        dit->active_block_count,
        dit->video_condition_rows != 0,
        dit->audio_condition_rows != 0,
        (uint64_t)sigmas->steps
    };
    gate_cache_hash(&hash, policy_identity, sizeof(policy_identity));
    gate_cache_hash(&hash, metadata, sizeof(metadata));
    size_t sigma_count = (size_t)sigmas->steps + 1;
    gate_cache_hash(&hash, sigmas->video,
                    sigma_count * sizeof(sigmas->video[0]));
    gate_cache_hash(&hash, sigmas->audio,
                    sigma_count * sizeof(sigmas->audio[0]));
    dit->gate_cache_fingerprint = hash;
    dit->gate_cache_identity_ready = 1;
    return 1;
}

static void trim_cache_line(char *line) {
    size_t length = strlen(line);
    while (length && (line[length - 1] == '\n' ||
                      line[length - 1] == '\r'))
        line[--length] = '\0';
}

static int gate_cache_has_only_whitespace(FILE *stream) {
    int character;
    while ((character = fgetc(stream)) != EOF)
        if (!isspace((unsigned char)character)) return 0;
    return 1;
}

static int configure_cached_gate_skip(h3_dit *dit) {
    if (!dit->gate_cache_identity_ready) return 0;
    const char *path = h3_runtime_getenv("H3_DIT_GATE_CACHE");
    if (!path || !*path) return 0;
    FILE *stream = fopen(path, "r");
    if (!stream) {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr, "h3: gate cache miss %s (%s)\n",
                    path, strerror(errno));
        return 0;
    }
    char magic[64], identity_line[64], skip_line[256];
    int valid = fgets(magic, sizeof(magic), stream) &&
        fgets(identity_line, sizeof(identity_line), stream) &&
        fgets(skip_line, sizeof(skip_line), stream);
    if (valid) {
        trim_cache_line(magic);
        trim_cache_line(identity_line);
        trim_cache_line(skip_line);
        static const char identity_prefix[] = "fingerprint=";
        static const char skip_prefix[] = "skip=";
        valid = !strcmp(magic, "h3-gate-cache-v1") &&
            !strncmp(identity_line, identity_prefix,
                     sizeof(identity_prefix) - 1) &&
            !strncmp(skip_line, skip_prefix, sizeof(skip_prefix) - 1) &&
            gate_cache_has_only_whitespace(stream);
        if (valid) {
            const char *identity_text =
                identity_line + sizeof(identity_prefix) - 1;
            errno = 0;
            char *tail = NULL;
            unsigned long long parsed = strtoull(identity_text, &tail, 16);
            valid = !errno && tail != identity_text && !*tail &&
                (uint64_t)parsed == dit->gate_cache_fingerprint;
        }
        if (valid) {
            unsigned expected = H3_DIT_BLOCKS - dit->active_block_count;
            valid = parse_cached_gate_skip(
                skip_line + sizeof(skip_prefix) - 1,
                expected, dit->block_active);
        }
    }
    fclose(stream);
    if (!valid) {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: gate cache ignored (identity/schema mismatch) %s\n",
                    path);
        return 0;
    }
    dit->cached_gate_skip = 1;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: gate cache hit %s fingerprint=%016" PRIx64
                        "\n", path, dit->gate_cache_fingerprint);
    return 1;
}

static void write_gate_cache(const h3_dit *dit) {
    if (!dit->gate_cache_identity_ready || dit->cached_gate_skip ||
        dit->explicit_gate_skip) return;
    const char *path = h3_runtime_getenv("H3_DIT_GATE_CACHE");
    if (!path || !*path) return;
    size_t path_length = strlen(path);
    static const char suffix[] = ".tmp.XXXXXX";
    if (path_length > SIZE_MAX - sizeof(suffix)) return;
    char *temporary = malloc(path_length + sizeof(suffix));
    if (!temporary) return;
    snprintf(temporary, path_length + sizeof(suffix), "%s%s", path, suffix);
    int descriptor = mkstemp(temporary);
    if (descriptor < 0) {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr, "h3: cannot create gate cache %s: %s\n",
                    path, strerror(errno));
        free(temporary);
        return;
    }
    FILE *stream = fdopen(descriptor, "w");
    if (!stream) {
        close(descriptor);
        unlink(temporary);
        free(temporary);
        return;
    }
    int ok = fprintf(stream, "h3-gate-cache-v1\n") > 0 &&
        fprintf(stream, "fingerprint=%016" PRIx64 "\n",
                dit->gate_cache_fingerprint) > 0 &&
        fprintf(stream, "skip=") > 0;
    unsigned written = 0;
    for (unsigned block = 0; ok && block < H3_DIT_BLOCKS; block++) {
        if (dit->block_active[block]) continue;
        if (written && fputc(',', stream) == EOF) ok = 0;
        if (ok && fprintf(stream, "%u", block) < 0) ok = 0;
        written++;
    }
    if (ok && fputc('\n', stream) == EOF) ok = 0;
    if (ok && fflush(stream) != 0) ok = 0;
    if (ok && fsync(descriptor) != 0) ok = 0;
    if (fclose(stream) != 0) ok = 0;
    if (ok && rename(temporary, path) == 0) {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr, "h3: wrote gate cache %s fingerprint=%016"
                            PRIx64 "\n",
                    path, dit->gate_cache_fingerprint);
    } else {
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr, "h3: cannot publish gate cache %s: %s\n",
                    path, strerror(errno));
        unlink(temporary);
    }
    free(temporary);
}

static unsigned first_active_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static unsigned active_block_ordinal(const h3_dit *dit, unsigned block) {
    unsigned ordinal = 0;
    if (!dit || block >= H3_DIT_BLOCKS || !dit->block_active[block])
        return UINT_MAX;
    for (unsigned index = 0; index < block; index++)
        if (dit->block_active[index]) ordinal++;
    return ordinal;
}

static int stream_block_pinned(const h3_dit *dit, unsigned block) {
    unsigned ordinal = active_block_ordinal(dit, block);
    return ordinal != UINT_MAX && ordinal < (unsigned)dit->ssd_pinned_prefix;
}

static unsigned first_streamed_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block] && !stream_block_pinned(dit, block))
            return block;
    return H3_DIT_BLOCKS;
}

static unsigned next_streamed_block(const h3_dit *dit, unsigned current) {
    for (unsigned offset = 1; offset <= H3_DIT_BLOCKS; offset++) {
        unsigned block = (current + offset) % H3_DIT_BLOCKS;
        if (dit->block_active[block] && !stream_block_pinned(dit, block))
            return block;
    }
    return H3_DIT_BLOCKS;
}

/* Return the BF16 payload size of one complete transformer block. This is
 * intentionally derived from the compiled model constants rather than a
 * checkpoint-specific file size so the policy remains valid with sharding. */
static uint64_t ssd_full_block_bytes(const h3_dit *dit) {
    uint64_t elements = (uint64_t)HIDDEN * 2u +
        (uint64_t)HEAD_DIM * 2u;
    if (dit && dit->ssd_quantized) {
        uint64_t weights = (uint64_t)INNER * 3u * HIDDEN +
            (uint64_t)HIDDEN * INNER +
            (uint64_t)FFN * 2u * HIDDEN +
            (uint64_t)HIDDEN * FFN;
        uint64_t scales = (uint64_t)INNER * 3u + HIDDEN +
            (uint64_t)FFN * 2u + HIDDEN;
        if (weights > UINT64_MAX - elements ||
            scales > (UINT64_MAX - elements - weights) / 4u)
            return UINT64_MAX;
        return elements * sizeof(uint16_t) + weights + scales * sizeof(float);
    }
    uint64_t matrices = (uint64_t)INNER * 3u * HIDDEN +
        (uint64_t)HIDDEN * INNER +
        (uint64_t)FFN * 2u * HIDDEN +
        (uint64_t)HIDDEN * FFN;
    if (matrices > UINT64_MAX - elements ||
        matrices + elements > UINT64_MAX / sizeof(uint16_t))
        return UINT64_MAX;
    return (matrices + elements) * sizeof(uint16_t);
}

static uint64_t ssd_activation_reserve_bytes(const h3_dit *dit) {
    if (!dit) return 0;
    /* The request arena contains multiple HIDDEN and INNER work buffers. Use
     * a conservative 8/4-buffer envelope plus a fixed 4 GiB margin for the
     * VAE, text encoder and allocator slack. This is a placement hint, not a
     * claim about the process hard limit. */
    uint64_t hidden = (uint64_t)dit->sequence * HIDDEN;
    uint64_t inner = (uint64_t)dit->sequence * INNER;
    if (hidden > UINT64_MAX / (8u * sizeof(uint16_t)) ||
        inner > UINT64_MAX / (4u * sizeof(uint16_t))) return UINT64_MAX;
    uint64_t bytes = hidden * 8u * sizeof(uint16_t) +
        inner * 4u * sizeof(uint16_t);
    return bytes > (4ull << 30) ? bytes : (4ull << 30);
}

static int configure_ssd_pinned_prefix(h3_dit *dit, int requested,
                                       uint64_t budget,
                                       char *error, size_t error_size) {
    if (!dit || requested < 0 || requested > H3_DIT_BLOCKS) {
        fail(error, error_size, "invalid H3 SSD pinned prefix");
        return 0;
    }
    if (!dit->ssd_streaming && (requested || budget)) {
        fail(error, error_size,
             "SSD pinned prefix/budget requires SSD streaming");
        return 0;
    }
    h3_stream_plan plan;
    h3_stream_plan_status status = h3_stream_plan_build(
        budget, ssd_activation_reserve_bytes(dit), ssd_full_block_bytes(dit),
        dit->active_block_count, (unsigned)requested, &plan);
    if (status != H3_STREAM_PLAN_OK) {
        fail(error, error_size, "cannot configure H3 SSD streaming plan: %s",
             h3_stream_plan_status_string(status));
        return 0;
    }
    dit->ssd_pinned_prefix = (int)plan.pinned_blocks;
    dit->ssd_memory_budget_bytes = budget;
    if (h3_runtime_getenv("H3_PROFILE")) {
        fprintf(stderr,
                "h3: SSD pinned-prefix=%d/%u budget=%llu block=%.3f GiB "
                "activation-reserve=%.3f GiB\n",
                dit->ssd_pinned_prefix, dit->active_block_count,
                (unsigned long long)budget,
                (double)plan.block_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)plan.activation_reserve_bytes /
                    (1024.0 * 1024.0 * 1024.0));
    }
    return 1;
}

static unsigned next_active_block(const h3_dit *dit, unsigned current) {
    for (unsigned block = current + 1; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static int configure_gate_ranked_blocks(h3_dit *dit) {
    const char *policy = h3_runtime_getenv("H3_DIT_LAYER_POLICY");
    if ((policy && !strcmp(policy, "uniform")) ||
        dit->active_block_count == H3_DIT_BLOCKS) return 1;
    typedef struct { unsigned block; double score; } block_score;
    /* The first two and final blocks establish/close the residual stream.
     * Block 1 has a small gate but proved structurally essential in decoded
     * A/B renders, so magnitude ranking must not treat it as disposable. */
    block_score scores[H3_DIT_BLOCKS - 3];
    for (unsigned block = 2; block + 1 < H3_DIT_BLOCKS; block++) {
        double score = h3_dit_schedule_gate_score(dit->schedule, block);
        if (score < 0.0) return 0;
        scores[block - 2] = (block_score){block, score};
    }
    unsigned count = H3_DIT_BLOCKS - 3;
    for (unsigned left = 0; left < count; left++) {
        unsigned least = left;
        for (unsigned right = left + 1; right < count; right++)
            if (scores[right].score < scores[least].score) least = right;
        block_score temporary = scores[left];
        scores[left] = scores[least];
        scores[least] = temporary;
    }
    memset(dit->block_active, 1, sizeof(dit->block_active));
    unsigned skipped = H3_DIT_BLOCKS - dit->active_block_count;
    for (unsigned index = 0; index < skipped; index++)
        dit->block_active[scores[index].block] = 0;
    if (h3_runtime_getenv("H3_PROFILE")) {
        fprintf(stderr, "h3: gate-ranked DiT skips");
        for (unsigned index = 0; index < skipped; index++)
            fprintf(stderr, " %u(%.4g)", scores[index].block,
                    scores[index].score);
        fputc('\n', stderr);
    }
    return 1;
}

static void profile_step_gate_scores(const h3_dit *dit) {
    if (!h3_runtime_getenv("H3_PROFILE_STEP_GATES")) return;
    int steps = h3_dit_schedule_steps(dit->schedule);
    double scores[H3_MAX_STEPS * H3_DIT_MODALITIES];
    double branch_scores[H3_MAX_STEPS * H3_DIT_MODALITIES * 2];
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (!dit->block_active[block] ||
            !h3_dit_schedule_gate_scores(
                dit->schedule, block, scores,
                (size_t)steps * H3_DIT_MODALITIES) ||
            !h3_dit_schedule_gate_branch_scores(
                dit->schedule, block, branch_scores,
                (size_t)steps * H3_DIT_MODALITIES * 2)) continue;
        fprintf(stderr, "h3 step gates: block=%u", block);
        for (int step = 0; step < steps; step++) {
            size_t offset = (size_t)step * H3_DIT_MODALITIES;
            fprintf(stderr, " s%d=%.6g/%.6g/%.6g", step + 1,
                    scores[offset], scores[offset + 1], scores[offset + 2]);
        }
        fputc('\n', stderr);
        fprintf(stderr, "h3 step branch gates: block=%u", block);
        for (int step = 0; step < steps; step++) {
            size_t offset = (size_t)step * H3_DIT_MODALITIES * 2;
            fprintf(stderr,
                    " s%d=v:%.6g/%.6g,t:%.6g/%.6g,a:%.6g/%.6g",
                    step + 1,
                    branch_scores[offset], branch_scores[offset + 1],
                    branch_scores[offset + 2], branch_scores[offset + 3],
                    branch_scores[offset + 4], branch_scores[offset + 5]);
        }
        fputc('\n', stderr);
    }
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    unsigned coreml_blocks = 0;
    unsigned requested_capture_block = UINT_MAX;
    int requested_capture_found = 0;
    if (dit->coreml_same_loaded_ab) {
        const char *value = h3_runtime_getenv("H3_BENCH_COREML_CAPTURE_BLOCK");
        if (value && *value) {
            char *tail = NULL;
            unsigned long parsed = strtoul(value, &tail, 10);
            if (tail == value || *tail || parsed >= H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "H3_BENCH_COREML_CAPTURE_BLOCK must be in [0, %u]",
                     H3_DIT_BLOCKS - 1);
                return 0;
            }
            requested_capture_block = (unsigned)parsed;
        }
    }
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (!dit->block_active[index]) {
            report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS);
            continue;
        }
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->ssd_streaming) {
            if (!load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size))
                return 0;
            if (stream_block_pinned(dit, index)) {
                if (dit->ssd_quantized) {
                    if (!load_quantized_block(
                            dit, &dit->blocks[index], index,
                            error, error_size)) return 0;
                } else if (!load_block_matrices(
                               dit, &dit->blocks[index], prefix, 1,
                               error, error_size)) return 0;
            } else if (!prepare_stream_layer(
                           dit, index, error, error_size)) return 0;
        } else {
            int use_coreml_mlp = coreml_mlp_enabled(
                index, error, error_size);
            int use_private_ane_mlp = private_ane_mlp_block_enabled(
                index, error, error_size);
            int use_coreml_qkv = coreml_qkv_enabled(
                index, error, error_size);
            int use_private_ane_qkv = private_ane_qkv_enabled(
                index, error, error_size);
            int use_private_ane_attention_out =
                private_ane_attention_out_enabled(
                    index, error, error_size);
            if (use_private_ane_mlp < 0 ||
                use_coreml_mlp < 0 || use_coreml_qkv < 0 ||
                use_private_ane_qkv < 0 ||
                use_private_ane_attention_out < 0 ||
                (use_coreml_mlp && use_private_ane_mlp) ||
                (use_coreml_qkv && use_private_ane_qkv) ||
                !load_block(dit, &dit->blocks[index], prefix,
                            !use_coreml_mlp && !use_private_ane_mlp,
                            error, error_size)) return 0;
            if (use_private_ane_mlp &&
                !configure_private_ane_mlp(
                    dit, &dit->blocks[index], index,
                    error, error_size)) return 0;
            if (use_coreml_mlp) {
                if (!configure_coreml_mlp(
                        dit, &dit->blocks[index], index,
                        error, error_size) ||
                    !load_coreml_benchmark_full_mlp(
                        dit, &dit->blocks[index], prefix,
                        error, error_size)) return 0;
                if (!coreml_blocks)
                    dit->coreml_benchmark_capture_block = index;
                if (index == requested_capture_block)
                    requested_capture_found = 1;
                coreml_blocks++;
            }
            if (use_coreml_qkv &&
                !configure_coreml_qkv(
                    dit, &dit->blocks[index], index,
                    error, error_size)) return 0;
            if (use_private_ane_qkv &&
                !configure_private_ane_qkv(
                    dit, &dit->blocks[index], index,
                    error, error_size)) return 0;
            if (use_private_ane_attention_out &&
                !configure_private_ane_attention_out(
                    dit, &dit->blocks[index], index,
                    error, error_size)) return 0;
            if (dit->int8_mlp && !dit->blocks[index].coreml_mlp &&
                !dit->blocks[index].ane_mlp &&
                !quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_qkv && !dit->blocks[index].coreml_qkv &&
                !dit->blocks[index].ane_qkv &&
                !quantize_block_qkv(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_attention_out &&
                !block_has_private_ane_attention_out(
                    &dit->blocks[index]) &&
                !quantize_block_attention_out(
                    dit, &dit->blocks[index], error, error_size)) return 0;
        }
        report(progress, opaque, "load transformer core", (int)index + 1,
               H3_DIT_BLOCKS);
    }
    if (dit->coreml_same_loaded_ab) {
        if (!coreml_blocks) {
            fail(error, error_size,
                 "H3_BENCH_COREML_SAME_LOADED_AB requires at least one "
                 "active Core ML block");
            return 0;
        }
        if (requested_capture_block != UINT_MAX) {
            if (!requested_capture_found) {
                fail(error, error_size,
                     "H3_BENCH_COREML_CAPTURE_BLOCK %u is not an active "
                     "Core ML block", requested_capture_block);
                return 0;
            }
            dit->coreml_benchmark_capture_block = requested_capture_block;
        }
    }
    if (dit->ssd_streaming) {
        if (!allocate_stream_slot(dit, &dit->stream_slots[0],
                                  error, error_size) ||
            !allocate_stream_slot(dit, &dit->stream_slots[1],
                                  error, error_size)) return 0;
        unsigned first = first_streamed_block(dit);
        if (first == H3_DIT_BLOCKS) {
            fail(error, error_size, "SSD stream has no active DiT block");
            return 0;
        }
        h3_dit_stream_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!read_stream_layer(&job)) {
            fail(error, error_size, "cannot prime DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = first;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;
        dit->stream_read_seconds += job.seconds;
    }
    dit->video_patch_w = f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    if (dit->bf16_final && dit->final_video_w && dit->final_video_b &&
        dit->final_audio_w && dit->final_audio_b) {
        h3_gpu_tensor *source[4] = {
            dit->final_video_w, dit->final_video_b,
            dit->final_audio_w, dit->final_audio_b
        };
        size_t elements[4] = {
            (size_t)VIDEO_PATCH * HIDDEN, VIDEO_PATCH,
            (size_t)AUDIO_CHANNELS * HIDDEN, AUDIO_CHANNELS
        };
        h3_gpu_tensor *target[4] = {0};
        int ok = 1;
        for (unsigned index = 0; index < 4; index++) {
            target[index] = h3_gpu_tensor_new_bf16(dit->gpu,
                                                    elements[index]);
            if (!target[index]) ok = 0;
        }
        if (ok) ok = h3_gpu_begin(dit->gpu);
        for (unsigned index = 0; ok && index < 4; index++)
            ok = h3_gpu_cast_f32_to_bf16(dit->gpu, target[index],
                                         source[index],
                                         (uint32_t)elements[index]);
        if (ok) ok = h3_gpu_submit(dit->gpu);
        if (ok) {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(source[index]);
            dit->final_video_w = target[0];
            dit->final_video_b = target[1];
            dit->final_audio_w = target[2];
            dit->final_audio_b = target[3];
        } else {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(target[index]);
            fail(error, error_size, "cannot convert DiT final weights: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int finish_coreml_loads(h3_dit *dit,
                               char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (dit->blocks[index].coreml_mlp &&
            !h3_coreml_mlp_finish_loading(
                dit->blocks[index].coreml_mlp, error, error_size)) return 0;
        if (dit->blocks[index].coreml_qkv &&
            !h3_coreml_mlp_finish_loading(
                dit->blocks[index].coreml_qkv, error, error_size)) return 0;
    }
    unsigned warmups = 0;
    const char *value = h3_runtime_getenv("H3_COREML_WARMUPS");
    if (value && *value) {
        char *tail = NULL;
        unsigned long parsed = strtoul(value, &tail, 10);
        if (tail == value || *tail || parsed > 100) {
            fail(error, error_size, "H3_COREML_WARMUPS must be in [0, 100]");
            return 0;
        }
        warmups = (unsigned)parsed;
    }
    for (unsigned round = 0; round < warmups; round++) {
        for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
            if (dit->blocks[index].coreml_mlp &&
                !h3_coreml_mlp_warmup(
                    dit->blocks[index].coreml_mlp, 1,
                    error, error_size)) return 0;
            if (dit->blocks[index].coreml_qkv &&
                !h3_coreml_mlp_warmup(
                    dit->blocks[index].coreml_qkv, 1,
                    error, error_size)) return 0;
        }
    }
    return 1;
}

static int finish_private_ane_loads(h3_dit *dit,
                                    char *error, size_t error_size) {
    if (!dit->ane_mlp_io) return 1;
    if (!dit->ane_mlp_blocks ||
        dit->ane_mlp_blocks > dit->active_block_count) {
        fail(error, error_size,
             "private ANE plan covers invalid block count %u of %u active",
             dit->ane_mlp_blocks, dit->active_block_count);
        return 0;
    }
    unsigned first = H3_DIT_BLOCKS;
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (dit->block_active[index] && dit->blocks[index].ane_mlp) {
            first = index;
            break;
        }
    }
    if (first >= H3_DIT_BLOCKS) {
        fail(error, error_size, "private ANE plan has no first active block");
        return 0;
    }
    return h3_ane_mlp_reload(
        dit->blocks[first].ane_mlp, error, error_size);
}

static int finish_private_ane_qkv_loads(h3_dit *dit,
                                        char *error, size_t error_size) {
    if (!dit->ane_qkv_io) return 1;
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (dit->block_active[index] && dit->blocks[index].ane_qkv)
            return h3_ane_linear_reload(
                dit->blocks[index].ane_qkv, error, error_size);
    }
    fail(error, error_size,
         "private ANE QKV plan has no active configured block");
    return 0;
}

static int finish_private_ane_attention_out_loads(
                                        h3_dit *dit,
                                        char *error, size_t error_size) {
    if (!dit->ane_attention_out_io) return 1;
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (dit->block_active[index] &&
            block_has_private_ane_attention_out(&dit->blocks[index])) {
            h3_dit_block *block = &dit->blocks[index];
            if (block->ane_attention_out)
                return h3_ane_linear_reload(
                    block->ane_attention_out, error, error_size);
            block->ane_attention_out =
                create_private_ane_attention_out_model(
                    dit, block, index, error, error_size);
            return block->ane_attention_out != NULL;
        }
    }
    fail(error, error_size,
         "private ANE attention-output plan has no active configured block");
    return 0;
}

static int eager_coreml_load_wait(h3_dit *dit,
                                  char *error, size_t error_size) {
    if (!finish_private_ane_loads(dit, error, error_size)) return 0;
    if (!finish_private_ane_qkv_loads(dit, error, error_size)) return 0;
    if (!finish_private_ane_attention_out_loads(
            dit, error, error_size)) return 0;
    const char *value = h3_runtime_getenv("H3_COREML_LAZY_LOAD");
    if (value && *value && strcmp(value, "0")) return 1;
    return finish_coreml_loads(dit, error, error_size);
}


static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
    dit->activation_aliases = !h3_runtime_getenv("H3_DISABLE_DIT_ACTIVATION_ALIAS");
    dit->fused_patch_projection =
        !h3_runtime_getenv("H3_DISABLE_FUSED_PATCH_CAST") && !h3_runtime_getenv("H3_SCALAR_PATCH");
    dit->fused_patch_pack = dit->fused_patch_projection &&
        !h3_runtime_getenv("H3_DISABLE_FUSED_PATCH_PACK");
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        F32(final_audio_inverse, audio),
        F32(final_video_inverse, video),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_pack) {
        dit->video_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected || !dit->audio_projected) {
            fail(error, error_size,
                 "cannot allocate packed patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_projection) {
        dit->video_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected_f32 || !dit->audio_projected_f32) {
            fail(error, error_size,
                 "cannot allocate separate patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->activation_aliases) {
        dit->attention_heads = dit->qkv;
        dit->mod_mlp = dit->qkv;
        dit->mlp_output = NULL;
    } else {
        dit->attention_heads = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * INNER);
        dit->mod_mlp = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
    }
    if (!dit->attention_heads || !dit->mod_mlp ||
        (!dit->activation_aliases && !dit->mlp_output)) {
        fail(error, error_size,
             "cannot allocate DiT activation buffers: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (dit->coreml_same_loaded_ab) {
        size_t elements = sequence * HIDDEN;
        dit->coreml_benchmark_mlp_input = h3_gpu_tensor_new_bf16(
            dit->gpu, elements);
        dit->coreml_benchmark_mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, elements);
        if (!dit->coreml_benchmark_mlp_input ||
            !dit->coreml_benchmark_mlp_output) {
            fail(error, error_size,
                 "cannot allocate Core ML benchmark MLP captures: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        dit->coreml_benchmark_capture_elements = elements;
    }
    if (h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_input = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_input = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_input || !dit->final_video_input) {
            fail(error, error_size,
                 "cannot allocate separate final DiT slices: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final || h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_HEAD") ||
        h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_norm || !dit->final_video_norm) {
            fail(error, error_size,
                 "cannot allocate separate final DiT normalization: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final) {
        dit->final_audio_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio * HIDDEN);
        dit->final_video_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video * HIDDEN);
        dit->audio_output = h3_gpu_tensor_new_f32(
            dit->gpu, audio * AUDIO_CHANNELS);
        dit->video_output = h3_gpu_tensor_new_f32(
            dit->gpu, video * VIDEO_PATCH);
        if (!dit->final_audio_f32 || !dit->final_video_f32 ||
            !dit->audio_output || !dit->video_output) {
            fail(error, error_size,
                 "cannot allocate F32 DiT final activations: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    }
    if (!dit->fused_mlp || dit->nax_mlp || dit->int8_mlp ||
        dit->ane_gpu_int8_mlp) {
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if ((!dit->fused_mlp && !dit->fc1) || !dit->activated) {
            fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->int8_mlp || dit->ane_gpu_int8_mlp || dit->int8_qkv ||
        dit->int8_attention_out) {
        size_t padded_sequence = (sequence + 127) & ~(size_t)127;
        dit->int8_activation = h3_gpu_tensor_new_i8(
            dit->gpu, padded_sequence * FFN);
        dit->int8_activation_scales = h3_gpu_tensor_new_f32(
            dit->gpu, padded_sequence * (FFN / 1024));
        if (!dit->int8_activation || !dit->int8_activation_scales) {
            fail(error, error_size,
                 "cannot allocate int8 DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->token_reduction) {
        size_t full_elements = sequence * HIDDEN;
        size_t qkv_capacity = sequence * INNER * 3;
        size_t qkv_used = (size_t)dit->reduced_sequence * INNER * 3;
        size_t baseline_elements =
            (size_t)dit->token_baseline_rows * HIDDEN;
        size_t attention_capacity = sequence * HIDDEN;
        size_t attention_used =
            (size_t)dit->reduced_sequence * HIDDEN;
        dit->token_original_in_qkv =
            qkv_used <= qkv_capacity &&
            full_elements <= qkv_capacity - qkv_used &&
            qkv_used <= UINT32_MAX &&
            full_elements <= UINT32_MAX - qkv_used;
        if (dit->token_original_in_qkv)
            dit->token_original_offset = qkv_used;
        else
            dit->token_original = h3_gpu_tensor_new_bf16(
                dit->gpu, full_elements);
        dit->token_baseline_offset = attention_used;
        if (attention_used > attention_capacity ||
            baseline_elements > attention_capacity - attention_used ||
            attention_used > UINT32_MAX ||
            baseline_elements > UINT32_MAX - attention_used ||
            (!dit->token_original_in_qkv && !dit->token_original)) {
            fail(error, error_size,
                 "cannot allocate token-reduction residual state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->core_reuse_interval > 1 || dit->first_block_cache ||
        dit->tea_cache) {
        dit->core_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->core_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->core_input || !dit->core_residual) {
            fail(error, error_size,
                 "cannot allocate DiT core residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->first_block_cache) {
        size_t elements = sequence * HIDDEN;
        if (elements > UINT32_MAX) {
            fail(error, error_size,
                 "FirstBlockCache activation size exceeds Metal limits");
            return 0;
        }
        dit->first_block_cache_partial_count =
            h3_gpu_relative_l1_partial_count((uint32_t)elements);
        dit->first_block_cache_previous = h3_gpu_tensor_new_bf16(
            dit->gpu, elements);
        dit->first_block_cache_partials = h3_gpu_tensor_new_f32(
            dit->gpu, (size_t)dit->first_block_cache_partial_count * 2);
        dit->first_block_cache_host_partials = malloc(
            (size_t)dit->first_block_cache_partial_count * 2 * sizeof(float));
        if (!dit->first_block_cache_partial_count ||
            !dit->first_block_cache_previous ||
            !dit->first_block_cache_partials ||
            !dit->first_block_cache_host_partials) {
            fail(error, error_size,
                 "cannot allocate FirstBlockCache state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->tea_cache) {
        size_t elements = sequence * HIDDEN;
        size_t audio_elements = (size_t)dit->audio_rows * HIDDEN;
        if (elements > UINT32_MAX || audio_elements > UINT32_MAX) {
            fail(error, error_size,
                 "TeaCache activation size exceeds Metal limits");
            return 0;
        }
        dit->tea_cache_partial_count =
            h3_gpu_relative_l1_partial_count((uint32_t)elements);
        if (dit->tea_cache_audio_threshold > 0.0f)
            dit->tea_cache_audio_partial_count =
                h3_gpu_relative_l1_partial_count((uint32_t)audio_elements);
        size_t partial_groups = (size_t)dit->tea_cache_partial_count +
            dit->tea_cache_audio_partial_count;
        dit->tea_cache_previous = h3_gpu_tensor_new_bf16(
            dit->gpu, elements);
        dit->tea_cache_partials = h3_gpu_tensor_new_f32(
            dit->gpu, partial_groups * 2);
        dit->tea_cache_host_partials = malloc(
            partial_groups * 2 * sizeof(float));
        if (!dit->tea_cache_partial_count || !dit->tea_cache_previous ||
            !dit->tea_cache_partials || !dit->tea_cache_host_partials) {
            fail(error, error_size, "cannot allocate TeaCache state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->sol_attention) {
        size_t blocks = (sequence + 63) / 64;
        size_t summary_elements = (size_t)HEADS * blocks * HEAD_DIM;
        size_t threshold_elements = (size_t)HEADS * blocks;
        size_t route_elements = threshold_elements * blocks;
        dit->sol_route_elements = route_elements;
        dit->sol_query_centroids = h3_gpu_tensor_new_f32(
            dit->gpu, summary_elements);
        dit->sol_key_centroids = h3_gpu_tensor_new_bf16(
            dit->gpu, summary_elements);
        dit->sol_value_sums = h3_gpu_tensor_new_bf16(
            dit->gpu, summary_elements);
        dit->sol_thresholds = h3_gpu_tensor_new_f32(
            dit->gpu, threshold_elements);
        dit->sol_routes = h3_gpu_tensor_new_f32(
            dit->gpu, route_elements);
        if (!dit->sol_query_centroids || !dit->sol_key_centroids ||
            !dit->sol_value_sums || !dit->sol_thresholds ||
            !dit->sol_routes) {
            fail(error, error_size,
                 "cannot allocate standalone Sol-Attn scratch: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        if (h3_runtime_getenv("H3_SOL_BQ64_VERIFY")) {
            size_t elements = sequence * INNER;
            if (elements > UINT32_MAX) {
                fail(error, error_size,
                     "BQ64 verification tensor exceeds Metal limits");
                return 0;
            }
            dit->sol_verify_elements = (uint32_t)elements;
            dit->sol_verify_partial_count =
                h3_gpu_compare_bf16_partial_count((uint32_t)elements);
            dit->sol_verify_dense = h3_gpu_tensor_new_bf16(
                dit->gpu, elements);
            dit->sol_verify_partials = h3_gpu_tensor_new_f32(
                dit->gpu, (size_t)dit->sol_verify_partial_count * 4);
            if (!dit->sol_verify_partial_count || !dit->sol_verify_dense ||
                !dit->sol_verify_partials) {
                fail(error, error_size,
                     "cannot allocate BQ64 verification scratch: %s",
                     h3_gpu_error(dit->gpu));
                return 0;
            }
        }
    }
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    report(state->callback, state->opaque, "precompute AdaLN", completed, total);
}

static h3_dit *load_dit(const char *weight_directory,
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
                        int defer_request_state,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        h3_dit_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas ||
        (defer_request_state != 0 && defer_request_state != 1) ||
        (ssd_streaming != 0 && ssd_streaming != 1) ||
        (ssd_quantized_cache_directory && !ssd_streaming) ||
        ssd_pinned_prefix < 0 || ssd_pinned_prefix > H3_DIT_BLOCKS ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f ||
        active_blocks < H3_DIT_BLOCKS / 2 ||
        active_blocks > H3_DIT_BLOCKS || core_reuse_interval < 1 ||
        core_reuse_interval > 6) {
        fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->weight_directory = strdup(weight_directory);
    if (!dit->weight_directory) {
        fail(error, error_size, "out of memory copying DiT weight directory");
        h3_dit_free(dit);
        return NULL;
    }
    dit->fused_mlp = h3_runtime_getenv("H3_DISABLE_FUSED_MLP") == NULL;
    h3_coreml_mlp_fallback_options fallback_options;
    if (!h3_coreml_mlp_configure_fallback(
            h3_runtime_getenv("H3_COREML_GPU_FALLBACK"),
            h3_runtime_getenv("H3_COREML_FORCE_GPU_FALLBACK"),
            h3_runtime_getenv("H3_COREML_FALLBACK_LATCH"),
            &fallback_options, error, error_size)) goto failed;
    dit->coreml_gpu_fallback = fallback_options.enabled;
    dit->coreml_force_gpu_fallback = fallback_options.forced;
    dit->coreml_fallback_latch = fallback_options.latch;
    dit->coreml_same_loaded_ab =
        h3_runtime_getenv("H3_BENCH_COREML_SAME_LOADED_AB") != NULL;
    if (dit->coreml_same_loaded_ab &&
        (!dit->coreml_gpu_fallback || dit->coreml_force_gpu_fallback ||
         dit->coreml_fallback_latch)) {
        fail(error, error_size,
             "H3_BENCH_COREML_SAME_LOADED_AB requires fallback enabled "
             "without global force or latch");
        goto failed;
    }
    /* The released final heads are F32, but their inputs are already BF16.
     * Converting these small weights once selects the Iris-derived tiled
     * linear and eliminates two full-width casts plus the scalar F32 kernel.
     * Keep the old path available for close-reference diagnosis. */
    dit->bf16_final = h3_runtime_getenv("H3_DIT_F32_FINAL") == NULL;
    dit->core_reuse_interval = core_reuse_interval;
    dit->ssd_streaming = ssd_streaming;
    dit->ssd_quantized = ssd_quantized_cache_directory &&
        *ssd_quantized_cache_directory;
    dit->ssd_pinned_prefix = ssd_pinned_prefix;
    dit->ssd_memory_budget_bytes = ssd_memory_budget_bytes;
    dit->spatial_rope_scale = spatial_rope_scale;
    configure_active_blocks(dit, active_blocks);
    if (!configure_explicit_gate_skip(dit, error, error_size) ||
        !copy_layout(dit, layout, error, error_size) ||
        !validate_layout(dit, text, error, error_size) ||
        !configure_token_reduction(dit, token_reduction,
                                   error, error_size) ||
        !configure_first_block_cache(dit, error, error_size) ||
        !configure_tea_cache(dit, error, error_size) ||
        !configure_sol_attention(dit, error, error_size)) goto failed;
    if (dit->first_block_cache && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: FirstBlockCache enabled at threshold %.6g; "
                "max consecutive hits %u%s; final step forced fresh\n",
                dit->first_block_cache_threshold,
                dit->first_block_cache_max_hits,
                dit->first_block_cache_max_hits ? "" : " (unlimited)");
    if (dit->tea_cache && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: TeaCache enabled at threshold %.6g; retain=%u "
                "cooldown=%u max consecutive hits %u%s; audio threshold "
                "%.6g%s; final step fresh\n",
                dit->tea_cache_threshold, dit->tea_cache_retain_steps,
                dit->tea_cache_cooldown_steps, dit->tea_cache_max_hits,
                dit->tea_cache_max_hits ? "" : " (unlimited)",
                dit->tea_cache_audio_threshold,
                dit->tea_cache_audio_threshold > 0.0f ? "" : " (disabled)");
    if (dit->sol_attention && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: standalone Sol-Attn enabled tau=%.6g "
                "dense-layers=%u dense-steps=%u dense-final-steps=%u%s\n",
                dit->sol_tau, dit->sol_dense_layers, dit->sol_dense_steps,
                dit->sol_dense_final_steps,
                h3_runtime_getenv("H3_SOL_FULL_SINK") ? " full-sink/all-exact" : "");
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(
        dit->weight_directory, error, error_size);
    if (!dit->weights) goto failed;
    if (dit->ssd_quantized) {
        uint64_t identity = 0;
        if (!h3_weight_store_identity(
                dit->weights, &identity, error, error_size) ||
            !h3_quant_cache_open(
                &dit->quant_cache, ssd_quantized_cache_directory,
                identity, H3_DIT_BLOCKS, HIDDEN, INNER, FFN,
                error, error_size)) goto failed;
        if (h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: provenance-bound INT8 SSD cache source=%016" PRIx64
                    " block=%.3f GiB\n", identity,
                    (double)dit->quant_cache.block_bytes /
                        (1024.0 * 1024.0 * 1024.0));
    }
    if (prepare_gate_cache_identity(dit, sigmas))
        (void)configure_cached_gate_skip(dit);
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) goto failed;
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu);
    dit->int8_mlp = (!dit->ssd_streaming || dit->ssd_quantized) &&
                    dit->fused_mlp &&
                    !use_slower_bf16_mlp &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_qkv = (!dit->ssd_streaming || dit->ssd_quantized) &&
                    !use_slower_bf16_qkv &&
                    dit->sequence >= 128 &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_attention_out = (!dit->ssd_streaming || dit->ssd_quantized) &&
                              !use_slower_bf16_attention_output &&
                              dit->sequence >= 128 &&
                              h3_gpu_has_int8_mlp(dit->gpu);
    if (private_ane_mlp_enabled()) {
        const char *gpu_precision =
            h3_runtime_getenv("H3_PRIVATE_ANE_MLP_GPU_PRECISION");
        if (gpu_precision && *gpu_precision &&
            strcmp(gpu_precision, "bf16") &&
            strcmp(gpu_precision, "int8")) {
            fail(error, error_size,
                 "H3_PRIVATE_ANE_MLP_GPU_PRECISION must be bf16 or int8");
            goto failed;
        }
        dit->ane_gpu_int8_mlp = gpu_precision &&
            !strcmp(gpu_precision, "int8");
        if (dit->ane_gpu_int8_mlp &&
            !h3_gpu_has_int8_mlp(dit->gpu)) {
            fail(error, error_size,
                 "private ANE GPU INT8 complement requires Metal "
                 "TensorOps support");
            goto failed;
        }
        dit->nax_mlp = 0;
    }
    dit->use_slower_row_major_attention_output =
        use_slower_row_major_attention_output;
    dit->use_slower_unfused_int8_inputs =
        use_slower_unfused_int8_inputs;
    dit->use_slower_unfused_qkv_rope =
        use_slower_unfused_qkv_rope;
    dit->use_slower_scalar_qkv_rms = use_slower_scalar_qkv_rms;
    dit->use_slower_uncached_int8_scales =
        use_slower_uncached_int8_scales;
    dit->use_slower_dynamic_fc1_k = use_slower_dynamic_fc1_k;
    dit->keep_bf16_attention_out = dit->int8_attention_out &&
        (h3_runtime_getenv("H3_INT8_KEEP_BF16_ATTENTION_OUT") ||
         h3_runtime_getenv("H3_BENCH_INT8_ATTENTION_OUT_AB"));
    dit->keep_bf16_qkv = dit->int8_qkv &&
        (h3_runtime_getenv("H3_INT8_KEEP_BF16_QKV") ||
         h3_runtime_getenv("H3_BENCH_INT8_QKV_AB"));
    dit->use_slower_grouped_quantizer = use_slower_grouped_quantizer;
    dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;
    dit->keep_bf16_mlp = dit->int8_mlp &&
        (h3_runtime_getenv("H3_INT8_KEEP_BF16_MLP") ||
         h3_runtime_getenv("H3_BENCH_INT8_MLP_AB") ||
         h3_runtime_getenv("H3_INT8_MLP_STAGE"));
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    if (!defer_request_state) {
        report(progress, progress_opaque, "refine text", 0, 1);
        if (!refine_text(dit, text, error, error_size)) goto failed;
        report(progress, progress_opaque, "refine text", 1, 1);
    }
    schedule_progress schedule_state = {progress, progress_opaque};
    int fixed_gate_skip = dit->explicit_gate_skip || dit->cached_gate_skip;
    dit->schedule = fixed_gate_skip ?
        h3_dit_schedule_precompute_active(
            dit->weights, dit->gpu, sigmas, dit->video_condition_rows != 0,
            dit->audio_condition_rows != 0, dit->block_active,
            H3_DIT_BLOCKS, schedule_report, &schedule_state,
            error, error_size) :
        h3_dit_schedule_precompute(
            dit->weights, dit->gpu, sigmas, dit->video_condition_rows != 0,
            dit->audio_condition_rows != 0, schedule_report, &schedule_state,
            error, error_size);
    if (dit->schedule) {
        if (!fixed_gate_skip) {
            if (configure_gate_ranked_blocks(dit)) {
                write_gate_cache(dit);
            } else if (h3_runtime_getenv("H3_PROFILE")) {
                fprintf(stderr,
                        "h3: gate ranking unavailable; cache not written\n");
            }
        }
        h3_dit_schedule_prune(dit->schedule, dit->block_active,
                              H3_DIT_BLOCKS);
    }
    profile_step_gate_scores(dit);
    if (!dit->schedule ||
        !configure_step_gate_skip(dit, sigmas, error, error_size) ||
        !configure_ssd_pinned_prefix(
            dit, ssd_pinned_prefix, ssd_memory_budget_bytes,
            error, error_size)) goto failed;
    if (defer_request_state) {
        if (!load_core(dit, progress, progress_opaque,
                       error, error_size) ||
            !eager_coreml_load_wait(dit, error, error_size)) goto failed;
        h3_gpu_profile_mark(dit->gpu, "core load");
        return dit;
    }
    if (!prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    if (!eager_coreml_load_wait(dit, error, error_size)) goto failed;
    dit->request_ready = 1;
    h3_gpu_profile_mark(dit->gpu, "load");
    return dit;
failed:
    h3_dit_free(dit);
    return NULL;
}

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
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming, ssd_pinned_prefix,
                    ssd_memory_budget_bytes,
                    ssd_quantized_cache_directory,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    0,
                    NULL, 0, NULL, 0, progress, progress_opaque,
                    error, error_size);
}

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
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming, ssd_pinned_prefix,
                    ssd_memory_budget_bytes,
                    ssd_quantized_cache_directory, spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2, 1,
                    NULL, 0, NULL, 0, progress, progress_opaque,
                    error, error_size);
}

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
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming, ssd_pinned_prefix,
                    ssd_memory_budget_bytes,
                    ssd_quantized_cache_directory,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    0,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    progress, progress_opaque, error, error_size);
}

static int enter_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_pool_bf16(
            dit->gpu, dit->attention_output, dit->hidden, 0,
            original, dit->token_original_offset, dit->attention_output,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_pool_pairs, dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN),
            error, error_size, "snapshot and pool video tokens")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int enter_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_pool_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            dit->hidden, 0, original, dit->token_original_offset,
            dit->attention_output, dit->token_baseline_offset,
            dit->token_baseline_indices, dit->token_pool_pairs,
            weight->norm1, modulation, dit->reduced_row_maps[step],
            dit->sequence, dit->reduced_sequence, dit->token_baseline_rows,
            HIDDEN, SLOTS, 0, 1, 1e-5f), error, error_size,
            "snapshot, pool, and apply attention AdaLN")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int leave_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_expand_delta_bf16(
            dit->gpu, dit->mod_attention, original,
            dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, dit->sequence,
            dit->reduced_sequence, dit->token_baseline_rows, HIDDEN,
            dit->video_target_start,
            dit->token_reduction_scale), error, error_size,
            "restore full video-token grid")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->mod_attention;
    dit->mod_attention = swap;
    dit->token_reduction_active = 0;
    return 1;
}

static int leave_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_expand_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            original, dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, weight->norm1, modulation,
            dit->row_maps[step], dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN, dit->video_target_start,
            dit->token_reduction_scale, SLOTS, 0, 1, 1e-5f),
            error, error_size, "restore tokens and apply attention AdaLN"))
        return 0;
    h3_gpu_tensor *reduced = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = reduced;
    dit->token_reduction_active = 0;
    return 1;
}

static int diagnose_coreml_range_block(unsigned index) {
    const char *value = h3_runtime_getenv("H3_DIAG_COREML_RANGE_BLOCK");
    if (!value || !*value) return 0;
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    return tail != value && !*tail && parsed == index;
}

static int dump_mlp_input_requested(int step, char *error,
                                    size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_MLP_INPUTS_DIR");
    if (!directory || !*directory) return 0;
    int requested_step = 0;
    const char *value = h3_runtime_getenv("H3_DUMP_MLP_INPUT_STEP");
    if (value && *value) {
        char *tail = NULL;
        long parsed = strtol(value, &tail, 10);
        if (tail == value || *tail || parsed < 0 || parsed >= H3_MAX_STEPS) {
            fail(error, error_size,
                 "H3_DUMP_MLP_INPUT_STEP must be in [0, %d)",
                 H3_MAX_STEPS);
            return -1;
        }
        requested_step = (int)parsed;
    }
    return step == requested_step;
}

static int dump_qkv_input_requested(int step, unsigned block,
                                    char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_QKV_INPUTS_DIR");
    if (!directory || !*directory) return 0;
    int requested_step = 0;
    const char *value = h3_runtime_getenv("H3_DUMP_QKV_INPUT_STEP");
    if (value && *value) {
        char *tail = NULL;
        long parsed = strtol(value, &tail, 10);
        if (tail == value || *tail || parsed < 0 || parsed >= H3_MAX_STEPS) {
            fail(error, error_size,
                 "H3_DUMP_QKV_INPUT_STEP must be in [0, %d)",
                 H3_MAX_STEPS);
            return -1;
        }
        requested_step = (int)parsed;
    }
    if (step != requested_step) return 0;
    const char *blocks = h3_runtime_getenv("H3_DUMP_QKV_INPUT_BLOCKS");
    if (!blocks || !*blocks) return 1;
    return coreml_block_in_list(
        "H3_DUMP_QKV_INPUT_BLOCKS", blocks, block, error, error_size);
}

static int dump_attention_output_input_requested(
                                    int step, unsigned block,
                                    char *error, size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_ATTN_OUT_INPUTS_DIR");
    if (!directory || !*directory) return 0;
    int requested_step = 0;
    const char *value = h3_runtime_getenv("H3_DUMP_ATTN_OUT_INPUT_STEP");
    if (value && *value) {
        char *tail = NULL;
        long parsed = strtol(value, &tail, 10);
        if (tail == value || *tail || parsed < 0 || parsed >= H3_MAX_STEPS) {
            fail(error, error_size,
                 "H3_DUMP_ATTN_OUT_INPUT_STEP must be in [0, %d)",
                 H3_MAX_STEPS);
            return -1;
        }
        requested_step = (int)parsed;
    }
    if (step != requested_step) return 0;
    const char *blocks = h3_runtime_getenv("H3_DUMP_ATTN_OUT_INPUT_BLOCKS");
    if (!blocks || !*blocks) return 1;
    return coreml_block_in_list(
        "H3_DUMP_ATTN_OUT_INPUT_BLOCKS", blocks, block,
        error, error_size);
}

static int dump_qkv_input_bf16(const h3_gpu_tensor *tensor, size_t elements,
                               unsigned block, char *error,
                               size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_QKV_INPUTS_DIR");
    const uint16_t *values = h3_gpu_tensor_host_pointer(
        (h3_gpu_tensor *)(void *)tensor);
    if (!directory || !*directory || !values) {
        fail(error, error_size, "QKV input dump tensor is not host-visible");
        return 0;
    }
    char path[PATH_MAX];
    int count = snprintf(path, sizeof(path), "%s/block-%02u.bf16",
                         directory, block);
    if (count < 0 || (size_t)count >= sizeof(path)) {
        fail(error, error_size, "QKV input dump path is too long");
        return 0;
    }
    FILE *stream = fopen(path, "wb");
    if (!stream) {
        fail(error, error_size, "cannot open QKV input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    size_t written = fwrite(values, sizeof(*values), elements, stream);
    int close_status = fclose(stream);
    if (written != elements || close_status != 0) {
        fail(error, error_size, "cannot write QKV input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: dumped real QKV input block=%u elements=%zu path=%s\n",
                block, elements, path);
    return 1;
}

static int dump_attention_output_input_bf16(
                               const h3_gpu_tensor *tensor, size_t elements,
                               unsigned block, char *error,
                               size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_ATTN_OUT_INPUTS_DIR");
    const uint16_t *values = h3_gpu_tensor_host_pointer(
        (h3_gpu_tensor *)(void *)tensor);
    if (!directory || !*directory || !values) {
        fail(error, error_size,
             "attention-output input dump tensor is not host-visible");
        return 0;
    }
    char path[PATH_MAX];
    int count = snprintf(path, sizeof(path), "%s/block-%02u.bf16",
                         directory, block);
    if (count < 0 || (size_t)count >= sizeof(path)) {
        fail(error, error_size,
             "attention-output input dump path is too long");
        return 0;
    }
    FILE *stream = fopen(path, "wb");
    if (!stream) {
        fail(error, error_size,
             "cannot open attention-output input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    size_t written = fwrite(values, sizeof(*values), elements, stream);
    int close_status = fclose(stream);
    if (written != elements || close_status != 0) {
        fail(error, error_size,
             "cannot write attention-output input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: dumped real attention-output input block=%u "
                "elements=%zu path=%s\n",
                block, elements, path);
    return 1;
}

static int dump_mlp_input_bf16(const h3_gpu_tensor *tensor, size_t elements,
                               unsigned block, char *error,
                               size_t error_size) {
    const char *directory = h3_runtime_getenv("H3_DUMP_MLP_INPUTS_DIR");
    const uint16_t *values = h3_gpu_tensor_host_pointer(
        (h3_gpu_tensor *)(void *)tensor);
    if (!directory || !*directory || !values) {
        fail(error, error_size, "MLP input dump tensor is not host-visible");
        return 0;
    }
    char path[PATH_MAX];
    int count = snprintf(path, sizeof(path), "%s/block-%02u.bf16",
                         directory, block);
    if (count < 0 || (size_t)count >= sizeof(path)) {
        fail(error, error_size, "MLP input dump path is too long");
        return 0;
    }
    FILE *stream = fopen(path, "wb");
    if (!stream) {
        fail(error, error_size, "cannot open MLP input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    size_t written = fwrite(values, sizeof(*values), elements, stream);
    int close_status = fclose(stream);
    int ok = written == elements && close_status == 0;
    if (!ok) {
        fail(error, error_size, "cannot write MLP input dump %s: %s",
             path, strerror(errno));
        return 0;
    }
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: dumped real MLP input block=%u elements=%zu path=%s\n",
                block, elements, path);
    return 1;
}

static int report_coreml_f16_range(const h3_gpu_tensor *tensor,
                                   size_t elements, unsigned block,
                                   const char *stage,
                                   char *error, size_t error_size) {
    const uint16_t *values = h3_gpu_tensor_host_pointer(
        (h3_gpu_tensor *)(void *)tensor);
    if (!values) {
        fail(error, error_size, "Core ML range tensor is not host-visible");
        return 0;
    }
    double minimum = INFINITY;
    double maximum = -INFINITY;
    double max_absolute = 0.0;
    size_t nonfinite = 0, over_60k = 0;
    for (size_t element = 0; element < elements; element++) {
        uint16_t bits = values[element];
        if ((bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            nonfinite++;
            continue;
        }
        _Float16 half = 0.0;
        memcpy(&half, &bits, sizeof(bits));
        double value = (double)half;
        double absolute = fabs(value);
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        if (absolute > max_absolute) max_absolute = absolute;
        if (absolute > 60000.0) over_60k++;
    }
    fprintf(stderr,
            "h3 Core ML range: block=%u stage=%s elements=%zu "
            "nonfinite=%zu min=%.9g max=%.9g maxabs=%.9g gt60k=%zu\n",
            block, stage, elements, nonfinite,
            minimum, maximum, max_absolute, over_60k);
    return 1;
}


static int run_block(h3_dit *dit, unsigned index, int step,
                     h3_dit_block *weight,
                     int attention_adaln_ready,
                     int attention_input_quantized,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready,
                     int *next_attention_input_quantized,
                     char *error, size_t error_size) {
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->token_reduction_active ?
        dit->reduced_row_maps[step] : dit->row_maps[step];
    h3_gpu_tensor *rope_cos = dit->token_reduction_active ?
        dit->reduced_rope_cos : dit->rope_cos;
    h3_gpu_tensor *rope_sin = dit->token_reduction_active ?
        dit->reduced_rope_sin : dit->rope_sin;
    uint32_t rows = dit->token_reduction_active ?
        dit->reduced_sequence : dit->sequence;
    int diagnose_coreml_range = diagnose_coreml_range_block(index);
    const char *profile_ops_value = h3_runtime_getenv("H3_PROFILE_OPS");
    int profile_ops = profile_ops_value && *profile_ops_value &&
                      strcmp(profile_ops_value, "0");
#define OP(call, label) do {                                                    \
    double op_started__ = profile_ops ? stream_now() : 0.0;                     \
    int op_ok__ = (call);                                                       \
    if (profile_ops)                                                            \
        fprintf(stderr,                                                         \
                "h3 op profile: step=%d block=%u rows=%u wall=%.6fs "         \
                "op=\"%s\"\n",                                               \
                step + 1, index, rows, stream_now() - op_started__, label);      \
    if (!gpu_op(dit, op_ok__, error, error_size, label)) return 0;               \
} while (0)
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
            weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS,
            0, 1, 1e-5f), "DiT attention AdaLN");
    int dump_qkv_input = dump_qkv_input_requested(
        step, index, error, error_size);
    if (dump_qkv_input < 0) return 0;
    if (dump_qkv_input) {
        size_t elements = (size_t)rows * HIDDEN;
        OP(h3_gpu_submit(dit->gpu), "submit real QKV input capture");
        if (!dump_qkv_input_bf16(
                dit->mod_attention, elements, index,
                error, error_size)) return 0;
        OP(h3_gpu_begin(dit->gpu), "resume after real QKV input capture");
    }
    if (weight->ane_qkv && rows == dit->sequence &&
        !h3_runtime_getenv("H3_DISABLE_PRIVATE_ANE_QKV")) {
        if (!run_private_ane_qkv(
                dit, weight, dit->mod_attention, rope_cos, rope_sin,
                rows, index, step, attention_input_quantized,
                error, error_size)) return 0;
    } else if (weight->coreml_qkv && rows == dit->sequence &&
        !h3_runtime_getenv("H3_DISABLE_COREML_QKV")) {
        uint32_t ane_heads = weight->coreml_qkv_heads;
        uint32_t gpu_heads = HEADS - ane_heads;
        uint32_t gpu_width = gpu_heads * 3u * HEAD_DIM;
        char prediction_error[512] = {0};
        OP(h3_gpu_pack_coreml_f16_transpose(
            dit->gpu, dit->coreml_qkv_input, dit->mod_attention,
            rows, HIDDEN), "pack Core ML QKV input");
        OP(h3_gpu_submit(dit->gpu), "submit Core ML QKV input");
        if (!h3_coreml_mlp_start(
                weight->coreml_qkv, prediction_error,
                sizeof(prediction_error))) {
            weight->coreml_qkv_prediction_failures++;
            fail(error, error_size, "%s", prediction_error);
            return 0;
        }
        weight->coreml_qkv_predictions++;
        OP(h3_gpu_begin(dit->gpu), "begin GPU QKV complement");
        OP(h3_gpu_linear_bf16(
            dit->gpu, dit->coreml_qkv_gpu_output, dit->mod_attention,
            weight->coreml_qkv_gpu_weight, NULL, rows, HIDDEN, gpu_width),
           "DiT GPU QKV complement projection");
        OP(h3_gpu_grouped_qkv_rope_bf16_segment(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->coreml_qkv_gpu_output, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, gpu_heads, HEADS, ane_heads,
            HEAD_DIM, ROPE_HALF, 1e-5f),
           "DiT GPU QKV complement norm/RoPE");
        OP(h3_gpu_submit(dit->gpu), "submit GPU QKV complement");
        if (!h3_coreml_mlp_wait(
                weight->coreml_qkv, prediction_error,
                sizeof(prediction_error))) {
            weight->coreml_qkv_prediction_failures++;
            fail(error, error_size, "%s", prediction_error);
            return 0;
        }
        OP(h3_gpu_begin(dit->gpu), "begin ANE QKV segment consumer");
        OP(h3_gpu_grouped_qkv_rope_coreml_f16_segment(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->coreml_qkv_output, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, dit->coreml_nonfinite,
            rows, ane_heads, HEADS, 0, HEAD_DIM, ROPE_HALF, 1e-5f, index),
           "DiT ANE QKV segment norm/RoPE");
        OP(h3_gpu_submit(dit->gpu), "submit ANE QKV segment consumer");
        OP(h3_gpu_begin(dit->gpu), "begin post-Core ML QKV attention");
    } else if (dit->int8_qkv && !h3_runtime_getenv("H3_DISABLE_INT8_QKV")) {
        OP(h3_gpu_grouped_qkv_linear_rope_int8(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->int8_activation, dit->int8_activation_scales,
            dit->mod_attention, weight->qkv_int8, weight->qkv_scales,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            attention_input_quantized,
            dit->use_slower_unfused_qkv_rope,
            dit->use_slower_scalar_qkv_rms,
            dit->use_slower_uncached_int8_scales),
           "DiT int8 QKV projection/norm/RoPE");
    } else {
        OP(h3_gpu_grouped_qkv_linear_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF,
            1e-5f), "DiT QKV projection/norm/RoPE");
    }
    int int8_attention_output = dit->int8_attention_out &&
        !block_has_private_ane_attention_out(weight) &&
        !h3_runtime_getenv("H3_DISABLE_INT8_ATTENTION_OUT");
    int head_major_attention_output = int8_attention_output &&
        !dit->use_slower_row_major_attention_output &&
        !dit->use_slower_uncached_int8_scales &&
        !h3_runtime_getenv("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT");
    int use_sol_attention = dit->sol_attention &&
        !h3_runtime_getenv("H3_SOL_RUNTIME_DISABLE") &&
        index >= dit->sol_dense_layers &&
        (unsigned)step >= dit->sol_dense_steps &&
        (unsigned)(dit->sigmas.steps - step) > dit->sol_dense_final_steps;
    int use_sol_bq64_exact = use_sol_attention &&
        h3_runtime_getenv("H3_SOL_BQ64_EXACT") != NULL;
    int use_sol_bq64_sparse = use_sol_attention &&
        h3_runtime_getenv("H3_SOL_BQ64_SPARSE") != NULL;
    int fused_sdpa_output = !use_sol_attention && !int8_attention_output &&
        !block_has_private_ane_attention_out(weight) &&
        h3_runtime_getenv("H3_FUSED_SDPA_OUT") != NULL;
    if (use_sol_bq64_exact || use_sol_bq64_sparse) {
        if (!head_major_attention_output) {
            fail(error, error_size,
                 "TensorOps BQ64 SDPA requires the head-major int8 "
                 "QKV/attention-output path");
            return 0;
        }
        if (dit->sol_verify_dense) {
            OP(h3_gpu_sdpa_bf16_head_major_output(
                dit->gpu, dit->sol_verify_dense,
                dit->query, dit->key, dit->value,
                rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
               "DiT dense verification attention");
        }
        if (use_sol_bq64_sparse) {
            uint32_t blocks = (rows + 63u) / 64u;
            uint32_t prefix_blocks =
                (dit->video_target_start + 63u) / 64u;
            if (prefix_blocks > blocks) prefix_blocks = blocks;
            uint32_t sink_end = h3_runtime_getenv("H3_SOL_FULL_SINK") ?
                blocks : prefix_blocks;
            OP(h3_gpu_sol_sdpa_bf16_nax_bq64(
                dit->gpu, dit->attention_heads,
                dit->query, dit->key, dit->value,
                dit->sol_query_centroids, dit->sol_key_centroids,
                dit->sol_value_sums, dit->sol_thresholds, dit->sol_routes,
                rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM),
                dit->sol_tau, 0, sink_end, 0, prefix_blocks),
               "DiT TensorOps BQ64 sparse Sol-Attn");
        } else {
            OP(h3_gpu_sdpa_bf16_nax_bq64_all_exact(
                dit->gpu, dit->attention_heads,
                dit->query, dit->key, dit->value,
                rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
               "DiT TensorOps BQ64 all-exact attention");
        }
        if (dit->sol_verify_dense) {
            uint32_t elements = rows * INNER;
            OP(h3_gpu_compare_bf16(
                dit->gpu, dit->sol_verify_partials, dit->attention_heads,
                dit->sol_verify_dense, elements),
               "compare BQ64 with dense attention");
            dit->sol_verify_elements = elements;
            dit->sol_verify_layer = index;
            dit->sol_verify_pending = 1;
        }
        dit->sol_calls++;
    } else if (use_sol_attention) {
        uint32_t blocks = (rows + 63u) / 64u;
        uint32_t prefix_blocks =
            (dit->video_target_start + 63u) / 64u;
        if (prefix_blocks > blocks) prefix_blocks = blocks;
        uint32_t sink_end = h3_runtime_getenv("H3_SOL_FULL_SINK") ?
            blocks : prefix_blocks;
        OP(h3_gpu_sol_sdpa_bf16(
            dit->gpu, dit->attention_heads,
            dit->query, dit->key, dit->value,
            dit->sol_query_centroids, dit->sol_key_centroids,
            dit->sol_value_sums, dit->sol_thresholds, dit->sol_routes,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM),
            dit->sol_tau, 0, sink_end, 0, prefix_blocks,
            head_major_attention_output),
           "DiT standalone Sol-Attn");
        dit->sol_calls++;
    } else if (fused_sdpa_output)
        OP(h3_gpu_sdpa_output_linear_bf16(
            dit->gpu, dit->attention_output, dit->query, dit->key, dit->value,
            weight->out, rows, HEADS, HEAD_DIM, HIDDEN,
            1.0f / sqrtf((float)HEAD_DIM)),
           "DiT fused full attention and BF16 output");
    else if (head_major_attention_output)
        OP(h3_gpu_sdpa_bf16_head_major_output(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT head-major full attention");
    else
        OP(h3_gpu_sdpa_bf16(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT full attention");
    if (!use_sol_attention) dit->sol_dense_calls++;
    if (use_sol_attention && h3_runtime_getenv("H3_SOL_ROUTE_PROFILE")) {
        dit->sol_route_profile_layer = index;
        dit->sol_route_profile_pending = 1;
    }
    int dump_attention_output_input =
        dump_attention_output_input_requested(
            step, index, error, error_size);
    if (dump_attention_output_input < 0) return 0;
    if (dump_attention_output_input) {
        if (fused_sdpa_output || head_major_attention_output) {
            fail(error, error_size,
                 "attention-output input dump requires row-major standalone "
                 "BF16 SDPA output");
            return 0;
        }
        size_t elements = (size_t)rows * INNER;
        OP(h3_gpu_submit(dit->gpu),
           "submit real attention-output input capture");
        if (!dump_attention_output_input_bf16(
                dit->attention_heads, elements, index,
                error, error_size)) return 0;
        OP(h3_gpu_begin(dit->gpu),
           "resume after real attention-output input capture");
    }
    if (block_has_private_ane_attention_out(weight)) {
        if (!run_private_ane_attention_out(
                dit, weight, dit->attention_output,
                dit->attention_heads, rows, index, step,
                error, error_size)) return 0;
    } else if (int8_attention_output) {
        if (head_major_attention_output)
            OP(h3_gpu_linear_int8_head_major_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, HEADS, HEAD_DIM,
                HIDDEN), "DiT head-major int8 attention output");
        else
            OP(h3_gpu_linear_int8_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, INNER, HIDDEN,
                dit->use_slower_uncached_int8_scales),
               "DiT int8 attention output");
    } else if (!fused_sdpa_output) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
            dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
           "DiT attention output");
    }
    int fused_int8_mlp_input = dit->int8_mlp &&
        !dit->use_slower_unfused_int8_inputs &&
        !h3_runtime_getenv("H3_DISABLE_FUSED_INT8_MLP_INPUT") &&
        !h3_runtime_getenv("H3_INT8_MLP_STAGE");
    if (fused_int8_mlp_input) {
        uint32_t padded_rows = (rows + 127u) & ~127u;
        OP(h3_gpu_gate_adaln_quantize_int8(
            dit->gpu, dit->hidden, dit->int8_activation,
            dit->int8_activation_scales, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map, rows, padded_rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate, MLP AdaLN and int8 quantization");
    } else if (!h3_runtime_getenv("H3_DISABLE_FUSED_GATE_ADALN")) {
        OP(h3_gpu_gate_adaln_bf16(
            dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map,
            rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate and MLP AdaLN");
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
            dit->attention_output, modulation, row_map, rows, HIDDEN,
            SLOTS, 2), "DiT attention gate");
        OP(h3_gpu_adaln_bf16(
            dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
            modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
           "DiT MLP AdaLN");
    }
    int dump_mlp_input = dump_mlp_input_requested(step, error, error_size);
    if (dump_mlp_input < 0) return 0;
    const char *dump_mlp_blocks = h3_runtime_getenv("H3_DUMP_MLP_INPUT_BLOCKS");
    if (dump_mlp_input && dump_mlp_blocks && *dump_mlp_blocks) {
        dump_mlp_input = coreml_block_in_list(
            "H3_DUMP_MLP_INPUT_BLOCKS", dump_mlp_blocks, index,
            error, error_size);
        if (dump_mlp_input < 0) return 0;
    }
    if (dump_mlp_input) {
        size_t elements = (size_t)rows * HIDDEN;
        OP(h3_gpu_submit(dit->gpu), "submit real MLP input capture");
        if (!dump_mlp_input_bf16(
                dit->mod_mlp, elements, index, error, error_size)) return 0;
        OP(h3_gpu_begin(dit->gpu), "resume after real MLP input capture");
    }
    h3_gpu_tensor *mlp_output = dit->activation_aliases ?
        dit->attention_output : dit->mlp_output;
    int coreml_benchmark_full = weight->coreml_mlp &&
        dit->coreml_same_loaded_ab &&
        dit->coreml_benchmark_route == H3_DIT_COREML_BENCHMARK_FULL_GPU;
    int private_ane_fused_epilogue = 0;
    if (weight->ane_mlp) {
        if (!run_private_ane_mlp(
                dit, weight, mlp_output, dit->mod_mlp,
                modulation, row_map, rows, index, step,
                fuse_next_attention, next_index,
                &private_ane_fused_epilogue,
                error, error_size)) return 0;
    } else if (coreml_benchmark_full) {
        OP(h3_gpu_mlp_bf16(
            dit->gpu, mlp_output, dit->mod_mlp,
            weight->coreml_full_fc1, weight->coreml_full_fc2,
            rows, HIDDEN, FFN, HIDDEN),
           "DiT benchmark full GPU MLP");
    } else if (weight->coreml_mlp) {
        if (rows != dit->sequence) {
            fail(error, error_size,
                 "Core ML MLP block %u does not support reduced rows %u",
                 index, rows);
            return 0;
        }
        uint32_t elements = rows * HIDDEN;
        uint32_t gpu_intermediate = FFN - weight->coreml_intermediate;
        int benchmark_exact_split = dit->coreml_same_loaded_ab &&
            dit->coreml_benchmark_route ==
                H3_DIT_COREML_BENCHMARK_EXACT_SPLIT;
        int benchmark_candidate = dit->coreml_same_loaded_ab &&
            dit->coreml_benchmark_route ==
                H3_DIT_COREML_BENCHMARK_CANDIDATE;
        int forced_fallback = benchmark_exact_split ||
            (!benchmark_candidate && dit->coreml_force_gpu_fallback);
        int latched_fallback = !benchmark_candidate &&
                               dit->coreml_fallback_latch &&
                               weight->coreml_fallback_latched;
        int use_fallback = forced_fallback || latched_fallback;
        const char *fallback_reason = forced_fallback ? "forced" :
                                      latched_fallback ? "latched" : NULL;
        int coreml_input_submitted = 0;
        char prediction_error[512] = {0};
        if (!use_fallback && h3_coreml_mlp_prediction_inflight(
                                 weight->coreml_mlp)) {
            int drained = h3_coreml_mlp_wait(
                weight->coreml_mlp, prediction_error,
                sizeof(prediction_error));
            if (!dit->coreml_gpu_fallback) {
                fail(error, error_size,
                     "Core ML MLP block %u had a stale prediction in flight%s%s",
                     index, prediction_error[0] ? ": " : "",
                     prediction_error);
                return 0;
            }
            if (drained)
                snprintf(prediction_error, sizeof(prediction_error),
                         "previous prediction was still in flight");
            use_fallback = 1;
            fallback_reason = drained ? "stale-inflight" :
                                        "stale-inflight-failed";
            weight->coreml_prediction_failures++;
        }
        if (!use_fallback) {
            OP(h3_gpu_pack_coreml_f16_transpose(
                dit->gpu, h3_coreml_mlp_input(weight->coreml_mlp),
                dit->mod_mlp, rows, HIDDEN),
               "pack Core ML MLP input");
            OP(h3_gpu_submit(dit->gpu), "submit Core ML MLP input");
            coreml_input_submitted = 1;
            if (diagnose_coreml_range &&
                !report_coreml_f16_range(
                    h3_coreml_mlp_input(weight->coreml_mlp), elements,
                    index, "input", error, error_size)) return 0;
            if (!h3_coreml_mlp_start(
                    weight->coreml_mlp, prediction_error,
                    sizeof(prediction_error))) {
                if (!dit->coreml_gpu_fallback) {
                    fail(error, error_size, "%s", prediction_error);
                    return 0;
                }
                use_fallback = 1;
                fallback_reason = "start-failed";
                weight->coreml_prediction_failures++;
            } else {
                weight->coreml_predictions++;
            }
        }
        if (coreml_input_submitted)
            OP(h3_gpu_begin(dit->gpu), "begin GPU MLP shard");
        OP(h3_gpu_mlp_bf16(
            dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN,
            gpu_intermediate, HIDDEN),
           "DiT GPU MLP shard");
        OP(h3_gpu_submit(dit->gpu), "submit GPU MLP shard");
        size_t nonfinite_count = 0;
        if (!use_fallback) {
            if (!h3_coreml_mlp_wait(
                    weight->coreml_mlp, prediction_error,
                    sizeof(prediction_error))) {
                if (!dit->coreml_gpu_fallback) {
                    fail(error, error_size, "%s", prediction_error);
                    return 0;
                }
                use_fallback = 1;
                fallback_reason = "prediction-failed";
                weight->coreml_prediction_failures++;
            } else {
                if (dit->coreml_gpu_fallback) {
                    const uint16_t *ane_values = h3_gpu_tensor_host_pointer(
                        (h3_gpu_tensor *)(void *)
                            h3_coreml_mlp_output(weight->coreml_mlp));
                    if (h3_coreml_mlp_fallback_required(
                            ane_values, elements, 0, &nonfinite_count)) {
                        use_fallback = 1;
                        fallback_reason = "nonfinite-output";
                        weight->coreml_nonfinite_values += nonfinite_count;
                    }
                }
            }
        }
        if (use_fallback) {
            if (!prepare_coreml_gpu_fallback(
                    dit, weight, error, error_size)) return 0;
            if (dit->coreml_fallback_latch && !forced_fallback &&
                !latched_fallback)
                weight->coreml_fallback_latched = 1;
            weight->coreml_fallbacks++;
            if (forced_fallback) weight->coreml_forced_fallbacks++;
            if (latched_fallback) weight->coreml_latched_fallbacks++;
            if (weight->coreml_fallbacks == 1 || nonfinite_count ||
                prediction_error[0])
                fprintf(stderr,
                        "h3: Core ML MLP exact GPU fallback step=%d "
                        "block=%u reason=%s nonfinite=%zu%s%s\n",
                        step + 1, index,
                        fallback_reason ? fallback_reason : "requested",
                        nonfinite_count,
                        prediction_error[0] ? " detail=" : "",
                        prediction_error);
            OP(h3_gpu_begin(dit->gpu), "begin exact GPU MLP fallback");
            OP(h3_gpu_mlp_bf16(
                dit->gpu, dit->coreml_fallback_output, dit->mod_mlp,
                weight->coreml_fallback_fc1,
                weight->coreml_fallback_fc2,
                rows, HIDDEN, weight->coreml_intermediate, HIDDEN),
               "DiT exact GPU ANE-prefix fallback");
            OP(h3_gpu_add_bf16(
                dit->gpu, mlp_output, mlp_output,
                dit->coreml_fallback_output, elements),
               "join GPU MLP complement and exact prefix fallback");
        } else {
            if (diagnose_coreml_range &&
                !report_coreml_f16_range(
                    h3_coreml_mlp_output(weight->coreml_mlp), elements,
                    index, "output", error, error_size)) return 0;
            OP(h3_gpu_begin(dit->gpu), "begin GPU+ANE MLP join");
            OP(h3_gpu_add_coreml_f16_bf16_transpose_checked(
                dit->gpu, mlp_output, mlp_output,
                h3_coreml_mlp_output(weight->coreml_mlp),
                dit->coreml_nonfinite, rows, HIDDEN,
                weight->coreml_output_scale, index),
               "join GPU+ANE MLP partials");
        }
    } else if (dit->int8_mlp &&
        (!h3_runtime_getenv("H3_DISABLE_INT8_MLP") ||
         !weight->fc1 || !weight->fc2)) {
        OP(h3_gpu_mlp_int8_bf16(
            dit->gpu, mlp_output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2,
            rows, HIDDEN, FFN, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, dit->use_int8_row_fc2,
            fused_int8_mlp_input),
           "DiT int8 fused MLP");
    } else if (dit->nax_mlp && !h3_runtime_getenv("H3_DISABLE_NAX_MLP")) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, mlp_output, dit->activated,
            dit->mod_mlp, weight->fc1, weight->fc2, rows, HIDDEN, FFN,
            HIDDEN), "DiT NAX fused MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    if (weight->coreml_mlp && dit->coreml_same_loaded_ab &&
        index == dit->coreml_benchmark_capture_block) {
        size_t elements = (size_t)rows * HIDDEN;
        if (rows != dit->sequence ||
            elements != dit->coreml_benchmark_capture_elements ||
            !dit->coreml_benchmark_mlp_input ||
            !dit->coreml_benchmark_mlp_output) {
            fail(error, error_size,
                 "Core ML benchmark MLP capture shape is unavailable");
            return 0;
        }
        OP(h3_gpu_copy_bf16(
            dit->gpu, dit->coreml_benchmark_mlp_input, 0,
            dit->mod_mlp, 0, elements),
           "capture Core ML benchmark MLP input");
        OP(h3_gpu_copy_bf16(
            dit->gpu, dit->coreml_benchmark_mlp_output, 0,
            mlp_output, 0, elements),
           "capture Core ML benchmark MLP output");
        dit->coreml_benchmark_capture_pending = 1;
    }
    if (private_ane_fused_epilogue) {
        *next_attention_adaln_ready = 1;
        *next_attention_input_quantized =
            private_ane_fused_epilogue == 2;
    } else if (fuse_next_attention) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        int fuse_int8_qkv_input = dit->int8_qkv &&
            !dit->use_slower_unfused_int8_inputs &&
            !h3_runtime_getenv("H3_DISABLE_INT8_QKV") &&
            !h3_runtime_getenv("H3_DISABLE_FUSED_INT8_QKV_INPUT");
        if (fuse_int8_qkv_input) {
            uint32_t padded_rows = (rows + 127u) & ~127u;
            OP(h3_gpu_gate_adaln_quantize_int8(
                dit->gpu, dit->hidden, dit->int8_activation,
                dit->int8_activation_scales, dit->hidden, mlp_output,
                next_weight->norm1, modulation, next_modulation, row_map,
                rows, padded_rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
               "DiT fused MLP gate, next attention AdaLN and int8 quantization");
            *next_attention_input_quantized = 1;
        } else {
            OP(h3_gpu_gate_adaln_bf16(
                dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
                mlp_output, next_weight->norm1, modulation,
                next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1,
                1e-5f), "DiT fused MLP gate and next attention AdaLN");
            *next_attention_input_quantized = 0;
        }
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(
            dit->gpu, dit->hidden, dit->hidden, mlp_output,
            modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
    }
#undef OP
    return 1;
}

static int first_block_cache_probe(h3_dit *dit, uint32_t elements,
                                   float *score, char *error,
                                   size_t error_size) {
    if (!h3_gpu_relative_l1_residual_bf16(
            dit->gpu, dit->first_block_cache_partials, dit->hidden,
            dit->core_input, dit->first_block_cache_previous, elements)) {
        fail(error, error_size, "FirstBlockCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    double wait_started = stream_now();
    if (!h3_gpu_submit(dit->gpu)) {
        fail(error, error_size, "submit FirstBlockCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    dit->first_block_cache_wait_seconds += stream_now() - wait_started;
    size_t partial_elements =
        (size_t)dit->first_block_cache_partial_count * 2;
    if (!h3_gpu_tensor_read_f32(dit->first_block_cache_partials,
                                dit->first_block_cache_host_partials,
                                partial_elements)) {
        fail(error, error_size,
             "cannot read FirstBlockCache relative-L1 partials");
        return 0;
    }
    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t group = 0;
         group < dit->first_block_cache_partial_count; group++) {
        numerator += dit->first_block_cache_host_partials[group * 2];
        denominator += dit->first_block_cache_host_partials[group * 2 + 1];
    }
    *score = (float)(numerator /
        (denominator > 1.0e-8 ? denominator : 1.0e-8));
    if (!isfinite(*score)) {
        fail(error, error_size,
             "FirstBlockCache relative-L1 score is not finite");
        return 0;
    }
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size, "resume after FirstBlockCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int tea_cache_probe(h3_dit *dit, uint32_t elements,
                           float *score, float *audio_score,
                           char *error, size_t error_size) {
    if (!h3_gpu_relative_l1_bf16(
            dit->gpu, dit->tea_cache_partials, dit->mod_attention,
            dit->tea_cache_previous, elements)) {
        fail(error, error_size, "TeaCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (dit->tea_cache_audio_threshold > 0.0f) {
        uint32_t audio_offset = dit->audio_target_start * HIDDEN;
        uint32_t audio_elements = dit->audio_rows * HIDDEN;
        if (!h3_gpu_relative_l1_bf16_range(
                dit->gpu, dit->tea_cache_partials,
                dit->tea_cache_partial_count, dit->mod_attention,
                dit->tea_cache_previous, audio_offset, audio_elements)) {
            fail(error, error_size, "TeaCache audio probe: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    double wait_started = stream_now();
    if (!h3_gpu_submit(dit->gpu)) {
        fail(error, error_size, "submit TeaCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    dit->tea_cache_wait_seconds += stream_now() - wait_started;
    size_t partial_elements =
        ((size_t)dit->tea_cache_partial_count +
         dit->tea_cache_audio_partial_count) * 2;
    if (!h3_gpu_tensor_read_f32(dit->tea_cache_partials,
                                dit->tea_cache_host_partials,
                                partial_elements)) {
        fail(error, error_size,
             "cannot read TeaCache relative-L1 partials");
        return 0;
    }
    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t group = 0; group < dit->tea_cache_partial_count; group++) {
        numerator += dit->tea_cache_host_partials[group * 2];
        denominator += dit->tea_cache_host_partials[group * 2 + 1];
    }
    *score = (float)(numerator /
        (denominator > 1.0e-8 ? denominator : 1.0e-8));
    if (!isfinite(*score)) {
        fail(error, error_size,
             "TeaCache relative-L1 score is not finite");
        return 0;
    }
    *audio_score = 0.0f;
    if (dit->tea_cache_audio_threshold > 0.0f) {
        numerator = 0.0;
        denominator = 0.0;
        for (uint32_t group = 0;
             group < dit->tea_cache_audio_partial_count; group++) {
            size_t index =
                (size_t)(dit->tea_cache_partial_count + group) * 2;
            numerator += dit->tea_cache_host_partials[index];
            denominator += dit->tea_cache_host_partials[index + 1];
        }
        *audio_score = (float)(numerator /
            (denominator > 1.0e-8 ? denominator : 1.0e-8));
        if (!isfinite(*audio_score)) {
            fail(error, error_size,
                 "TeaCache audio relative-L1 score is not finite");
            return 0;
        }
    }
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size, "resume after TeaCache probe: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split,
                          h3_dit_progress progress, void *progress_opaque,
                          char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (begin) OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (dit->fused_patch_pack) {
        if (dit->video_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->video_input, dit->video_patch_w,
                dit->video_patch_b, dit->video_projection_map, dit->sequence,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "project mapped video sources");
        if (dit->audio_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->audio_input, dit->audio_patch_w,
                dit->audio_patch_b, dit->audio_projection_map, dit->sequence,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "project mapped audio sources");
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                if (!dit->video_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->video_input,
                        video_offset * VIDEO_PATCH, dit->video_patch_w,
                        dit->video_patch_b, (uint32_t)segment_rows,
                        VIDEO_PATCH, HIDDEN), "project packed video source");
                video_offset += segment_rows;
            } else {
                if (!dit->audio_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->audio_input,
                        audio_offset * AUDIO_CHANNELS, dit->audio_patch_w,
                        dit->audio_patch_b, (uint32_t)segment_rows,
                        AUDIO_CHANNELS, HIDDEN),
                       "project packed audio source");
                audio_offset += segment_rows;
            }
        }
    } else {
        if (dit->fused_patch_projection) {
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->video_projected, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "fused video patch projection");
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->audio_projected, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "fused audio patch projection");
        } else {
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->video_projected_f32, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "video patch projection");
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->audio_projected_f32, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "audio patch projection");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->video_projected, dit->video_projected_f32,
                dit->video_total_rows * HIDDEN), "video BF16 cast");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->audio_projected, dit->audio_projected_f32,
                dit->audio_total_rows * HIDDEN), "audio BF16 cast");
        }
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->video_projected,
                    video_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack video source");
                video_offset += segment_rows;
            } else {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->audio_projected,
                    audio_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack audio source");
                audio_offset += segment_rows;
            }
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    int evaluate_core = dit->core_reuse_interval == 1 ||
        !dit->core_residual_ready ||
        dit->core_forward_count % dit->core_reuse_interval == 0 ||
        step == h3_dit_schedule_steps(dit->schedule) - 1;
    int use_token_reduction = evaluate_core && dit->token_reduction &&
        !h3_runtime_getenv("H3_DISABLE_TOKEN_REDUCTION");
    int use_first_block_cache = dit->first_block_cache &&
        !h3_runtime_getenv("H3_FBC_RUNTIME_DISABLE");
    int use_tea_cache = dit->tea_cache &&
        !h3_runtime_getenv("H3_TEACACHE_RUNTIME_DISABLE");
    int use_step_gate_skip = dit->step_gate_skip &&
        !h3_runtime_getenv("H3_STEP_GATE_RUNTIME_DISABLE");
    dit->first_block_cache_last_reused = 0;
    dit->tea_cache_last_reused = 0;
    unsigned token_reduction_end =
        dit->token_reduction_early_steps &&
        (unsigned)step < dit->token_reduction_early_steps ?
            dit->token_reduction_early_end : dit->token_reduction_end;
    uint32_t hidden_elements = dit->sequence * HIDDEN;
    unsigned first_cache_block = use_first_block_cache ?
        first_active_block(dit) : H3_DIT_BLOCKS;
    unsigned tea_cache_block = use_tea_cache ?
        first_active_block(dit) : H3_DIT_BLOCKS;
    int first_block_cache_reused = 0;
    int tea_cache_reused = 0;
    int tea_cache_attention_adaln_ready = 0;
    if (evaluate_core &&
        (dit->core_reuse_interval > 1 || use_first_block_cache))
        OP(h3_gpu_copy_bf16(dit->gpu, dit->core_input, 0, dit->hidden, 0,
                            hidden_elements), "save DiT core input");
    if (evaluate_core) {
        if (use_tea_cache) {
            h3_dit_block *weight = &dit->blocks[tea_cache_block];
            const h3_gpu_tensor *modulation = h3_dit_schedule_block(
                dit->schedule, tea_cache_block);
            OP(h3_gpu_adaln_bf16(
                dit->gpu, dit->mod_attention, dit->hidden, weight->norm1,
                modulation, dit->row_maps[step], dit->sequence, HIDDEN,
                SLOTS, 0, 1, 1e-5f),
               "TeaCache first attention AdaLN");
            tea_cache_attention_adaln_ready = 1;

            unsigned steps = (unsigned)h3_dit_schedule_steps(dit->schedule);
            int last_step = (unsigned)step + 1 == steps;
            int warmup = (unsigned)step < dit->tea_cache_retain_steps;
            int cooldown = dit->tea_cache_cooldown_steps &&
                (unsigned)step >= steps -
                    (dit->tea_cache_cooldown_steps < steps ?
                        dit->tea_cache_cooldown_steps : steps);
            int forced_fresh = h3_runtime_getenv("H3_TEACACHE_FORCE_FRESH") != NULL;
            int hit_cap = dit->tea_cache_max_hits &&
                dit->tea_cache_consecutive_hits >=
                    dit->tea_cache_max_hits;
            int fresh = warmup || cooldown ||
                !dit->tea_cache_previous_ready ||
                !dit->tea_cache_residual_ready || last_step ||
                forced_fresh || hit_cap;
            const char *reason = warmup ? "warmup" :
                cooldown || last_step ? "cooldown" :
                !dit->tea_cache_previous_ready ||
                    !dit->tea_cache_residual_ready ? "initialize" :
                forced_fresh ? "forced" :
                hit_cap ? "hit-cap" : "threshold";
            float score = 0.0f;
            float audio_score = 0.0f;
            int score_available = 0;
            double decision_accumulator = dit->tea_cache_accumulator;
            double decision_audio_accumulator =
                dit->tea_cache_audio_accumulator;
            if (!fresh) {
                if (!tea_cache_probe(dit, hidden_elements, &score,
                                     &audio_score,
                                     error, error_size)) return 0;
                score_available = 1;
                dit->tea_cache_accumulator += score;
                dit->tea_cache_audio_accumulator += audio_score;
                decision_accumulator = dit->tea_cache_accumulator;
                decision_audio_accumulator =
                    dit->tea_cache_audio_accumulator;
                int global_fresh = dit->tea_cache_accumulator >=
                    dit->tea_cache_threshold;
                int audio_fresh = dit->tea_cache_audio_threshold > 0.0f &&
                    dit->tea_cache_audio_accumulator >=
                        dit->tea_cache_audio_threshold;
                fresh = global_fresh || audio_fresh;
                reason = audio_fresh && !global_fresh ?
                    "audio-threshold" :
                    fresh ? "threshold" : "below-threshold";
            }
            dit->tea_cache_calls++;
            if (fresh) {
                dit->tea_cache_fresh++;
                if (hit_cap) dit->tea_cache_hit_cap_forced++;
                dit->tea_cache_consecutive_hits = 0;
                dit->tea_cache_accumulator = 0.0;
                dit->tea_cache_audio_accumulator = 0.0;
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->core_input, 0, dit->hidden, 0,
                    hidden_elements), "save TeaCache core input");
            } else {
                dit->tea_cache_reuse++;
                dit->tea_cache_consecutive_hits++;
                OP(h3_gpu_add_bf16(
                    dit->gpu, dit->hidden, dit->hidden,
                    dit->core_residual, hidden_elements),
                   "reuse TeaCache core residual");
                tea_cache_reused = 1;
                dit->tea_cache_last_reused = 1;
            }
            OP(h3_gpu_copy_bf16(
                dit->gpu, dit->tea_cache_previous, 0,
                dit->mod_attention, 0, hidden_elements),
               "save TeaCache modulated input");
            dit->tea_cache_previous_ready = 1;
            if (h3_runtime_getenv("H3_PROFILE")) {
                fprintf(stderr,
                        "h3 TeaCache: step=%d/%u action=%s reason=%s ",
                        step + 1, steps, fresh ? "fresh" : "reuse", reason);
                if (score_available)
                    fprintf(stderr, "relative-l1=%.8g ", score);
                else
                    fprintf(stderr, "relative-l1=n/a ");
                fprintf(stderr, "accumulator=%.8g threshold=%.8g",
                        decision_accumulator, dit->tea_cache_threshold);
                if (dit->tea_cache_audio_threshold > 0.0f) {
                    if (score_available)
                        fprintf(stderr, " audio-relative-l1=%.8g",
                                audio_score);
                    else
                        fprintf(stderr, " audio-relative-l1=n/a");
                    fprintf(stderr,
                            " audio-accumulator=%.8g audio-threshold=%.8g",
                            decision_audio_accumulator,
                            dit->tea_cache_audio_threshold);
                }
                fputc('\n', stderr);
            }
        }
        if (!tea_cache_reused) {
        unsigned command_blocks = disable_command_split
            ? 0 : command_block_interval(dit);
        if (dit->ssd_streaming) command_blocks = 0;
        unsigned eviction_interval = final_eviction_interval();
        int final_eviction = eviction_interval && !dit->ssd_streaming &&
            !dit->ane_mlp_io &&
            step == h3_dit_schedule_steps(dit->schedule) - 1;
        unsigned eviction_begin = 0;
        unsigned completed_blocks = 0;
        int carried_attention_adaln = 0;
        int carried_attention_input_quantized = 0;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            int fused_token_adaln = carried_attention_adaln;
            int fused_attention_input_quantized =
                carried_attention_input_quantized;
            carried_attention_adaln = 0;
            carried_attention_input_quantized = 0;
            if (tea_cache_attention_adaln_ready &&
                block == tea_cache_block) {
                fused_token_adaln = 1;
                fused_attention_input_quantized = 0;
                tea_cache_attention_adaln_ready = 0;
            }
            if (use_token_reduction &&
                block == dit->token_reduction_begin) {
                fused_token_adaln = block_active_at_step(
                    dit, block, step, use_step_gate_skip) &&
                    !h3_runtime_getenv("H3_DISABLE_FUSED_TOKEN_POOL_ADALN");
                if (fused_token_adaln) {
                    if (!enter_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!enter_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (use_token_reduction && block == token_reduction_end) {
                fused_token_adaln = block_active_at_step(
                    dit, block, step, use_step_gate_skip) &&
                    !h3_runtime_getenv("H3_DISABLE_FUSED_TOKEN_ADALN");
                if (fused_token_adaln) {
                    if (!leave_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!leave_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (!block_active_at_step(
                    dit, block, step, use_step_gate_skip)) continue;
            unsigned next_block = block + 1;
            int next_is_token_boundary = use_token_reduction &&
                (next_block == dit->token_reduction_begin ||
                 next_block == token_reduction_end);
            int fuse_next_attention =
                !h3_runtime_getenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                next_block < H3_DIT_BLOCKS &&
                block_active_at_step(
                    dit, next_block, step, use_step_gate_skip) &&
                !next_is_token_boundary;
            h3_dit_block streamed_weight;
            h3_dit_block *weight = &dit->blocks[block];
            h3_dit_stream_job stream_job;
            pthread_t stream_thread;
            int stream_started = 0;
            if (dit->ssd_streaming && !stream_block_pinned(dit, block)) {
                if (dit->stream_ready_layer != block ||
                    dit->stream_ready_slot > 1) {
                    fail(error, error_size,
                         "DiT SSD stream expected block %u, has block %u",
                         block, dit->stream_ready_layer);
                    return 0;
                }
                h3_dit_block *slot =
                    &dit->stream_slots[dit->stream_ready_slot];
                streamed_weight = dit->blocks[block];
                if (dit->ssd_quantized) {
                    streamed_weight.qkv_int8 = slot->qkv_int8;
                    streamed_weight.qkv_scales = slot->qkv_scales;
                    streamed_weight.out_int8 = slot->out_int8;
                    streamed_weight.out_scales = slot->out_scales;
                    streamed_weight.fc1_int8 = slot->fc1_int8;
                    streamed_weight.fc1_scales = slot->fc1_scales;
                    streamed_weight.fc2_int8 = slot->fc2_int8;
                    streamed_weight.fc2_scales = slot->fc2_scales;
                } else {
                    streamed_weight.qkv = slot->qkv;
                    streamed_weight.out = slot->out;
                    streamed_weight.fc1 = slot->fc1;
                    streamed_weight.fc2 = slot->fc2;
                }
                weight = &streamed_weight;

                unsigned future = next_streamed_block(dit, block);
                if (future != H3_DIT_BLOCKS) {
                    stream_job = (h3_dit_stream_job){
                        .dit = dit,
                        .layer = future,
                        .slot = dit->stream_ready_slot ^ 1u
                    };
                    int thread_error = pthread_create(
                        &stream_thread, NULL, read_stream_layer_thread,
                        &stream_job);
                    if (thread_error) {
                        fail(error, error_size,
                             "cannot start DiT SSD prefetch for block %u: %s",
                             future, strerror(thread_error));
                        return 0;
                    }
                    stream_started = 1;
                }
            }
            int block_ok = run_block(
                dit, block, step, weight, fused_token_adaln,
                fused_attention_input_quantized,
                fuse_next_attention, next_block,
                &carried_attention_adaln,
                &carried_attention_input_quantized,
                error, error_size);
            if (!block_ok) {
                if (stream_started) (void)pthread_join(stream_thread, NULL);
                return 0;
            }
            completed_blocks++;
            if (use_first_block_cache && block == first_cache_block) {
                int last_step =
                    step == h3_dit_schedule_steps(dit->schedule) - 1;
                int forced_fresh = h3_runtime_getenv("H3_FBC_FORCE_FRESH") != NULL;
                int hit_cap = dit->first_block_cache_max_hits &&
                    dit->first_block_cache_consecutive_hits >=
                        dit->first_block_cache_max_hits;
                int fresh = !dit->first_block_cache_previous_ready ||
                    !dit->first_block_cache_tail_ready || last_step ||
                    forced_fresh || hit_cap;
                float score = 0.0f;
                int score_available = 0;
                const char *reason = !dit->first_block_cache_previous_ready ||
                    !dit->first_block_cache_tail_ready ? "initialize" :
                    last_step ? "final" :
                    forced_fresh ? "forced" :
                    hit_cap ? "hit-cap" : "threshold";
                if (!fresh) {
                    if (!first_block_cache_probe(
                            dit, hidden_elements, &score,
                            error, error_size)) return 0;
                    score_available = 1;
                    fresh = score > dit->first_block_cache_threshold;
                    reason = fresh ? "threshold" : "below-threshold";
                }
                dit->first_block_cache_calls++;
                if (fresh) {
                    dit->first_block_cache_fresh++;
                    if (hit_cap) dit->first_block_cache_hit_cap_forced++;
                    dit->first_block_cache_consecutive_hits = 0;
                    OP(h3_gpu_sub_bf16(
                        dit->gpu, dit->first_block_cache_previous,
                        dit->hidden, dit->core_input, hidden_elements),
                       "save FirstBlockCache head residual");
                    OP(h3_gpu_copy_bf16(
                        dit->gpu, dit->core_input, 0, dit->hidden, 0,
                        hidden_elements),
                       "save FirstBlockCache head output");
                    dit->first_block_cache_previous_ready = 1;
                } else {
                    dit->first_block_cache_reuse++;
                    dit->first_block_cache_consecutive_hits++;
                    OP(h3_gpu_add_bf16(
                        dit->gpu, dit->hidden, dit->hidden,
                        dit->core_residual, hidden_elements),
                       "reuse FirstBlockCache tail residual");
                    first_block_cache_reused = 1;
                    dit->first_block_cache_last_reused = 1;
                }
                if (h3_runtime_getenv("H3_PROFILE")) {
                    fprintf(stderr,
                            "h3 FBC: step=%d/%d action=%s reason=%s ",
                            step + 1,
                            h3_dit_schedule_steps(dit->schedule),
                            fresh ? "fresh" : "reuse", reason);
                    if (score_available)
                        fprintf(stderr, "relative-l1=%.8g ", score);
                    else
                        fprintf(stderr, "relative-l1=n/a ");
                    fprintf(stderr, "threshold=%.8g\n",
                            dit->first_block_cache_threshold);
                }
                if (first_block_cache_reused) break;
            }
            if (command_blocks &&
                completed_blocks < (use_step_gate_skip ?
                    dit->step_active_block_count[step] :
                    dit->active_block_count) &&
                completed_blocks % command_blocks == 0)
                OP(h3_gpu_continue(dit->gpu), "continue DiT command chain");
            if (stream_started) {
                int gpu_ok = gpu_op(dit, h3_gpu_submit(dit->gpu),
                                    error, error_size,
                                    "submit streamed DiT block");
                double wait_started = stream_now();
                int join_error = pthread_join(stream_thread, NULL);
                dit->stream_wait_seconds += stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    fail(error, error_size,
                         "cannot join DiT SSD prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->stream_bytes += stream_job.bytes;
                dit->stream_read_seconds += stream_job.seconds;
                if (!stream_job.ok) {
                    fail(error, error_size,
                         "cannot stream DiT block %u: %s",
                         stream_job.layer, stream_job.error);
                    return 0;
                }
                dit->stream_ready_layer = stream_job.layer;
                dit->stream_ready_slot = stream_job.slot;
                OP(h3_gpu_begin(dit->gpu),
                   "continue after streamed DiT block");
            }
            if (final_eviction &&
                ((block + 1) % eviction_interval == 0 ||
                 block + 1 == H3_DIT_BLOCKS)) {
                if (!evict_final_block_group(
                        dit, eviction_begin, block + 1,
                        error, error_size)) return 0;
                eviction_begin = block + 1;
                report(progress, progress_opaque,
                       "denoise final eviction", (int)eviction_begin,
                       H3_DIT_BLOCKS);
            }
        }
        if (use_token_reduction &&
            token_reduction_end == H3_DIT_BLOCKS &&
            !leave_token_reduction(dit, error, error_size)) return 0;
        }
        if (use_tea_cache && !tea_cache_reused) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache TeaCache core residual");
            dit->tea_cache_residual_ready = 1;
        } else if (use_first_block_cache && !first_block_cache_reused) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache FirstBlockCache tail residual");
            dit->first_block_cache_tail_ready = 1;
        } else if (dit->core_reuse_interval > 1) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache DiT core residual");
            dit->core_residual_ready = 1;
        }
    } else {
        OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                           dit->core_residual, hidden_elements),
           "reuse DiT core residual");
    }
    dit->core_forward_count++;
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    int fused_final_head = dit->bf16_final &&
        !h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_HEAD") &&
        !h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_SLICE");
    if (fused_final_head) {
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->audio_output_bf16, dit->final_audio_inverse,
            dit->hidden, (size_t)dit->audio_target_start * HIDDEN,
            dit->final_norm, final, dit->final_audio_maps[step],
            dit->final_audio_w, dit->final_audio_b, dit->audio_rows, HIDDEN,
            AUDIO_CHANNELS, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final audio AdaLN/head");
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->video_output_bf16, dit->final_video_inverse,
            dit->hidden, (size_t)dit->video_target_start * HIDDEN,
            dit->final_norm, final, dit->final_video_maps[step],
            dit->final_video_w, dit->final_video_b, dit->video_rows, HIDDEN,
            VIDEO_PATCH, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final video AdaLN/head");
    } else if (h3_runtime_getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN,
            (size_t)dit->audio_rows * HIDDEN), "slice final audio");
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN,
            (size_t)dit->video_rows * HIDDEN), "slice final video");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
            dit->final_audio_input, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final audio AdaLN");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
            dit->final_video_input, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final video AdaLN");
    } else {
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_audio_norm, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final audio slice/AdaLN");
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_video_norm, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final video slice/AdaLN");
    }
    if (dit->bf16_final && !fused_final_head) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->audio_output_bf16,
            dit->final_audio_norm, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS),
           "BF16 final audio head");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->video_output_bf16,
            dit->final_video_norm, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH),
           "BF16 final video head");
    } else if (!dit->bf16_final) {
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
            dit->final_audio_norm, dit->audio_rows * HIDDEN),
           "final audio F32 cast");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
            dit->final_video_norm, dit->video_rows * HIDDEN),
           "final video F32 cast");
        OP(h3_gpu_linear_f32(dit->gpu, dit->audio_output,
            dit->final_audio_f32, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS), "final audio head");
        OP(h3_gpu_linear_f32(dit->gpu, dit->video_output,
            dit->final_video_f32, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH), "final video head");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
            dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
           "final audio output cast");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
            dit->video_output, dit->video_rows * VIDEO_PATCH),
           "final video output cast");
    }
    if (submit) OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#undef OP
    return 1;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

static void reset_run_state(h3_dit *dit) {
    dit->core_forward_count = 0;
    dit->core_residual_ready = 0;
    dit->first_block_cache_previous_ready = 0;
    dit->first_block_cache_tail_ready = 0;
    dit->first_block_cache_calls = 0;
    dit->first_block_cache_fresh = 0;
    dit->first_block_cache_reuse = 0;
    dit->first_block_cache_consecutive_hits = 0;
    dit->first_block_cache_hit_cap_forced = 0;
    dit->first_block_cache_wait_seconds = 0.0;
    dit->first_block_cache_last_reused = 0;
    dit->tea_cache_previous_ready = 0;
    dit->tea_cache_residual_ready = 0;
    dit->tea_cache_calls = 0;
    dit->tea_cache_fresh = 0;
    dit->tea_cache_reuse = 0;
    dit->tea_cache_consecutive_hits = 0;
    dit->tea_cache_hit_cap_forced = 0;
    dit->tea_cache_accumulator = 0.0;
    dit->tea_cache_audio_accumulator = 0.0;
    dit->tea_cache_wait_seconds = 0.0;
    dit->tea_cache_last_reused = 0;
    dit->sol_calls = 0;
    dit->sol_dense_calls = 0;
    dit->sol_route_profile_pending = 0;
    dit->sol_verify_pending = 0;
    dit->token_reduction_active = 0;
    dit->coreml_benchmark_capture_pending = 0;
    dit->coreml_benchmark_capture_ready = 0;
}

static void free_request_state(h3_dit *dit) {
    if (!dit) return;
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->reduced_row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->reduced_row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->reduced_row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    dit->row_maps = NULL;
    dit->reduced_row_maps = NULL;
    dit->final_audio_maps = NULL;
    dit->final_video_maps = NULL;
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->reduced_rope_cos);
    free_tensor(&dit->reduced_rope_sin);
#define FREE_REQUEST(field) free_tensor(&dit->field)
    if (dit->activation_aliases) {
        dit->attention_heads = NULL;
        dit->mod_mlp = NULL;
    }
    FREE_REQUEST(video_input); FREE_REQUEST(audio_input);
    FREE_REQUEST(video_projected_f32); FREE_REQUEST(audio_projected_f32);
    FREE_REQUEST(video_projected); FREE_REQUEST(audio_projected);
    FREE_REQUEST(video_projection_map); FREE_REQUEST(audio_projection_map);
    FREE_REQUEST(hidden); FREE_REQUEST(core_input); FREE_REQUEST(core_residual);
    FREE_REQUEST(first_block_cache_previous);
    FREE_REQUEST(first_block_cache_partials);
    FREE_REQUEST(tea_cache_previous);
    FREE_REQUEST(tea_cache_partials);
    FREE_REQUEST(sol_query_centroids); FREE_REQUEST(sol_key_centroids);
    FREE_REQUEST(sol_value_sums); FREE_REQUEST(sol_thresholds);
    FREE_REQUEST(sol_routes); FREE_REQUEST(sol_verify_dense);
    FREE_REQUEST(sol_verify_partials); FREE_REQUEST(mod_attention);
    FREE_REQUEST(qkv); FREE_REQUEST(query); FREE_REQUEST(key);
    FREE_REQUEST(value); FREE_REQUEST(attention_heads);
    FREE_REQUEST(attention_output); FREE_REQUEST(token_pool_pairs);
    FREE_REQUEST(token_baseline_indices); FREE_REQUEST(token_expand_parents);
    FREE_REQUEST(token_original); FREE_REQUEST(mod_mlp); FREE_REQUEST(fc1);
    FREE_REQUEST(activated); FREE_REQUEST(mlp_output);
    FREE_REQUEST(coreml_benchmark_mlp_input);
    FREE_REQUEST(coreml_benchmark_mlp_output);
    FREE_REQUEST(int8_activation); FREE_REQUEST(int8_activation_scales);
    FREE_REQUEST(final_audio_input); FREE_REQUEST(final_video_input);
    FREE_REQUEST(final_audio_inverse); FREE_REQUEST(final_video_inverse);
    FREE_REQUEST(final_audio_norm); FREE_REQUEST(final_video_norm);
    FREE_REQUEST(final_audio_f32); FREE_REQUEST(final_video_f32);
    FREE_REQUEST(audio_output); FREE_REQUEST(video_output);
    FREE_REQUEST(audio_output_bf16); FREE_REQUEST(video_output_bf16);
    FREE_REQUEST(previous_audio_velocity); FREE_REQUEST(previous_video_velocity);
#undef FREE_REQUEST
    free(dit->first_block_cache_host_partials);
    dit->first_block_cache_host_partials = NULL;
    free(dit->tea_cache_host_partials);
    dit->tea_cache_host_partials = NULL;
    h3_layout_free(&dit->layout);
    dit->activation_aliases = 0;
    dit->fused_patch_projection = 0;
    dit->fused_patch_pack = 0;
    dit->token_reduction_active = 0;
    dit->token_original_in_qkv = 0;
    dit->token_original_offset = 0;
    dit->token_baseline_offset = 0;
    dit->first_block_cache_partial_count = 0;
    dit->tea_cache_partial_count = 0;
    dit->tea_cache_audio_partial_count = 0;
    dit->sol_route_elements = 0;
    dit->sol_verify_elements = 0;
    dit->sol_verify_partial_count = 0;
    dit->coreml_benchmark_capture_elements = 0;
    dit->coreml_benchmark_capture_pending = 0;
    dit->coreml_benchmark_capture_ready = 0;
    dit->request_ready = 0;
}

static int same_sigmas(const h3_sigma_schedule *left,
                       const h3_sigma_schedule *right) {
    if (!left || !right || left->steps != right->steps ||
        left->steps < 0 || left->steps > H3_MAX_STEPS) return 0;
    size_t bytes = ((size_t)left->steps + 1) * sizeof(left->video[0]);
    return !memcmp(left->video, right->video, bytes) &&
           !memcmp(left->audio, right->audio, bytes);
}

void h3_dit_release_request(h3_dit *dit) {
    if (!dit) return;
    free_request_state(dit);
    reset_run_state(dit);
}

static int same_token_reduction_policy(const h3_dit *left,
                                       const h3_dit *right) {
    return left->token_reduction == right->token_reduction &&
        (!left->token_reduction ||
         (left->token_reduction_begin == right->token_reduction_begin &&
          left->token_reduction_end == right->token_reduction_end &&
          left->token_reduction_early_steps ==
              right->token_reduction_early_steps &&
          left->token_reduction_early_end ==
              right->token_reduction_early_end &&
          left->token_reduction_scale == right->token_reduction_scale));
}

static void take_request_geometry(h3_dit *dit, h3_dit *request,
                                  float spatial_rope_scale) {
    dit->layout = request->layout;
    memset(&request->layout, 0, sizeof(request->layout));
    dit->latent_t = request->latent_t;
    dit->latent_h = request->latent_h;
    dit->latent_w = request->latent_w;
    dit->audio_t = request->audio_t;
    dit->text_rows = request->text_rows;
    dit->video_condition_rows = request->video_condition_rows;
    dit->audio_condition_rows = request->audio_condition_rows;
    dit->audio_rows = request->audio_rows;
    dit->video_rows = request->video_rows;
    dit->video_total_rows = request->video_total_rows;
    dit->audio_total_rows = request->audio_total_rows;
    dit->audio_target_start = request->audio_target_start;
    dit->video_target_start = request->video_target_start;
    dit->sequence = request->sequence;
    dit->reduced_sequence = request->reduced_sequence;
    dit->reduced_video_rows = request->reduced_video_rows;
    dit->token_baseline_rows = request->token_baseline_rows;
    dit->token_reduction = request->token_reduction;
    dit->token_reduction_begin = request->token_reduction_begin;
    dit->token_reduction_end = request->token_reduction_end;
    dit->token_reduction_early_steps = request->token_reduction_early_steps;
    dit->token_reduction_early_end = request->token_reduction_early_end;
    dit->token_reduction_scale = request->token_reduction_scale;
    dit->spatial_rope_scale = spatial_rope_scale;
}

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
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || dit->final_evicted_blocks || !text || !layout || !sigmas ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f) {
        fail(error, error_size, "invalid resident DiT reprepare arguments");
        return 0;
    }
    if (!same_sigmas(&dit->sigmas, sigmas)) {
        fail(error, error_size,
             "resident DiT sigma schedule changed; full reload required");
        return 0;
    }
    if (dit->ssd_streaming &&
        (dit->stream_ready_layer != first_streamed_block(dit) ||
         dit->stream_ready_slot > 1)) {
        fail(error, error_size,
             "resident SSD stream is not reset to its first active block");
        return 0;
    }

    h3_dit request;
    memset(&request, 0, sizeof(request));
    if (!copy_layout(&request, layout, error, error_size) ||
        !validate_layout(&request, text, error, error_size) ||
        !configure_token_reduction(&request, dit->token_reduction,
                                   error, error_size)) {
        h3_layout_free(&request.layout);
        return 0;
    }
    if ((request.video_condition_rows != 0) !=
            (dit->video_condition_rows != 0) ||
        (request.audio_condition_rows != 0) !=
            (dit->audio_condition_rows != 0)) {
        h3_layout_free(&request.layout);
        fail(error, error_size,
             "resident DiT conditioning modality changed; full reload required");
        return 0;
    }
    if (!same_token_reduction_policy(dit, &request)) {
        h3_layout_free(&request.layout);
        fail(error, error_size,
             "resident DiT token-reduction policy changed; full reload required");
        return 0;
    }
    if ((dit->sequence >= 128) != (request.sequence >= 128)) {
        h3_layout_free(&request.layout);
        fail(error, error_size,
             "resident DiT crossed a sequence-dependent backend boundary");
        return 0;
    }
    if ((dit->ane_mlp_io || dit->ane_qkv_io ||
         dit->ane_attention_out_io) &&
        dit->sequence != request.sequence) {
        h3_layout_free(&request.layout);
        fail(error, error_size,
             "private ANE runtime has fixed rows; changed sequence requires a "
             "full reload");
        return 0;
    }
    size_t wanted_video =
        (size_t)request.video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)request.audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        h3_layout_free(&request.layout);
        fail(error, error_size,
             "resident DiT condition rows do not match the new layout");
        return 0;
    }

    double started = stream_now();
    free_request_state(dit);
    take_request_geometry(dit, &request, spatial_rope_scale);
    report(progress, progress_opaque, "resident DiT request", 0, 5);
    if (!refine_text(dit, text, error, error_size)) goto failed;
    report(progress, progress_opaque, "resident DiT request", 1, 5);
    if (!prepare_rope(dit, error, error_size)) goto failed;
    report(progress, progress_opaque, "resident DiT request", 2, 5);
    if (!prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size)) goto failed;
    report(progress, progress_opaque, "resident DiT request", 3, 5);
    if (!allocate_activations(dit, error, error_size)) goto failed;
    report(progress, progress_opaque, "resident DiT request", 4, 5);
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        fail(error, error_size,
             "cannot write resident DiT condition rows");
        goto failed;
    }
    dit->request_ready = 1;
    reset_run_state(dit);
    report(progress, progress_opaque, "resident DiT request", 5, 5);
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident DiT reprepare %.3fs\n",
                stream_now() - started);
    return 1;
failed:
    h3_layout_free(&request.layout);
    return 0;
}

int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !dit->request_ready || dit->final_evicted_blocks) {
        fail(error, error_size, "prepared DiT is absent");
        return 0;
    }
    size_t wanted_video =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        fail(error, error_size, "prepared DiT condition rows do not match");
        return 0;
    }
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        fail(error, error_size, "cannot refresh prepared DiT conditions");
        return 0;
    }
    reset_run_state(dit);
    return 1;
}

int h3_dit_set_coreml_benchmark_route(
                     h3_dit *dit, h3_dit_coreml_benchmark_route route,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (route < H3_DIT_COREML_BENCHMARK_DEFAULT ||
        route > H3_DIT_COREML_BENCHMARK_CANDIDATE) {
        fail(error, error_size, "invalid Core ML benchmark route");
        return 0;
    }
    if (!dit) {
        fail(error, error_size, "Core ML benchmark DiT is absent");
        return 0;
    }
    if (!dit->coreml_same_loaded_ab) {
        fail(error, error_size,
             "Core ML benchmark routes require "
             "H3_BENCH_COREML_SAME_LOADED_AB");
        return 0;
    }
    unsigned selected = 0;
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        h3_dit_block *block = &dit->blocks[index];
        if (!block->coreml_mlp) continue;
        selected++;
        if (!block->coreml_full_fc1 || !block->coreml_full_fc2) {
            fail(error, error_size,
                 "Core ML benchmark full-GPU weights are absent");
            return 0;
        }
        if (h3_coreml_mlp_prediction_inflight(block->coreml_mlp)) {
            fail(error, error_size,
                 "cannot switch Core ML benchmark route with a prediction "
                 "in flight");
            return 0;
        }
    }
    if (!selected) {
        fail(error, error_size,
             "Core ML benchmark routes require at least one loaded block");
        return 0;
    }
    dit->coreml_benchmark_route = route;
    dit->coreml_benchmark_capture_pending = 0;
    dit->coreml_benchmark_capture_ready = 0;
    return 1;
}

size_t h3_dit_coreml_benchmark_capture_elements(const h3_dit *dit) {
    if (!dit || !dit->coreml_same_loaded_ab ||
        !dit->coreml_benchmark_mlp_input ||
        !dit->coreml_benchmark_mlp_output) return 0;
    return dit->coreml_benchmark_capture_elements;
}

int h3_dit_read_coreml_benchmark_capture(
                     const h3_dit *dit,
                     uint16_t *modulated_mlp_input, size_t input_elements,
                     uint16_t *mlp_branch_output, size_t output_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!modulated_mlp_input || !mlp_branch_output || !input_elements ||
        input_elements != output_elements) {
        fail(error, error_size,
             "invalid Core ML benchmark capture buffers or counts");
        return 0;
    }
    size_t expected = h3_dit_coreml_benchmark_capture_elements(dit);
    if (!expected) {
        fail(error, error_size,
             "Core ML benchmark MLP capture is unavailable");
        return 0;
    }
    if (input_elements != expected) {
        fail(error, error_size,
             "Core ML benchmark capture requires exactly %zu elements",
             expected);
        return 0;
    }
    if (!dit->coreml_benchmark_capture_ready) {
        fail(error, error_size,
             "Core ML benchmark MLP capture is not ready");
        return 0;
    }
    if (!h3_gpu_tensor_read_bf16(
            dit->coreml_benchmark_mlp_input,
            modulated_mlp_input, input_elements) ||
        !h3_gpu_tensor_read_bf16(
            dit->coreml_benchmark_mlp_output,
            mlp_branch_output, output_elements)) {
        fail(error, error_size,
             "cannot read Core ML benchmark MLP capture");
        return 0;
    }
    return 1;
}


static int h3_dit_forward_progress(
                   h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !dit->request_ready || dit->final_evicted_blocks || step < 0 ||
        step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    float *video_rows = malloc(video_row_elements * sizeof(*video_rows));
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    uint16_t *video_out = malloc(video_row_elements * sizeof(*video_out));
    uint16_t *audio_out = malloc(audio_row_elements * sizeof(*audio_out));
    float *video_f32 = malloc(video_row_elements * sizeof(*video_f32));
    float *audio_f32 = malloc(audio_row_elements * sizeof(*audio_f32));
    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        fail(error, error_size, "out of memory packing DiT latents");
        free(video_rows); free(audio_rows); free(video_out); free(audio_out);
        free(video_f32); free(audio_f32);
        return 0;
    }
    if (dit->coreml_nonfinite)
        memset(h3_gpu_tensor_host_pointer(dit->coreml_nonfinite), 0,
               sizeof(uint32_t));
    dit->coreml_benchmark_capture_pending = 0;
    dit->coreml_benchmark_capture_ready = 0;
    dit->sol_verify_pending = 0;
    dit->sol_route_profile_pending = 0;
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = encode_forward(
        dit, step, 1, 1, 0, progress, progress_opaque,
        error, error_size);
    if (ok && dit->coreml_nonfinite) {
        float encoded = 0.0f;
        uint32_t block_plus_one = 0;
        if (!h3_gpu_tensor_read_f32(
                dit->coreml_nonfinite, &encoded, 1)) {
            fail(error, error_size,
                 "cannot read Core ML MLP nonfinite flag");
            ok = 0;
        } else {
            memcpy(&block_plus_one, &encoded, sizeof(block_plus_one));
            if (block_plus_one) {
                fail(error, error_size,
                     "Core ML MLP produced nonfinite output at step %d "
                     "block %u", step + 1, block_plus_one - 1u);
                ok = 0;
            }
        }
    }
    if (ok && dit->sol_verify_pending) {
        size_t values_count = (size_t)dit->sol_verify_partial_count * 4;
        float *values = malloc(values_count * sizeof(*values));
        if (!values || !h3_gpu_tensor_read_f32(
                dit->sol_verify_partials, values, values_count)) {
            free(values);
            fail(error, error_size,
                 "cannot read BQ64 attention verification partials");
            ok = 0;
        } else {
            double square_error = 0.0;
            double square_reference = 0.0;
            double maximum_error = 0.0;
            double maximum_reference = 0.0;
            for (uint32_t group = 0;
                 group < dit->sol_verify_partial_count; group++) {
                square_error += values[(size_t)group * 4];
                square_reference += values[(size_t)group * 4 + 1];
                if (values[(size_t)group * 4 + 2] > maximum_error)
                    maximum_error = values[(size_t)group * 4 + 2];
                if (values[(size_t)group * 4 + 3] > maximum_reference)
                    maximum_reference = values[(size_t)group * 4 + 3];
            }
            fprintf(stderr,
                    "h3: BQ64 attention layer %u elements %u relL2 %.9g "
                    "relMax %.9g abs %.9g\n",
                    dit->sol_verify_layer, dit->sol_verify_elements,
                    sqrt(square_error / fmax(square_reference, 1e-30)),
                    maximum_error / fmax(maximum_reference, 1e-30),
                    maximum_error);
            free(values);
        }
    }
    if (ok && dit->sol_route_profile_pending) {
        float *routes = malloc(dit->sol_route_elements * sizeof(*routes));
        if (!routes || !h3_gpu_tensor_read_f32(
                dit->sol_routes, routes, dit->sol_route_elements)) {
            free(routes);
            fail(error, error_size, "cannot read Sol route profile");
            ok = 0;
        } else {
            size_t exact = 0;
            for (size_t index = 0; index < dit->sol_route_elements; index++)
                exact += routes[index] != 0.0f;
            fprintf(stderr,
                    "h3: Sol route layer %u exact %zu/%zu density %.6f\n",
                    dit->sol_route_profile_layer, exact,
                    dit->sol_route_elements,
                    (double)exact / (double)dit->sol_route_elements);
            free(routes);
        }
    }
    if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot read DiT output");
    if (ok) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot unpack DiT output");
    if (dit->coreml_same_loaded_ab)
        dit->coreml_benchmark_capture_ready = ok &&
            dit->coreml_benchmark_capture_pending;
    free(video_rows); free(audio_rows); free(video_out); free(audio_out);
    free(video_f32); free(audio_f32);
    return ok;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    return h3_dit_forward_progress(
        dit, step, video_latent, audio_latent,
        video_velocity, audio_velocity, NULL, NULL, error, error_size);
}

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

int h3_dit_get_streaming_info(const h3_dit *dit,
                              h3_dit_streaming_info *info) {
    if (!dit || !info) return 0;
    memset(info, 0, sizeof(*info));
    info->enabled = dit->ssd_streaming;
    info->active_blocks = dit->active_block_count;
    if (!dit->ssd_streaming) return 1;
    info->pinned_blocks = (unsigned)dit->ssd_pinned_prefix;
    info->streamed_blocks = info->active_blocks - info->pinned_blocks;
    info->memory_budget_bytes = dit->ssd_memory_budget_bytes;
    info->block_bytes = ssd_full_block_bytes(dit);
    info->quantized = dit->ssd_quantized;
    info->activation_reserve_bytes = ssd_activation_reserve_bytes(dit);
    info->bytes_read = dit->stream_bytes;
    info->read_seconds = dit->stream_read_seconds;
    info->wait_seconds = dit->stream_wait_seconds;
    return 1;
}

int h3_dit_final_evicted(const h3_dit *dit) {
    return dit && dit->final_evicted_blocks != 0;
}

static float extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) return 0.0f;
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    return ratio;
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float ratio = extrapolation_ratio(current_sigma, last_sigma,
                                      previous_sigma, have_previous);
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio * (last[index] - previous[index]);
}

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps) return -1;
    memset(selected, 0, (size_t)steps);

    int count = 0;
    for (int step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static int parse_reuse_steps(int steps, uint8_t *selected) {
    const char *text = h3_runtime_getenv("H3_REUSE_STEPS");
    if (!text || !*text) return 0;
    memset(selected, 0, (size_t)steps);
    int count = 0;
    int previous = -1;
    while (*text) {
        char *end = NULL;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value >= steps ||
            value <= previous) return -1;
        selected[value] = 1;
        previous = (int)value;
        count++;
        if (!*end) break;
        if (*end != ',') return -1;
        text = end + 1;
        if (!*text) return -1;
    }
    return selected[0] && selected[steps - 1] ? count : -1;
}

static int gpu_sampler_requested(const h3_dit *dit) {
    const char *cpu = h3_runtime_getenv("H3_CPU_SAMPLER");
    if (cpu && *cpu && strcmp(cpu, "0")) return 0;
    const char *value = h3_runtime_getenv("H3_GPU_SAMPLER");
    if (value) return *value && strcmp(value, "0");
    return h3_gpu_is_m5(dit->gpu);
}

static unsigned gpu_sampler_window(void) {
    const char *value = h3_runtime_getenv("H3_GPU_SAMPLER_WINDOW");
    if (!value || !*value) return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end != value && !*end && parsed >= 0 && parsed <= H3_MAX_STEPS
        ? (unsigned)parsed : 1;
}

static int ensure_previous_velocities(h3_dit *dit, char *error,
                                      size_t error_size) {
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    dit->previous_video_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->video_rows * VIDEO_PATCH);
    dit->previous_audio_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->audio_rows * AUDIO_CHANNELS);
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    fail(error, error_size, "cannot allocate GPU Euler velocity cache: %s",
         h3_gpu_error(dit->gpu));
    return 0;
}

static int denoise_euler_gpu(h3_dit *dit, float *video_latent,
                             float *audio_latent, int reuse_interval,
                             h3_dit_progress progress, void *progress_opaque,
                             h3_dit_preview preview, void *preview_opaque,
                             char *error, size_t error_size) {
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s GPU reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    unsigned window = gpu_sampler_window();
    int profile_steps = profile_steps_enabled();
    int disable_command_split = window == 1 &&
                                h3_runtime_getenv("H3_DIT_COMMAND_BLOCKS") == NULL;
    if (h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: GPU sampler encode window is %s; internal split "
                "%s\n", window ? "bounded" : "unbounded",
                disable_command_split ? "disabled" : "enabled");

    size_t video_count = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_count = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    size_t video_offset = (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t audio_offset = (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (video_count > UINT32_MAX || audio_count > UINT32_MAX ||
        video_offset > UINT32_MAX - video_count ||
        audio_offset > UINT32_MAX - audio_count ||
        (reuse_interval > 1 &&
         !ensure_previous_velocities(dit, error, error_size))) return 0;

    float *video_rows = malloc(video_count * sizeof(*video_rows));
    float *audio_rows = malloc(audio_count * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        fail(error, error_size, "out of memory packing GPU Euler latents");
        free(video_rows);
        free(audio_rows);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows, video_count) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_count) &&
        h3_gpu_tensor_write_f32_range(dit->video_input, video_offset,
                                      video_rows, video_count) &&
        h3_gpu_tensor_write_f32_range(dit->audio_input, audio_offset,
                                      audio_rows, audio_count);
    if (!ok) fail(error, error_size, "cannot pack/write GPU Euler latents");

    int last_evaluated = -1;
    int previous_evaluated = -1;
    unsigned pending_evaluations = 0;
    int command_active = 0;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        double step_wall_start = stream_now();
        h3_gpu_stats step_stats_start = {0};
        if (profile_steps)
            (void)h3_gpu_get_stats(dit->gpu, &step_stats_start);
        report(progress, progress_opaque, "denoise enqueue", step,
               dit->sigmas.steps);
        if (!command_active) {
            ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                        "begin GPU Euler command chain");
            command_active = ok;
        }
        if (!ok) break;
        int evaluate = selected[step];
        if (evaluate) {
            if (last_evaluated >= 0 && reuse_interval > 1) {
                ok = gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_video_velocity, 0,
                    dit->video_output_bf16, 0, video_count),
                    error, error_size, "cache previous video velocity") &&
                    gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_audio_velocity, 0,
                    dit->audio_output_bf16, 0, audio_count),
                    error, error_size, "cache previous audio velocity");
                if (ok) previous_evaluated = last_evaluated;
            }
            if (ok) ok = encode_forward(
                dit, step, 0, 0, disable_command_split,
                progress, progress_opaque, error, error_size);
            if (ok) {
                last_evaluated = step;
                pending_evaluations++;
            }
        }
        if (!ok) break;

        float video_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.video[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        float audio_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.audio[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        const h3_gpu_tensor *previous_video = previous_evaluated >= 0
            ? dit->previous_video_velocity : dit->video_output_bf16;
        const h3_gpu_tensor *previous_audio = previous_evaluated >= 0
            ? dit->previous_audio_velocity : dit->audio_output_bf16;
        ok = gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->video_input, video_offset,
                dit->video_output_bf16, previous_video, (uint32_t)video_count,
                dit->sigmas.video[step] - dit->sigmas.video[step + 1],
                video_ratio), error, error_size, "GPU video Euler step") &&
             gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->audio_input, audio_offset,
                dit->audio_output_bf16, previous_audio, (uint32_t)audio_count,
                dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                audio_ratio), error, error_size, "GPU audio Euler step");
        if (ok && (evaluate || preview)) {
            int finish = profile_steps || preview ||
                         step + 1 == dit->sigmas.steps ||
                         (window && pending_evaluations >= window);
            ok = gpu_op(dit, finish ? h3_gpu_submit(dit->gpu)
                                    : h3_gpu_continue(dit->gpu),
                        error, error_size,
                        finish ? "submit GPU Euler window"
                               : "continue GPU Euler command chain");
            if (ok && finish) {
                command_active = 0;
                pending_evaluations = 0;
            }
        }
        if (ok && profile_steps && command_active) {
            ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                        "submit profiled GPU Euler step");
            if (ok) {
                command_active = 0;
                pending_evaluations = 0;
            }
        }
        if (ok && profile_steps)
            profile_denoise_step(dit, step, dit->sigmas.steps, evaluate,
                                 step_wall_start, &step_stats_start);
        if (ok && preview) {
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit));
            if (!ok) {
                fail(error, error_size,
                     "cannot read GPU Euler preview latent at step %d", step);
            } else if (preview(step + 1, dit->sigmas.steps, video_latent,
                               h3_dit_video_elements(dit), preview_opaque)) {
                fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok) report(progress, progress_opaque, "denoise enqueue", step + 1,
                       dit->sigmas.steps);
    }
    if (ok && command_active)
        ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit GPU Euler denoise");
    if (ok) ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count);
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot read GPU Euler latents");
    if (ok) ok = h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent, h3_dit_video_elements(dit)) &&
                 h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, dit->audio_t,
                                     audio_latent,
                                     h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot unpack GPU Euler latents");
    free(video_rows);
    free(audio_rows);
    if (ok) report(progress, progress_opaque, "denoise", dit->sigmas.steps,
                   dit->sigmas.steps);
    h3_gpu_profile_mark(dit->gpu, "GPU Euler denoise");
    return ok;
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !dit->request_ready || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        ok = h3_dit_forward_progress(
            dit, step, video_latent, audio_latent,
            video_velocity, audio_velocity, progress, progress_opaque,
            error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT);
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            report(progress, progress_opaque, "denoise", step + 1,
                   dit->sigmas.steps);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !dit->request_ready || !video_latent || !audio_latent ||
        reuse_interval < 1 ||
        reuse_interval > 32 ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    if ((dit->first_block_cache || dit->tea_cache) &&
        reuse_interval > 1) {
        fail(error, error_size,
             "adaptive cache cannot be combined with denoiser reuse");
        return 0;
    }
    if (gpu_sampler_requested(dit))
        return denoise_euler_gpu(dit, video_latent, audio_latent,
                                 reuse_interval, progress, progress_opaque,
                                 preview, preview_opaque,
                                 error, error_size);
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr, "h3: %s reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *last_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*last_video)) : NULL;
    float *previous_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*previous_video)) : NULL;
    float *last_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*last_audio)) : NULL;
    float *previous_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*previous_audio)) : NULL;
    if (!video_velocity || !audio_velocity ||
        (reuse_interval > 1 &&
         (!last_video || !previous_video || !last_audio || !previous_audio))) {
        fail(error, error_size, "out of memory allocating Euler velocities");
        free(video_velocity);
        free(audio_velocity);
        free(last_video);
        free(previous_video);
        free(last_audio);
        free(previous_audio);
        return 0;
    }
    int ok = 1;
    int last_evaluated = -1;
    int previous_evaluated = -1;
    int profile_steps = profile_steps_enabled();
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        double step_wall_start = stream_now();
        h3_gpu_stats step_stats_start = {0};
        if (profile_steps)
            (void)h3_gpu_get_stats(dit->gpu, &step_stats_start);
        report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        int evaluate = selected[step];
        if (evaluate) {
            ok = h3_dit_forward_progress(
                dit, step, video_latent, audio_latent,
                video_velocity, audio_velocity, progress, progress_opaque,
                error, error_size);
            if (ok && reuse_interval > 1) {
                if (last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
        } else {
            extrapolate_velocity(
                video_velocity, last_video, previous_video, video_count,
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
        }
        if (ok) {
            ok = h3_euler_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 h3_euler_velocity_step(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            if (!ok) fail(error, error_size,
                          "Euler solver rejected step %d", step);
        }
        if (ok && preview &&
            preview(step + 1, dit->sigmas.steps, video_latent, video_count,
                    preview_opaque)) {
            fail(error, error_size, "denoising preview stopped at step %d",
                 step + 1);
            ok = 0;
        }
        if (ok) report(progress, progress_opaque, "denoise", step + 1,
                       dit->sigmas.steps);
        if (ok && profile_steps)
            profile_denoise_step(dit, step, dit->sigmas.steps, evaluate,
                                 step_wall_start, &step_stats_start);
    }
    free(video_velocity);
    free(audio_velocity);
    free(last_video);
    free(previous_video);
    free(last_audio);
    free(previous_audio);
    h3_gpu_profile_mark(dit->gpu, "Euler denoise");
    return ok;
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_preview(
        dit, video_latent, audio_latent, reuse_interval,
        progress, progress_opaque, NULL, NULL, error, error_size);
}

void h3_dit_free(h3_dit *dit) {
    if (!dit) return;
    if (dit->first_block_cache && h3_runtime_getenv("H3_PROFILE")) {
        fprintf(stderr,
                "h3: FirstBlockCache summary calls=%u fresh=%u reuse=%u "
                "reuse-rate=%.1f%% hit-cap-forced=%u decision-wait=%.3fs\n",
                dit->first_block_cache_calls,
                dit->first_block_cache_fresh,
                dit->first_block_cache_reuse,
                dit->first_block_cache_calls ?
                    100.0 * (double)dit->first_block_cache_reuse /
                        dit->first_block_cache_calls : 0.0,
                dit->first_block_cache_hit_cap_forced,
                dit->first_block_cache_wait_seconds);
    }
    if (dit->tea_cache && h3_runtime_getenv("H3_PROFILE")) {
        fprintf(stderr,
                "h3: TeaCache summary calls=%u fresh=%u reuse=%u "
                "reuse-rate=%.1f%% hit-cap-forced=%u decision-wait=%.3fs\n",
                dit->tea_cache_calls, dit->tea_cache_fresh,
                dit->tea_cache_reuse,
                dit->tea_cache_calls ?
                    100.0 * (double)dit->tea_cache_reuse /
                        dit->tea_cache_calls : 0.0,
                dit->tea_cache_hit_cap_forced,
                dit->tea_cache_wait_seconds);
    }
    if (dit->sol_attention && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: standalone Sol-Attn summary sparse-calls=%u "
                "dense-calls=%u\n",
                dit->sol_calls, dit->sol_dense_calls);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        const h3_dit_block *value = &dit->blocks[block];
        if (value->coreml_mlp &&
            (h3_runtime_getenv("H3_PROFILE") || value->coreml_fallbacks))
            fprintf(stderr,
                    "h3: Core ML MLP telemetry block=%u predictions=%u "
                    "fallbacks=%u forced=%u latched=%u prediction-failures=%u "
                    "nonfinite-values=%zu latch-active=%d lazy-maps=%u "
                    "lazy-map-sha-first=%.6fs lazy-map-sha-total=%.6fs\n",
                    block, value->coreml_predictions,
                    value->coreml_fallbacks,
                    value->coreml_forced_fallbacks,
                    value->coreml_latched_fallbacks,
                    value->coreml_prediction_failures,
                    value->coreml_nonfinite_values,
                    value->coreml_fallback_latched,
                    value->coreml_fallback_lazy_maps,
                    value->coreml_fallback_first_map_seconds,
                    value->coreml_fallback_map_seconds);
        if (value->coreml_qkv && h3_runtime_getenv("H3_PROFILE"))
            fprintf(stderr,
                    "h3: Core ML QKV telemetry block=%u predictions=%u "
                    "prediction-failures=%u ANE-heads=%u\n",
                    block, value->coreml_qkv_predictions,
                    value->coreml_qkv_prediction_failures,
                    value->coreml_qkv_heads);
        if (value->ane_qkv &&
            (h3_runtime_getenv("H3_PROFILE") || value->ane_qkv_prediction_failures ||
             value->ane_qkv_nonfinite_failures)) {
            double calls = value->ane_qkv_predictions ?
                (double)value->ane_qkv_predictions : 1.0;
            fprintf(stderr,
                    "h3: private ANE QKV telemetry block=%u precision=%s "
                    "GPU-complement=%s calls=%u failures=%u nonfinite=%u "
                    "ANE-heads=%u "
                    "pack-mean=%.6fs overlap-mean=%.6fs "
                    "join-mean=%.6fs unload-mean=%.6fs\n",
                    block,
                    value->ane_qkv_int8_weights ? "int8" : "fp16",
                    value->ane_qkv_gpu_int8_weights ? "int8" : "bf16",
                    value->ane_qkv_predictions,
                    value->ane_qkv_prediction_failures,
                    value->ane_qkv_nonfinite_failures,
                    value->ane_qkv_heads,
                    value->ane_qkv_pack_seconds / calls,
                    value->ane_qkv_overlap_seconds / calls,
                    value->ane_qkv_join_seconds / calls,
                    value->ane_qkv_unload_seconds / calls);
        }
        if (block_has_private_ane_attention_out(value) &&
            (h3_runtime_getenv("H3_PROFILE") ||
             value->ane_attention_out_prediction_failures ||
             value->ane_attention_out_nonfinite_failures)) {
            double calls = value->ane_attention_out_predictions ?
                (double)value->ane_attention_out_predictions : 1.0;
            fprintf(stderr,
                    "h3: private ANE attention-output telemetry block=%u "
                    "precision=%s calls=%u failures=%u nonfinite=%u "
                    "ANE-width=%u pack-mean=%.6fs overlap-mean=%.6fs "
                    "join-mean=%.6fs unload-mean=%.6fs\n",
                    block,
                    value->ane_attention_out_int8_weights ? "int8" : "fp16",
                    value->ane_attention_out_predictions,
                    value->ane_attention_out_prediction_failures,
                    value->ane_attention_out_nonfinite_failures,
                    value->ane_attention_out_width,
                    value->ane_attention_out_pack_seconds / calls,
                    value->ane_attention_out_overlap_seconds / calls,
                    value->ane_attention_out_join_seconds / calls,
                    value->ane_attention_out_unload_seconds / calls);
        }
        if (value->ane_mlp &&
            (h3_runtime_getenv("H3_PROFILE") || value->ane_prediction_failures ||
             value->ane_nonfinite_failures)) {
            double calls = value->ane_predictions ?
                (double)value->ane_predictions : 1.0;
            fprintf(stderr,
                    "h3: private ANE MLP telemetry block=%u precision=%s "
                    "calls=%u "
                    "failures=%u nonfinite=%u range-retries=%u "
                    "headroom-retries=%u range-recoveries=%u "
                    "raw-peak-max=%.9g headroom=%.9g runtime-scale=%.9g "
                    "range-retry-total=%.6fs pack-mean=%.6fs "
                    "overlap-mean=%.6fs join-mean=%.6fs "
                    "unload-mean=%.6fs\n",
                    block, value->ane_int8_weights ? "int8" : "fp16",
                    value->ane_predictions,
                    value->ane_prediction_failures,
                    value->ane_nonfinite_failures,
                    value->ane_range_retries,
                    value->ane_range_headroom_retries,
                    value->ane_range_recoveries,
                    value->ane_range_peak_max,
                    dit->ane_mlp_range_headroom,
                    h3_ane_mlp_runtime_scale(value->ane_mlp),
                    value->ane_range_retry_seconds,
                    value->ane_pack_seconds / calls,
                    value->ane_overlap_seconds / calls,
                    value->ane_join_seconds / calls,
                    value->ane_unload_seconds / calls);
        }
    }
    if (dit->ssd_streaming && h3_runtime_getenv("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: %s SSD stream %.3f GiB read in %.3fs (%.3f GiB/s), "
                "unhidden wait %.3fs\n",
                dit->ssd_quantized ? "INT8/F32" : "BF16",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds : 0.0,
                dit->stream_wait_seconds);
    }
    if (dit->final_eviction_groups && h3_runtime_getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: final-pass DiT eviction summary groups=%u blocks=%u "
                "released=%.3f GiB\n",
                dit->final_eviction_groups, dit->final_evicted_blocks,
                (double)dit->final_evicted_bytes /
                    (1024.0 * 1024.0 * 1024.0));
    free_request_state(dit);
    free_tensor(&dit->video_patch_w); free_tensor(&dit->video_patch_b);
    free_tensor(&dit->audio_patch_w); free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        free_block(&dit->blocks[block]);
    free_tensor(&dit->coreml_input);
    free_tensor(&dit->coreml_output);
    free_tensor(&dit->coreml_qkv_input);
    free_tensor(&dit->coreml_qkv_output);
    free_tensor(&dit->coreml_qkv_gpu_output);
    free_tensor(&dit->ane_qkv_micro_partials);
    free_tensor(&dit->ane_attention_out_gpu_output);
    free_tensor(&dit->ane_attention_out_micro_reference);
    free_tensor(&dit->ane_attention_out_micro_partials);
    free_tensor(&dit->coreml_fallback_output);
    free_tensor(&dit->coreml_nonfinite);
    free_tensor(&dit->ane_nonfinite);
    h3_ane_mlp_io_free(dit->ane_mlp_io);
    dit->ane_mlp_io = NULL;
    h3_ane_linear_io_free(dit->ane_qkv_io);
    dit->ane_qkv_io = NULL;
    h3_ane_linear_io_free(dit->ane_attention_out_io);
    dit->ane_attention_out_io = NULL;
    free_block(&dit->stream_slots[0]);
    free_block(&dit->stream_slots[1]);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->final_video_w); free_tensor(&dit->final_video_b);
    free_tensor(&dit->final_audio_w); free_tensor(&dit->final_audio_b);
    h3_dit_schedule_free(dit->schedule);
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_quant_cache_close(&dit->quant_cache);
    free(dit->weight_directory);
    free(dit);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}
