#ifndef H3_SUPER_H
#define H3_SUPER_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

#define H3_SUPER_PATH_MAX 4096

typedef enum {
    H3_SUPER_PROFILE_480P = 0,
    H3_SUPER_PROFILE_480P_FAST = 1,
    H3_SUPER_PROFILE_480P_QUALITY = 2,
    H3_SUPER_PROFILE_480P_MOTION = 3,
    H3_SUPER_PROFILE_V2 = 4
} h3_super_profile;

typedef enum {
    H3_SUPER_DECODER_TAEHV = 0,
    H3_SUPER_DECODER_OFFICIAL = 1
} h3_super_decoder;

typedef enum {
    H3_SUPER_STAGE2_SCHEDULE_DEFAULT = 0,
    H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9 = 1,
    H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42 = 2
} h3_super_stage2_schedule;

typedef struct {
    const char *endpoint;
    const char *input_dir;
    const char *output_dir;
    char input_dir_storage[H3_SUPER_PATH_MAX];
    char output_dir_storage[H3_SUPER_PATH_MAX];
    const char *transformer;
    const char *refiner_lora;
    double refiner_lora_strength;
    const char *text_encoder;
    const char *video_vae;
    const char *audio_vae;
    const char *latent_upscaler;
    const char *taehv_checkpoint;
    const char *taehv_compute_dtype;
    const char *taehv_temporal_mode;
    const char *negative_prompt;
    const char *first_frame;
    const char *stage1_model_dir;
    const char *telemetry_path;
    h3_super_profile profile;
    h3_super_decoder decoder;
    h3_super_stage2_schedule stage2_schedule;
    int stage1_width;
    int stage1_height;
    int stage1_frames;
    int stage1_steps;
    int input_width;
    int input_height;
    int refined_width;
    int refined_height;
    int output_width;
    int output_height;
    int output_frames;
    int output_fps;
    double timeout_seconds;
    double poll_seconds;
    int taehv_keep_on_device;
    int prefetch_transformer;
    int prefetch_conditioning;
    int prefetch_input_models;
    int prefetch_start_step;
    int prefetch_start_block;
    int final_evict_blocks;
    int sol_attention;
    int fuse_denoise_decode;
    int direct_handoff;
    int prepared_handoff;
    int keep_stage1;
    int allow_swap;
    int face_quality_gate;
    int person_quality_gate;
    int output_quality_gate;
    int strict_person_quality;
    int quality_attempts;
    double min_face_area;
} h3_super_params;

typedef struct {
    char prompt_id[128];
    char backend_output[H3_SUPER_PATH_MAX];
    double preflight_seconds;
    double queue_seconds;
    double stage2_seconds;
    double conditioning_seconds;
    double input_prepare_seconds;
    double denoise_seconds;
    double decode_seconds;
    double denoise_decode_seconds;
    double output_copy_seconds;
    double output_quality_seconds;
    double backend_release_seconds;
    double transformer_prefetch_seconds;
    double transformer_prefetch_wait_seconds;
    double conditioning_prefetch_seconds;
    double conditioning_prefetch_wait_seconds;
    double input_model_prefetch_seconds;
    double handoff_seconds;
    double first_frame_import_seconds;
    uint64_t handoff_bytes;
    uint64_t first_frame_bytes;
    uint64_t output_bytes;
    uint64_t backend_ram_total;
    uint64_t backend_ram_free_before;
    uint64_t backend_ram_free_min;
    uint64_t backend_ram_free_after_release;
    uint64_t system_swapouts_before;
    uint64_t system_swapouts_after;
    int swapouts_available;
    int no_swap_gate_passed;
    int transformer_prefetch_completed;
    int conditioning_prefetch_completed;
    int input_model_prefetch_completed;
    int prefetch_started_step;
    int prefetch_total_steps;
    int prefetch_started_block;
    int prefetch_total_blocks;
    int original_first_frame_conditioning;
    int face_quality_gate_applied;
    int face_quality_gate_passed;
    int face_quality_sampled_frames;
    int face_quality_detected_frames;
    int face_quality_duplicate_frames;
    double face_quality_detection_rate;
    double face_quality_capture_mean;
    double face_quality_capture_min;
    double face_quality_area_mean;
    double face_quality_area_min;
    double face_quality_feature_mean;
    double face_quality_feature_max;
    double face_quality_landmark_mean;
    double face_quality_landmark_max;
    double face_quality_mouth_landmark_max;
    double face_quality_contour_landmark_max;
    double face_quality_center_jump_mean;
    double face_quality_center_jump_max;
    double face_quality_log_area_jump_mean;
    double face_quality_log_area_jump_max;
    int body_quality_gate_applied;
    int body_quality_gate_passed;
    int body_quality_sampled_frames;
    int body_quality_detected_frames;
    int body_quality_duplicate_frames;
    double body_quality_detection_rate;
    double body_quality_joint_coverage_mean;
    double body_quality_joint_coverage_min;
    double body_quality_core_coverage_min;
    double body_quality_complete_chains_mean;
    double body_quality_complete_chains_min;
    double body_quality_limb_asymmetry_max;
    uint64_t stage1_first_seed;
    uint64_t stage1_accepted_seed;
    int stage1_model_load_attempts;
    double stage1_model_load_seconds;
    int stage1_quality_attempts;
    int stage1_quality_rejections;
    double stage1_quality_rejected_seconds;
    double stage1_quality_accepted_seconds;
    char stage1_quality_last_rejection[512];
    int refine_quality_attempts;
    int refine_quality_rejections;
    double refine_quality_rejected_seconds;
    double refine_quality_accepted_seconds;
    double refine_quality_total_seconds;
    double output_quality_total_seconds;
    char refine_quality_last_rejection[512];
    int output_quality_gate_applied;
    int output_quality_gate_passed;
    int output_face_sampled_frames;
    int output_face_detected_frames;
    int output_face_duplicate_frames;
    double output_face_detection_rate;
    double output_face_capture_mean;
    double output_face_capture_min;
    double output_face_area_mean;
    double output_face_area_min;
    double output_face_feature_mean;
    double output_face_feature_max;
    double output_face_landmark_max;
    double output_face_mouth_landmark_max;
    double output_face_contour_landmark_max;
    double output_face_center_jump_mean;
    double output_face_center_jump_max;
    double output_face_log_area_jump_mean;
    double output_face_log_area_jump_max;
    int output_body_gate_applied;
    int output_body_gate_passed;
    int output_body_sampled_frames;
    int output_body_detected_frames;
    int output_body_duplicate_frames;
    double output_body_detection_rate;
    double output_body_joint_coverage_mean;
    double output_body_joint_coverage_min;
    double output_body_core_coverage_min;
    double output_body_complete_chains_mean;
    double output_body_complete_chains_min;
    double output_body_limb_asymmetry_max;
    char conditioning_prefetch_state[256];
    char first_frame_sha256[65];
    char first_frame_copy_sha256[65];
} h3_super_result;

void h3_super_params_init(h3_super_params *params);
void h3_super_params_set_profile(h3_super_params *params,
                                 h3_super_profile profile);
/* Lock the measured, fail-closed high-quality person recipe.  This uses the
 * v1.1 H3 Stage-1 artifact, the official INT8 LTX base with the BF16 LoRA
 * applied as a runtime bypass, and a sequential TAEHV decode. */
void h3_super_params_apply_person_quality_preset(h3_super_params *params);
/* Select the measured conditioning-prefetch Pareto point for the configured
 * Stage-2 schedule without overriding an explicit step/block deadline. */
void h3_super_params_apply_prefetch_defaults(
    h3_super_params *params,
    int start_step_given,
    int start_block_given);
const char *h3_super_profile_name(h3_super_profile profile);
const char *h3_super_decoder_name(h3_super_decoder decoder);
const char *h3_super_stage2_schedule_name(h3_super_stage2_schedule schedule);

/* Read and validate the Stage-1 merge manifest, then return the exact
 * flow-matching shifts required by that pinned LightX2V artifact. */
int h3_super_stage1_schedule(const h3_super_params *params,
                             double *video_shift,
                             double *audio_shift,
                             char *error, size_t error_size);

/* Require the quality preset's pinned LightX2V v1.1 768p merge contract. */
int h3_super_stage1_quality_contract(const h3_super_params *params,
                                     char *error, size_t error_size);

/* Validate the local ComfyUI service, required node classes, configured model
 * filenames, and shared input/output directories before the expensive H3
 * stage starts. */
int h3_super_preflight(const h3_super_params *params,
                       h3_super_result *result,
                       char *error, size_t error_size);

/* Reserve a unique path directly in the ComfyUI input directory. The caller
 * writes the H3 Stage-1 MP4 to this path, avoiding a second large media copy. */
int h3_super_make_stage1_path(const h3_super_params *params,
                              char *path, size_t path_size,
                              char *error, size_t error_size);

/* Load and pin the configured LTX Transformer in ComfyUI without releasing it.
 * The caller may run this function on a background thread during native H3
 * denoising, then enable params->prefetch_transformer for Stage 2 only when it
 * succeeds. */
int h3_super_prefetch_transformer(const h3_super_params *params,
                                  double *seconds,
                                  char *error, size_t error_size);

/* Encode LTX/Gemma conditioning and persist the compact CPU artifact while
 * native H3 is denoising. The function releases the Comfy model again before
 * returning so Stage 1 and Stage 2 do not remain jointly resident. */
int h3_super_prefetch_conditioning(const h3_super_params *params,
                                   const char *prompt,
                                   h3_super_result *result,
                                   double *seconds,
                                   char *error, size_t error_size);

/* Preload only the Video VAE, Audio VAE, and learned latent upscaler wrappers
 * on their CPU/offload device for the later Stage-2 input prompt. */
int h3_super_prefetch_input_models(const h3_super_params *params,
                                   h3_super_result *result,
                                   double *seconds,
                                   char *error, size_t error_size);

/* Copy an existing diagnostic Stage-1 MP4 or direct .h3sh artifact into the
 * reserved ComfyUI input path selected by params->direct_handoff. */
int h3_super_import_stage1(const char *source_path,
                           const char *stage1_path,
                           char *error, size_t error_size);

/* Serialize retained native TAEH3 frames and original FP32 H3 PCM into the
 * mmap-friendly local handoff. prepared_handoff selects the experimental
 * center-cropped, Stage-2-sized BF16 v2 payload; otherwise v1 preserves F32. */
int h3_super_write_handoff(const h3_super_params *params,
                           const h3_result *stage1,
                           const char *handoff_path,
                           uint64_t *bytes,
                           char *error, size_t error_size);

/* Sample retained native Stage-1 RGB frames with Apple Vision before paying
 * the LTX Stage-2 cost. The face gate verifies opening-identity continuity;
 * the person gate additionally verifies full-body pose completeness. */
int h3_super_check_stage1_face_quality(
    const h3_super_params *params,
    const h3_result *stage1,
    h3_super_result *result,
    char *error, size_t error_size);

/* Recheck the copied refined MP4 with a delivery-oriented face/body envelope.
 * This catches LTX amplification that was not visible at the Stage-1 gate. */
int h3_super_check_output_person_quality(
    const h3_super_params *params,
    const char *output_path,
    h3_super_result *result,
    char *error, size_t error_size);

/* Submit the selected fixed Super workflow, wait for completion, and
 * atomically copy the backend result to output_path. The 480p profile uses
 * four independently released Comfy phases and TAEHV wide by default; the
 * experimental fused path keeps the input boundary but combines denoise and
 * TAEHV decode into one completion-ordered prompt. */
int h3_super_refine_comfy(const h3_super_params *params,
                          const char *stage1_path,
                          const char *prompt,
                          uint64_t seed,
                          const char *output_path,
                          h3_super_result *result,
                          char *error, size_t error_size);

/* Write request telemetry. stage1_seconds and total_seconds are measured by
 * the CLI around the native H3 stage and the complete pipeline. */
int h3_super_write_telemetry(const h3_super_params *params,
                             const h3_super_result *result,
                             const char *stage1_path,
                             const char *output_path,
                             uint64_t seed,
                             double stage1_seconds,
                             double total_seconds,
                             char *error, size_t error_size);

#endif
