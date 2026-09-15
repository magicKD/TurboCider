#ifndef LTX_GEMMA_ENCODER_H
#define LTX_GEMMA_ENCODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_gemma_encoder ltx_gemma_encoder;
typedef int (*ltx_gemma_progress)(const char *phase, int current, int total,
                                  void *opaque);

typedef struct {
    const char *checkpoint;
    const char *tokenizer_json;
    const char *shader_source;
    uint32_t max_tokens;
    /* Optional directory (or one layer manifest) containing compiled
     * ltx-gemma-ane-mlp-v1 artifacts.  A bounded procedure cache is retained
     * across warm encodes, with exact per-layer GPU fallback on failure. */
    const char *ane_manifest;
} ltx_gemma_encoder_options;

typedef struct {
    uint32_t rows;
    uint32_t ane_layers_available;
    uint32_t ane_layers_attempted;
    uint32_t ane_layers_succeeded;
    uint32_t ane_layers_fallback;
    uint32_t ane_cache_hits;
    uint32_t ane_cache_misses;
    uint32_t ane_preload_models_session_total;
    uint32_t ane_preload_workers;
    uint32_t ane_selected_bucket;
    uint32_t ane_padding_rows;
    uint32_t ane_minimum_profitable_rows;
    uint64_t ane_execution_rows;
    uint64_t resident_weight_bytes;
    uint32_t resident_weight_cache_hits;
    uint32_t resident_weight_cache_misses;
    double ane_total_seconds;
    double ane_preload_seconds_session_total;
    int ane_requested;
    int ane_used;
    int ane_output_backing_used;
    int resident_weights_enabled;
    char ane_plan_reason[48];
} ltx_gemma_encoder_telemetry;

#define LTX_GEMMA_MLP_PROBE_MAX_RUNS 50u

/* Diagnostic-only isolated GPU baseline for one resident Gemma gated MLP.
 * The caller supplies deterministic BF16 input and receives the final BF16
 * down-projection output after one first pass plus warm_runs measured passes. */
typedef struct {
    uint32_t layer;
    uint32_t rows;
    uint32_t warm_runs;
    uint32_t hidden;
    uint32_t intermediate;
    double checkpoint_validation_seconds;
    double checkpoint_open_seconds;
    double gpu_setup_seconds;
    double weight_load_seconds;
    double workspace_setup_seconds;
    double first_seconds;
    double warm_seconds[LTX_GEMMA_MLP_PROBE_MAX_RUNS];
    uint64_t resident_weight_bytes;
    uint64_t workspace_bytes;
} ltx_gemma_mlp_probe_result;

ltx_gemma_encoder *ltx_gemma_encoder_create(
    const ltx_gemma_encoder_options *options,
    char *error, size_t error_size);
void ltx_gemma_encoder_free(ltx_gemma_encoder *encoder);
/* Eagerly prepare the fixed-shape ANE procedures needed by a prompt.  A
 * failed or unavailable procedure is deliberately non-fatal: encode continues
 * with its exact GPU fallback. */
int ltx_gemma_encoder_prepare_prompt(
    ltx_gemma_encoder *encoder, const char *prompt,
    char *error, size_t error_size);
int ltx_gemma_encoder_get_telemetry(
    const ltx_gemma_encoder *encoder,
    ltx_gemma_encoder_telemetry *telemetry);

/* Encode a text prompt into the raw Gemma video/audio projection streams
 * consumed by the LTX connector. Outputs contain exactly output_rows valid
 * rows (no padding); mask uses LTX additive-BF16 semantics (valid rows 0). */
int ltx_gemma_encoder_encode(
    ltx_gemma_encoder *encoder, const char *prompt,
    uint16_t *video_output, size_t video_output_elements,
    uint16_t *audio_output, size_t audio_output_elements,
    uint16_t *mask_output, size_t mask_output_elements,
    uint32_t *output_rows, ltx_gemma_progress progress, void *opaque,
    char *error, size_t error_size);

int ltx_gemma_mlp_gpu_probe(
    const char *checkpoint, const char *shader_source,
    uint32_t layer, uint32_t rows, uint32_t warm_runs,
    const uint16_t *input, size_t input_elements,
    uint16_t *output, size_t output_elements,
    ltx_gemma_mlp_probe_result *result,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
