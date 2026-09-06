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
} ltx_gemma_encoder_options;

ltx_gemma_encoder *ltx_gemma_encoder_create(
    const ltx_gemma_encoder_options *options,
    char *error, size_t error_size);
void ltx_gemma_encoder_free(ltx_gemma_encoder *encoder);

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

#ifdef __cplusplus
}
#endif

#endif
