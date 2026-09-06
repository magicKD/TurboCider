#ifndef LTX_GEMMA_TOKENIZER_H
#define LTX_GEMMA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltx_gemma_tokenizer ltx_gemma_tokenizer;

/* Load the tokenizers.json BPE specification used by Gemma4. */
ltx_gemma_tokenizer *ltx_gemma_tokenizer_load(
    const char *tokenizer_json, char *error, size_t error_size);
void ltx_gemma_tokenizer_free(ltx_gemma_tokenizer *tokenizer);

/* max_length=0 returns the unpadded sequence. Otherwise output is exactly
 * max_length entries: overflow is truncated from the left and shorter input
 * is left-padded with the tokenizer's native pad token. */
int ltx_gemma_tokenizer_encode(
    const ltx_gemma_tokenizer *tokenizer, const char *utf8,
    uint32_t max_length, uint32_t **ids, uint8_t **mask, size_t *count,
    char *error, size_t error_size);
void ltx_gemma_tokenizer_ids_free(uint32_t *ids, uint8_t *mask);

#ifdef __cplusplus
}
#endif

#endif
