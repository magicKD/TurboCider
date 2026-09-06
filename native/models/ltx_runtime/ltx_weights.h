#ifndef LTX_WEIGHTS_H
#define LTX_WEIGHTS_H

#include "ltx_safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char path[4096];
    char model_class[96];
    uint32_t num_layers;
    uint32_t video_dim;
    uint32_t audio_dim;
    uint32_t video_heads;
    uint32_t audio_heads;
    uint32_t video_head_dim;
    uint32_t audio_head_dim;
    uint32_t video_channels;
    uint32_t audio_channels;
    uint32_t video_ff_dim;
    uint32_t audio_ff_dim;
    uint32_t transformer_blocks_found;
    size_t tensor_count;
    size_t dtype_counts[LTX_DTYPE_F64 + 1];
    int ff_bias;
    int audio_ff_bias;
    int split_rope;
    int double_precision_rope;
    int keyframe_absolute_positions;
    int quantized_int8;
    int comfy_quantized;
    int appears_distilled;
} ltx_checkpoint_info;

typedef struct {
    const ltx_st_tensor *weight;
    const ltx_st_tensor *weight_scale;
    const ltx_st_tensor *bias;
    const ltx_st_tensor *quant_metadata;
    uint32_t input_dim;
    uint32_t output_dim;
    int quantized_int8;
    int convrot;
    uint32_t convrot_group_size;
} ltx_linear_weight_info;

typedef struct {
    char path[4096];
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_layers;
    uint32_t attention_heads;
    uint32_t sliding_kv_heads;
    uint32_t full_kv_heads;
    uint32_t sliding_head_dim;
    uint32_t full_head_dim;
    uint32_t projection_video_dim;
    uint32_t projection_audio_dim;
    uint32_t projection_input_dim;
    int attention_k_eq_v;
    int quantized_int8;
    int all_convrot_group_256;
    size_t tensor_count;
} ltx_gemma_checkpoint_info;

int ltx_checkpoint_inspect(const char *path, ltx_checkpoint_info *info,
                           char *error, size_t error_size);
int ltx_checkpoint_validate_transformer(const ltx_checkpoint_info *info,
                                        char *error, size_t error_size);
int ltx_linear_weight_resolve(const ltx_st_header *header,
                              const ltx_st_mapping *mapping,
                              const char *prefix,
                              ltx_linear_weight_info *info,
                              char *error, size_t error_size);
int ltx_gemma_checkpoint_inspect(const char *path,
                                 ltx_gemma_checkpoint_info *info,
                                 char *error, size_t error_size);
int ltx_gemma_checkpoint_validate(const ltx_gemma_checkpoint_info *info,
                                  char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
