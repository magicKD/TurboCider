#ifndef LTX_CONDITIONING_H
#define LTX_CONDITIONING_H

#include "ltx_gpu.h"
#include "ltx_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_adaln_single ltx_adaln_single;
typedef struct ltx_transformer_conditioning ltx_transformer_conditioning;

typedef struct {
    uint32_t video_dim;
    uint32_t audio_dim;
    const uint16_t *video_adaln;
    const uint16_t *audio_adaln;
    const uint16_t *video_prompt;
    const uint16_t *audio_prompt;
    const uint16_t *av_video;
    const uint16_t *av_audio;
    const uint16_t *a2v_gate;
    const uint16_t *v2a_gate;
    ltx_gpu_buffer *video_embedded;
    ltx_gpu_buffer *audio_embedded;
} ltx_transformer_conditioning_values;

ltx_adaln_single *ltx_adaln_single_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size);
void ltx_adaln_single_free(ltx_adaln_single *adaln);

uint32_t ltx_adaln_single_timestep_dim(const ltx_adaln_single *adaln);
uint32_t ltx_adaln_single_hidden_dim(const ltx_adaln_single *adaln);
uint32_t ltx_adaln_single_parameter_count(const ltx_adaln_single *adaln);
size_t ltx_adaln_single_embedded_bytes(const ltx_adaln_single *adaln);
size_t ltx_adaln_single_parameter_bytes(const ltx_adaln_single *adaln);

int ltx_adaln_single_eval_scalar(
    ltx_adaln_single *adaln,
    ltx_gpu_buffer *parameters,
    ltx_gpu_buffer *embedded_timestep,
    float timestep,
    char *error, size_t error_size);

ltx_transformer_conditioning *ltx_transformer_conditioning_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size);
void ltx_transformer_conditioning_free(
    ltx_transformer_conditioning *conditioning);
int ltx_transformer_conditioning_eval_scalar(
    ltx_transformer_conditioning *conditioning, float sigma,
    ltx_transformer_conditioning_values *values,
    char *error, size_t error_size);

#endif
