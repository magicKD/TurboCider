#ifndef LTX_CONNECTOR_H
#define LTX_CONNECTOR_H

#include "ltx_gpu.h"
#include "ltx_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_connector ltx_connector;

ltx_connector *ltx_connector_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size);
void ltx_connector_free(ltx_connector *connector);

uint32_t ltx_connector_video_dim(const ltx_connector *connector);
uint32_t ltx_connector_audio_dim(const ltx_connector *connector);
uint32_t ltx_connector_register_rows(const ltx_connector *connector);
/*
 * Connector outputs are padded with tiled learnable registers to the next
 * register boundary, with a minimum sequence length of 1024 tokens.
 */
uint32_t ltx_connector_output_rows(const ltx_connector *connector,
                                   uint32_t input_rows);

int ltx_connector_run_bf16(
    ltx_connector *connector,
    ltx_gpu_buffer *video_output, ltx_gpu_buffer *audio_output,
    const ltx_gpu_buffer *video_input,
    const ltx_gpu_buffer *audio_input,
    uint32_t input_rows,
    char *error, size_t error_size);

#endif
