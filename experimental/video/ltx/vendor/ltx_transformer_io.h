#ifndef LTX_TRANSFORMER_IO_H
#define LTX_TRANSFORMER_IO_H

#include "ltx_gpu.h"
#include "ltx_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_transformer_io ltx_transformer_io;

ltx_transformer_io *ltx_transformer_io_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size);
void ltx_transformer_io_free(ltx_transformer_io *io);

uint32_t ltx_transformer_io_video_patch_dim(const ltx_transformer_io *io);
uint32_t ltx_transformer_io_audio_patch_dim(const ltx_transformer_io *io);
uint32_t ltx_transformer_io_video_hidden_dim(const ltx_transformer_io *io);
uint32_t ltx_transformer_io_audio_hidden_dim(const ltx_transformer_io *io);
size_t ltx_transformer_io_weight_bytes(const ltx_transformer_io *io);

int ltx_transformer_io_patchify_video(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, uint32_t rows,
    char *error, size_t error_size);
int ltx_transformer_io_patchify_audio(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, uint32_t rows,
    char *error, size_t error_size);

int ltx_transformer_io_output_video(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *embedded_timestep,
    uint32_t rows, char *error, size_t error_size);
int ltx_transformer_io_output_video_split(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *generated_embedded_timestep,
    const ltx_gpu_buffer *conditioned_embedded_timestep,
    uint32_t rows, uint32_t conditioned_prefix_rows,
    char *error, size_t error_size);
int ltx_transformer_io_output_audio(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *embedded_timestep,
    uint32_t rows, char *error, size_t error_size);

#endif
