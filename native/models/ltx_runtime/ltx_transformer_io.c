#include "ltx_transformer_io.h"

#include "ltx_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
    size_t bytes;
} ltx_io_linear;

typedef struct {
    ltx_io_linear patchify;
    ltx_io_linear output;
    ltx_gpu_buffer *shift_table;
    ltx_gpu_buffer *scale_table;
    uint32_t patch_dim;
    uint32_t hidden_dim;
} ltx_io_modality;

struct ltx_transformer_io {
    ltx_gpu *gpu;
    ltx_io_modality video;
    ltx_io_modality audio;
    size_t weight_bytes;
};

static int ltx_io_fail(char *error, size_t error_size,
                       const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static int ltx_io_name(char *name, size_t name_size,
                       const char *prefix, const char *suffix,
                       char *error, size_t error_size) {
    int length = snprintf(name, name_size, "%s.%s", prefix, suffix);
    if (length < 0 || (size_t)length >= name_size)
        return ltx_io_fail(error, error_size,
                           "Transformer I/O tensor name is too long");
    return 1;
}

static uint16_t ltx_io_f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float ltx_io_f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (!exponent) {
        if (!mantissa) {
            bits = sign;
        } else {
            int32_t shift = 0;
            while (!(mantissa & 0x0400u)) {
                mantissa <<= 1u;
                shift++;
            }
            mantissa &= 0x03ffu;
            uint32_t f32_exponent = (uint32_t)(113 - shift);
            bits = sign | (f32_exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int ltx_io_convert_bf16(const void *source, ltx_dtype dtype,
                               uint64_t elements, uint16_t **converted,
                               char *error, size_t error_size) {
    if (!source || !converted || !elements ||
        elements > SIZE_MAX / sizeof(uint16_t))
        return ltx_io_fail(error, error_size,
                           "invalid Transformer I/O tensor conversion");
    uint16_t *values = malloc((size_t)elements * sizeof(*values));
    if (!values)
        return ltx_io_fail(error, error_size,
                           "out of memory converting Transformer I/O tensor");
    if (dtype == LTX_DTYPE_BF16) {
        memcpy(values, source, (size_t)elements * sizeof(*values));
    } else if (dtype == LTX_DTYPE_F16) {
        const uint16_t *input = source;
        for (uint64_t index = 0; index < elements; index++)
            values[index] = ltx_io_f32_to_bf16(
                ltx_io_f16_to_f32(input[index]));
    } else if (dtype == LTX_DTYPE_F32) {
        const float *input = source;
        for (uint64_t index = 0; index < elements; index++)
            values[index] = ltx_io_f32_to_bf16(input[index]);
    } else {
        free(values);
        return ltx_io_fail(error, error_size,
                           "unsupported Transformer I/O tensor dtype");
    }
    *converted = values;
    return 1;
}

static void ltx_io_linear_free(ltx_io_linear *linear) {
    if (!linear) return;
    ltx_gpu_buffer_free(linear->weight);
    ltx_gpu_buffer_free(linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int ltx_io_linear_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix, ltx_io_linear *linear,
        char *error, size_t error_size) {
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(
            header, mapping, prefix, &info, error, error_size)) return 0;
    if (info.quantized_int8 || !info.bias)
        return ltx_io_fail(error, error_size,
                           "Transformer I/O linear must be dense with bias");
    size_t weight_source_bytes = 0;
    size_t bias_source_bytes = 0;
    const void *weight_source = ltx_st_map_tensor(
        mapping, info.weight, &weight_source_bytes, error, error_size);
    const void *bias_source = ltx_st_map_tensor(
        mapping, info.bias, &bias_source_bytes, error, error_size);
    if (!weight_source || !bias_source) return 0;
    uint64_t weight_elements = (uint64_t)info.input_dim * info.output_dim;
    uint16_t *weight = NULL;
    uint16_t *bias = NULL;
    if (!ltx_io_convert_bf16(
            weight_source, info.weight->dtype, weight_elements,
            &weight, error, error_size) ||
        !ltx_io_convert_bf16(
            bias_source, info.bias->dtype, info.output_dim,
            &bias, error, error_size)) {
        free(weight);
        free(bias);
        return 0;
    }
    size_t weight_bytes = (size_t)weight_elements * sizeof(*weight);
    size_t bias_bytes = (size_t)info.output_dim * sizeof(*bias);
    linear->weight = ltx_gpu_buffer_new_copy(
        gpu, weight, weight_bytes, error, error_size);
    linear->bias = ltx_gpu_buffer_new_copy(
        gpu, bias, bias_bytes, error, error_size);
    free(weight);
    free(bias);
    if (!linear->weight || !linear->bias) {
        ltx_io_linear_free(linear);
        return 0;
    }
    linear->input_dim = info.input_dim;
    linear->output_dim = info.output_dim;
    linear->bytes = weight_bytes + bias_bytes;
    return 1;
}

static void ltx_io_modality_free(ltx_io_modality *modality) {
    if (!modality) return;
    ltx_gpu_buffer_free(modality->shift_table);
    ltx_gpu_buffer_free(modality->scale_table);
    ltx_io_linear_free(&modality->output);
    ltx_io_linear_free(&modality->patchify);
    memset(modality, 0, sizeof(*modality));
}

static int ltx_io_table_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *name, uint32_t hidden_dim,
        ltx_gpu_buffer **shift_table, ltx_gpu_buffer **scale_table,
        size_t *bytes, char *error, size_t error_size) {
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor || tensor->ndim != 2u || tensor->shape[0] != 2u ||
        tensor->shape[1] != hidden_dim)
        return ltx_io_fail(error, error_size,
                           "invalid Transformer output scale/shift table");
    size_t source_bytes = 0;
    const void *source = ltx_st_map_tensor(
        mapping, tensor, &source_bytes, error, error_size);
    if (!source) return 0;
    uint16_t *values = NULL;
    if (!ltx_io_convert_bf16(
            source, tensor->dtype, (uint64_t)hidden_dim * 2u,
            &values, error, error_size)) return 0;
    size_t row_bytes = (size_t)hidden_dim * sizeof(*values);
    *shift_table = ltx_gpu_buffer_new_copy(
        gpu, values, row_bytes, error, error_size);
    *scale_table = ltx_gpu_buffer_new_copy(
        gpu, values + hidden_dim, row_bytes, error, error_size);
    free(values);
    if (!*shift_table || !*scale_table) {
        ltx_gpu_buffer_free(*shift_table);
        ltx_gpu_buffer_free(*scale_table);
        *shift_table = NULL;
        *scale_table = NULL;
        return 0;
    }
    *bytes = row_bytes * 2u;
    return 1;
}

static int ltx_io_modality_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix,
        const char *patchify_suffix, const char *output_suffix,
        const char *table_suffix, ltx_io_modality *modality,
        char *error, size_t error_size) {
    char name[4096];
    size_t table_bytes = 0;
    if (!ltx_io_name(name, sizeof(name), prefix, patchify_suffix,
                     error, error_size) ||
        !ltx_io_linear_load(
            header, mapping, gpu, name, &modality->patchify,
            error, error_size) ||
        !ltx_io_name(name, sizeof(name), prefix, output_suffix,
                     error, error_size) ||
        !ltx_io_linear_load(
            header, mapping, gpu, name, &modality->output,
            error, error_size)) {
        ltx_io_modality_free(modality);
        return 0;
    }
    modality->patch_dim = modality->patchify.input_dim;
    modality->hidden_dim = modality->patchify.output_dim;
    if (modality->output.input_dim != modality->hidden_dim ||
        modality->output.output_dim != modality->patch_dim ||
        !ltx_io_name(name, sizeof(name), prefix, table_suffix,
                     error, error_size) ||
        !ltx_io_table_load(
            header, mapping, gpu, name, modality->hidden_dim,
            &modality->shift_table, &modality->scale_table,
            &table_bytes, error, error_size)) {
        if (!error || !error_size || !error[0])
            ltx_io_fail(error, error_size,
                        "incompatible Transformer I/O geometry");
        ltx_io_modality_free(modality);
        return 0;
    }
    return 1;
}

ltx_transformer_io *ltx_transformer_io_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size) {
    if (!header || !mapping || !gpu || !prefix) {
        ltx_io_fail(error, error_size,
                    "missing Transformer I/O load argument");
        return NULL;
    }
    ltx_transformer_io *io = calloc(1, sizeof(*io));
    if (!io) {
        ltx_io_fail(error, error_size,
                    "out of memory loading Transformer I/O");
        return NULL;
    }
    io->gpu = gpu;
    if (!ltx_io_modality_load(
            header, mapping, gpu, prefix,
            "patchify_proj", "proj_out", "scale_shift_table",
            &io->video, error, error_size) ||
        !ltx_io_modality_load(
            header, mapping, gpu, prefix,
            "audio_patchify_proj", "audio_proj_out",
            "audio_scale_shift_table",
            &io->audio, error, error_size)) {
        ltx_transformer_io_free(io);
        return NULL;
    }
    io->weight_bytes = io->video.patchify.bytes + io->video.output.bytes +
        io->audio.patchify.bytes + io->audio.output.bytes +
        ((size_t)io->video.hidden_dim + io->audio.hidden_dim) *
            2u * sizeof(uint16_t);
    return io;
}

void ltx_transformer_io_free(ltx_transformer_io *io) {
    if (!io) return;
    ltx_io_modality_free(&io->audio);
    ltx_io_modality_free(&io->video);
    free(io);
}

uint32_t ltx_transformer_io_video_patch_dim(const ltx_transformer_io *io) {
    return io ? io->video.patch_dim : 0u;
}

uint32_t ltx_transformer_io_audio_patch_dim(const ltx_transformer_io *io) {
    return io ? io->audio.patch_dim : 0u;
}

uint32_t ltx_transformer_io_video_hidden_dim(const ltx_transformer_io *io) {
    return io ? io->video.hidden_dim : 0u;
}

uint32_t ltx_transformer_io_audio_hidden_dim(const ltx_transformer_io *io) {
    return io ? io->audio.hidden_dim : 0u;
}

size_t ltx_transformer_io_weight_bytes(const ltx_transformer_io *io) {
    return io ? io->weight_bytes : 0u;
}

static int ltx_io_patchify(
        ltx_transformer_io *io, const ltx_io_modality *modality,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, char *error, size_t error_size) {
    if (!io || !modality)
        return ltx_io_fail(error, error_size,
                           "missing Transformer patchify runtime");
    return ltx_gpu_linear_bf16(
        io->gpu, output, input,
        modality->patchify.weight, modality->patchify.bias,
        rows, modality->patch_dim, modality->hidden_dim,
        error, error_size);
}

int ltx_transformer_io_patchify_video(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, uint32_t rows,
    char *error, size_t error_size) {
    return ltx_io_patchify(
        io, io ? &io->video : NULL, output, input, rows,
        error, error_size);
}

int ltx_transformer_io_patchify_audio(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    const ltx_gpu_buffer *input, uint32_t rows,
    char *error, size_t error_size) {
    return ltx_io_patchify(
        io, io ? &io->audio : NULL, output, input, rows,
        error, error_size);
}

static int ltx_io_output(
        ltx_transformer_io *io, const ltx_io_modality *modality,
        ltx_gpu_buffer *output, ltx_gpu_buffer *workspace,
        const ltx_gpu_buffer *hidden,
        const ltx_gpu_buffer *embedded_timestep,
        uint32_t rows, char *error, size_t error_size) {
    if (!io || !modality)
        return ltx_io_fail(error, error_size,
                           "missing Transformer output runtime");
    return ltx_gpu_output_adaln_bf16(
            io->gpu, workspace, hidden, embedded_timestep,
            modality->shift_table, modality->scale_table,
            rows, modality->hidden_dim, 1e-6f, error, error_size) &&
        ltx_gpu_linear_bf16(
            io->gpu, output, workspace,
            modality->output.weight, modality->output.bias,
            rows, modality->hidden_dim, modality->patch_dim,
            error, error_size);
}

int ltx_transformer_io_output_video(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *embedded_timestep,
    uint32_t rows, char *error, size_t error_size) {
    return ltx_io_output(
        io, io ? &io->video : NULL, output, workspace,
        hidden, embedded_timestep, rows, error, error_size);
}

int ltx_transformer_io_output_video_split(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *generated_embedded_timestep,
    const ltx_gpu_buffer *conditioned_embedded_timestep,
    uint32_t rows, uint32_t conditioned_prefix_rows,
    char *error, size_t error_size) {
    const ltx_io_modality *modality = io ? &io->video : NULL;
    if (!io || !modality || !conditioned_prefix_rows ||
        conditioned_prefix_rows > rows)
        return ltx_io_fail(error, error_size,
                           "invalid split Transformer video output");
    return ltx_gpu_output_adaln_bf16_split(
            io->gpu, workspace, hidden,
            generated_embedded_timestep, conditioned_embedded_timestep,
            modality->shift_table, modality->scale_table,
            rows, modality->hidden_dim, conditioned_prefix_rows,
            1e-6f, error, error_size) &&
        ltx_gpu_linear_bf16(
            io->gpu, output, workspace,
            modality->output.weight, modality->output.bias,
            rows, modality->hidden_dim, modality->patch_dim,
            error, error_size);
}

int ltx_transformer_io_output_audio(
    ltx_transformer_io *io, ltx_gpu_buffer *output,
    ltx_gpu_buffer *workspace,
    const ltx_gpu_buffer *hidden,
    const ltx_gpu_buffer *embedded_timestep,
    uint32_t rows, char *error, size_t error_size) {
    return ltx_io_output(
        io, io ? &io->audio : NULL, output, workspace,
        hidden, embedded_timestep, rows, error, error_size);
}
