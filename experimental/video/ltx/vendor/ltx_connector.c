#include "ltx_connector.h"

#include "ltx.h"
#include "ltx_weights.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LTX_CONNECTOR_BLOCKS = 8 };

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *scale;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
} ltx_connector_linear;

typedef struct {
    ltx_connector_linear query;
    ltx_connector_linear key;
    ltx_connector_linear value;
    ltx_connector_linear output;
    ltx_gpu_buffer *query_norm;
    ltx_gpu_buffer *key_norm;
    ltx_gpu_buffer *gate_weight;
    ltx_gpu_buffer *gate_bias;
    uint32_t heads;
    uint32_t head_dim;
} ltx_connector_attention;

typedef struct {
    ltx_connector_linear first;
    ltx_connector_linear second;
    uint32_t hidden_dim;
} ltx_connector_mlp;

typedef struct {
    ltx_connector_attention attention;
    ltx_connector_mlp mlp;
} ltx_connector_block;

typedef struct {
    uint32_t dim;
    uint32_t register_rows;
    uint16_t *register_values;
    ltx_connector_block blocks[LTX_CONNECTOR_BLOCKS];
} ltx_connector_modality;

struct ltx_connector {
    ltx_gpu *gpu;
    ltx_connector_modality video;
    ltx_connector_modality audio;
};

static int connector_fail(char *error, size_t error_size,
                          const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static int checked_bytes(uint64_t elements, size_t element_size,
                         size_t *bytes) {
    if (!bytes || (element_size && elements > SIZE_MAX / element_size))
        return 0;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int make_name(char *name, size_t name_size,
                     const char *prefix, const char *suffix,
                     char *error, size_t error_size) {
    int length = snprintf(name, name_size, "%s.%s", prefix, suffix);
    if (length < 0 || (size_t)length >= name_size)
        return connector_fail(error, error_size,
                              "connector tensor name is too long");
    return 1;
}

static ltx_gpu_buffer *upload_tensor(
        const ltx_st_mapping *mapping, const ltx_st_tensor *tensor,
        ltx_gpu *gpu, char *error, size_t error_size) {
    size_t bytes = 0;
    const void *data = tensor ? ltx_st_map_tensor(
        mapping, tensor, &bytes, error, error_size) : NULL;
    return data ? ltx_gpu_buffer_new_copy(
        gpu, data, bytes, error, error_size) : NULL;
}

static void linear_free(ltx_connector_linear *linear) {
    if (!linear) return;
    ltx_gpu_buffer_free(linear->weight);
    ltx_gpu_buffer_free(linear->scale);
    ltx_gpu_buffer_free(linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int linear_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix, ltx_connector_linear *linear,
        char *error, size_t error_size) {
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(
            header, mapping, prefix, &info, error, error_size)) return 0;
    if (!info.quantized_int8 || !info.convrot ||
        info.convrot_group_size != 256u || !info.weight_scale ||
        info.weight->dtype != LTX_DTYPE_I8 ||
        info.weight_scale->dtype != LTX_DTYPE_F32 ||
        (info.bias && info.bias->dtype != LTX_DTYPE_BF16)) {
        snprintf(error, error_size,
                 "%s is not a supported ConvRot INT8 linear", prefix);
        return 0;
    }
    linear->weight = upload_tensor(
        mapping, info.weight, gpu, error, error_size);
    linear->scale = upload_tensor(
        mapping, info.weight_scale, gpu, error, error_size);
    linear->bias = info.bias ? upload_tensor(
        mapping, info.bias, gpu, error, error_size) : NULL;
    linear->input_dim = info.input_dim;
    linear->output_dim = info.output_dim;
    if (!linear->weight || !linear->scale || (info.bias && !linear->bias)) {
        linear_free(linear);
        return 0;
    }
    return 1;
}

static void attention_free(ltx_connector_attention *attention) {
    if (!attention) return;
    linear_free(&attention->query);
    linear_free(&attention->key);
    linear_free(&attention->value);
    linear_free(&attention->output);
    ltx_gpu_buffer_free(attention->query_norm);
    ltx_gpu_buffer_free(attention->key_norm);
    ltx_gpu_buffer_free(attention->gate_weight);
    ltx_gpu_buffer_free(attention->gate_bias);
    memset(attention, 0, sizeof(*attention));
}

static int attention_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix,
        ltx_connector_attention *attention,
        char *error, size_t error_size) {
    memset(attention, 0, sizeof(*attention));
    char name[1024];
#define LOAD_LINEAR(FIELD, SUFFIX) \
    (make_name(name, sizeof(name), prefix, (SUFFIX), error, error_size) && \
     linear_load(header, mapping, gpu, name, &(attention)->FIELD, \
                 error, error_size))
    if (!LOAD_LINEAR(query, "to_q") ||
        !LOAD_LINEAR(key, "to_k") ||
        !LOAD_LINEAR(value, "to_v") ||
        !LOAD_LINEAR(output, "to_out.0")) {
#undef LOAD_LINEAR
        attention_free(attention);
        return 0;
    }
#undef LOAD_LINEAR
    uint32_t dim = attention->query.input_dim;
    uint32_t inner_dim = attention->query.output_dim;
    if (attention->key.input_dim != dim ||
        attention->value.input_dim != dim ||
        attention->output.output_dim != dim ||
        attention->key.output_dim != inner_dim ||
        attention->value.output_dim != inner_dim ||
        attention->output.input_dim != inner_dim) {
        snprintf(error, error_size,
                 "%s has incompatible projection geometry", prefix);
        attention_free(attention);
        return 0;
    }
    char query_norm_name[1024];
    char key_norm_name[1024];
    char gate_weight_name[1024];
    char gate_bias_name[1024];
    if (!make_name(query_norm_name, sizeof(query_norm_name), prefix,
                   "q_norm.weight", error, error_size) ||
        !make_name(key_norm_name, sizeof(key_norm_name), prefix,
                   "k_norm.weight", error, error_size) ||
        !make_name(gate_weight_name, sizeof(gate_weight_name), prefix,
                   "to_gate_logits.weight", error, error_size) ||
        !make_name(gate_bias_name, sizeof(gate_bias_name), prefix,
                   "to_gate_logits.bias", error, error_size)) {
        attention_free(attention);
        return 0;
    }
    const ltx_st_tensor *query_norm = ltx_st_find(header, query_norm_name);
    const ltx_st_tensor *key_norm = ltx_st_find(header, key_norm_name);
    const ltx_st_tensor *gate_weight = ltx_st_find(header, gate_weight_name);
    const ltx_st_tensor *gate_bias = ltx_st_find(header, gate_bias_name);
    if (!query_norm || !key_norm || !gate_weight || !gate_bias ||
        query_norm->dtype != LTX_DTYPE_BF16 || query_norm->ndim != 1u ||
        query_norm->shape[0] != inner_dim ||
        key_norm->dtype != LTX_DTYPE_BF16 || key_norm->ndim != 1u ||
        key_norm->shape[0] != inner_dim ||
        gate_weight->dtype != LTX_DTYPE_BF16 || gate_weight->ndim != 2u ||
        gate_weight->shape[1] != dim || !gate_weight->shape[0] ||
        gate_weight->shape[0] > UINT32_MAX ||
        gate_bias->dtype != LTX_DTYPE_BF16 || gate_bias->ndim != 1u ||
        gate_bias->shape[0] != gate_weight->shape[0]) {
        snprintf(error, error_size,
                 "%s has invalid norm/gate geometry", prefix);
        attention_free(attention);
        return 0;
    }
    attention->heads = (uint32_t)gate_weight->shape[0];
    if (inner_dim % attention->heads) {
        snprintf(error, error_size,
                 "%s has invalid attention head geometry", prefix);
        attention_free(attention);
        return 0;
    }
    attention->head_dim = inner_dim / attention->heads;
    attention->query_norm = upload_tensor(
        mapping, query_norm, gpu, error, error_size);
    attention->key_norm = upload_tensor(
        mapping, key_norm, gpu, error, error_size);
    attention->gate_weight = upload_tensor(
        mapping, gate_weight, gpu, error, error_size);
    attention->gate_bias = upload_tensor(
        mapping, gate_bias, gpu, error, error_size);
    if (!attention->query_norm || !attention->key_norm ||
        !attention->gate_weight || !attention->gate_bias) {
        attention_free(attention);
        return 0;
    }
    return 1;
}

static void mlp_free(ltx_connector_mlp *mlp) {
    if (!mlp) return;
    linear_free(&mlp->first);
    linear_free(&mlp->second);
    memset(mlp, 0, sizeof(*mlp));
}

static int mlp_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix, ltx_connector_mlp *mlp,
        char *error, size_t error_size) {
    memset(mlp, 0, sizeof(*mlp));
    char first_name[1024];
    char second_name[1024];
    if (!make_name(first_name, sizeof(first_name), prefix,
                   "net.0.proj", error, error_size) ||
        !make_name(second_name, sizeof(second_name), prefix,
                   "net.2", error, error_size) ||
        !linear_load(header, mapping, gpu, first_name, &mlp->first,
                     error, error_size) ||
        !linear_load(header, mapping, gpu, second_name, &mlp->second,
                     error, error_size)) {
        mlp_free(mlp);
        return 0;
    }
    mlp->hidden_dim = mlp->first.output_dim;
    if (mlp->first.input_dim != mlp->second.output_dim ||
        mlp->second.input_dim != mlp->hidden_dim) {
        snprintf(error, error_size,
                 "%s has incompatible MLP geometry", prefix);
        mlp_free(mlp);
        return 0;
    }
    return 1;
}

static void block_free(ltx_connector_block *block) {
    if (!block) return;
    attention_free(&block->attention);
    mlp_free(&block->mlp);
}

static void modality_free(ltx_connector_modality *modality) {
    if (!modality) return;
    for (uint32_t index = 0; index < LTX_CONNECTOR_BLOCKS; index++)
        block_free(&modality->blocks[index]);
    free(modality->register_values);
    memset(modality, 0, sizeof(*modality));
}

static int modality_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix, ltx_connector_modality *modality,
        char *error, size_t error_size) {
    memset(modality, 0, sizeof(*modality));
    char register_name[1024];
    if (!make_name(register_name, sizeof(register_name), prefix,
                   "learnable_registers", error, error_size)) return 0;
    const ltx_st_tensor *registers = ltx_st_find(header, register_name);
    if (!registers || registers->dtype != LTX_DTYPE_BF16 ||
        registers->ndim != 2u || !registers->shape[0] ||
        !registers->shape[1] || registers->shape[0] > UINT32_MAX ||
        registers->shape[1] > UINT32_MAX) {
        snprintf(error, error_size, "%s has invalid register geometry",
                 prefix);
        return 0;
    }
    modality->register_rows = (uint32_t)registers->shape[0];
    modality->dim = (uint32_t)registers->shape[1];
    size_t register_bytes = 0;
    size_t mapped_bytes = 0;
    const void *register_data = ltx_st_map_tensor(
        mapping, registers, &mapped_bytes, error, error_size);
    if (!checked_bytes(
            (uint64_t)modality->register_rows * modality->dim,
            sizeof(uint16_t), &register_bytes) ||
        !register_data || mapped_bytes != register_bytes) {
        modality_free(modality);
        return 0;
    }
    modality->register_values = malloc(register_bytes);
    if (!modality->register_values) {
        modality_free(modality);
        return connector_fail(error, error_size,
                              "out of memory copying connector registers");
    }
    memcpy(modality->register_values, register_data, register_bytes);

    for (uint32_t index = 0; index < LTX_CONNECTOR_BLOCKS; index++) {
        char block_prefix[1024];
        int length = snprintf(
            block_prefix, sizeof(block_prefix),
            "%s.transformer_1d_blocks.%u", prefix, index);
        char attention_prefix[1024];
        char mlp_prefix[1024];
        if (length < 0 || (size_t)length >= sizeof(block_prefix) ||
            !make_name(attention_prefix, sizeof(attention_prefix),
                       block_prefix, "attn1", error, error_size) ||
            !make_name(mlp_prefix, sizeof(mlp_prefix),
                       block_prefix, "ff", error, error_size) ||
            !attention_load(
                header, mapping, gpu, attention_prefix,
                &modality->blocks[index].attention, error, error_size) ||
            !mlp_load(
                header, mapping, gpu, mlp_prefix,
                &modality->blocks[index].mlp, error, error_size)) {
            modality_free(modality);
            return 0;
        }
        if (modality->blocks[index].attention.query.input_dim !=
                modality->dim ||
            modality->blocks[index].mlp.first.input_dim != modality->dim) {
            snprintf(error, error_size,
                     "%s block %u differs from register dimension",
                     prefix, index);
            modality_free(modality);
            return 0;
        }
    }
    return 1;
}

ltx_connector *ltx_connector_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size) {
    if (!header || !mapping || !gpu || !prefix) {
        connector_fail(error, error_size,
                       "missing connector load argument");
        return NULL;
    }
    ltx_connector *connector = calloc(1, sizeof(*connector));
    if (!connector) {
        connector_fail(error, error_size,
                       "out of memory creating connector");
        return NULL;
    }
    connector->gpu = gpu;
    char video_prefix[1024];
    char audio_prefix[1024];
    if (!make_name(video_prefix, sizeof(video_prefix), prefix,
                   "video_embeddings_connector", error, error_size) ||
        !make_name(audio_prefix, sizeof(audio_prefix), prefix,
                   "audio_embeddings_connector", error, error_size) ||
        !modality_load(header, mapping, gpu, video_prefix,
                       &connector->video, error, error_size) ||
        !modality_load(header, mapping, gpu, audio_prefix,
                       &connector->audio, error, error_size) ||
        connector->video.register_rows != connector->audio.register_rows) {
        if (error && error_size && !error[0])
            snprintf(error, error_size,
                     "video/audio connector register rows differ");
        ltx_connector_free(connector);
        return NULL;
    }
    return connector;
}

void ltx_connector_free(ltx_connector *connector) {
    if (!connector) return;
    modality_free(&connector->video);
    modality_free(&connector->audio);
    free(connector);
}

uint32_t ltx_connector_video_dim(const ltx_connector *connector) {
    return connector ? connector->video.dim : 0u;
}

uint32_t ltx_connector_audio_dim(const ltx_connector *connector) {
    return connector ? connector->audio.dim : 0u;
}

uint32_t ltx_connector_register_rows(const ltx_connector *connector) {
    return connector ? connector->video.register_rows : 0u;
}

uint32_t ltx_connector_output_rows(const ltx_connector *connector,
                                   uint32_t input_rows) {
    if (!connector || !input_rows || !connector->video.register_rows)
        return 0u;
    uint32_t rows = input_rows < 1024u ? 1024u : input_rows;
    uint32_t remainder = rows % connector->video.register_rows;
    if (!remainder) return rows;
    uint32_t padding = connector->video.register_rows - remainder;
    return rows <= UINT32_MAX - padding ? rows + padding : 0u;
}

static int modality_run(
        ltx_gpu *gpu, const ltx_connector_modality *modality,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t input_rows, char *error, size_t error_size) {
    uint32_t rows = input_rows < 1024u ? 1024u : input_rows;
    uint32_t remainder = rows % modality->register_rows;
    if (remainder) {
        uint32_t padding = modality->register_rows - remainder;
        if (rows > UINT32_MAX - padding)
            return connector_fail(error, error_size,
                                  "connector row count overflow");
        rows += padding;
    }
    uint64_t elements = (uint64_t)rows * modality->dim;
    size_t bytes = 0;
    size_t input_bytes = 0;
    size_t register_bytes = 0;
    uint64_t frequency_elements =
        (uint64_t)modality->blocks[0].attention.heads * rows *
        (modality->blocks[0].attention.head_dim / 2u);
    size_t frequency_bytes = 0;
    if (!input_rows || rows < input_rows || elements > UINT32_MAX ||
        !checked_bytes(elements, sizeof(uint16_t), &bytes) ||
        !checked_bytes((uint64_t)input_rows * modality->dim,
                       sizeof(uint16_t), &input_bytes) ||
        !checked_bytes(
            (uint64_t)modality->register_rows * modality->dim,
            sizeof(uint16_t), &register_bytes) ||
        !checked_bytes(frequency_elements, sizeof(uint16_t),
                       &frequency_bytes) ||
        ltx_gpu_buffer_bytes(output) < bytes ||
        ltx_gpu_buffer_bytes(input) < input_bytes) {
        return connector_fail(error, error_size,
                              "invalid connector run geometry/buffers");
    }

    uint16_t *combined = malloc(bytes);
    float *positions = malloc((size_t)rows * sizeof(float));
    uint16_t *cosine = malloc(frequency_bytes);
    uint16_t *sine = malloc(frequency_bytes);
    uint16_t *ones = malloc((size_t)modality->dim * sizeof(uint16_t));
    if (!combined || !positions || !cosine || !sine || !ones) {
        free(combined);
        free(positions);
        free(cosine);
        free(sine);
        free(ones);
        return connector_fail(error, error_size,
                              "out of memory creating connector workspace");
    }
    if (!ltx_gpu_buffer_read(input, combined, input_bytes,
                             error, error_size)) {
        free(combined);
        free(positions);
        free(cosine);
        free(sine);
        free(ones);
        return 0;
    }
    for (uint32_t row = input_rows; row < rows; row++) {
        uint32_t register_row = row % modality->register_rows;
        memcpy(combined + (uint64_t)row * modality->dim,
               modality->register_values +
                   (uint64_t)register_row * modality->dim,
               (size_t)modality->dim * sizeof(uint16_t));
    }
    for (uint32_t row = 0; row < rows; row++) positions[row] = (float)row;
    for (uint32_t column = 0; column < modality->dim; column++)
        ones[column] = 0x3f80u;
    float max_positions[1] = {4096.0f};
    int rope_ok = ltx_compute_rope_split_bf16(
        cosine, sine, frequency_elements, positions, rows, 1u,
        modality->blocks[0].attention.heads,
        modality->blocks[0].attention.head_dim,
        10000.0, max_positions, 1, error, error_size);
    free(positions);
    if (!rope_ok) {
        free(combined);
        free(cosine);
        free(sine);
        free(ones);
        return 0;
    }

    ltx_gpu_buffer *state[2] = {
        ltx_gpu_buffer_new_copy(gpu, combined, bytes, error, error_size),
        ltx_gpu_buffer_new(gpu, bytes, error, error_size),
    };
    ltx_gpu_buffer *normed = ltx_gpu_buffer_new(
        gpu, bytes, error, error_size);
    ltx_gpu_buffer *branch = ltx_gpu_buffer_new(
        gpu, bytes, error, error_size);
    ltx_gpu_buffer *cosine_buffer = ltx_gpu_buffer_new_copy(
        gpu, cosine, frequency_bytes, error, error_size);
    ltx_gpu_buffer *sine_buffer = ltx_gpu_buffer_new_copy(
        gpu, sine, frequency_bytes, error, error_size);
    ltx_gpu_buffer *ones_buffer = ltx_gpu_buffer_new_copy(
        gpu, ones, (size_t)modality->dim * sizeof(uint16_t),
        error, error_size);
    free(combined);
    free(cosine);
    free(sine);
    free(ones);
    int ok = state[0] && state[1] && normed && branch && cosine_buffer &&
        sine_buffer && ones_buffer;
    uint32_t current = 0u;
    for (uint32_t index = 0; ok && index < LTX_CONNECTOR_BLOCKS; index++) {
        const ltx_connector_block *block = &modality->blocks[index];
        const ltx_connector_attention *attention = &block->attention;
        ok = ltx_gpu_rms_norm_bf16(
                gpu, normed, state[current], rows, modality->dim, 1e-6f,
                error, error_size) &&
            ltx_gpu_self_attention_int8_mps_bf16(
                gpu, branch, normed,
                attention->query.weight, attention->query.scale,
                attention->query.bias,
                attention->key.weight, attention->key.scale,
                attention->key.bias,
                attention->value.weight, attention->value.scale,
                attention->value.bias,
                attention->query_norm, attention->key_norm,
                attention->gate_weight, attention->gate_bias,
                attention->output.weight, attention->output.scale,
                attention->output.bias,
                cosine_buffer, sine_buffer,
                rows, attention->heads, attention->head_dim,
                256u, 1e-5f, error, error_size) &&
            ltx_gpu_residual_gate_bf16(
                gpu, state[current ^ 1u], state[current], branch,
                ones_buffer, rows, modality->dim, 1u,
                error, error_size);
        current ^= 1u;
        if (!ok) break;
        ok = ltx_gpu_rms_norm_bf16(
                gpu, normed, state[current], rows, modality->dim, 1e-6f,
                error, error_size) &&
            ltx_gpu_mlp_int8_convrot_mps_bf16(
                gpu, branch, normed,
                block->mlp.first.weight, block->mlp.first.scale,
                block->mlp.first.bias,
                block->mlp.second.weight, block->mlp.second.scale,
                block->mlp.second.bias,
                rows, modality->dim, block->mlp.hidden_dim,
                modality->dim, 256u, error, error_size) &&
            ltx_gpu_residual_gate_bf16(
                gpu, state[current ^ 1u], state[current], branch,
                ones_buffer, rows, modality->dim, 1u,
                error, error_size);
        current ^= 1u;
    }
    if (ok)
        ok = ltx_gpu_rms_norm_bf16(
            gpu, output, state[current], rows, modality->dim, 1e-6f,
            error, error_size);
    ltx_gpu_buffer_free(state[0]);
    ltx_gpu_buffer_free(state[1]);
    ltx_gpu_buffer_free(normed);
    ltx_gpu_buffer_free(branch);
    ltx_gpu_buffer_free(cosine_buffer);
    ltx_gpu_buffer_free(sine_buffer);
    ltx_gpu_buffer_free(ones_buffer);
    return ok;
}

int ltx_connector_run_bf16(
    ltx_connector *connector,
    ltx_gpu_buffer *video_output, ltx_gpu_buffer *audio_output,
    const ltx_gpu_buffer *video_input,
    const ltx_gpu_buffer *audio_input,
    uint32_t input_rows,
    char *error, size_t error_size) {
    if (!connector || !video_output || !audio_output ||
        !video_input || !audio_input ||
        !ltx_connector_output_rows(connector, input_rows))
        return connector_fail(error, error_size,
                              "invalid connector run arguments");
    return modality_run(
            connector->gpu, &connector->video,
            video_output, video_input, input_rows, error, error_size) &&
        modality_run(
            connector->gpu, &connector->audio,
            audio_output, audio_input, input_rows, error, error_size);
}
