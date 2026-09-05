#include "ltx_conditioning.h"

#include "ltx.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
} ltx_bf16_linear;

struct ltx_adaln_single {
    ltx_gpu *gpu;
    ltx_bf16_linear linear1;
    ltx_bf16_linear linear2;
    ltx_bf16_linear parameter;
    ltx_gpu_buffer *timestep_embedding;
    uint32_t timestep_dim;
    uint32_t hidden_dim;
    uint32_t parameter_count;
};

enum {
    LTX_CONDITIONING_VIDEO = 0,
    LTX_CONDITIONING_AUDIO,
    LTX_CONDITIONING_VIDEO_PROMPT,
    LTX_CONDITIONING_AUDIO_PROMPT,
    LTX_CONDITIONING_AV_VIDEO,
    LTX_CONDITIONING_AV_AUDIO,
    LTX_CONDITIONING_A2V_GATE,
    LTX_CONDITIONING_V2A_GATE,
    LTX_CONDITIONING_COUNT
};

struct ltx_transformer_conditioning {
    ltx_adaln_single *module[LTX_CONDITIONING_COUNT];
    ltx_gpu_buffer *parameters[LTX_CONDITIONING_COUNT];
    ltx_gpu_buffer *embedded[LTX_CONDITIONING_COUNT];
    uint16_t *host_parameters[LTX_CONDITIONING_COUNT];
    uint32_t video_dim;
    uint32_t audio_dim;
};

static int ltx_conditioning_fail(char *error, size_t error_size,
                                 const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static int ltx_conditioning_name(
        char *name, size_t name_size,
        const char *prefix, const char *suffix,
        char *error, size_t error_size) {
    int length = snprintf(name, name_size, "%s.%s", prefix, suffix);
    if (length < 0 || (size_t)length >= name_size)
        return ltx_conditioning_fail(
            error, error_size, "AdaLN tensor name is too long");
    return 1;
}

static void ltx_bf16_linear_free(ltx_bf16_linear *linear) {
    if (!linear) return;
    ltx_gpu_buffer_free(linear->weight);
    ltx_gpu_buffer_free(linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int ltx_bf16_linear_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix, ltx_bf16_linear *linear,
        char *error, size_t error_size) {
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(
            header, mapping, prefix, &info, error, error_size)) return 0;
    if (info.quantized_int8 || info.weight->dtype != LTX_DTYPE_BF16 ||
        !info.bias || info.bias->dtype != LTX_DTYPE_BF16)
        return ltx_conditioning_fail(
            error, error_size,
            "AdaLN requires BF16 weights and BF16 biases");
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    const void *weight = ltx_st_map_tensor(
        mapping, info.weight, &weight_bytes, error, error_size);
    const void *bias = ltx_st_map_tensor(
        mapping, info.bias, &bias_bytes, error, error_size);
    if (!weight || !bias) return 0;
    linear->weight = ltx_gpu_buffer_new_copy(
        gpu, weight, weight_bytes, error, error_size);
    linear->bias = ltx_gpu_buffer_new_copy(
        gpu, bias, bias_bytes, error, error_size);
    linear->input_dim = info.input_dim;
    linear->output_dim = info.output_dim;
    if (!linear->weight || !linear->bias) {
        ltx_bf16_linear_free(linear);
        return 0;
    }
    return 1;
}

ltx_adaln_single *ltx_adaln_single_load(
    const ltx_st_header *header, const ltx_st_mapping *mapping,
    ltx_gpu *gpu, const char *prefix,
    char *error, size_t error_size) {
    if (!header || !mapping || !gpu || !prefix) {
        ltx_conditioning_fail(
            error, error_size, "missing AdaLN load argument");
        return NULL;
    }
    ltx_adaln_single *adaln = calloc(1, sizeof(*adaln));
    if (!adaln) {
        ltx_conditioning_fail(
            error, error_size, "out of memory loading AdaLN");
        return NULL;
    }
    char name[4096];
    if (!ltx_conditioning_name(
            name, sizeof(name), prefix,
            "emb.timestep_embedder.linear_1", error, error_size) ||
        !ltx_bf16_linear_load(
            header, mapping, gpu, name, &adaln->linear1,
            error, error_size) ||
        !ltx_conditioning_name(
            name, sizeof(name), prefix,
            "emb.timestep_embedder.linear_2", error, error_size) ||
        !ltx_bf16_linear_load(
            header, mapping, gpu, name, &adaln->linear2,
            error, error_size) ||
        !ltx_conditioning_name(
            name, sizeof(name), prefix, "linear", error, error_size) ||
        !ltx_bf16_linear_load(
            header, mapping, gpu, name, &adaln->parameter,
            error, error_size)) {
        ltx_adaln_single_free(adaln);
        return NULL;
    }
    adaln->gpu = gpu;
    adaln->timestep_dim = adaln->linear1.input_dim;
    adaln->hidden_dim = adaln->linear1.output_dim;
    if (adaln->linear2.input_dim != adaln->hidden_dim ||
        adaln->linear2.output_dim != adaln->hidden_dim ||
        adaln->parameter.input_dim != adaln->hidden_dim ||
        adaln->parameter.output_dim % adaln->hidden_dim) {
        ltx_conditioning_fail(
            error, error_size, "incompatible AdaLN weight geometry");
        ltx_adaln_single_free(adaln);
        return NULL;
    }
    adaln->parameter_count =
        adaln->parameter.output_dim / adaln->hidden_dim;
    adaln->timestep_embedding = ltx_gpu_buffer_new(
        gpu, (size_t)adaln->timestep_dim * sizeof(uint16_t),
        error, error_size);
    if (!adaln->timestep_embedding) {
        ltx_adaln_single_free(adaln);
        return NULL;
    }
    return adaln;
}

void ltx_adaln_single_free(ltx_adaln_single *adaln) {
    if (!adaln) return;
    ltx_gpu_buffer_free(adaln->timestep_embedding);
    ltx_bf16_linear_free(&adaln->parameter);
    ltx_bf16_linear_free(&adaln->linear2);
    ltx_bf16_linear_free(&adaln->linear1);
    free(adaln);
}

uint32_t ltx_adaln_single_timestep_dim(const ltx_adaln_single *adaln) {
    return adaln ? adaln->timestep_dim : 0u;
}

uint32_t ltx_adaln_single_hidden_dim(const ltx_adaln_single *adaln) {
    return adaln ? adaln->hidden_dim : 0u;
}

uint32_t ltx_adaln_single_parameter_count(const ltx_adaln_single *adaln) {
    return adaln ? adaln->parameter_count : 0u;
}

size_t ltx_adaln_single_embedded_bytes(const ltx_adaln_single *adaln) {
    return adaln ? (size_t)adaln->hidden_dim * sizeof(uint16_t) : 0u;
}

size_t ltx_adaln_single_parameter_bytes(const ltx_adaln_single *adaln) {
    return adaln ? (size_t)adaln->hidden_dim * adaln->parameter_count *
        sizeof(uint16_t) : 0u;
}

int ltx_adaln_single_eval_scalar(
    ltx_adaln_single *adaln,
    ltx_gpu_buffer *parameters,
    ltx_gpu_buffer *embedded_timestep,
    float timestep,
    char *error, size_t error_size) {
    if (!adaln || !parameters || !embedded_timestep)
        return ltx_conditioning_fail(
            error, error_size, "missing AdaLN evaluation argument");
    size_t bytes = (size_t)adaln->timestep_dim * sizeof(uint16_t);
    uint16_t *embedding = malloc(bytes);
    if (!embedding)
        return ltx_conditioning_fail(
            error, error_size, "out of memory computing timestep embedding");
    int ok = ltx_compute_timestep_embedding_bf16(
            embedding, adaln->timestep_dim, &timestep,
            1u, adaln->timestep_dim, 1, 0.0f, 1.0f, 10000.0f,
            error, error_size) &&
        ltx_gpu_buffer_write(
            adaln->timestep_embedding, embedding, bytes,
            error, error_size) &&
        ltx_gpu_adaln_single_mps_bf16(
            adaln->gpu, parameters, embedded_timestep,
            adaln->timestep_embedding,
            adaln->linear1.weight, adaln->linear1.bias,
            adaln->linear2.weight, adaln->linear2.bias,
            adaln->parameter.weight, adaln->parameter.bias,
            1u, adaln->timestep_dim, adaln->hidden_dim,
            adaln->parameter_count, error, error_size);
    free(embedding);
    return ok;
}

void ltx_transformer_conditioning_free(
        ltx_transformer_conditioning *conditioning) {
    if (!conditioning) return;
    for (unsigned index = 0; index < LTX_CONDITIONING_COUNT; index++) {
        free(conditioning->host_parameters[index]);
        ltx_gpu_buffer_free(conditioning->embedded[index]);
        ltx_gpu_buffer_free(conditioning->parameters[index]);
        ltx_adaln_single_free(conditioning->module[index]);
    }
    free(conditioning);
}

ltx_transformer_conditioning *ltx_transformer_conditioning_load(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        ltx_gpu *gpu, const char *prefix,
        char *error, size_t error_size) {
    if (!header || !mapping || !gpu || !prefix) {
        ltx_conditioning_fail(
            error, error_size, "missing Transformer conditioning argument");
        return NULL;
    }
    static const char *const suffix[LTX_CONDITIONING_COUNT] = {
        "adaln_single",
        "audio_adaln_single",
        "prompt_adaln_single",
        "audio_prompt_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_v2a_gate_adaln_single",
    };
    static const uint32_t expected_count[LTX_CONDITIONING_COUNT] = {
        9u, 9u, 2u, 2u, 4u, 4u, 1u, 1u,
    };
    ltx_transformer_conditioning *conditioning =
        calloc(1, sizeof(*conditioning));
    if (!conditioning) {
        ltx_conditioning_fail(
            error, error_size, "out of memory loading Transformer conditioning");
        return NULL;
    }
    for (unsigned index = 0; index < LTX_CONDITIONING_COUNT; index++) {
        char name[4096];
        if (!ltx_conditioning_name(
                name, sizeof(name), prefix, suffix[index],
                error, error_size)) {
            ltx_transformer_conditioning_free(conditioning);
            return NULL;
        }
        conditioning->module[index] = ltx_adaln_single_load(
            header, mapping, gpu, name, error, error_size);
        if (!conditioning->module[index] ||
            ltx_adaln_single_parameter_count(
                conditioning->module[index]) != expected_count[index]) {
            if (!error || !error_size || !error[0])
                ltx_conditioning_fail(
                    error, error_size,
                    "incompatible Transformer conditioning geometry");
            ltx_transformer_conditioning_free(conditioning);
            return NULL;
        }
        size_t parameter_bytes = ltx_adaln_single_parameter_bytes(
            conditioning->module[index]);
        size_t embedded_bytes = ltx_adaln_single_embedded_bytes(
            conditioning->module[index]);
        conditioning->parameters[index] = ltx_gpu_buffer_new(
            gpu, parameter_bytes, error, error_size);
        conditioning->embedded[index] = ltx_gpu_buffer_new(
            gpu, embedded_bytes, error, error_size);
        conditioning->host_parameters[index] = malloc(parameter_bytes);
        if (!conditioning->parameters[index] ||
            !conditioning->embedded[index] ||
            !conditioning->host_parameters[index]) {
            if (!error || !error_size || !error[0])
                ltx_conditioning_fail(
                    error, error_size,
                    "out of memory allocating Transformer conditioning");
            ltx_transformer_conditioning_free(conditioning);
            return NULL;
        }
    }

    conditioning->video_dim = ltx_adaln_single_hidden_dim(
        conditioning->module[LTX_CONDITIONING_VIDEO]);
    conditioning->audio_dim = ltx_adaln_single_hidden_dim(
        conditioning->module[LTX_CONDITIONING_AUDIO]);
    static const unsigned video_modules[] = {
        LTX_CONDITIONING_VIDEO_PROMPT,
        LTX_CONDITIONING_AV_VIDEO,
        LTX_CONDITIONING_A2V_GATE,
    };
    static const unsigned audio_modules[] = {
        LTX_CONDITIONING_AUDIO_PROMPT,
        LTX_CONDITIONING_AV_AUDIO,
        LTX_CONDITIONING_V2A_GATE,
    };
    for (size_t index = 0;
         index < sizeof(video_modules) / sizeof(video_modules[0]); index++)
        if (ltx_adaln_single_hidden_dim(
                conditioning->module[video_modules[index]]) !=
            conditioning->video_dim) {
            ltx_conditioning_fail(
                error, error_size,
                "video Transformer conditioning dimensions differ");
            ltx_transformer_conditioning_free(conditioning);
            return NULL;
        }
    for (size_t index = 0;
         index < sizeof(audio_modules) / sizeof(audio_modules[0]); index++)
        if (ltx_adaln_single_hidden_dim(
                conditioning->module[audio_modules[index]]) !=
            conditioning->audio_dim) {
            ltx_conditioning_fail(
                error, error_size,
                "audio Transformer conditioning dimensions differ");
            ltx_transformer_conditioning_free(conditioning);
            return NULL;
        }
    return conditioning;
}

int ltx_transformer_conditioning_eval_scalar(
        ltx_transformer_conditioning *conditioning, float sigma,
        ltx_transformer_conditioning_values *values,
        char *error, size_t error_size) {
    if (!conditioning || !values || !isfinite(sigma) ||
        sigma < 0.0f || sigma > 1.0f)
        return ltx_conditioning_fail(
            error, error_size,
            "invalid Transformer scalar conditioning arguments");
    memset(values, 0, sizeof(*values));
    for (unsigned index = 0; index < LTX_CONDITIONING_COUNT; index++) {
        float timestep = index <= LTX_CONDITIONING_AUDIO_PROMPT ?
            sigma * 1000.0f : sigma;
        size_t parameter_bytes = ltx_adaln_single_parameter_bytes(
            conditioning->module[index]);
        if (!ltx_adaln_single_eval_scalar(
                conditioning->module[index],
                conditioning->parameters[index],
                conditioning->embedded[index], timestep,
                error, error_size) ||
            !ltx_gpu_buffer_read(
                conditioning->parameters[index],
                conditioning->host_parameters[index], parameter_bytes,
                error, error_size))
            return 0;
    }
    values->video_dim = conditioning->video_dim;
    values->audio_dim = conditioning->audio_dim;
    values->video_adaln =
        conditioning->host_parameters[LTX_CONDITIONING_VIDEO];
    values->audio_adaln =
        conditioning->host_parameters[LTX_CONDITIONING_AUDIO];
    values->video_prompt =
        conditioning->host_parameters[LTX_CONDITIONING_VIDEO_PROMPT];
    values->audio_prompt =
        conditioning->host_parameters[LTX_CONDITIONING_AUDIO_PROMPT];
    values->av_video =
        conditioning->host_parameters[LTX_CONDITIONING_AV_VIDEO];
    values->av_audio =
        conditioning->host_parameters[LTX_CONDITIONING_AV_AUDIO];
    values->a2v_gate =
        conditioning->host_parameters[LTX_CONDITIONING_A2V_GATE];
    values->v2a_gate =
        conditioning->host_parameters[LTX_CONDITIONING_V2A_GATE];
    values->video_embedded =
        conditioning->embedded[LTX_CONDITIONING_VIDEO];
    values->audio_embedded =
        conditioning->embedded[LTX_CONDITIONING_AUDIO];
    return 1;
}
