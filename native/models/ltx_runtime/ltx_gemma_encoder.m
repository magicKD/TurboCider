#include "ltx_gemma_encoder.h"

#include "ltx_gemma_tokenizer.h"
#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GEMMA_VOCAB = 262144,
    GEMMA_HIDDEN = 3840,
    GEMMA_INTERMEDIATE = 15360,
    GEMMA_LAYERS = 48,
    GEMMA_HEADS = 16,
    GEMMA_SLIDING_KV = 8,
    GEMMA_FULL_KV = 1,
    GEMMA_SLIDING_DIM = 256,
    GEMMA_FULL_DIM = 512,
    GEMMA_PROJECTION_INPUT = 188160,
    GEMMA_VIDEO_DIM = 4096,
    GEMMA_AUDIO_DIM = 2048,
    GEMMA_CONVROT_GROUP = 256
};

struct ltx_gemma_encoder {
    ltx_gpu *gpu;
    ltx_gemma_tokenizer *tokenizer;
    ltx_st_header header;
    ltx_st_mapping mapping;
    int mapping_open;
    uint32_t max_tokens;
};

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *scale;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
} gemma_linear_weights;

static int gemma_fail(char *error, size_t error_size,
                      const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static uint16_t gemma_f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float gemma_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void gemma_free_buffer(ltx_gpu_buffer **buffer) {
    if (buffer && *buffer) {
        ltx_gpu_buffer_free(*buffer);
        *buffer = NULL;
    }
}

static void gemma_free_linear(gemma_linear_weights *linear) {
    if (!linear) return;
    gemma_free_buffer(&linear->weight);
    gemma_free_buffer(&linear->scale);
    gemma_free_buffer(&linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int gemma_tensor_bytes(ltx_gemma_encoder *encoder,
                              const ltx_st_tensor *tensor,
                              const void **data, size_t *bytes,
                              char *error, size_t error_size) {
    if (!tensor || !data || !bytes)
        return gemma_fail(error, error_size, "missing Gemma tensor");
    *data = ltx_st_map_tensor(&encoder->mapping, tensor, bytes,
                              error, error_size);
    return *data != NULL;
}

static ltx_gpu_buffer *gemma_load_tensor(
        ltx_gemma_encoder *encoder, const ltx_st_tensor *tensor,
        char *error, size_t error_size) {
    const void *data = NULL;
    size_t bytes = 0;
    if (!gemma_tensor_bytes(encoder, tensor, &data, &bytes,
                            error, error_size)) return NULL;
    return ltx_gpu_buffer_new_copy(encoder->gpu, data, bytes,
                                   error, error_size);
}

static int gemma_load_linear(ltx_gemma_encoder *encoder, const char *prefix,
                             uint32_t input_dim, uint32_t output_dim,
                             gemma_linear_weights *linear,
                             char *error, size_t error_size) {
    if (!linear) return gemma_fail(error, error_size,
                                   "missing Gemma linear destination");
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(&encoder->header, &encoder->mapping,
                                   prefix, &info, error, error_size))
        return 0;
    if (!info.quantized_int8 || !info.convrot ||
        info.convrot_group_size != GEMMA_CONVROT_GROUP ||
        info.input_dim != input_dim || info.output_dim != output_dim ||
        info.bias)
        return gemma_fail(error, error_size,
                          "invalid Gemma ConvRot linear %s", prefix);
    linear->weight = gemma_load_tensor(encoder, info.weight,
                                       error, error_size);
    linear->scale = gemma_load_tensor(encoder, info.weight_scale,
                                      error, error_size);
    if (!linear->weight || !linear->scale) {
        gemma_free_linear(linear);
        return 0;
    }
    linear->input_dim = input_dim;
    linear->output_dim = output_dim;
    return 1;
}

static int gemma_load_vector(ltx_gemma_encoder *encoder, const char *name,
                             uint32_t elements, ltx_gpu_buffer **result,
                             char *error, size_t error_size) {
    const ltx_st_tensor *tensor = ltx_st_find(&encoder->header, name);
    if (!tensor || tensor->dtype != LTX_DTYPE_BF16 ||
        tensor->ndim != 1u || tensor->shape[0] != elements)
        return gemma_fail(error, error_size,
                          "invalid Gemma vector %s", name);
    *result = gemma_load_tensor(encoder, tensor, error, error_size);
    return *result != NULL;
}

static int gemma_apply_linear(ltx_gemma_encoder *encoder, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input, const gemma_linear_weights *linear,
                        uint32_t rows, char *error, size_t error_size) {
    return ltx_gpu_linear_int8_convrot_mps_bf16(
        encoder->gpu, output, input, linear->weight, linear->scale,
        linear->bias, rows, linear->input_dim, linear->output_dim,
        GEMMA_CONVROT_GROUP, error, error_size);
}

static int gemma_norm(ltx_gemma_encoder *encoder, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *input,
                      const ltx_gpu_buffer *weight, uint32_t rows,
                      uint32_t columns, char *error, size_t error_size) {
    return ltx_gpu_rms_norm_weighted_bf16(
        encoder->gpu, output, input, weight, rows, columns, 1e-6f,
        error, error_size);
}

static int gemma_make_rope(uint32_t rows, uint32_t position_offset,
                           uint32_t head_dim, double theta,
                           double partial_rotary_factor,
                           uint16_t **cosine, uint16_t **sine) {
    uint32_t half = head_dim / 2u;
    uint32_t rotating = partial_rotary_factor >= 1.0 ?
        half : (uint32_t)(partial_rotary_factor * (double)head_dim / 2.0);
    size_t count = (size_t)rows * half;
    uint16_t *c = calloc(count ? count : 1u, sizeof(*c));
    uint16_t *s = calloc(count ? count : 1u, sizeof(*s));
    if (!c || !s) {
        free(c); free(s);
        return 0;
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t pair = 0; pair < half; pair++) {
            float cv = 1.0f;
            float sv = 0.0f;
            if (pair < rotating) {
                double exponent = (2.0 * (double)pair) / (double)head_dim;
                double angle = (double)(position_offset + row) *
                    pow(theta, -exponent);
                cv = (float)cos(angle);
                sv = (float)sin(angle);
            }
            c[(size_t)row * half + pair] = gemma_f32_to_bf16(cv);
            s[(size_t)row * half + pair] = gemma_f32_to_bf16(sv);
        }
    }
    *cosine = c;
    *sine = s;
    return 1;
}

static int gemma_build_projection_input(
        const uint16_t *states, uint32_t rows, uint16_t *flat,
        uint32_t target_dim) {
    if (!states || !flat || !rows ||
        (target_dim != GEMMA_VIDEO_DIM && target_dim != GEMMA_AUDIO_DIM))
        return 0;
    float multiplier = sqrtf((float)target_dim / (float)GEMMA_HIDDEN);
    size_t layer_stride = (size_t)rows * GEMMA_HIDDEN;
    for (uint32_t row = 0; row < rows; row++) {
        float inverse_rms[GEMMA_LAYERS + 1u];
        for (uint32_t layer = 0; layer <= GEMMA_LAYERS; layer++) {
            float sum = 0.0f;
            const uint16_t *state =
                states + (size_t)layer * layer_stride +
                (size_t)row * GEMMA_HIDDEN;
            for (uint32_t hidden = 0; hidden < GEMMA_HIDDEN; hidden++) {
                float value = gemma_bf16_to_f32(state[hidden]);
                sum += value * value;
            }
            inverse_rms[layer] = 1.0f /
                sqrtf(sum / (float)GEMMA_HIDDEN + 1e-6f);
        }
        for (uint32_t hidden = 0; hidden < GEMMA_HIDDEN; hidden++) {
            for (uint32_t layer = 0; layer <= GEMMA_LAYERS; layer++) {
                float value = gemma_bf16_to_f32(
                    states[(size_t)layer * layer_stride +
                           (size_t)row * GEMMA_HIDDEN + hidden]);
                float normalized = value * inverse_rms[layer];
                size_t index = ((size_t)row * GEMMA_HIDDEN + hidden) *
                               (GEMMA_LAYERS + 1u) + layer;
                flat[index] = gemma_f32_to_bf16(normalized * multiplier);
            }
        }
    }
    return 1;
}

static int gemma_project(
        ltx_gemma_encoder *encoder, const ltx_gpu_buffer *input,
        uint32_t rows, uint32_t output_dim, const char *weight_name,
        const char *bias_name, uint16_t *output, size_t output_elements,
        char *error, size_t error_size) {
    const ltx_st_tensor *weight = ltx_st_find(&encoder->header, weight_name);
    const ltx_st_tensor *bias = ltx_st_find(&encoder->header, bias_name);
    if (!weight || weight->dtype != LTX_DTYPE_BF16 || weight->ndim != 2u ||
        weight->shape[0] != output_dim ||
        weight->shape[1] != GEMMA_PROJECTION_INPUT ||
        !bias || bias->dtype != LTX_DTYPE_BF16 || bias->ndim != 1u ||
        bias->shape[0] != output_dim ||
        output_elements != (size_t)rows * output_dim)
        return gemma_fail(error, error_size,
                          "invalid Gemma projection geometry");
    ltx_gpu_buffer *weight_gpu = gemma_load_tensor(
        encoder, weight, error, error_size);
    ltx_gpu_buffer *bias_gpu = gemma_load_tensor(
        encoder, bias, error, error_size);
    ltx_gpu_buffer *output_gpu = ltx_gpu_buffer_new(
        encoder->gpu, output_elements * sizeof(uint16_t), error, error_size);
    int ok = weight_gpu && bias_gpu && output_gpu &&
        ltx_gpu_linear_mps_bf16(
            encoder->gpu, output_gpu, input, weight_gpu, bias_gpu, rows,
            GEMMA_PROJECTION_INPUT, output_dim, error, error_size) &&
        ltx_gpu_buffer_read(output_gpu, output,
                            output_elements * sizeof(uint16_t),
                            error, error_size);
    gemma_free_buffer(&output_gpu);
    gemma_free_buffer(&bias_gpu);
    gemma_free_buffer(&weight_gpu);
    return ok;
}

ltx_gemma_encoder *ltx_gemma_encoder_create(
    const ltx_gemma_encoder_options *options,
    char *error, size_t error_size) {
    if (!options || !options->checkpoint || !options->tokenizer_json ||
        !options->shader_source)
        {
            gemma_fail(error, error_size, "missing Gemma encoder options");
            return NULL;
        }
    ltx_gemma_checkpoint_info info;
    if (!ltx_gemma_checkpoint_inspect(options->checkpoint, &info,
                                      error, error_size) ||
        !ltx_gemma_checkpoint_validate(&info, error, error_size))
        return NULL;
    ltx_gemma_encoder *encoder = calloc(1, sizeof(*encoder));
    if (!encoder) {
        gemma_fail(error, error_size, "out of memory creating Gemma encoder");
        return NULL;
    }
    if (!ltx_st_read_header(options->checkpoint, &encoder->header,
                            error, error_size) ||
        !ltx_st_map_open(&encoder->header, &encoder->mapping,
                         error, error_size)) {
        ltx_st_free_header(&encoder->header);
        free(encoder);
        return NULL;
    }
    encoder->mapping_open = 1;
    encoder->tokenizer = ltx_gemma_tokenizer_load(
        options->tokenizer_json, error, error_size);
    encoder->gpu = ltx_gpu_create(options->shader_source,
                                  error, error_size);
    if (!encoder->tokenizer || !encoder->gpu) {
        ltx_gemma_encoder_free(encoder);
        return NULL;
    }
    encoder->max_tokens = options->max_tokens ? options->max_tokens : 1024u;
    return encoder;
}

void ltx_gemma_encoder_free(ltx_gemma_encoder *encoder) {
    if (!encoder) return;
    ltx_gpu_free(encoder->gpu);
    ltx_gemma_tokenizer_free(encoder->tokenizer);
    if (encoder->mapping_open) ltx_st_map_close(&encoder->mapping);
    ltx_st_free_header(&encoder->header);
    free(encoder);
}

int ltx_gemma_encoder_encode(
    ltx_gemma_encoder *encoder, const char *prompt,
    uint16_t *video_output, size_t video_output_elements,
    uint16_t *audio_output, size_t audio_output_elements,
    uint16_t *mask_output, size_t mask_output_elements,
    uint32_t *output_rows, ltx_gemma_progress progress, void *opaque,
    char *error, size_t error_size) {
    if (!encoder || !prompt || !video_output || !audio_output ||
        !mask_output || !output_rows)
        return gemma_fail(error, error_size,
                          "invalid Gemma encoder arguments");
    if (error && error_size) error[0] = '\0';
    *output_rows = 0;
    uint32_t *ids = NULL;
    uint8_t *mask = NULL;
    size_t token_count = 0;
    if (!ltx_gemma_tokenizer_encode(encoder->tokenizer, prompt,
                                    encoder->max_tokens,
                                    &ids, &mask, &token_count,
                                    error, error_size))
        return 0;
    if (!token_count || token_count > UINT32_MAX) {
        ltx_gemma_tokenizer_ids_free(ids, mask);
        return gemma_fail(error, error_size,
                          "Gemma prompt token count is out of range");
    }
    size_t first = 0;
    while (first < token_count && !mask[first]) first++;
    size_t real_count = token_count - first;
    if (!real_count || real_count > UINT32_MAX) {
        ltx_gemma_tokenizer_ids_free(ids, mask);
        return gemma_fail(error, error_size,
                          "Gemma prompt contains no valid tokens");
    }
    uint32_t position_offset = (uint32_t)first;
    memmove(ids, ids + first, real_count * sizeof(*ids));
    memmove(mask, mask + first, real_count * sizeof(*mask));
    token_count = real_count;
    uint32_t rows = (uint32_t)token_count;
    size_t video_count = (size_t)rows * GEMMA_VIDEO_DIM;
    size_t audio_count = (size_t)rows * GEMMA_AUDIO_DIM;
    if (video_output_elements < video_count ||
        audio_output_elements < audio_count ||
        mask_output_elements < rows) {
        ltx_gemma_tokenizer_ids_free(ids, mask);
        return gemma_fail(error, error_size,
                          "Gemma encoder output buffers are too small");
    }

    int ok = 0;
    ltx_gpu_buffer *x = NULL;
    uint16_t *states = NULL;
    uint16_t *embedding = NULL;
    uint16_t *cosine_host = NULL;
    uint16_t *sine_host = NULL;
    ltx_gpu_buffer *cosine = NULL;
    ltx_gpu_buffer *sine = NULL;
    ltx_gpu_buffer *zero = NULL;
    ltx_gpu_buffer *flat_gpu = NULL;
    char name[512];

    const ltx_st_tensor *embedding_tensor = ltx_st_find(
        &encoder->header, "model.embed_tokens.weight");
    const void *embedding_data = NULL;
    size_t embedding_bytes = 0;
    if (!embedding_tensor || embedding_tensor->dtype != LTX_DTYPE_BF16 ||
        !gemma_tensor_bytes(encoder, embedding_tensor, &embedding_data,
                             &embedding_bytes, error, error_size))
        goto cleanup;
    embedding = malloc((size_t)rows * GEMMA_HIDDEN * sizeof(*embedding));
    if (!embedding) {
        gemma_fail(error, error_size, "out of memory for Gemma embeddings");
        goto cleanup;
    }
    const uint16_t *embedding_values = embedding_data;
    for (uint32_t row = 0; row < rows; row++) {
        if (ids[row] >= GEMMA_VOCAB) {
            gemma_fail(error, error_size, "Gemma token ID is out of range");
            goto cleanup;
        }
        const uint16_t *source =
            embedding_values + (size_t)ids[row] * GEMMA_HIDDEN;
        uint16_t *destination =
            embedding + (size_t)row * GEMMA_HIDDEN;
        for (uint32_t column = 0; column < GEMMA_HIDDEN; column++)
            destination[column] = gemma_f32_to_bf16(
                gemma_bf16_to_f32(source[column]) * sqrtf((float)GEMMA_HIDDEN));
    }
    x = ltx_gpu_buffer_new_copy(
        encoder->gpu, embedding,
        (size_t)rows * GEMMA_HIDDEN * sizeof(*embedding),
        error, error_size);
    if (!x) goto cleanup;
    states = calloc((size_t)(GEMMA_LAYERS + 1u) * rows * GEMMA_HIDDEN,
                    sizeof(*states));
    if (!states) {
        gemma_fail(error, error_size, "out of memory for Gemma hidden states");
        goto cleanup;
    }
    if (!ltx_gpu_buffer_read(
            x, states, (size_t)rows * GEMMA_HIDDEN * sizeof(*states),
            error, error_size)) goto cleanup;
    zero = ltx_gpu_buffer_new(
        encoder->gpu, (size_t)rows * GEMMA_INTERMEDIATE * sizeof(uint16_t),
        error, error_size);
    if (!zero) goto cleanup;
    uint16_t *zero_host = calloc((size_t)rows * GEMMA_INTERMEDIATE,
                                 sizeof(*zero_host));
    if (!zero_host ||
        !ltx_gpu_buffer_write(zero, zero_host,
                              (size_t)rows * GEMMA_INTERMEDIATE *
                                  sizeof(*zero_host),
                              error, error_size)) {
        free(zero_host);
        goto cleanup;
    }
    free(zero_host);

    for (uint32_t layer = 0; layer < GEMMA_LAYERS; layer++) {
        if (progress && progress("gemma_layer", (int)layer,
                                 GEMMA_LAYERS, opaque)) {
            gemma_fail(error, error_size, "Gemma encode cancelled");
            goto cleanup;
        }
        int sliding = (layer % 6u) != 5u;
        int layer_ok = 0;
        uint32_t head_dim = sliding ? GEMMA_SLIDING_DIM : GEMMA_FULL_DIM;
        uint32_t kv_heads = sliding ? GEMMA_SLIDING_KV : GEMMA_FULL_KV;
        uint32_t query_dim = GEMMA_HEADS * head_dim;
        uint32_t kv_dim = kv_heads * head_dim;
        gemma_linear_weights q = {0}, k = {0}, v = {0}, o = {0};
        gemma_linear_weights gate = {0}, up = {0}, down = {0};
        ltx_gpu_buffer *input_norm = NULL, *q_raw = NULL, *k_raw = NULL;
        ltx_gpu_buffer *v_raw = NULL, *q_norm = NULL, *k_norm = NULL;
        ltx_gpu_buffer *v_norm = NULL, *attn = NULL, *o_raw = NULL;
        ltx_gpu_buffer *post_attn = NULL, *residual = NULL;
        ltx_gpu_buffer *gate_raw = NULL, *up_raw = NULL, *gelu = NULL;
        ltx_gpu_buffer *product = NULL, *down_raw = NULL, *post_ff = NULL;
        ltx_gpu_buffer *next = NULL, *layer_scale = NULL, *layer_shift = NULL;
        int batch = 0;
        const char *layer_stage = "load normalization vectors";
        snprintf(name, sizeof(name), "model.layers.%u.input_layernorm.weight", layer);
        ltx_gpu_buffer *input_weight = NULL;
        snprintf(name, sizeof(name), "model.layers.%u.input_layernorm.weight", layer);
        if (!gemma_load_vector(encoder, name, GEMMA_HIDDEN, &input_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.post_attention_layernorm.weight", layer);
        ltx_gpu_buffer *post_attn_weight = NULL;
        if (!gemma_load_vector(encoder, name, GEMMA_HIDDEN, &post_attn_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.pre_feedforward_layernorm.weight", layer);
        ltx_gpu_buffer *pre_ff_weight = NULL;
        if (!gemma_load_vector(encoder, name, GEMMA_HIDDEN, &pre_ff_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.post_feedforward_layernorm.weight", layer);
        ltx_gpu_buffer *post_ff_weight = NULL;
        if (!gemma_load_vector(encoder, name, GEMMA_HIDDEN, &post_ff_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.q_norm.weight", layer);
        ltx_gpu_buffer *q_weight = NULL;
        if (!gemma_load_vector(encoder, name, head_dim, &q_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.k_norm.weight", layer);
        ltx_gpu_buffer *k_weight = NULL;
        if (!gemma_load_vector(encoder, name, head_dim, &k_weight,
                               error, error_size)) goto layer_cleanup;
        snprintf(name, sizeof(name), "model.layers.%u.layer_scalar", layer);
        ltx_gpu_buffer *scalar = NULL;
        if (!gemma_load_vector(encoder, name, 1u, &scalar,
                               error, error_size)) goto layer_cleanup;
        uint16_t scalar_host = 0;
        layer_stage = "read layer scalar";
        if (!ltx_gpu_buffer_read(scalar, &scalar_host, sizeof(scalar_host),
                                 error, error_size)) goto layer_cleanup;
        float scalar_value = gemma_bf16_to_f32(scalar_host) - 1.0f;
        uint16_t *scalar_vector = malloc((size_t)GEMMA_HIDDEN * sizeof(*scalar_vector));
        uint16_t *shift_vector = calloc(GEMMA_HIDDEN, sizeof(*shift_vector));
        if (!scalar_vector || !shift_vector) {
            free(scalar_vector); free(shift_vector);
            gemma_fail(error, error_size, "out of memory for Gemma scalar");
            goto layer_cleanup;
        }
        for (uint32_t column = 0; column < GEMMA_HIDDEN; column++)
            scalar_vector[column] = gemma_f32_to_bf16(scalar_value);
        layer_stage = "allocate layer modulation";
        layer_scale = ltx_gpu_buffer_new_copy(
            encoder->gpu, scalar_vector,
            (size_t)GEMMA_HIDDEN * sizeof(*scalar_vector), error, error_size);
        layer_shift = ltx_gpu_buffer_new_copy(
            encoder->gpu, shift_vector,
            (size_t)GEMMA_HIDDEN * sizeof(*shift_vector), error, error_size);
        free(scalar_vector); free(shift_vector);
        if (!layer_scale || !layer_shift) goto layer_cleanup;

        layer_stage = "load q projection";
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.q_proj", layer);
        if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN, query_dim,
                               &q, error, error_size)) goto layer_cleanup;
        layer_stage = "load k projection";
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.k_proj", layer);
        if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN, kv_dim,
                               &k, error, error_size)) goto layer_cleanup;
        if (sliding) {
            layer_stage = "load v projection";
            snprintf(name, sizeof(name), "model.layers.%u.self_attn.v_proj", layer);
            if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN, kv_dim,
                                   &v, error, error_size)) goto layer_cleanup;
        }
        layer_stage = "load o projection";
        snprintf(name, sizeof(name), "model.layers.%u.self_attn.o_proj", layer);
        if (!gemma_load_linear(encoder, name, query_dim, GEMMA_HIDDEN,
                               &o, error, error_size)) goto layer_cleanup;
        layer_stage = "load gate projection";
        snprintf(name, sizeof(name), "model.layers.%u.mlp.gate_proj", layer);
        if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN,
                               GEMMA_INTERMEDIATE, &gate,
                               error, error_size)) goto layer_cleanup;
        layer_stage = "load up projection";
        snprintf(name, sizeof(name), "model.layers.%u.mlp.up_proj", layer);
        if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN,
                               GEMMA_INTERMEDIATE, &up,
                               error, error_size)) goto layer_cleanup;
        layer_stage = "load down projection";
        snprintf(name, sizeof(name), "model.layers.%u.mlp.down_proj", layer);
        if (!gemma_load_linear(encoder, name, GEMMA_INTERMEDIATE,
                               GEMMA_HIDDEN, &down,
                               error, error_size)) goto layer_cleanup;

        layer_stage = "build RoPE";
        if (!gemma_make_rope(rows, position_offset, head_dim,
                             sliding ? 10000.0 : 1000000.0,
                             sliding ? 1.0 : 0.25,
                             &cosine_host, &sine_host)) {
            gemma_fail(error, error_size, "out of memory for Gemma RoPE");
            goto layer_cleanup;
        }
        layer_stage = "upload RoPE";
        cosine = ltx_gpu_buffer_new_copy(
            encoder->gpu, cosine_host,
            (size_t)rows * head_dim / 2u * sizeof(uint16_t),
            error, error_size);
        sine = ltx_gpu_buffer_new_copy(
            encoder->gpu, sine_host,
            (size_t)rows * head_dim / 2u * sizeof(uint16_t),
            error, error_size);
        free(cosine_host); cosine_host = NULL;
        free(sine_host); sine_host = NULL;
        if (!cosine || !sine) goto layer_cleanup;

#define GEMMA_NEW(NAME, BYTES) \
        layer_stage = "allocate layer workspace"; \
        NAME = ltx_gpu_buffer_new(encoder->gpu, (BYTES), error, error_size); \
        if (!NAME) goto layer_cleanup
        GEMMA_NEW(input_norm, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(q_raw, (size_t)rows * query_dim * 2u);
        GEMMA_NEW(k_raw, (size_t)rows * kv_dim * 2u);
        if (sliding) GEMMA_NEW(v_raw, (size_t)rows * kv_dim * 2u);
        GEMMA_NEW(q_norm, (size_t)rows * query_dim * 2u);
        GEMMA_NEW(k_norm, (size_t)rows * kv_dim * 2u);
        GEMMA_NEW(v_norm, (size_t)rows * kv_dim * 2u);
        GEMMA_NEW(attn, (size_t)rows * query_dim * 2u);
        GEMMA_NEW(o_raw, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(post_attn, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(residual, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(gate_raw, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
        GEMMA_NEW(up_raw, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
        GEMMA_NEW(gelu, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
        GEMMA_NEW(product, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
        GEMMA_NEW(down_raw, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(post_ff, (size_t)rows * GEMMA_HIDDEN * 2u);
        GEMMA_NEW(next, (size_t)rows * GEMMA_HIDDEN * 2u);
#undef GEMMA_NEW
        layer_stage = "begin GPU batch";
        if (!ltx_gpu_batch_begin(encoder->gpu, error, error_size)) goto layer_cleanup;
        batch = 1;
#define GEMMA_CALL(LABEL, EXPR) do { \
            if (error && error_size) error[0] = '\0'; \
            if (!(EXPR)) { \
                if (error && error_size && !error[0]) \
                    snprintf(error, error_size, "Gemma layer %u %s failed", \
                             layer, LABEL); \
                goto layer_cleanup; \
            } \
        } while (0)
        GEMMA_CALL("input norm", gemma_norm(encoder, input_norm, x,
                        input_weight, rows, GEMMA_HIDDEN, error, error_size));
        GEMMA_CALL("q projection", gemma_apply_linear(encoder, q_raw,
                        input_norm, &q, rows, error, error_size));
        GEMMA_CALL("k projection", gemma_apply_linear(encoder, k_raw,
                        input_norm, &k, rows, error, error_size));
        if (sliding)
            GEMMA_CALL("v projection", gemma_apply_linear(encoder, v_raw,
                        input_norm, &v, rows, error, error_size));
        GEMMA_CALL("q norm", gemma_norm(encoder, q_norm, q_raw, q_weight,
                        rows * GEMMA_HEADS, head_dim, error, error_size));
        GEMMA_CALL("k norm", gemma_norm(encoder, k_norm, k_raw, k_weight,
                        rows * kv_heads, head_dim, error, error_size));
        GEMMA_CALL("v norm", ltx_gpu_rms_norm_bf16(encoder->gpu, v_norm,
                sliding ? v_raw : k_raw, rows * kv_heads, head_dim, 1e-6f,
                error, error_size));
        GEMMA_CALL("attention", ltx_gpu_gemma_attention_mps_bf16(
                encoder->gpu, attn, q_norm, k_norm, v_norm, cosine, sine,
                rows, GEMMA_HEADS, kv_heads, head_dim, sliding ? 1024u : 0u,
                error, error_size));
        GEMMA_CALL("o projection", gemma_apply_linear(encoder, o_raw, attn,
                        &o, rows, error, error_size));
        GEMMA_CALL("post-attention norm", gemma_norm(encoder, post_attn,
                        o_raw, post_attn_weight, rows, GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("attention residual", ltx_gpu_add_bf16(encoder->gpu,
                        residual, x, post_attn, rows * GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("pre-feedforward norm", gemma_norm(encoder, input_norm,
                        residual, pre_ff_weight, rows, GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("gate projection", gemma_apply_linear(encoder, gate_raw,
                        input_norm, &gate, rows, error, error_size));
        GEMMA_CALL("up projection", gemma_apply_linear(encoder, up_raw,
                        input_norm, &up, rows, error, error_size));
        GEMMA_CALL("gelu", ltx_gpu_gelu_tanh_bf16(encoder->gpu, gelu,
                        gate_raw, rows * GEMMA_INTERMEDIATE,
                        error, error_size));
        GEMMA_CALL("gated product", ltx_gpu_residual_gate_bf16(
                encoder->gpu, product, zero, gelu, up_raw,
                rows, GEMMA_INTERMEDIATE, rows, error, error_size));
        GEMMA_CALL("down projection", gemma_apply_linear(encoder, down_raw,
                        product, &down, rows, error, error_size));
        GEMMA_CALL("post-feedforward norm", gemma_norm(encoder, post_ff,
                        down_raw, post_ff_weight, rows, GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("feedforward residual", ltx_gpu_add_bf16(encoder->gpu,
                        residual, residual, post_ff, rows * GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("layer affine", ltx_gpu_affine_bf16(
                encoder->gpu, next, residual, layer_scale, layer_shift,
                rows, GEMMA_HIDDEN, 1u, error, error_size));
        if (error && error_size) error[0] = '\0';
        int batch_ok = ltx_gpu_batch_end(encoder->gpu, error, error_size);
        batch = 0;
        if (!batch_ok) {
            if (error && error_size && !error[0])
                snprintf(error, error_size,
                         "Gemma layer %u batch completion failed", layer);
            goto layer_cleanup;
        }
#undef GEMMA_CALL
        layer_stage = "read hidden state";
        if (!ltx_gpu_buffer_read(
                next, states + (size_t)(layer + 1u) * rows * GEMMA_HIDDEN,
                (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t),
                error, error_size)) goto layer_cleanup;
        gemma_free_buffer(&x);
        x = next; next = NULL;
        layer_ok = 1;

layer_cleanup:
        if (batch) ltx_gpu_batch_end(encoder->gpu, error, error_size);
        free(cosine_host); cosine_host = NULL;
        free(sine_host); sine_host = NULL;
        gemma_free_buffer(&cosine);
        gemma_free_buffer(&sine);
        gemma_free_buffer(&input_weight);
        gemma_free_buffer(&post_attn_weight);
        gemma_free_buffer(&pre_ff_weight);
        gemma_free_buffer(&post_ff_weight);
        gemma_free_buffer(&q_weight);
        gemma_free_buffer(&k_weight);
        gemma_free_buffer(&scalar);
        gemma_free_buffer(&input_norm);
        gemma_free_buffer(&q_raw);
        gemma_free_buffer(&k_raw);
        gemma_free_buffer(&v_raw);
        gemma_free_buffer(&q_norm);
        gemma_free_buffer(&k_norm);
        gemma_free_buffer(&v_norm);
        gemma_free_buffer(&attn);
        gemma_free_buffer(&o_raw);
        gemma_free_buffer(&post_attn);
        gemma_free_buffer(&residual);
        gemma_free_buffer(&gate_raw);
        gemma_free_buffer(&up_raw);
        gemma_free_buffer(&gelu);
        gemma_free_buffer(&product);
        gemma_free_buffer(&down_raw);
        gemma_free_buffer(&post_ff);
        gemma_free_buffer(&next);
        gemma_free_linear(&q);
        gemma_free_linear(&k);
        gemma_free_linear(&v);
        gemma_free_linear(&o);
        gemma_free_linear(&gate);
        gemma_free_linear(&up);
        gemma_free_linear(&down);
        if (!layer_ok) {
            if (error && error_size && !error[0])
                snprintf(error, error_size,
                         "Gemma layer %u failed during %s without a backend error",
                         layer, layer_stage);
            goto cleanup;
        }
    }

    snprintf(name, sizeof(name), "model.norm.weight");
    ltx_gpu_buffer *final_norm = NULL;
    if (!gemma_load_vector(encoder, name, GEMMA_HIDDEN, &final_norm,
                           error, error_size)) goto cleanup;
    ltx_gpu_buffer *final_gpu = ltx_gpu_buffer_new(
        encoder->gpu, (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t),
        error, error_size);
    if (!final_gpu ||
        !gemma_norm(encoder, final_gpu, x, final_norm, rows, GEMMA_HIDDEN,
                    error, error_size) ||
        !ltx_gpu_buffer_read(
            final_gpu, states + (size_t)GEMMA_LAYERS * rows * GEMMA_HIDDEN,
            (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t),
            error, error_size)) {
        gemma_free_buffer(&final_norm);
        gemma_free_buffer(&final_gpu);
        goto cleanup;
    }
    gemma_free_buffer(&final_norm);
    gemma_free_buffer(&final_gpu);
    if (progress && progress("gemma_layer", GEMMA_LAYERS,
                             GEMMA_LAYERS, opaque)) {
        gemma_fail(error, error_size, "Gemma encode cancelled");
        goto cleanup;
    }

    uint16_t *flat = malloc((size_t)rows * GEMMA_PROJECTION_INPUT *
                            sizeof(*flat));
    if (!flat || !gemma_build_projection_input(
            states, rows, flat, GEMMA_VIDEO_DIM)) {
        free(flat);
        gemma_fail(error, error_size, "failed to build Gemma projection input");
        goto cleanup;
    }
    flat_gpu = ltx_gpu_buffer_new_copy(
        encoder->gpu, flat,
        (size_t)rows * GEMMA_PROJECTION_INPUT * sizeof(*flat),
        error, error_size);
    if (!flat_gpu || !gemma_project(
            encoder, flat_gpu, rows, GEMMA_VIDEO_DIM,
            "text_embedding_projection.video_aggregate_embed.weight",
            "text_embedding_projection.video_aggregate_embed.bias",
            video_output, video_count, error, error_size)) {
        free(flat);
        goto cleanup;
    }
    gemma_free_buffer(&flat_gpu);
    if (!gemma_build_projection_input(
            states, rows, flat, GEMMA_AUDIO_DIM)) {
        free(flat);
        gemma_fail(error, error_size,
                   "failed to build Gemma audio projection input");
        goto cleanup;
    }
    flat_gpu = ltx_gpu_buffer_new_copy(
        encoder->gpu, flat,
        (size_t)rows * GEMMA_PROJECTION_INPUT * sizeof(*flat),
        error, error_size);
    free(flat);
    if (!flat_gpu || !gemma_project(
            encoder, flat_gpu, rows, GEMMA_AUDIO_DIM,
            "text_embedding_projection.audio_aggregate_embed.weight",
            "text_embedding_projection.audio_aggregate_embed.bias",
            audio_output, audio_count, error, error_size))
        goto cleanup;
    /* LTX consumes an additive BF16 cross-attention mask. The valid token
     * rows are unmasked (zero); connector padding rows are appended by the
     * native connector and are also zero in the reference artifact format. */
    for (uint32_t row = 0; row < rows; row++)
        mask_output[row] = 0u;
    *output_rows = rows;
    ok = 1;

cleanup:
    gemma_free_buffer(&x);
    gemma_free_buffer(&zero);
    gemma_free_buffer(&flat_gpu);
    free(embedding);
    free(states);
    free(cosine_host);
    free(sine_host);
    ltx_gemma_tokenizer_ids_free(ids, mask);
    return ok;
}
