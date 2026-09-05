#include "ltx.h"
#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    CROSS_BACKEND_FUSED = 0,
    CROSS_BACKEND_STAGED,
} cross_backend;

typedef enum {
    ROPE_NONE = 0,
    ROPE_VIDEO_AUDIO,
    ROPE_AUDIO_VIDEO,
} rope_mode;

typedef struct {
    const void *weight;
    size_t weight_bytes;
    const void *scale;
    size_t scale_bytes;
    const void *bias;
    size_t bias_bytes;
} mapped_linear;

typedef struct {
    ltx_gpu_buffer *rotated_query;
    ltx_gpu_buffer *rotated_key_value;
    ltx_gpu_buffer *query_projection;
    ltx_gpu_buffer *key_projection;
    ltx_gpu_buffer *value_projection;
    ltx_gpu_buffer *query_normalized;
    ltx_gpu_buffer *key_normalized;
    ltx_gpu_buffer *query_heads;
    ltx_gpu_buffer *key_heads;
    ltx_gpu_buffer *value_heads;
    ltx_gpu_buffer *attention_heads;
    ltx_gpu_buffer *gate_logits;
    ltx_gpu_buffer *gate;
    ltx_gpu_buffer *row_output;
    ltx_gpu_buffer *rotated_output;
} staged_buffers;

typedef struct {
    double query;
    double key_value;
    double sdpa;
    double gate_unpack;
    double output;
} cross_timing;

static uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !value || value > UINT32_MAX) {
        fprintf(stderr, "bench_cross_attention: invalid %s: %s\n",
                label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static int checked_bytes(uint64_t elements, size_t element_size,
                         size_t *bytes) {
    if (!bytes || elements > SIZE_MAX / element_size) return 0;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int map_linear(const ltx_st_mapping *mapping,
                      const ltx_linear_weight_info *linear,
                      mapped_linear *mapped,
                      char *error, size_t error_size) {
    memset(mapped, 0, sizeof(*mapped));
    mapped->weight = ltx_st_map_tensor(mapping, linear->weight,
                                       &mapped->weight_bytes,
                                       error, error_size);
    mapped->scale = ltx_st_map_tensor(mapping, linear->weight_scale,
                                      &mapped->scale_bytes,
                                      error, error_size);
    if (linear->bias)
        mapped->bias = ltx_st_map_tensor(mapping, linear->bias,
                                         &mapped->bias_bytes,
                                         error, error_size);
    return mapped->weight && mapped->scale &&
        (!linear->bias || mapped->bias);
}

static int resolve_projection(const ltx_st_header *header,
                              const ltx_st_mapping *mapping,
                              const char *attention_prefix,
                              const char *suffix,
                              ltx_linear_weight_info *linear,
                              char *error, size_t error_size) {
    char prefix[1024];
    int length = snprintf(prefix, sizeof(prefix), "%s.%s",
                          attention_prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        snprintf(error, error_size, "attention prefix is too long");
        return 0;
    }
    return ltx_linear_weight_resolve(header, mapping, prefix, linear,
                                     error, error_size);
}

static const ltx_st_tensor *find_tensor(const ltx_st_header *header,
                                        const char *attention_prefix,
                                        const char *suffix,
                                        char *error, size_t error_size) {
    char name[1024];
    int length = snprintf(name, sizeof(name), "%s.%s",
                          attention_prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        snprintf(error, error_size, "attention prefix is too long");
        return NULL;
    }
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor) snprintf(error, error_size, "missing %s", name);
    return tensor;
}

static int validate_projection(const ltx_linear_weight_info *linear,
                               uint32_t input_dim, uint32_t output_dim) {
    return linear->quantized_int8 && linear->convrot &&
        linear->convrot_group_size == 256u &&
        linear->input_dim == input_dim &&
        linear->output_dim == output_dim &&
        (!linear->bias || linear->bias->dtype == LTX_DTYPE_BF16);
}

static rope_mode parse_rope_mode(const char *value) {
    if (!strcmp(value, "none")) return ROPE_NONE;
    if (!strcmp(value, "video-audio")) return ROPE_VIDEO_AUDIO;
    if (!strcmp(value, "audio-video")) return ROPE_AUDIO_VIDEO;
    fprintf(stderr,
            "bench_cross_attention: rope must be none, video-audio, "
            "or audio-video\n");
    exit(2);
}

static int fill_temporal_positions(float *positions, uint32_t rows,
                                   int video,
                                   char *error, size_t error_size) {
    if (!video)
        return ltx_compute_audio_positions(
            positions, rows, rows, error, error_size);
    uint32_t frames = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    if (rows == 1001u) {
        frames = 13u;
        height = 7u;
        width = 11u;
    } else if (rows == 4004u) {
        frames = 13u;
        height = 14u;
        width = 22u;
    } else {
        snprintf(error, error_size,
                 "video cross-RoPE requires 1001 or 4004 video rows");
        return 0;
    }
    float *video_positions = malloc((size_t)rows * 3u * sizeof(float));
    if (!video_positions) {
        snprintf(error, error_size, "out of memory creating video positions");
        return 0;
    }
    int ok = ltx_compute_video_positions(
        video_positions, (uint64_t)rows * 3u,
        frames, height, width, 24.0f, error, error_size);
    if (ok)
        for (uint32_t row = 0; row < rows; row++)
            positions[row] = video_positions[(uint64_t)row * 3u];
    free(video_positions);
    return ok;
}

static int generate_cross_rope(uint16_t *cosine, uint16_t *sine,
                               uint32_t rows, uint32_t heads,
                               uint32_t head_dim, int video,
                               char *error, size_t error_size) {
    float *positions = malloc((size_t)rows * sizeof(float));
    if (!positions) {
        snprintf(error, error_size, "out of memory creating positions");
        return 0;
    }
    float max_position[1] = {20.0f};
    uint64_t frequency_elements =
        (uint64_t)heads * rows * (head_dim / 2u);
    int ok = fill_temporal_positions(
        positions, rows, video, error, error_size) &&
        ltx_compute_rope_split_bf16(
            cosine, sine, frequency_elements,
            positions, rows, 1u, heads, head_dim,
            10000.0, max_position, 1, error, error_size);
    free(positions);
    return ok;
}

static int run_cross_attention(
        cross_backend backend, ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *query_input,
        const ltx_gpu_buffer *key_value_input,
        const ltx_gpu_buffer *query_weight,
        const ltx_gpu_buffer *query_scale,
        const ltx_gpu_buffer *query_bias,
        const ltx_gpu_buffer *key_weight,
        const ltx_gpu_buffer *key_scale,
        const ltx_gpu_buffer *key_bias,
        const ltx_gpu_buffer *value_weight,
        const ltx_gpu_buffer *value_scale,
        const ltx_gpu_buffer *value_bias,
        const ltx_gpu_buffer *query_norm,
        const ltx_gpu_buffer *key_norm,
        const ltx_gpu_buffer *gate_weight,
        const ltx_gpu_buffer *gate_bias,
        const ltx_gpu_buffer *output_weight,
        const ltx_gpu_buffer *output_scale,
        const ltx_gpu_buffer *output_bias,
        const ltx_gpu_buffer *query_cosine,
        const ltx_gpu_buffer *query_sine,
        const ltx_gpu_buffer *key_cosine,
        const ltx_gpu_buffer *key_sine,
        staged_buffers *scratch,
        uint32_t query_rows, uint32_t key_value_rows,
        uint32_t query_dim, uint32_t key_value_dim,
        uint32_t heads, uint32_t head_dim, uint32_t output_dim,
        cross_timing *timing,
        char *error, size_t error_size) {
    uint32_t inner_dim = heads * head_dim;
    if (backend == CROSS_BACKEND_FUSED)
        return ltx_gpu_cross_attention_int8_mps_bf16(
            gpu, output, query_input, key_value_input,
            query_weight, query_scale, query_bias,
            key_weight, key_scale, key_bias,
            value_weight, value_scale, value_bias,
            query_norm, key_norm, gate_weight, gate_bias,
            output_weight, output_scale, output_bias,
            query_cosine, query_sine, key_cosine, key_sine,
            query_rows, key_value_rows, query_dim, key_value_dim,
            heads, head_dim, output_dim, 256u, 1e-6f,
            error, error_size);

    int use_rope = query_cosine != NULL;
    double start = now_seconds();
    if (!ltx_gpu_convrot_bf16(
            gpu, scratch->rotated_query, query_input,
            query_rows, query_dim, 256u, error, error_size) ||
        ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, scratch->query_projection, scratch->rotated_query,
            query_weight, query_scale, query_bias,
            query_rows, query_dim, inner_dim, error, error_size) == 0 ||
        ltx_gpu_rms_norm_weighted_bf16(
            gpu, scratch->query_normalized, scratch->query_projection,
            query_norm, query_rows, inner_dim, 1e-6f,
            error, error_size) == 0 ||
        !(use_rope ? ltx_gpu_pack_rope_split_bf16(
            gpu, scratch->query_heads, scratch->query_normalized,
            query_cosine, query_sine,
            query_rows, heads, head_dim, error, error_size) :
          ltx_gpu_pack_heads_bf16(
            gpu, scratch->query_heads, scratch->query_normalized,
            query_rows, heads, head_dim, error, error_size))) return 0;
    if (timing) timing->query = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_convrot_bf16(
            gpu, scratch->rotated_key_value, key_value_input,
            key_value_rows, key_value_dim, 256u, error, error_size) ||
        ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, scratch->key_projection, scratch->rotated_key_value,
            key_weight, key_scale, key_bias,
            key_value_rows, key_value_dim, inner_dim,
            error, error_size) == 0 ||
        ltx_gpu_rms_norm_weighted_bf16(
            gpu, scratch->key_normalized, scratch->key_projection,
            key_norm, key_value_rows, inner_dim, 1e-6f,
            error, error_size) == 0 ||
        !(use_rope ? ltx_gpu_pack_rope_split_bf16(
            gpu, scratch->key_heads, scratch->key_normalized,
            key_cosine, key_sine,
            key_value_rows, heads, head_dim, error, error_size) :
          ltx_gpu_pack_heads_bf16(
            gpu, scratch->key_heads, scratch->key_normalized,
            key_value_rows, heads, head_dim, error, error_size)) ||
        ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, scratch->value_projection, scratch->rotated_key_value,
            value_weight, value_scale, value_bias,
            key_value_rows, key_value_dim, inner_dim,
            error, error_size) == 0 ||
        !ltx_gpu_pack_heads_bf16(
            gpu, scratch->value_heads, scratch->value_projection,
            key_value_rows, heads, head_dim, error, error_size)) return 0;
    if (timing) timing->key_value = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_sdpa_mps_bf16(
            gpu, scratch->attention_heads,
            scratch->query_heads, scratch->key_heads,
            scratch->value_heads, heads, query_rows,
            key_value_rows, head_dim,
            1.0f / sqrtf((float)head_dim), error, error_size)) return 0;
    if (timing) timing->sdpa = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_linear_bf16(
            gpu, scratch->gate_logits, query_input,
            gate_weight, gate_bias,
            query_rows, query_dim, heads, error, error_size) ||
        ltx_gpu_sigmoid2_bf16(
            gpu, scratch->gate, scratch->gate_logits,
            query_rows * heads, error, error_size) == 0 ||
        !ltx_gpu_unpack_heads_gate_bf16(
            gpu, scratch->row_output, scratch->attention_heads,
            scratch->gate, query_rows, heads, head_dim,
            error, error_size)) return 0;
    if (timing) timing->gate_unpack = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_convrot_bf16(
            gpu, scratch->rotated_output, scratch->row_output,
            query_rows, inner_dim, 256u, error, error_size) ||
        !ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, output, scratch->rotated_output,
            output_weight, output_scale, output_bias,
            query_rows, inner_dim, output_dim, error, error_size)) return 0;
    if (timing) timing->output = now_seconds() - start;
    return 1;
}

static void free_staged(staged_buffers *buffers) {
    ltx_gpu_buffer_free(buffers->rotated_query);
    ltx_gpu_buffer_free(buffers->rotated_key_value);
    ltx_gpu_buffer_free(buffers->query_projection);
    ltx_gpu_buffer_free(buffers->key_projection);
    ltx_gpu_buffer_free(buffers->value_projection);
    ltx_gpu_buffer_free(buffers->query_normalized);
    ltx_gpu_buffer_free(buffers->key_normalized);
    ltx_gpu_buffer_free(buffers->query_heads);
    ltx_gpu_buffer_free(buffers->key_heads);
    ltx_gpu_buffer_free(buffers->value_heads);
    ltx_gpu_buffer_free(buffers->attention_heads);
    ltx_gpu_buffer_free(buffers->gate_logits);
    ltx_gpu_buffer_free(buffers->gate);
    ltx_gpu_buffer_free(buffers->row_output);
    ltx_gpu_buffer_free(buffers->rotated_output);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT ATTENTION_PREFIX QUERY_ROWS KV_ROWS "
        "[rope] [warmup] [iterations]\n"
        "rope: none (default), video-audio, or audio-video.\n"
        "Set LTX_CROSS_BACKEND=staged for the multi-dispatch baseline.\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 8) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint_path = argv[1];
    const char *prefix = argv[2];
    uint32_t query_rows = parse_u32(argv[3], "query rows");
    uint32_t key_value_rows = parse_u32(argv[4], "key/value rows");
    rope_mode rope = argc > 5 ? parse_rope_mode(argv[5]) : ROPE_NONE;
    uint32_t warmup = argc > 6 ? parse_u32(argv[6], "warmup") : 2u;
    uint32_t iterations = argc > 7 ?
        parse_u32(argv[7], "iterations") : 5u;
    const char *backend_value = getenv("LTX_CROSS_BACKEND");
    cross_backend backend = backend_value && !strcmp(backend_value, "staged") ?
        CROSS_BACKEND_STAGED : CROSS_BACKEND_FUSED;
    char error[1024] = {0};
    int result = 1;

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint_path, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: %s\n", error);
        return 1;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: %s\n", error);
        ltx_st_free_header(&header);
        return 1;
    }

    ltx_linear_weight_info query_info;
    ltx_linear_weight_info key_info;
    ltx_linear_weight_info value_info;
    ltx_linear_weight_info output_info;
    if (!resolve_projection(&header, &mapping, prefix, "to_q", &query_info,
                            error, sizeof(error)) ||
        !resolve_projection(&header, &mapping, prefix, "to_k", &key_info,
                            error, sizeof(error)) ||
        !resolve_projection(&header, &mapping, prefix, "to_v", &value_info,
                            error, sizeof(error)) ||
        !resolve_projection(&header, &mapping, prefix, "to_out.0",
                            &output_info, error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: resolve projections: %s\n",
                error);
        goto cleanup_mapping;
    }

    uint32_t query_dim = query_info.input_dim;
    uint32_t key_value_dim = key_info.input_dim;
    uint32_t inner_dim = query_info.output_dim;
    uint32_t output_dim = output_info.output_dim;
    const ltx_st_tensor *gate_weight_tensor = find_tensor(
        &header, prefix, "to_gate_logits.weight", error, sizeof(error));
    const ltx_st_tensor *gate_bias_tensor = find_tensor(
        &header, prefix, "to_gate_logits.bias", error, sizeof(error));
    const ltx_st_tensor *query_norm_tensor = find_tensor(
        &header, prefix, "q_norm.weight", error, sizeof(error));
    const ltx_st_tensor *key_norm_tensor = find_tensor(
        &header, prefix, "k_norm.weight", error, sizeof(error));
    if (!gate_weight_tensor || !gate_bias_tensor ||
        !query_norm_tensor || !key_norm_tensor ||
        gate_weight_tensor->dtype != LTX_DTYPE_BF16 ||
        gate_weight_tensor->ndim != 2u ||
        gate_weight_tensor->shape[1] != query_dim ||
        !gate_weight_tensor->shape[0] ||
        gate_weight_tensor->shape[0] > UINT32_MAX ||
        gate_bias_tensor->dtype != LTX_DTYPE_BF16 ||
        gate_bias_tensor->ndim != 1u ||
        gate_bias_tensor->shape[0] != gate_weight_tensor->shape[0] ||
        query_norm_tensor->dtype != LTX_DTYPE_BF16 ||
        query_norm_tensor->ndim != 1u ||
        query_norm_tensor->shape[0] != inner_dim ||
        key_norm_tensor->dtype != LTX_DTYPE_BF16 ||
        key_norm_tensor->ndim != 1u ||
        key_norm_tensor->shape[0] != inner_dim) {
        fprintf(stderr, "bench_cross_attention: incompatible small weights\n");
        goto cleanup_mapping;
    }
    uint32_t heads = (uint32_t)gate_weight_tensor->shape[0];
    if (inner_dim % heads) {
        fputs("bench_cross_attention: invalid head geometry\n", stderr);
        goto cleanup_mapping;
    }
    uint32_t head_dim = inner_dim / heads;
    if (!validate_projection(&query_info, query_dim, inner_dim) ||
        !validate_projection(&key_info, key_value_dim, inner_dim) ||
        !validate_projection(&value_info, key_value_dim, inner_dim) ||
        !validate_projection(&output_info, inner_dim, output_dim)) {
        fputs("bench_cross_attention: projections must be ConvRot INT8\n",
              stderr);
        goto cleanup_mapping;
    }

    mapped_linear query_mapped;
    mapped_linear key_mapped;
    mapped_linear value_mapped;
    mapped_linear output_mapped;
    if (!map_linear(&mapping, &query_info, &query_mapped,
                    error, sizeof(error)) ||
        !map_linear(&mapping, &key_info, &key_mapped,
                    error, sizeof(error)) ||
        !map_linear(&mapping, &value_info, &value_mapped,
                    error, sizeof(error)) ||
        !map_linear(&mapping, &output_info, &output_mapped,
                    error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: map projections: %s\n",
                error);
        goto cleanup_mapping;
    }

#define LTX_MAP_TENSOR(TENSOR, DATA, BYTES) \
    size_t BYTES = 0; \
    const void *DATA = ltx_st_map_tensor( \
        &mapping, (TENSOR), &BYTES, error, sizeof(error))
    LTX_MAP_TENSOR(gate_weight_tensor, gate_weight_data, gate_weight_bytes);
    LTX_MAP_TENSOR(gate_bias_tensor, gate_bias_data, gate_bias_bytes);
    LTX_MAP_TENSOR(query_norm_tensor, query_norm_data, query_norm_bytes);
    LTX_MAP_TENSOR(key_norm_tensor, key_norm_data, key_norm_bytes);
#undef LTX_MAP_TENSOR
    if (!gate_weight_data || !gate_bias_data ||
        !query_norm_data || !key_norm_data) {
        fprintf(stderr, "bench_cross_attention: map small weights: %s\n",
                error);
        goto cleanup_mapping;
    }

    size_t query_input_bytes = 0;
    size_t key_value_input_bytes = 0;
    size_t query_inner_bytes = 0;
    size_t key_value_inner_bytes = 0;
    size_t gate_bytes = 0;
    size_t output_bytes = 0;
    size_t output_row_bytes = 0;
    size_t query_frequency_bytes = 0;
    size_t key_frequency_bytes = 0;
    if (!checked_bytes((uint64_t)query_rows * query_dim, sizeof(uint16_t),
                       &query_input_bytes) ||
        !checked_bytes((uint64_t)key_value_rows * key_value_dim,
                       sizeof(uint16_t), &key_value_input_bytes) ||
        !checked_bytes((uint64_t)query_rows * inner_dim, sizeof(uint16_t),
                       &query_inner_bytes) ||
        !checked_bytes((uint64_t)key_value_rows * inner_dim,
                       sizeof(uint16_t), &key_value_inner_bytes) ||
        !checked_bytes((uint64_t)query_rows * heads, sizeof(uint16_t),
                       &gate_bytes) ||
        !checked_bytes((uint64_t)query_rows * output_dim, sizeof(uint16_t),
                       &output_bytes) ||
        !checked_bytes(output_dim, sizeof(uint16_t), &output_row_bytes) ||
        !checked_bytes((uint64_t)heads * query_rows * (head_dim / 2u),
                       sizeof(uint16_t), &query_frequency_bytes) ||
        !checked_bytes((uint64_t)heads * key_value_rows * (head_dim / 2u),
                       sizeof(uint16_t), &key_frequency_bytes)) {
        fputs("bench_cross_attention: allocation size overflow\n", stderr);
        goto cleanup_mapping;
    }

    uint16_t *host_query_input = malloc(query_input_bytes);
    uint16_t *host_key_value_input = malloc(key_value_input_bytes);
    uint16_t *host_query_cosine = rope != ROPE_NONE ?
        malloc(query_frequency_bytes) : NULL;
    uint16_t *host_query_sine = rope != ROPE_NONE ?
        malloc(query_frequency_bytes) : NULL;
    uint16_t *host_key_cosine = rope != ROPE_NONE ?
        malloc(key_frequency_bytes) : NULL;
    uint16_t *host_key_sine = rope != ROPE_NONE ?
        malloc(key_frequency_bytes) : NULL;
    uint16_t *reference_output = malloc(output_row_bytes);
    uint16_t *candidate_output = malloc(output_row_bytes);
    double *timings = calloc(iterations, sizeof(double));
    if (!host_query_input || !host_key_value_input ||
        !reference_output || !candidate_output || !timings ||
        (rope != ROPE_NONE &&
         (!host_query_cosine || !host_query_sine ||
          !host_key_cosine || !host_key_sine))) {
        fputs("bench_cross_attention: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (uint64_t index = 0;
         index < (uint64_t)query_rows * query_dim; index++)
        host_query_input[index] = f32_to_bf16(
            sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f);
    for (uint64_t index = 0;
         index < (uint64_t)key_value_rows * key_value_dim; index++)
        host_key_value_input[index] = f32_to_bf16(
            cosf((float)(index % 4096u) * 0.011f) * 0.625f -
            sinf((float)(index % 2048u) * 0.005f) * 0.125f);
    if (rope != ROPE_NONE) {
        int query_is_video = rope == ROPE_VIDEO_AUDIO;
        int key_is_video = rope == ROPE_AUDIO_VIDEO;
        if (!generate_cross_rope(
                host_query_cosine, host_query_sine,
                query_rows, heads, head_dim, query_is_video,
                error, sizeof(error)) ||
            !generate_cross_rope(
                host_key_cosine, host_key_sine,
                key_value_rows, heads, head_dim, key_is_video,
                error, sizeof(error))) {
            fprintf(stderr, "bench_cross_attention: RoPE: %s\n", error);
            goto cleanup_host;
        }
    }

    double setup_start = now_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    ltx_gpu_buffer *query_input = NULL;
    ltx_gpu_buffer *key_value_input = NULL;
    ltx_gpu_buffer *query_weight = NULL;
    ltx_gpu_buffer *query_scale = NULL;
    ltx_gpu_buffer *query_bias = NULL;
    ltx_gpu_buffer *key_weight = NULL;
    ltx_gpu_buffer *key_scale = NULL;
    ltx_gpu_buffer *key_bias = NULL;
    ltx_gpu_buffer *value_weight = NULL;
    ltx_gpu_buffer *value_scale = NULL;
    ltx_gpu_buffer *value_bias = NULL;
    ltx_gpu_buffer *output_weight = NULL;
    ltx_gpu_buffer *output_scale = NULL;
    ltx_gpu_buffer *output_bias = NULL;
    ltx_gpu_buffer *query_norm = NULL;
    ltx_gpu_buffer *key_norm = NULL;
    ltx_gpu_buffer *gate_weight = NULL;
    ltx_gpu_buffer *gate_bias = NULL;
    ltx_gpu_buffer *query_cosine = NULL;
    ltx_gpu_buffer *query_sine = NULL;
    ltx_gpu_buffer *key_cosine = NULL;
    ltx_gpu_buffer *key_sine = NULL;
    ltx_gpu_buffer *output = NULL;
    staged_buffers scratch = {0};
#define LTX_UPLOAD_LINEAR(NAME, MAPPED) \
    NAME##_weight = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).weight, (MAPPED).weight_bytes, error, sizeof(error)); \
    NAME##_scale = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).scale, (MAPPED).scale_bytes, error, sizeof(error)); \
    if ((MAPPED).bias) NAME##_bias = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).bias, (MAPPED).bias_bytes, error, sizeof(error))
    if (gpu) {
        query_input = ltx_gpu_buffer_new_copy(
            gpu, host_query_input, query_input_bytes, error, sizeof(error));
        key_value_input = ltx_gpu_buffer_new_copy(
            gpu, host_key_value_input, key_value_input_bytes,
            error, sizeof(error));
        LTX_UPLOAD_LINEAR(query, query_mapped);
        LTX_UPLOAD_LINEAR(key, key_mapped);
        LTX_UPLOAD_LINEAR(value, value_mapped);
        LTX_UPLOAD_LINEAR(output, output_mapped);
        query_norm = ltx_gpu_buffer_new_copy(
            gpu, query_norm_data, query_norm_bytes, error, sizeof(error));
        key_norm = ltx_gpu_buffer_new_copy(
            gpu, key_norm_data, key_norm_bytes, error, sizeof(error));
        gate_weight = ltx_gpu_buffer_new_copy(
            gpu, gate_weight_data, gate_weight_bytes, error, sizeof(error));
        gate_bias = ltx_gpu_buffer_new_copy(
            gpu, gate_bias_data, gate_bias_bytes, error, sizeof(error));
        if (rope != ROPE_NONE) {
            query_cosine = ltx_gpu_buffer_new_copy(
                gpu, host_query_cosine, query_frequency_bytes,
                error, sizeof(error));
            query_sine = ltx_gpu_buffer_new_copy(
                gpu, host_query_sine, query_frequency_bytes,
                error, sizeof(error));
            key_cosine = ltx_gpu_buffer_new_copy(
                gpu, host_key_cosine, key_frequency_bytes,
                error, sizeof(error));
            key_sine = ltx_gpu_buffer_new_copy(
                gpu, host_key_sine, key_frequency_bytes,
                error, sizeof(error));
        }
        output = ltx_gpu_buffer_new(gpu, output_bytes, error, sizeof(error));
        scratch.rotated_query = ltx_gpu_buffer_new(
            gpu, query_input_bytes, error, sizeof(error));
        scratch.rotated_key_value = ltx_gpu_buffer_new(
            gpu, key_value_input_bytes, error, sizeof(error));
        scratch.query_projection = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
        scratch.key_projection = ltx_gpu_buffer_new(
            gpu, key_value_inner_bytes, error, sizeof(error));
        scratch.value_projection = ltx_gpu_buffer_new(
            gpu, key_value_inner_bytes, error, sizeof(error));
        scratch.query_normalized = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
        scratch.key_normalized = ltx_gpu_buffer_new(
            gpu, key_value_inner_bytes, error, sizeof(error));
        scratch.query_heads = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
        scratch.key_heads = ltx_gpu_buffer_new(
            gpu, key_value_inner_bytes, error, sizeof(error));
        scratch.value_heads = ltx_gpu_buffer_new(
            gpu, key_value_inner_bytes, error, sizeof(error));
        scratch.attention_heads = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
        scratch.gate_logits = ltx_gpu_buffer_new(
            gpu, gate_bytes, error, sizeof(error));
        scratch.gate = ltx_gpu_buffer_new(
            gpu, gate_bytes, error, sizeof(error));
        scratch.row_output = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
        scratch.rotated_output = ltx_gpu_buffer_new(
            gpu, query_inner_bytes, error, sizeof(error));
    }
#undef LTX_UPLOAD_LINEAR
    double setup_seconds = now_seconds() - setup_start;
    if (!gpu || !query_input || !key_value_input ||
        !query_weight || !query_scale || !key_weight || !key_scale ||
        !value_weight || !value_scale || !output_weight || !output_scale ||
        !query_norm || !key_norm || !gate_weight || !gate_bias || !output ||
        !scratch.rotated_query || !scratch.rotated_key_value ||
        !scratch.query_projection || !scratch.key_projection ||
        !scratch.value_projection || !scratch.query_normalized ||
        !scratch.key_normalized || !scratch.query_heads ||
        !scratch.key_heads || !scratch.value_heads ||
        !scratch.attention_heads || !scratch.gate_logits || !scratch.gate ||
        !scratch.row_output || !scratch.rotated_output ||
        (query_mapped.bias && !query_bias) ||
        (key_mapped.bias && !key_bias) ||
        (value_mapped.bias && !value_bias) ||
        (output_mapped.bias && !output_bias) ||
        (rope != ROPE_NONE &&
         (!query_cosine || !query_sine || !key_cosine || !key_sine))) {
        fprintf(stderr, "bench_cross_attention: GPU setup failed: %s\n",
                error);
        goto cleanup_gpu;
    }

#define LTX_RUN(BACKEND, TIMING) run_cross_attention( \
    (BACKEND), gpu, output, query_input, key_value_input, \
    query_weight, query_scale, query_bias, \
    key_weight, key_scale, key_bias, \
    value_weight, value_scale, value_bias, \
    query_norm, key_norm, gate_weight, gate_bias, \
    output_weight, output_scale, output_bias, \
    query_cosine, query_sine, key_cosine, key_sine, &scratch, \
    query_rows, key_value_rows, query_dim, key_value_dim, \
    heads, head_dim, output_dim, (TIMING), error, sizeof(error))
    if (!LTX_RUN(CROSS_BACKEND_STAGED, NULL) ||
        !ltx_gpu_buffer_read(output, reference_output, output_row_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: staged reference failed: %s\n",
                error);
        goto cleanup_gpu;
    }
    for (uint32_t index = 0; index < warmup; index++)
        if (!LTX_RUN(backend, NULL)) {
            fprintf(stderr, "bench_cross_attention: warmup failed: %s\n",
                    error);
            goto cleanup_gpu;
        }
    cross_timing detail_sum = {0};
    for (uint32_t index = 0; index < iterations; index++) {
        cross_timing detail = {0};
        double start = now_seconds();
        if (!LTX_RUN(backend, &detail)) {
            fprintf(stderr, "bench_cross_attention: iteration failed: %s\n",
                    error);
            goto cleanup_gpu;
        }
        timings[index] = now_seconds() - start;
        detail_sum.query += detail.query;
        detail_sum.key_value += detail.key_value;
        detail_sum.sdpa += detail.sdpa;
        detail_sum.gate_unpack += detail.gate_unpack;
        detail_sum.output += detail.output;
    }
#undef LTX_RUN
    if (!ltx_gpu_buffer_read(output, candidate_output, output_row_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_cross_attention: output read failed: %s\n",
                error);
        goto cleanup_gpu;
    }

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite = 0;
    for (uint32_t index = 0; index < output_dim; index++) {
        double reference = bf16_to_f32(reference_output[index]);
        double candidate = bf16_to_f32(candidate_output[index]);
        if (!isfinite(candidate)) nonfinite++;
        double difference = candidate - reference;
        diff2 += difference * difference;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    qsort(timings, iterations, sizeof(*timings), compare_double);
    size_t p50_index = (size_t)(iterations - 1u) / 2u;
    size_t p95_index = (size_t)ceil(0.95 * (double)iterations) - 1u;
    if (p95_index >= iterations) p95_index = iterations - 1u;
    double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
    double cosine_similarity = reference2 > 0.0 && candidate2 > 0.0 ?
        dot / sqrt(reference2 * candidate2) : 0.0;
    const char *rope_name = rope == ROPE_NONE ? "none" :
        (rope == ROPE_VIDEO_AUDIO ? "video-audio-temporal" :
                                    "audio-video-temporal");
    printf("checkpoint=%s\n", checkpoint_path);
    printf("attention=%s query_rows=%u kv_rows=%u query_dim=%u kv_dim=%u "
           "inner_dim=%u output_dim=%u heads=%u head_dim=%u\n",
           prefix, query_rows, key_value_rows, query_dim, key_value_dim,
           inner_dim, output_dim, heads, head_dim);
    printf("backend=%s rope=%s projection_weight_bytes=%zu "
           "setup_seconds=%.6f\n",
           backend == CROSS_BACKEND_FUSED ?
               "fused-full-mpsgraph" : "staged-mpsgraph",
           rope_name,
           query_mapped.weight_bytes + key_mapped.weight_bytes +
           value_mapped.weight_bytes + output_mapped.weight_bytes,
           setup_seconds);
    printf("warmup=%u iterations=%u p50_ms=%.3f p95_ms=%.3f\n",
           warmup, iterations,
           timings[p50_index] * 1000.0, timings[p95_index] * 1000.0);
    if (backend == CROSS_BACKEND_STAGED) {
        double scale = 1000.0 / (double)iterations;
        printf("mean_query_ms=%.3f mean_key_value_ms=%.3f "
               "mean_sdpa_ms=%.3f mean_gate_unpack_ms=%.3f "
               "mean_output_ms=%.3f\n",
               detail_sum.query * scale,
               detail_sum.key_value * scale,
               detail_sum.sdpa * scale,
               detail_sum.gate_unpack * scale,
               detail_sum.output * scale);
    }
    printf("row0_vs_staged_rel_l2=%.9g row0_vs_staged_cosine=%.9g "
           "max_abs=%.9g nonfinite=%llu\n",
           rel_l2, cosine_similarity, max_abs,
           (unsigned long long)nonfinite);
    result = nonfinite || cosine_similarity < 0.99 ? 1 : 0;

cleanup_gpu:
    ltx_gpu_buffer_free(query_input);
    ltx_gpu_buffer_free(key_value_input);
    ltx_gpu_buffer_free(query_weight);
    ltx_gpu_buffer_free(query_scale);
    ltx_gpu_buffer_free(query_bias);
    ltx_gpu_buffer_free(key_weight);
    ltx_gpu_buffer_free(key_scale);
    ltx_gpu_buffer_free(key_bias);
    ltx_gpu_buffer_free(value_weight);
    ltx_gpu_buffer_free(value_scale);
    ltx_gpu_buffer_free(value_bias);
    ltx_gpu_buffer_free(output_weight);
    ltx_gpu_buffer_free(output_scale);
    ltx_gpu_buffer_free(output_bias);
    ltx_gpu_buffer_free(query_norm);
    ltx_gpu_buffer_free(key_norm);
    ltx_gpu_buffer_free(gate_weight);
    ltx_gpu_buffer_free(gate_bias);
    ltx_gpu_buffer_free(query_cosine);
    ltx_gpu_buffer_free(query_sine);
    ltx_gpu_buffer_free(key_cosine);
    ltx_gpu_buffer_free(key_sine);
    ltx_gpu_buffer_free(output);
    free_staged(&scratch);
    ltx_gpu_free(gpu);
cleanup_host:
    free(host_query_input);
    free(host_key_value_input);
    free(host_query_cosine);
    free(host_query_sine);
    free(host_key_cosine);
    free(host_key_sine);
    free(reference_output);
    free(candidate_output);
    free(timings);
cleanup_mapping:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
