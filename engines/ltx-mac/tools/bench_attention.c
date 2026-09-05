#include "ltx.h"
#ifdef LTX_ENABLE_ANE_QKV
#include "ltx_ane_qkv.h"
#endif
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
    ATTENTION_BACKEND_FUSED = 0,
    ATTENTION_BACKEND_STAGED,
    ATTENTION_BACKEND_SOL,
} attention_backend;

typedef struct {
    const void *weight;
    size_t weight_bytes;
    const void *scale;
    size_t scale_bytes;
    const void *bias;
    size_t bias_bytes;
} mapped_linear;

typedef struct {
    double qkv;
    double gate;
    double core;
    double output_convrot;
    double output_linear;
} attention_timing;

typedef struct {
    double gather;
    double core;
    double output_convrot;
    double output_linear;
} attention_shard_timing;

typedef struct {
    uint32_t head_start;
    uint32_t heads;
    uint32_t inner_dim;
    ltx_gpu_buffer *query;
    ltx_gpu_buffer *key;
    ltx_gpu_buffer *value;
    ltx_gpu_buffer *gate;
    ltx_gpu_buffer *cosine;
    ltx_gpu_buffer *sine;
    ltx_gpu_buffer *core_output;
    ltx_gpu_buffer *rotated_output;
    ltx_gpu_buffer *output_weight;
    ltx_gpu_buffer *partial_output;
} attention_shard;

typedef struct {
    double qkv;
    double gate;
    attention_shard_timing shard[2];
    double join;
} attention_split_timing;

#ifdef LTX_ENABLE_ANE_QKV
typedef struct {
    ltx_ane_qkv_timing qkv;
    double gpu_qkv;
    double gate;
    double core;
    double output_convrot;
    double output_linear;
} attention_sequence_timing;
#endif

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

static void free_attention_shard(attention_shard *shard) {
    if (!shard) return;
    ltx_gpu_buffer_free(shard->query);
    ltx_gpu_buffer_free(shard->key);
    ltx_gpu_buffer_free(shard->value);
    ltx_gpu_buffer_free(shard->gate);
    ltx_gpu_buffer_free(shard->cosine);
    ltx_gpu_buffer_free(shard->sine);
    ltx_gpu_buffer_free(shard->core_output);
    ltx_gpu_buffer_free(shard->rotated_output);
    ltx_gpu_buffer_free(shard->output_weight);
    ltx_gpu_buffer_free(shard->partial_output);
    memset(shard, 0, sizeof(*shard));
}

static int attention_shard_ready(const attention_shard *shard) {
    return shard && shard->heads && shard->inner_dim &&
        shard->query && shard->key && shard->value && shard->gate &&
        shard->cosine && shard->sine && shard->core_output &&
        shard->rotated_output && shard->output_weight &&
        shard->partial_output;
}

static int8_t *slice_output_weight(const mapped_linear *mapped,
                                   uint32_t output_dim,
                                   uint32_t input_dim,
                                   uint32_t start_column,
                                   uint32_t columns) {
    uint64_t expected = (uint64_t)output_dim * input_dim;
    uint64_t result_bytes = (uint64_t)output_dim * columns;
    if (!mapped || !mapped->weight || mapped->weight_bytes != expected ||
        !columns || start_column >= input_dim ||
        columns > input_dim - start_column || result_bytes > SIZE_MAX)
        return NULL;
    int8_t *result = malloc((size_t)result_bytes);
    if (!result) return NULL;
    const int8_t *source = mapped->weight;
    for (uint32_t row = 0; row < output_dim; row++)
        memcpy(result + (uint64_t)row * columns,
               source + (uint64_t)row * input_dim + start_column,
               columns);
    return result;
}

static int run_attention_shard(
        ltx_gpu *gpu, attention_shard *shard,
        const ltx_gpu_buffer *full_query,
        const ltx_gpu_buffer *full_key,
        const ltx_gpu_buffer *full_value,
        const ltx_gpu_buffer *full_gate,
        const ltx_gpu_buffer *output_scale,
        const ltx_gpu_buffer *output_bias,
        uint32_t rows, uint32_t full_inner_dim,
        uint32_t full_heads, uint32_t head_dim,
        attention_shard_timing *timing,
        char *error, size_t error_size) {
    if (!gpu || !attention_shard_ready(shard)) return 0;
    double start = now_seconds();
    if (!ltx_gpu_batch_begin(gpu, error, error_size)) return 0;
    int ok = ltx_gpu_slice_columns_bf16(
            gpu, shard->query, full_query, rows, full_inner_dim,
            shard->head_start * head_dim, shard->inner_dim,
            error, error_size) &&
        ltx_gpu_slice_columns_bf16(
            gpu, shard->key, full_key, rows, full_inner_dim,
            shard->head_start * head_dim, shard->inner_dim,
            error, error_size) &&
        ltx_gpu_slice_columns_bf16(
            gpu, shard->value, full_value, rows, full_inner_dim,
            shard->head_start * head_dim, shard->inner_dim,
            error, error_size) &&
        ltx_gpu_slice_columns_bf16(
            gpu, shard->gate, full_gate, rows, full_heads,
            shard->head_start, shard->heads,
            error, error_size);
    char batch_error[1024] = {0};
    int batch_ok = ltx_gpu_batch_end(
        gpu, batch_error, sizeof(batch_error));
    if (!batch_ok && ok)
        snprintf(error, error_size, "%s",
                 batch_error[0] ? batch_error :
                 "attention shard gather batch failed");
    if (!ok || !batch_ok) return 0;
    if (timing) timing->gather = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_self_attention_core_mps_bf16(
            gpu, shard->core_output,
            shard->query, shard->key, shard->value,
            shard->cosine, shard->sine, shard->gate,
            rows, shard->heads, head_dim,
            1.0f / sqrtf((float)head_dim), error, error_size)) return 0;
    if (timing) timing->core = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_convrot_bf16(
            gpu, shard->rotated_output, shard->core_output,
            rows, shard->inner_dim, 256u, error, error_size)) return 0;
    if (timing) timing->output_convrot = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, shard->partial_output, shard->rotated_output,
            shard->output_weight, output_scale, output_bias,
            rows, shard->inner_dim, full_inner_dim,
            error, error_size)) return 0;
    if (timing) timing->output_linear = now_seconds() - start;
    return 1;
}

static int run_attention_split(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input,
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
        const ltx_gpu_buffer *output_scale,
        const ltx_gpu_buffer *output_bias,
        ltx_gpu_buffer *query, ltx_gpu_buffer *key,
        ltx_gpu_buffer *value, ltx_gpu_buffer *gate_logits,
        ltx_gpu_buffer *gate, attention_shard shard[2],
        uint32_t rows, uint32_t heads, uint32_t head_dim,
        attention_split_timing *timing,
        char *error, size_t error_size) {
    uint32_t inner_dim = heads * head_dim;
    double start = now_seconds();
    if (!ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu, query, key, value, input,
            query_weight, query_scale, query_bias,
            key_weight, key_scale, key_bias,
            value_weight, value_scale, value_bias,
            query_norm, key_norm,
            rows, inner_dim, inner_dim, 256u, 1e-6f,
            error, error_size)) return 0;
    if (timing) timing->qkv = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_linear_bf16(
            gpu, gate_logits, input, gate_weight, gate_bias,
            rows, inner_dim, heads, error, error_size) ||
        !ltx_gpu_sigmoid2_bf16(
            gpu, gate, gate_logits, rows * heads,
            error, error_size)) return 0;
    if (timing) timing->gate = now_seconds() - start;

    if (!run_attention_shard(
            gpu, &shard[0], query, key, value, gate,
            output_scale, output_bias, rows, inner_dim, heads, head_dim,
            timing ? &timing->shard[0] : NULL, error, error_size) ||
        !run_attention_shard(
            gpu, &shard[1], query, key, value, gate,
            output_scale, NULL, rows, inner_dim, heads, head_dim,
            timing ? &timing->shard[1] : NULL, error, error_size)) return 0;

    start = now_seconds();
    uint64_t elements = (uint64_t)rows * inner_dim;
    if (elements > UINT32_MAX || !ltx_gpu_add_bf16(
            gpu, output, shard[0].partial_output, shard[1].partial_output,
            (uint32_t)elements, error, error_size)) return 0;
    if (timing) timing->join = now_seconds() - start;
    return 1;
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
        fprintf(stderr, "bench_attention: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static float parse_f32(const char *text, const char *label) {
    char *end = NULL;
    float value = strtof(text, &end);
    if (!text[0] || !end || *end || !isfinite(value)) {
        fprintf(stderr, "bench_attention: invalid %s: %s\n", label, text);
        exit(2);
    }
    return value;
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

static const ltx_st_tensor *resolve_tensor(
        const ltx_st_header *header, const char *attention_prefix,
        const char *suffix, ltx_dtype dtype,
        uint32_t dimensions, uint64_t first, uint64_t second,
        char *error, size_t error_size) {
    char name[1024];
    int length = snprintf(name, sizeof(name), "%s.%s",
                          attention_prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        snprintf(error, error_size, "attention prefix is too long");
        return NULL;
    }
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor || tensor->dtype != dtype || tensor->ndim != dimensions ||
        tensor->shape[0] != first ||
        (dimensions > 1u && tensor->shape[1] != second)) {
        snprintf(error, error_size, "invalid or missing %s", name);
        return NULL;
    }
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

static int run_attention(
        attention_backend backend, ltx_gpu *gpu,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        const ltx_gpu_buffer *query_weight,
        const ltx_gpu_buffer *query_scale,
        const ltx_gpu_buffer *query_bias,
        const ltx_gpu_buffer *key_weight,
        const ltx_gpu_buffer *key_scale,
        const ltx_gpu_buffer *key_bias,
        const ltx_gpu_buffer *value_weight,
        const ltx_gpu_buffer *value_scale,
        const ltx_gpu_buffer *value_bias,
        const ltx_gpu_buffer *packed_qkv_weight,
        const ltx_gpu_buffer *packed_qkv_scale,
        const ltx_gpu_buffer *packed_qkv_bias,
        const ltx_gpu_buffer *query_norm,
        const ltx_gpu_buffer *key_norm,
        const ltx_gpu_buffer *gate_weight,
        const ltx_gpu_buffer *gate_bias,
        const ltx_gpu_buffer *output_weight,
        const ltx_gpu_buffer *output_scale,
        const ltx_gpu_buffer *output_bias,
        const ltx_gpu_buffer *cosine,
        const ltx_gpu_buffer *sine,
        ltx_gpu_buffer *query,
        ltx_gpu_buffer *key,
        ltx_gpu_buffer *value,
        ltx_gpu_buffer *gate_logits,
        ltx_gpu_buffer *gate,
        ltx_gpu_buffer *core_output,
        ltx_gpu_buffer *rotated_output,
        ltx_gpu_buffer *packed_query,
        ltx_gpu_buffer *packed_key,
        ltx_gpu_buffer *packed_value,
        ltx_gpu_buffer *packed_output,
        ltx_gpu_buffer *query_centroids,
        ltx_gpu_buffer *key_centroids,
        ltx_gpu_buffer *value_sums,
        ltx_gpu_buffer *thresholds,
        ltx_gpu_buffer *routes,
        uint32_t rows, uint32_t heads, uint32_t head_dim,
        float sol_tau, int use_packed_qkv, int fuse_output_projection,
        int batch_qkv_gate,
        attention_timing *timing,
        char *error, size_t error_size) {
    uint32_t inner_dim = heads * head_dim;
    if (backend == ATTENTION_BACKEND_FUSED && !use_packed_qkv)
        return ltx_gpu_self_attention_int8_mps_bf16(
            gpu, output, input,
            query_weight, query_scale, query_bias,
            key_weight, key_scale, key_bias,
            value_weight, value_scale, value_bias,
            query_norm, key_norm, gate_weight, gate_bias,
            output_weight, output_scale, output_bias,
            cosine, sine, rows, heads, head_dim, 256u, 1e-6f,
            error, error_size);
#define LTX_RUN_QKV() (use_packed_qkv ? \
        ltx_gpu_qkv_packed_int8_convrot_mps_bf16( \
            gpu, query, key, value, input, \
            packed_qkv_weight, packed_qkv_scale, packed_qkv_bias, \
            query_norm, key_norm, \
            rows, inner_dim, inner_dim, 256u, 1e-6f, \
            error, error_size) : \
        ltx_gpu_qkv_int8_convrot_mps_bf16( \
            gpu, query, key, value, input, \
            query_weight, query_scale, query_bias, \
            key_weight, key_scale, key_bias, \
            value_weight, value_scale, value_bias, \
            query_norm, key_norm, \
            rows, inner_dim, inner_dim, 256u, 1e-6f, \
            error, error_size))
    double start = now_seconds();
    if (batch_qkv_gate) {
        if (!ltx_gpu_batch_begin(gpu, error, error_size)) return 0;
        int prep_ok = LTX_RUN_QKV();
        if (prep_ok)
            prep_ok = ltx_gpu_linear_bf16(
                gpu, gate_logits, input, gate_weight, gate_bias,
                rows, inner_dim, heads, error, error_size);
        if (prep_ok)
            prep_ok = ltx_gpu_sigmoid2_bf16(
                gpu, gate, gate_logits, rows * heads, error, error_size);
        char batch_error[1024] = {0};
        int batch_ok = ltx_gpu_batch_end(
            gpu, batch_error, sizeof(batch_error));
        if (!prep_ok) return 0;
        if (!batch_ok) {
            snprintf(error, error_size, "%s", batch_error[0] ?
                     batch_error : "QKV/gate command batch failed");
            return 0;
        }
        if (timing) timing->qkv = now_seconds() - start;
    } else {
        if (!LTX_RUN_QKV()) return 0;
        if (timing) timing->qkv = now_seconds() - start;

        start = now_seconds();
        if (!ltx_gpu_linear_bf16(
                gpu, gate_logits, input, gate_weight, gate_bias,
                rows, inner_dim, heads, error, error_size) ||
            !ltx_gpu_sigmoid2_bf16(
                gpu, gate, gate_logits, rows * heads,
                error, error_size)) return 0;
        if (timing) timing->gate = now_seconds() - start;
    }
#undef LTX_RUN_QKV

    start = now_seconds();
    int core_ok = backend == ATTENTION_BACKEND_SOL ?
        ltx_gpu_self_attention_core_sol_bf16(
               gpu, core_output, query, key, value, cosine, sine, gate,
               packed_query, packed_key, packed_value, packed_output,
               query_centroids, key_centroids, value_sums, thresholds, routes,
               rows, heads, head_dim, 1.0f / sqrtf((float)head_dim), sol_tau,
               0u, 0u, 0u, 0u, error, error_size) :
        ltx_gpu_self_attention_core_mps_bf16(
               gpu, core_output, query, key, value, cosine, sine, gate,
               rows, heads, head_dim, 1.0f / sqrtf((float)head_dim),
               error, error_size);
    if (!core_ok) return 0;
    if (timing) timing->core = now_seconds() - start;

    if (fuse_output_projection) {
        start = now_seconds();
        if (!ltx_gpu_linear_int8_convrot_mps_bf16(
                gpu, output, core_output,
                output_weight, output_scale, output_bias,
                rows, inner_dim, inner_dim, 256u,
                error, error_size)) return 0;
        if (timing) timing->output_linear = now_seconds() - start;
    } else {
        start = now_seconds();
        if (!ltx_gpu_convrot_bf16(
                gpu, rotated_output, core_output, rows, inner_dim, 256u,
                error, error_size)) return 0;
        if (timing) timing->output_convrot = now_seconds() - start;

        start = now_seconds();
        if (!ltx_gpu_linear_int8_weight_mps_bf16(
                gpu, output, rotated_output,
                output_weight, output_scale, output_bias,
                rows, inner_dim, inner_dim, error, error_size)) return 0;
        if (timing) timing->output_linear = now_seconds() - start;
    }
    return 1;
}

#ifdef LTX_ENABLE_ANE_QKV
static int run_attention_sequence_split(
        ltx_ane_qkv *ane_qkv, ltx_gpu *gpu,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
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
        const ltx_gpu_buffer *cosine,
        const ltx_gpu_buffer *sine,
        ltx_gpu_buffer *query_prefix,
        ltx_gpu_buffer *key_prefix,
        ltx_gpu_buffer *value_prefix,
        ltx_gpu_buffer *query,
        ltx_gpu_buffer *key,
        ltx_gpu_buffer *value,
        ltx_gpu_buffer *gate_logits,
        ltx_gpu_buffer *gate,
        ltx_gpu_buffer *core_output,
        ltx_gpu_buffer *rotated_output,
        uint32_t rows, uint32_t prefix_rows,
        uint32_t heads, uint32_t head_dim,
        attention_sequence_timing *timing,
        char *error, size_t error_size) {
    uint32_t inner_dim = heads * head_dim;
    if (!ltx_ane_qkv_start(
            ane_qkv, gpu, input, rows, prefix_rows,
            error, error_size)) return 0;

    double start = now_seconds();
    if (!ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu, query_prefix, key_prefix, value_prefix, input,
            query_weight, query_scale, query_bias,
            key_weight, key_scale, key_bias,
            value_weight, value_scale, value_bias,
            query_norm, key_norm,
            prefix_rows, inner_dim, inner_dim, 256u, 1e-6f,
            error, error_size)) return 0;
    if (timing) timing->gpu_qkv = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_linear_bf16(
            gpu, gate_logits, input, gate_weight, gate_bias,
            rows, inner_dim, heads, error, error_size) ||
        !ltx_gpu_sigmoid2_bf16(
            gpu, gate, gate_logits, rows * heads,
            error, error_size)) return 0;
    if (timing) timing->gate = now_seconds() - start;

    ltx_ane_qkv_timing qkv_timing = {0};
    if (!ltx_ane_qkv_wait(
            ane_qkv, gpu, query, key, value,
            query_prefix, key_prefix, value_prefix, prefix_rows,
            &qkv_timing, error, error_size)) return 0;
    if (timing) timing->qkv = qkv_timing;

    start = now_seconds();
    if (!ltx_gpu_self_attention_core_mps_bf16(
            gpu, core_output, query, key, value, cosine, sine, gate,
            rows, heads, head_dim, 1.0f / sqrtf((float)head_dim),
            error, error_size)) return 0;
    if (timing) timing->core = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_convrot_bf16(
            gpu, rotated_output, core_output, rows, inner_dim, 256u,
            error, error_size)) return 0;
    if (timing) timing->output_convrot = now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, output, rotated_output,
            output_weight, output_scale, output_bias,
            rows, inner_dim, inner_dim, error, error_size)) return 0;
    if (timing) timing->output_linear = now_seconds() - start;
    return 1;
}
#endif

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s CHECKPOINT ATTENTION_PREFIX [rows] [warmup] [iterations]\n"
        "Set LTX_ATTENTION_BACKEND=fused, staged, or sol.\n"
        "Set LTX_QKV_BACKEND=separate or packed (experimental).\n"
        "Set LTX_OUTPUT_BACKEND=staged or fused (experimental).\n"
        "Set LTX_PREP_BATCH=1 to batch QKV and gate submission.\n"
        "Set LTX_ATTENTION_BATCH=1 to wait once after the full attention.\n"
        "LTX_ATTENTION_SPLIT_HEADS=2 or 4 runs an exact shared-QKV split.\n"
#ifdef LTX_ENABLE_ANE_QKV
        "LTX_ANE_QKV_MANIFEST enables exact token-sequence GPU+ANE QKV.\n"
        "LTX_ANE_QKV_VARIANT selects its artifact (default int8_pc).\n"
#endif
        "LTX_SOL_TAU controls Sol sparsity (default 1.0).\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    const char *checkpoint_path = argv[1];
    const char *prefix = argv[2];
    uint32_t rows = argc > 3 ? parse_u32(argv[3], "rows") : 1001u;
    uint32_t warmup = argc > 4 ? parse_u32(argv[4], "warmup") : 2u;
    uint32_t iterations = argc > 5 ? parse_u32(argv[5], "iterations") : 5u;
    const char *backend_value = getenv("LTX_ATTENTION_BACKEND");
    attention_backend backend = ATTENTION_BACKEND_FUSED;
    if (backend_value && !strcmp(backend_value, "staged"))
        backend = ATTENTION_BACKEND_STAGED;
    else if (backend_value && !strcmp(backend_value, "sol"))
        backend = ATTENTION_BACKEND_SOL;
    else if (backend_value && strcmp(backend_value, "fused")) {
        fprintf(stderr, "bench_attention: invalid backend: %s\n",
                backend_value);
        return 2;
    }
    const char *qkv_backend_value = getenv("LTX_QKV_BACKEND");
    int packed_qkv = qkv_backend_value &&
        strcmp(qkv_backend_value, "packed") == 0;
    if (qkv_backend_value && strcmp(qkv_backend_value, "separate") != 0 &&
        strcmp(qkv_backend_value, "packed") != 0) {
        fprintf(stderr, "bench_attention: invalid QKV backend: %s\n",
                qkv_backend_value);
        return 2;
    }
    const char *output_backend_value = getenv("LTX_OUTPUT_BACKEND");
    int fused_output_projection = output_backend_value &&
        strcmp(output_backend_value, "fused") == 0;
    if (output_backend_value &&
        strcmp(output_backend_value, "staged") != 0 &&
        strcmp(output_backend_value, "fused") != 0) {
        fprintf(stderr, "bench_attention: invalid output backend: %s\n",
                output_backend_value);
        return 2;
    }
    const char *prep_batch_value = getenv("LTX_PREP_BATCH");
    int batch_qkv_gate = prep_batch_value &&
        strcmp(prep_batch_value, "0") != 0;
    const char *attention_batch_value = getenv("LTX_ATTENTION_BATCH");
    int batch_attention = attention_batch_value &&
        strcmp(attention_batch_value, "0") != 0;
    if (batch_attention && batch_qkv_gate) {
        fputs("bench_attention: full attention batch includes prep batch\n",
              stderr);
        return 2;
    }
    const char *tau_value = getenv("LTX_SOL_TAU");
    float sol_tau = tau_value ? parse_f32(tau_value, "LTX_SOL_TAU") : 1.0f;
    const char *split_value = getenv("LTX_ATTENTION_SPLIT_HEADS");
    uint32_t split_heads = split_value ?
        parse_u32(split_value, "LTX_ATTENTION_SPLIT_HEADS") : 0u;
    if (packed_qkv && split_heads) {
        fputs("bench_attention: packed QKV and head splitting cannot be "
              "benchmarked together\n", stderr);
        return 2;
    }
#ifdef LTX_ENABLE_ANE_QKV
    const char *ane_qkv_manifest = getenv("LTX_ANE_QKV_MANIFEST");
    const char *ane_qkv_variant = getenv("LTX_ANE_QKV_VARIANT");
    if (!ane_qkv_variant || !ane_qkv_variant[0])
        ane_qkv_variant = "int8_pc";
    if (ane_qkv_manifest && ane_qkv_manifest[0] &&
        (backend != ATTENTION_BACKEND_STAGED || split_heads || packed_qkv)) {
        fputs("bench_attention: ANE QKV requires staged backend without "
              "head splitting\n", stderr);
        return 2;
    }
#endif
    char error[1024] = {0};
    int result = 1;
    attention_shard split_shard[2] = {{0}};
    int8_t *split_output_weight_host[2] = {NULL, NULL};
    double *split_timings = NULL;

    ltx_st_header header;
    if (!ltx_st_read_header(checkpoint_path, &header, error, sizeof(error))) {
        fprintf(stderr, "bench_attention: %s\n", error);
        return 1;
    }
    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, sizeof(error))) {
        fprintf(stderr, "bench_attention: %s\n", error);
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
        fprintf(stderr, "bench_attention: resolve projections: %s\n", error);
        goto cleanup_mapping;
    }
    uint32_t inner_dim = query_info.output_dim;
    const ltx_st_tensor *gate_weight_tensor = resolve_tensor(
        &header, prefix, "to_gate_logits.weight", LTX_DTYPE_BF16,
        2u, 32u, inner_dim, error, sizeof(error));
    if (!gate_weight_tensor) {
        fprintf(stderr, "bench_attention: %s\n", error);
        goto cleanup_mapping;
    }
    uint32_t heads = (uint32_t)gate_weight_tensor->shape[0];
    if (!heads || inner_dim % heads) {
        fputs("bench_attention: invalid head geometry\n", stderr);
        goto cleanup_mapping;
    }
    uint32_t head_dim = inner_dim / heads;
    if (split_heads &&
        (split_heads >= heads || split_heads % 2u ||
         (heads - split_heads) % 2u)) {
        fprintf(stderr,
                "bench_attention: split heads must be an even proper shard "
                "with an even GPU complement\n");
        goto cleanup_mapping;
    }
    const ltx_st_tensor *gate_bias_tensor = resolve_tensor(
        &header, prefix, "to_gate_logits.bias", LTX_DTYPE_BF16,
        1u, heads, 0u, error, sizeof(error));
    const ltx_st_tensor *query_norm_tensor = resolve_tensor(
        &header, prefix, "q_norm.weight", LTX_DTYPE_BF16,
        1u, inner_dim, 0u, error, sizeof(error));
    const ltx_st_tensor *key_norm_tensor = resolve_tensor(
        &header, prefix, "k_norm.weight", LTX_DTYPE_BF16,
        1u, inner_dim, 0u, error, sizeof(error));
    if (!gate_bias_tensor || !query_norm_tensor || !key_norm_tensor ||
        query_info.input_dim != inner_dim ||
        !validate_projection(&query_info, inner_dim, inner_dim) ||
        !validate_projection(&key_info, inner_dim, inner_dim) ||
        !validate_projection(&value_info, inner_dim, inner_dim) ||
        !validate_projection(&output_info, inner_dim, inner_dim)) {
        fprintf(stderr, "bench_attention: incompatible attention: %s\n",
                error[0] ? error : "requires square ConvRot INT8 projections");
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
        fprintf(stderr, "bench_attention: map projections: %s\n", error);
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
        fprintf(stderr, "bench_attention: map small weights: %s\n", error);
        goto cleanup_mapping;
    }

    uint64_t tensor_elements = (uint64_t)rows * inner_dim;
    uint64_t frequency_elements =
        (uint64_t)heads * rows * (head_dim / 2u);
    uint32_t sol_blocks = (rows + 63u) / 64u;
    uint64_t sol_summary_elements =
        (uint64_t)heads * sol_blocks * head_dim;
    uint64_t sol_threshold_elements = (uint64_t)heads * sol_blocks;
    uint64_t sol_route_elements = sol_threshold_elements * sol_blocks;
    if (tensor_elements > SIZE_MAX / sizeof(uint16_t) ||
        frequency_elements > SIZE_MAX / sizeof(uint16_t) ||
        sol_summary_elements > SIZE_MAX / sizeof(float) ||
        sol_route_elements > SIZE_MAX / sizeof(float)) {
        fputs("bench_attention: allocation size overflow\n", stderr);
        goto cleanup_mapping;
    }
    size_t tensor_bytes = (size_t)tensor_elements * sizeof(uint16_t);
    size_t frequency_bytes =
        (size_t)frequency_elements * sizeof(uint16_t);
    size_t gate_bytes = (size_t)rows * heads * sizeof(uint16_t);
    size_t sol_query_summary_bytes =
        (size_t)sol_summary_elements * sizeof(float);
    size_t sol_bf16_summary_bytes =
        (size_t)sol_summary_elements * sizeof(uint16_t);
    size_t sol_threshold_bytes =
        (size_t)sol_threshold_elements * sizeof(float);
    size_t sol_route_bytes = (size_t)sol_route_elements * sizeof(float);
    size_t packed_qkv_weight_bytes = 0;
    size_t packed_qkv_scale_bytes = 0;
    size_t packed_qkv_bias_bytes = 0;
    int8_t *host_packed_qkv_weight = NULL;
    float *host_packed_qkv_scale = NULL;
    uint16_t *host_packed_qkv_bias = NULL;
    if (packed_qkv) {
        if (query_mapped.weight_bytes != key_mapped.weight_bytes ||
            query_mapped.weight_bytes != value_mapped.weight_bytes ||
            query_mapped.scale_bytes != key_mapped.scale_bytes ||
            query_mapped.scale_bytes != value_mapped.scale_bytes ||
            query_mapped.weight_bytes > SIZE_MAX / 3u ||
            query_mapped.scale_bytes > SIZE_MAX / 3u ||
            (size_t)inner_dim > SIZE_MAX / (3u * sizeof(uint16_t))) {
            fputs("bench_attention: packed QKV size mismatch/overflow\n",
                  stderr);
            goto cleanup_mapping;
        }
        packed_qkv_weight_bytes = query_mapped.weight_bytes * 3u;
        packed_qkv_scale_bytes = query_mapped.scale_bytes * 3u;
        packed_qkv_bias_bytes =
            (size_t)inner_dim * 3u * sizeof(uint16_t);
        host_packed_qkv_weight = malloc(packed_qkv_weight_bytes);
        host_packed_qkv_scale = malloc(packed_qkv_scale_bytes);
        host_packed_qkv_bias = calloc(1u, packed_qkv_bias_bytes);
        if (host_packed_qkv_weight && host_packed_qkv_scale &&
            host_packed_qkv_bias) {
            memcpy(host_packed_qkv_weight, query_mapped.weight,
                   query_mapped.weight_bytes);
            memcpy(host_packed_qkv_weight + query_mapped.weight_bytes,
                   key_mapped.weight, key_mapped.weight_bytes);
            memcpy(host_packed_qkv_weight + query_mapped.weight_bytes * 2u,
                   value_mapped.weight, value_mapped.weight_bytes);
            memcpy(host_packed_qkv_scale, query_mapped.scale,
                   query_mapped.scale_bytes);
            memcpy((uint8_t *)host_packed_qkv_scale +
                       query_mapped.scale_bytes,
                   key_mapped.scale, key_mapped.scale_bytes);
            memcpy((uint8_t *)host_packed_qkv_scale +
                       query_mapped.scale_bytes * 2u,
                   value_mapped.scale, value_mapped.scale_bytes);
            if (query_mapped.bias)
                memcpy(host_packed_qkv_bias, query_mapped.bias,
                       query_mapped.bias_bytes);
            if (key_mapped.bias)
                memcpy(host_packed_qkv_bias + inner_dim, key_mapped.bias,
                       key_mapped.bias_bytes);
            if (value_mapped.bias)
                memcpy(host_packed_qkv_bias + inner_dim * 2u,
                       value_mapped.bias, value_mapped.bias_bytes);
        }
    }
    uint16_t *host_input = malloc(tensor_bytes);
    uint16_t *host_cosine = malloc(frequency_bytes);
    uint16_t *host_sine = malloc(frequency_bytes);
    float *host_positions = NULL;
    uint16_t *reference_output = malloc(tensor_bytes);
    uint16_t *candidate_output = malloc(tensor_bytes);
    float *host_routes = backend == ATTENTION_BACKEND_SOL ?
        malloc(sol_route_bytes) : NULL;
    double *timings = calloc(iterations, sizeof(double));
    if (split_heads) {
        split_shard[0].head_start = 0u;
        split_shard[0].heads = heads - split_heads;
        split_shard[0].inner_dim = split_shard[0].heads * head_dim;
        split_shard[1].head_start = heads - split_heads;
        split_shard[1].heads = split_heads;
        split_shard[1].inner_dim = split_heads * head_dim;
        split_output_weight_host[0] = slice_output_weight(
            &output_mapped, inner_dim, inner_dim, 0u,
            split_shard[0].inner_dim);
        split_output_weight_host[1] = slice_output_weight(
            &output_mapped, inner_dim, inner_dim,
            split_shard[1].head_start * head_dim,
            split_shard[1].inner_dim);
        split_timings = calloc(iterations, sizeof(*split_timings));
    }
    if (!host_input || !host_cosine || !host_sine ||
        !reference_output || !candidate_output || !timings ||
        (packed_qkv &&
         (!host_packed_qkv_weight || !host_packed_qkv_scale ||
          !host_packed_qkv_bias)) ||
        (backend == ATTENTION_BACKEND_SOL && !host_routes) ||
        (split_heads &&
         (!split_output_weight_host[0] || !split_output_weight_host[1] ||
          !split_timings))) {
        fputs("bench_attention: out of host memory\n", stderr);
        goto cleanup_host;
    }
    for (uint64_t index = 0; index < tensor_elements; index++)
        host_input[index] = f32_to_bf16(
            sinf((float)(index % 8192u) * 0.013f) * 0.75f +
            cosf((float)(index % 4096u) * 0.007f) * 0.25f);
    const char *rope_source = "synthetic";
    uint32_t latent_frames = 0;
    uint32_t latent_height = 0;
    uint32_t latent_width = 0;
    if (rows == 1001u) {
        latent_frames = 13u;
        latent_height = 7u;
        latent_width = 11u;
    } else if (rows == 4004u) {
        latent_frames = 13u;
        latent_height = 14u;
        latent_width = 22u;
    }
    if (latent_frames) {
        host_positions = malloc((size_t)rows * 3u * sizeof(float));
        float max_positions[3] = {20.0f, 2048.0f, 2048.0f};
        if (!host_positions || !ltx_compute_video_positions(
                host_positions, (uint64_t)rows * 3u,
                latent_frames, latent_height, latent_width, 24.0f,
                error, sizeof(error)) ||
            !ltx_compute_rope_split_bf16(
                host_cosine, host_sine, frequency_elements,
                host_positions, rows, 3u, heads, head_dim,
                10000.0, max_positions, 1, error, sizeof(error))) {
            fprintf(stderr, "bench_attention: real RoPE generation: %s\n",
                    error[0] ? error : "out of memory");
            goto cleanup_host;
        }
        rope_source = "ltx-video-f64-grid";
    } else {
        for (uint64_t index = 0; index < frequency_elements; index++) {
            float angle = (float)(index % 4093u) * 0.0007f;
            host_cosine[index] = f32_to_bf16(cosf(angle));
            host_sine[index] = f32_to_bf16(sinf(angle));
        }
    }

    double setup_start = now_seconds();
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    ltx_gpu_buffer *input = NULL;
    ltx_gpu_buffer *query_weight = NULL;
    ltx_gpu_buffer *query_scale = NULL;
    ltx_gpu_buffer *query_bias = NULL;
    ltx_gpu_buffer *key_weight = NULL;
    ltx_gpu_buffer *key_scale = NULL;
    ltx_gpu_buffer *key_bias = NULL;
    ltx_gpu_buffer *value_weight = NULL;
    ltx_gpu_buffer *value_scale = NULL;
    ltx_gpu_buffer *value_bias = NULL;
    ltx_gpu_buffer *packed_qkv_weight = NULL;
    ltx_gpu_buffer *packed_qkv_scale = NULL;
    ltx_gpu_buffer *packed_qkv_bias = NULL;
    ltx_gpu_buffer *output_weight = NULL;
    ltx_gpu_buffer *output_scale = NULL;
    ltx_gpu_buffer *output_bias = NULL;
    ltx_gpu_buffer *query_norm = NULL;
    ltx_gpu_buffer *key_norm = NULL;
    ltx_gpu_buffer *gate_weight = NULL;
    ltx_gpu_buffer *gate_bias = NULL;
    ltx_gpu_buffer *cosine = NULL;
    ltx_gpu_buffer *sine = NULL;
    ltx_gpu_buffer *query = NULL;
    ltx_gpu_buffer *key = NULL;
    ltx_gpu_buffer *value = NULL;
    ltx_gpu_buffer *gate_logits = NULL;
    ltx_gpu_buffer *gate = NULL;
    ltx_gpu_buffer *core_output = NULL;
    ltx_gpu_buffer *rotated_output = NULL;
    ltx_gpu_buffer *output = NULL;
    ltx_gpu_buffer *packed_query = NULL;
    ltx_gpu_buffer *packed_key = NULL;
    ltx_gpu_buffer *packed_value = NULL;
    ltx_gpu_buffer *packed_output = NULL;
    ltx_gpu_buffer *query_centroids = NULL;
    ltx_gpu_buffer *key_centroids = NULL;
    ltx_gpu_buffer *value_sums = NULL;
    ltx_gpu_buffer *thresholds = NULL;
    ltx_gpu_buffer *routes = NULL;
#ifdef LTX_ENABLE_ANE_QKV
    ltx_ane_qkv *ane_qkv = NULL;
    uint32_t ane_prefix_rows = 0u;
    ltx_gpu_buffer *query_prefix = NULL;
    ltx_gpu_buffer *key_prefix = NULL;
    ltx_gpu_buffer *value_prefix = NULL;
#endif
#define LTX_UPLOAD_LINEAR(NAME, MAPPED) \
    NAME##_weight = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).weight, (MAPPED).weight_bytes, error, sizeof(error)); \
    NAME##_scale = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).scale, (MAPPED).scale_bytes, error, sizeof(error)); \
    if ((MAPPED).bias) NAME##_bias = ltx_gpu_buffer_new_copy( \
        gpu, (MAPPED).bias, (MAPPED).bias_bytes, error, sizeof(error))
    if (gpu) {
        input = ltx_gpu_buffer_new_copy(
            gpu, host_input, tensor_bytes, error, sizeof(error));
        LTX_UPLOAD_LINEAR(query, query_mapped);
        LTX_UPLOAD_LINEAR(key, key_mapped);
        LTX_UPLOAD_LINEAR(value, value_mapped);
        LTX_UPLOAD_LINEAR(output, output_mapped);
        if (packed_qkv) {
            packed_qkv_weight = ltx_gpu_buffer_new_copy(
                gpu, host_packed_qkv_weight, packed_qkv_weight_bytes,
                error, sizeof(error));
            packed_qkv_scale = ltx_gpu_buffer_new_copy(
                gpu, host_packed_qkv_scale, packed_qkv_scale_bytes,
                error, sizeof(error));
            packed_qkv_bias = ltx_gpu_buffer_new_copy(
                gpu, host_packed_qkv_bias, packed_qkv_bias_bytes,
                error, sizeof(error));
        }
        query_norm = ltx_gpu_buffer_new_copy(
            gpu, query_norm_data, query_norm_bytes, error, sizeof(error));
        key_norm = ltx_gpu_buffer_new_copy(
            gpu, key_norm_data, key_norm_bytes, error, sizeof(error));
        gate_weight = ltx_gpu_buffer_new_copy(
            gpu, gate_weight_data, gate_weight_bytes, error, sizeof(error));
        gate_bias = ltx_gpu_buffer_new_copy(
            gpu, gate_bias_data, gate_bias_bytes, error, sizeof(error));
        cosine = ltx_gpu_buffer_new_copy(
            gpu, host_cosine, frequency_bytes, error, sizeof(error));
        sine = ltx_gpu_buffer_new_copy(
            gpu, host_sine, frequency_bytes, error, sizeof(error));
        query = ltx_gpu_buffer_new(gpu, tensor_bytes, error, sizeof(error));
        key = ltx_gpu_buffer_new(gpu, tensor_bytes, error, sizeof(error));
        value = ltx_gpu_buffer_new(gpu, tensor_bytes, error, sizeof(error));
        gate_logits = ltx_gpu_buffer_new(
            gpu, gate_bytes, error, sizeof(error));
        gate = ltx_gpu_buffer_new(gpu, gate_bytes, error, sizeof(error));
        core_output = ltx_gpu_buffer_new(
            gpu, tensor_bytes, error, sizeof(error));
        rotated_output = ltx_gpu_buffer_new(
            gpu, tensor_bytes, error, sizeof(error));
        output = ltx_gpu_buffer_new(gpu, tensor_bytes, error, sizeof(error));
#ifdef LTX_ENABLE_ANE_QKV
        if (ane_qkv_manifest && ane_qkv_manifest[0]) {
            ane_qkv = ltx_ane_qkv_create(
                gpu, ane_qkv_manifest, ane_qkv_variant,
                error, sizeof(error));
            const ltx_ane_qkv_shape *ane_shape =
                ltx_ane_qkv_get_shape(ane_qkv);
            if (ane_shape && ane_shape->hidden == inner_dim &&
                ane_shape->rows < rows) {
                ane_prefix_rows = rows - ane_shape->rows;
                size_t prefix_bytes =
                    (size_t)ane_prefix_rows * inner_dim * sizeof(uint16_t);
                query_prefix = ltx_gpu_buffer_new(
                    gpu, prefix_bytes, error, sizeof(error));
                key_prefix = ltx_gpu_buffer_new(
                    gpu, prefix_bytes, error, sizeof(error));
                value_prefix = ltx_gpu_buffer_new(
                    gpu, prefix_bytes, error, sizeof(error));
            } else if (ane_qkv && !error[0]) {
                snprintf(error, sizeof(error),
                         "ANE QKV rows/hidden do not match attention");
            }
        }
#endif
        if (split_heads) {
            for (uint32_t shard_index = 0u; shard_index < 2u;
                 shard_index++) {
                attention_shard *shard = &split_shard[shard_index];
                size_t compact_bytes =
                    (size_t)rows * shard->inner_dim * sizeof(uint16_t);
                size_t shard_gate_bytes =
                    (size_t)rows * shard->heads * sizeof(uint16_t);
                size_t shard_frequency_elements =
                    (size_t)shard->heads * rows * (head_dim / 2u);
                size_t shard_frequency_bytes =
                    shard_frequency_elements * sizeof(uint16_t);
                size_t frequency_offset =
                    (size_t)shard->head_start * rows * (head_dim / 2u);
                size_t shard_weight_bytes =
                    (size_t)inner_dim * shard->inner_dim;
                shard->query = ltx_gpu_buffer_new(
                    gpu, compact_bytes, error, sizeof(error));
                shard->key = ltx_gpu_buffer_new(
                    gpu, compact_bytes, error, sizeof(error));
                shard->value = ltx_gpu_buffer_new(
                    gpu, compact_bytes, error, sizeof(error));
                shard->gate = ltx_gpu_buffer_new(
                    gpu, shard_gate_bytes, error, sizeof(error));
                shard->cosine = ltx_gpu_buffer_new_copy(
                    gpu, host_cosine + frequency_offset,
                    shard_frequency_bytes, error, sizeof(error));
                shard->sine = ltx_gpu_buffer_new_copy(
                    gpu, host_sine + frequency_offset,
                    shard_frequency_bytes, error, sizeof(error));
                shard->core_output = ltx_gpu_buffer_new(
                    gpu, compact_bytes, error, sizeof(error));
                shard->rotated_output = ltx_gpu_buffer_new(
                    gpu, compact_bytes, error, sizeof(error));
                shard->output_weight = ltx_gpu_buffer_new_copy(
                    gpu, split_output_weight_host[shard_index],
                    shard_weight_bytes, error, sizeof(error));
                shard->partial_output = ltx_gpu_buffer_new(
                    gpu, tensor_bytes, error, sizeof(error));
            }
        }
        if (backend == ATTENTION_BACKEND_SOL) {
            packed_query = ltx_gpu_buffer_new(
                gpu, tensor_bytes, error, sizeof(error));
            packed_key = ltx_gpu_buffer_new(
                gpu, tensor_bytes, error, sizeof(error));
            packed_value = ltx_gpu_buffer_new(
                gpu, tensor_bytes, error, sizeof(error));
            packed_output = ltx_gpu_buffer_new(
                gpu, tensor_bytes, error, sizeof(error));
            query_centroids = ltx_gpu_buffer_new(
                gpu, sol_query_summary_bytes, error, sizeof(error));
            key_centroids = ltx_gpu_buffer_new(
                gpu, sol_bf16_summary_bytes, error, sizeof(error));
            value_sums = ltx_gpu_buffer_new(
                gpu, sol_bf16_summary_bytes, error, sizeof(error));
            thresholds = ltx_gpu_buffer_new(
                gpu, sol_threshold_bytes, error, sizeof(error));
            routes = ltx_gpu_buffer_new(
                gpu, sol_route_bytes, error, sizeof(error));
        }
    }
#undef LTX_UPLOAD_LINEAR
    double setup_seconds = now_seconds() - setup_start;
    if (!gpu || !input || !query_weight || !query_scale ||
        !key_weight || !key_scale || !value_weight || !value_scale ||
        !output_weight || !output_scale || !query_norm || !key_norm ||
        !gate_weight || !gate_bias || !cosine || !sine ||
        !query || !key || !value || !gate_logits || !gate ||
        !core_output || !rotated_output || !output ||
        (query_mapped.bias && !query_bias) ||
        (key_mapped.bias && !key_bias) ||
        (value_mapped.bias && !value_bias) ||
        (output_mapped.bias && !output_bias) ||
        (packed_qkv &&
         (!packed_qkv_weight || !packed_qkv_scale || !packed_qkv_bias)) ||
#ifdef LTX_ENABLE_ANE_QKV
        ((ane_qkv_manifest && ane_qkv_manifest[0]) &&
         (!ane_qkv || !ane_prefix_rows || !query_prefix ||
          !key_prefix || !value_prefix)) ||
#endif
        (split_heads &&
         (!attention_shard_ready(&split_shard[0]) ||
          !attention_shard_ready(&split_shard[1]))) ||
        (backend == ATTENTION_BACKEND_SOL &&
         (!packed_query || !packed_key || !packed_value || !packed_output ||
          !query_centroids || !key_centroids || !value_sums ||
          !thresholds || !routes))) {
        fprintf(stderr, "bench_attention: GPU setup failed: %s\n", error);
        goto cleanup_gpu;
    }

#define LTX_RUN(BACKEND, PACKED_QKV, FUSED_OUTPUT, PREP_BATCH, TIMING) \
    run_attention( \
    (BACKEND), gpu, output, input, \
    query_weight, query_scale, query_bias, \
    key_weight, key_scale, key_bias, \
    value_weight, value_scale, value_bias, \
    packed_qkv_weight, packed_qkv_scale, packed_qkv_bias, \
    query_norm, key_norm, gate_weight, gate_bias, \
    output_weight, output_scale, output_bias, cosine, sine, \
    query, key, value, gate_logits, gate, core_output, rotated_output, \
    packed_query, packed_key, packed_value, packed_output, \
    query_centroids, key_centroids, value_sums, thresholds, routes, \
    rows, heads, head_dim, sol_tau, (PACKED_QKV), (FUSED_OUTPUT), \
    (PREP_BATCH), \
    (TIMING), error, sizeof(error))
    if (!LTX_RUN(ATTENTION_BACKEND_STAGED, 0, 0, 0, NULL) ||
        !ltx_gpu_buffer_read(output, reference_output, tensor_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_attention: staged reference failed: %s\n",
                error);
        goto cleanup_gpu;
    }
    for (uint32_t index = 0; index < warmup; index++) {
        int warmup_ok = 0;
#ifdef LTX_ENABLE_ANE_QKV
        if (ane_qkv)
            warmup_ok = run_attention_sequence_split(
                ane_qkv, gpu, output, input,
                query_weight, query_scale, query_bias,
                key_weight, key_scale, key_bias,
                value_weight, value_scale, value_bias,
                query_norm, key_norm, gate_weight, gate_bias,
                output_weight, output_scale, output_bias, cosine, sine,
                query_prefix, key_prefix, value_prefix,
                query, key, value, gate_logits, gate,
                core_output, rotated_output,
                rows, ane_prefix_rows, heads, head_dim,
                NULL, error, sizeof(error));
        else
#endif
        if (batch_attention) {
            warmup_ok = ltx_gpu_batch_begin(
                gpu, error, sizeof(error));
            if (warmup_ok)
                warmup_ok = LTX_RUN(
                    backend, packed_qkv, fused_output_projection, 0, NULL);
            char batch_error[1024] = {0};
            int batch_ok = ltx_gpu_batch_end(
                gpu, batch_error, sizeof(batch_error));
            if (!batch_ok && warmup_ok)
                snprintf(error, sizeof(error), "%s", batch_error[0] ?
                         batch_error : "attention batch failed");
            warmup_ok = warmup_ok && batch_ok;
        } else {
            warmup_ok = LTX_RUN(
                backend, packed_qkv, fused_output_projection,
                batch_qkv_gate, NULL);
        }
        if (!warmup_ok) {
            fprintf(stderr, "bench_attention: warmup failed: %s\n", error);
            goto cleanup_gpu;
        }
    }
    attention_timing detail_sum = {0};
#ifdef LTX_ENABLE_ANE_QKV
    attention_sequence_timing sequence_sum = {0};
#endif
    for (uint32_t index = 0; index < iterations; index++) {
        attention_timing detail = {0};
        double start = now_seconds();
        int iteration_ok = 0;
#ifdef LTX_ENABLE_ANE_QKV
        attention_sequence_timing sequence_detail = {0};
        if (ane_qkv)
            iteration_ok = run_attention_sequence_split(
                ane_qkv, gpu, output, input,
                query_weight, query_scale, query_bias,
                key_weight, key_scale, key_bias,
                value_weight, value_scale, value_bias,
                query_norm, key_norm, gate_weight, gate_bias,
                output_weight, output_scale, output_bias, cosine, sine,
                query_prefix, key_prefix, value_prefix,
                query, key, value, gate_logits, gate,
                core_output, rotated_output,
                rows, ane_prefix_rows, heads, head_dim,
                &sequence_detail, error, sizeof(error));
        else
#endif
        if (batch_attention) {
            iteration_ok = ltx_gpu_batch_begin(
                gpu, error, sizeof(error));
            if (iteration_ok)
                iteration_ok = LTX_RUN(
                    backend, packed_qkv, fused_output_projection,
                    0, &detail);
            char batch_error[1024] = {0};
            int batch_ok = ltx_gpu_batch_end(
                gpu, batch_error, sizeof(batch_error));
            if (!batch_ok && iteration_ok)
                snprintf(error, sizeof(error), "%s", batch_error[0] ?
                         batch_error : "attention batch failed");
            iteration_ok = iteration_ok && batch_ok;
        } else {
            iteration_ok = LTX_RUN(
                backend, packed_qkv, fused_output_projection,
                batch_qkv_gate, &detail);
        }
        if (!iteration_ok) {
            fprintf(stderr, "bench_attention: iteration failed: %s\n", error);
            goto cleanup_gpu;
        }
        timings[index] = now_seconds() - start;
#ifdef LTX_ENABLE_ANE_QKV
        if (ane_qkv) {
            sequence_sum.qkv.pack_ms += sequence_detail.qkv.pack_ms;
            sequence_sum.qkv.ane_ms += sequence_detail.qkv.ane_ms;
            sequence_sum.qkv.unpack_ms += sequence_detail.qkv.unpack_ms;
            sequence_sum.qkv.total_ms += sequence_detail.qkv.total_ms;
            sequence_sum.gpu_qkv += sequence_detail.gpu_qkv;
            sequence_sum.gate += sequence_detail.gate;
            sequence_sum.core += sequence_detail.core;
            sequence_sum.output_convrot +=
                sequence_detail.output_convrot;
            sequence_sum.output_linear += sequence_detail.output_linear;
            sequence_sum.qkv.query_output_backing_used +=
                sequence_detail.qkv.query_output_backing_used;
            sequence_sum.qkv.key_output_backing_used +=
                sequence_detail.qkv.key_output_backing_used;
            sequence_sum.qkv.value_output_backing_used +=
                sequence_detail.qkv.value_output_backing_used;
            continue;
        }
#endif
        detail_sum.qkv += detail.qkv;
        detail_sum.gate += detail.gate;
        detail_sum.core += detail.core;
        detail_sum.output_convrot += detail.output_convrot;
        detail_sum.output_linear += detail.output_linear;
    }
#undef LTX_RUN
    if (!ltx_gpu_buffer_read(output, candidate_output, tensor_bytes,
                             error, sizeof(error))) {
        fprintf(stderr, "bench_attention: output read failed: %s\n", error);
        goto cleanup_gpu;
    }
    uint64_t exact_routes = 0;
    if (backend == ATTENTION_BACKEND_SOL) {
        if (!ltx_gpu_buffer_read(routes, host_routes, sol_route_bytes,
                                 error, sizeof(error))) {
            fprintf(stderr, "bench_attention: route read failed: %s\n",
                    error);
            goto cleanup_gpu;
        }
        for (uint64_t index = 0; index < sol_route_elements; index++)
            if (host_routes[index] != 0.0f) exact_routes++;
    }

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite = 0;
    for (uint64_t index = 0; index < tensor_elements; index++) {
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
    const char *backend_name = backend == ATTENTION_BACKEND_FUSED ?
        "fused-full-mpsgraph" :
        (backend == ATTENTION_BACKEND_SOL ? "sol-metal" :
         "staged-mpsgraph");
#ifdef LTX_ENABLE_ANE_QKV
    if (ane_qkv) backend_name = "gpu-ane-sequence-qkv";
#endif
    printf("checkpoint=%s\n", checkpoint_path);
    printf("attention=%s rows=%u heads=%u head_dim=%u backend=%s rope=%s\n",
           prefix, rows, heads, head_dim, backend_name, rope_source);
    printf("qkv_backend=%s\n", packed_qkv ? "packed-3x" : "separate-3x");
    printf("output_backend=%s\n",
           fused_output_projection ? "fused-convrot-linear" : "staged");
    printf("prep_batch=%d\n", batch_qkv_gate);
    printf("attention_batch=%d\n", batch_attention);
    printf("projection_weight_bytes=%zu setup_seconds=%.6f\n",
           query_mapped.weight_bytes + key_mapped.weight_bytes +
           value_mapped.weight_bytes + output_mapped.weight_bytes,
           setup_seconds);
    printf("warmup=%u iterations=%u p50_ms=%.3f p95_ms=%.3f\n",
           warmup, iterations,
           timings[p50_index] * 1000.0, timings[p95_index] * 1000.0);
    if (backend != ATTENTION_BACKEND_FUSED) {
        double divisor = 1000.0 / (double)iterations;
#ifdef LTX_ENABLE_ANE_QKV
        if (ane_qkv) {
            printf("sequence_gpu_rows=%u sequence_ane_rows=%u "
                   "mean_pack_ms=%.3f mean_gpu_qkv_ms=%.3f "
                   "mean_ane_ms=%.3f mean_concat_ms=%.3f "
                   "mean_qkv_wall_ms=%.3f\n",
                   ane_prefix_rows, rows - ane_prefix_rows,
                   sequence_sum.qkv.pack_ms / (double)iterations,
                   sequence_sum.gpu_qkv * divisor,
                   sequence_sum.qkv.ane_ms / (double)iterations,
                   sequence_sum.qkv.unpack_ms / (double)iterations,
                   sequence_sum.qkv.total_ms / (double)iterations);
            printf("mean_gate_ms=%.3f mean_core_ms=%.3f "
                   "mean_output_convrot_ms=%.3f "
                   "mean_output_linear_ms=%.3f "
                   "output_backings=%d/%d/%d of %u\n",
                   sequence_sum.gate * divisor,
                   sequence_sum.core * divisor,
                   sequence_sum.output_convrot * divisor,
                   sequence_sum.output_linear * divisor,
                   sequence_sum.qkv.query_output_backing_used,
                   sequence_sum.qkv.key_output_backing_used,
                   sequence_sum.qkv.value_output_backing_used,
                   iterations);
        } else
#endif
        printf("mean_qkv_norm_ms=%.3f mean_gate_ms=%.3f "
               "mean_core_ms=%.3f mean_output_convrot_ms=%.3f "
               "mean_output_linear_ms=%.3f\n",
               detail_sum.qkv * divisor, detail_sum.gate * divisor,
               detail_sum.core * divisor,
               detail_sum.output_convrot * divisor,
               detail_sum.output_linear * divisor);
    }
    printf("all_vs_staged_rel_l2=%.9g all_vs_staged_cosine=%.9g "
           "max_abs=%.9g nonfinite=%llu\n",
           rel_l2, cosine_similarity, max_abs,
           (unsigned long long)nonfinite);
    if (backend == ATTENTION_BACKEND_SOL)
        printf("sol_tau=%.9g blocks=%u exact_routes=%llu/%llu "
               "exact_fraction=%.6f\n",
               sol_tau, sol_blocks, (unsigned long long)exact_routes,
               (unsigned long long)sol_route_elements,
               (double)exact_routes / (double)sol_route_elements);
    result = nonfinite || cosine_similarity < 0.99 ? 1 : 0;

    if (split_heads) {
#define LTX_RUN_SPLIT(TIMING) run_attention_split( \
        gpu, output, input, \
        query_weight, query_scale, query_bias, \
        key_weight, key_scale, key_bias, \
        value_weight, value_scale, value_bias, \
        query_norm, key_norm, gate_weight, gate_bias, \
        output_scale, output_bias, \
        query, key, value, gate_logits, gate, split_shard, \
        rows, heads, head_dim, (TIMING), error, sizeof(error))
        for (uint32_t index = 0u; index < warmup; index++)
            if (!LTX_RUN_SPLIT(NULL)) {
                fprintf(stderr,
                        "bench_attention: split warmup failed: %s\n", error);
                goto cleanup_gpu;
            }
        attention_split_timing split_sum = {0};
        for (uint32_t index = 0u; index < iterations; index++) {
            attention_split_timing detail = {0};
            double start = now_seconds();
            if (!LTX_RUN_SPLIT(&detail)) {
                fprintf(stderr,
                        "bench_attention: split iteration failed: %s\n",
                        error);
                goto cleanup_gpu;
            }
            split_timings[index] = now_seconds() - start;
            split_sum.qkv += detail.qkv;
            split_sum.gate += detail.gate;
            split_sum.join += detail.join;
            for (uint32_t shard_index = 0u; shard_index < 2u;
                 shard_index++) {
                split_sum.shard[shard_index].gather +=
                    detail.shard[shard_index].gather;
                split_sum.shard[shard_index].core +=
                    detail.shard[shard_index].core;
                split_sum.shard[shard_index].output_convrot +=
                    detail.shard[shard_index].output_convrot;
                split_sum.shard[shard_index].output_linear +=
                    detail.shard[shard_index].output_linear;
            }
        }
#undef LTX_RUN_SPLIT
        if (!ltx_gpu_buffer_read(output, candidate_output, tensor_bytes,
                                 error, sizeof(error))) {
            fprintf(stderr,
                    "bench_attention: split output read failed: %s\n",
                    error);
            goto cleanup_gpu;
        }
        double split_diff2 = 0.0;
        double split_reference2 = 0.0;
        double split_candidate2 = 0.0;
        double split_dot = 0.0;
        double split_max_abs = 0.0;
        uint64_t split_nonfinite = 0u;
        for (uint64_t index = 0u; index < tensor_elements; index++) {
            double reference = bf16_to_f32(reference_output[index]);
            double candidate = bf16_to_f32(candidate_output[index]);
            if (!isfinite(candidate)) split_nonfinite++;
            double difference = candidate - reference;
            split_diff2 += difference * difference;
            split_reference2 += reference * reference;
            split_candidate2 += candidate * candidate;
            split_dot += reference * candidate;
            if (fabs(difference) > split_max_abs)
                split_max_abs = fabs(difference);
        }
        double split_rel_l2 = split_reference2 > 0.0 ?
            sqrt(split_diff2 / split_reference2) : 0.0;
        double split_cosine =
            split_reference2 > 0.0 && split_candidate2 > 0.0 ?
            split_dot / sqrt(split_reference2 * split_candidate2) : 0.0;
        qsort(split_timings, iterations, sizeof(*split_timings),
              compare_double);
        double split_scale = 1000.0 / (double)iterations;
        double shard_total_ms[2] = {0.0, 0.0};
        for (uint32_t shard_index = 0u; shard_index < 2u; shard_index++) {
            attention_shard_timing *shard_timing =
                &split_sum.shard[shard_index];
            shard_total_ms[shard_index] =
                (shard_timing->gather + shard_timing->core +
                 shard_timing->output_convrot +
                 shard_timing->output_linear) * split_scale;
        }
        double exact_gpu_parallel_bound_ms =
            (split_sum.qkv + split_sum.gate + split_sum.join) * split_scale +
            fmax(shard_total_ms[0], shard_total_ms[1]);
        printf("exact_shared_norm_split=%u+%u sequential_p50_ms=%.3f "
               "gpu_shape_parallel_bound_ms=%.3f\n",
               split_shard[0].heads, split_shard[1].heads,
               split_timings[p50_index] * 1000.0,
               exact_gpu_parallel_bound_ms);
        for (uint32_t shard_index = 0u; shard_index < 2u; shard_index++) {
            attention_shard_timing *shard_timing =
                &split_sum.shard[shard_index];
            printf("split_shard=%u heads=%u mean_gather_ms=%.3f "
                   "mean_core_ms=%.3f mean_output_convrot_ms=%.3f "
                   "mean_output_linear_ms=%.3f mean_total_ms=%.3f\n",
                   shard_index, split_shard[shard_index].heads,
                   shard_timing->gather * split_scale,
                   shard_timing->core * split_scale,
                   shard_timing->output_convrot * split_scale,
                   shard_timing->output_linear * split_scale,
                   shard_total_ms[shard_index]);
        }
        printf("split_mean_qkv_norm_ms=%.3f split_mean_gate_ms=%.3f "
               "split_mean_join_ms=%.3f\n",
               split_sum.qkv * split_scale,
               split_sum.gate * split_scale,
               split_sum.join * split_scale);
        printf("split_vs_full_rel_l2=%.9g split_vs_full_cosine=%.9g "
               "max_abs=%.9g nonfinite=%llu\n",
               split_rel_l2, split_cosine, split_max_abs,
               (unsigned long long)split_nonfinite);
        if (split_nonfinite || split_cosine < 0.99) result = 1;
    }

cleanup_gpu:
#ifdef LTX_ENABLE_ANE_QKV
    ltx_ane_qkv_free(ane_qkv);
    ltx_gpu_buffer_free(query_prefix);
    ltx_gpu_buffer_free(key_prefix);
    ltx_gpu_buffer_free(value_prefix);
#endif
    ltx_gpu_buffer_free(input);
    ltx_gpu_buffer_free(query_weight);
    ltx_gpu_buffer_free(query_scale);
    ltx_gpu_buffer_free(query_bias);
    ltx_gpu_buffer_free(key_weight);
    ltx_gpu_buffer_free(key_scale);
    ltx_gpu_buffer_free(key_bias);
    ltx_gpu_buffer_free(value_weight);
    ltx_gpu_buffer_free(value_scale);
    ltx_gpu_buffer_free(value_bias);
    ltx_gpu_buffer_free(packed_qkv_weight);
    ltx_gpu_buffer_free(packed_qkv_scale);
    ltx_gpu_buffer_free(packed_qkv_bias);
    ltx_gpu_buffer_free(output_weight);
    ltx_gpu_buffer_free(output_scale);
    ltx_gpu_buffer_free(output_bias);
    ltx_gpu_buffer_free(query_norm);
    ltx_gpu_buffer_free(key_norm);
    ltx_gpu_buffer_free(gate_weight);
    ltx_gpu_buffer_free(gate_bias);
    ltx_gpu_buffer_free(cosine);
    ltx_gpu_buffer_free(sine);
    ltx_gpu_buffer_free(query);
    ltx_gpu_buffer_free(key);
    ltx_gpu_buffer_free(value);
    ltx_gpu_buffer_free(gate_logits);
    ltx_gpu_buffer_free(gate);
    ltx_gpu_buffer_free(core_output);
    ltx_gpu_buffer_free(rotated_output);
    ltx_gpu_buffer_free(output);
    ltx_gpu_buffer_free(packed_query);
    ltx_gpu_buffer_free(packed_key);
    ltx_gpu_buffer_free(packed_value);
    ltx_gpu_buffer_free(packed_output);
    ltx_gpu_buffer_free(query_centroids);
    ltx_gpu_buffer_free(key_centroids);
    ltx_gpu_buffer_free(value_sums);
    ltx_gpu_buffer_free(thresholds);
    ltx_gpu_buffer_free(routes);
    free_attention_shard(&split_shard[0]);
    free_attention_shard(&split_shard[1]);
    ltx_gpu_free(gpu);
cleanup_host:
    free(host_input);
    free(host_cosine);
    free(host_sine);
    free(host_positions);
    free(reference_output);
    free(candidate_output);
    free(host_routes);
    free(timings);
    free(split_timings);
    free(split_output_weight_host[0]);
    free(split_output_weight_host[1]);
    free(host_packed_qkv_weight);
    free(host_packed_qkv_scale);
    free(host_packed_qkv_bias);
cleanup_mapping:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}
