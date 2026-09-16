#include "ltx_gemma_encoder.h"

#include "ltx_gemma_ane_mlp.h"
#include "ltx_gemma_tokenizer.h"
#include "ltx_gpu.h"
#include "ltx_safetensors.h"
#include "ltx_weights.h"

#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

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
    char *ane_manifest;
    struct gemma_ane_cache_entry *ane_cache;
    uint32_t ane_cache_count;
    uint32_t ane_cache_capacity;
    uint64_t ane_cache_tick;
    uint32_t ane_preload_models_session_total;
    uint32_t ane_preload_workers;
    double ane_preload_seconds_session_total;
    uint32_t ane_preload_rows;
    int ane_preload_attempted;
    ltx_gpu_buffer **resident_weight_cache;
    size_t resident_weight_cache_count;
    uint64_t resident_weight_bytes;
    ltx_gemma_encoder_telemetry telemetry;
};

typedef struct gemma_ane_cache_entry {
    uint32_t layer;
    uint32_t requested_rows;
    uint64_t last_used;
    char path[PATH_MAX];
    ltx_gemma_ane_mlp *mlp;
} gemma_ane_cache_entry;

static double gemma_probe_now(void);

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

static char *gemma_copy_string(const char *value) {
    if (!value || !*value) return NULL;
    size_t bytes = strlen(value) + 1u;
    char *copy = malloc(bytes);
    if (copy) memcpy(copy, value, bytes);
    return copy;
}

static int gemma_regular_file(const char *path) {
    struct stat status = {};
    return path && stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static int gemma_directory(const char *path) {
    struct stat status = {};
    return path && stat(path, &status) == 0 && S_ISDIR(status.st_mode);
}

static uint32_t gemma_ane_cache_capacity(void) {
    const char *configured = getenv("TURBOCIDER_LTX_GEMMA_ANE_CACHE_LAYERS");
    if (configured && *configured) {
        char *end = NULL;
        unsigned long value = strtoul(configured, &end, 10);
        if (end && *end == '\0' && value >= 1u && value <= GEMMA_LAYERS)
            return (uint32_t)value;
    }
    return 8u;
}

static uint32_t gemma_ane_max_padding_percent(void) {
    uint32_t maximum_padding_percent = 0u;
    const char *configured = getenv(
        "TURBOCIDER_LTX_GEMMA_ANE_MAX_PADDING_PERCENT");
    if (configured && *configured) {
        char *end = NULL;
        unsigned long value = strtoul(configured, &end, 10);
        if (end && *end == '\0' && value <= 1000u)
            maximum_padding_percent = (uint32_t)value;
    }
    return maximum_padding_percent;
}

static int gemma_ane_padding_allowed(uint32_t rows,
                                     uint32_t execution_rows) {
    if (!rows || execution_rows < rows) return 0;
    return (uint64_t)(execution_rows - rows) * 100u <=
           (uint64_t)rows * gemma_ane_max_padding_percent();
}

static void gemma_ane_cache_clear(ltx_gemma_encoder *encoder) {
    if (!encoder || !encoder->ane_cache) return;
    for (uint32_t index = 0; index < encoder->ane_cache_count; index++)
        ltx_gemma_ane_mlp_free(encoder->ane_cache[index].mlp);
    free(encoder->ane_cache);
    encoder->ane_cache = NULL;
    encoder->ane_cache_count = 0;
    encoder->ane_cache_capacity = 0;
}

static ltx_gemma_ane_mlp *gemma_ane_cache_put(
        ltx_gemma_encoder *encoder, uint32_t layer, uint32_t rows,
        const char *path, ltx_gemma_ane_mlp *mlp,
        char *error, size_t error_size) {
    if (!encoder || !path || !mlp)
        return (gemma_fail(error, error_size,
                           "invalid Gemma ANE cache insertion"), NULL);
    if (!encoder->ane_cache_capacity)
        encoder->ane_cache_capacity = gemma_ane_cache_capacity();
    if (!encoder->ane_cache) {
        encoder->ane_cache = calloc(encoder->ane_cache_capacity,
                                    sizeof(*encoder->ane_cache));
        if (!encoder->ane_cache)
            return (gemma_fail(error, error_size,
                               "out of memory for Gemma ANE cache"), NULL);
    }
    for (uint32_t index = 0; index < encoder->ane_cache_count; index++) {
        gemma_ane_cache_entry *entry = &encoder->ane_cache[index];
        if (entry->layer == layer && entry->requested_rows == rows &&
            strcmp(entry->path, path) == 0) {
            ltx_gemma_ane_mlp_free(mlp);
            entry->last_used = ++encoder->ane_cache_tick;
            return entry->mlp;
        }
    }
    if (encoder->ane_cache_count == encoder->ane_cache_capacity) {
        uint64_t oldest = UINT64_MAX;
        uint32_t slot = 0;
        for (uint32_t index = 0; index < encoder->ane_cache_count; index++)
            if (encoder->ane_cache[index].last_used < oldest) {
                oldest = encoder->ane_cache[index].last_used;
                slot = index;
            }
        ltx_gemma_ane_mlp_free(encoder->ane_cache[slot].mlp);
        encoder->ane_cache_count--;
        if (slot != encoder->ane_cache_count)
            encoder->ane_cache[slot] =
                encoder->ane_cache[encoder->ane_cache_count];
        memset(&encoder->ane_cache[encoder->ane_cache_count], 0,
               sizeof(*encoder->ane_cache));
    }
    gemma_ane_cache_entry *entry =
        &encoder->ane_cache[encoder->ane_cache_count++];
    entry->mlp = mlp;
    entry->layer = layer;
    entry->requested_rows = rows;
    entry->last_used = ++encoder->ane_cache_tick;
    snprintf(entry->path, sizeof(entry->path), "%s", path);
    return entry->mlp;
}

static ltx_gemma_ane_mlp *gemma_ane_cache_get(
        ltx_gemma_encoder *encoder, uint32_t layer, uint32_t rows,
        const char *path, char *error, size_t error_size, int *cache_hit) {
    if (cache_hit) *cache_hit = 0;
    if (!encoder || !path) return NULL;
    if (!encoder->ane_cache_capacity)
        encoder->ane_cache_capacity = gemma_ane_cache_capacity();
    for (uint32_t index = 0; index < encoder->ane_cache_count; index++) {
        gemma_ane_cache_entry *entry = &encoder->ane_cache[index];
        if (entry->layer == layer && entry->requested_rows == rows &&
            strcmp(entry->path, path) == 0) {
            entry->last_used = ++encoder->ane_cache_tick;
            encoder->telemetry.ane_cache_hits++;
            if (cache_hit) *cache_hit = 1;
            return entry->mlp;
        }
    }
    encoder->telemetry.ane_cache_misses++;
    ltx_gemma_ane_mlp *mlp = ltx_gemma_ane_mlp_create(
        encoder->gpu, path, encoder->header.path, layer, rows,
        error, error_size);
    if (!mlp) return NULL;
    ltx_gemma_ane_mlp *cached = gemma_ane_cache_put(
        encoder, layer, rows, path, mlp, error, error_size);
    if (!cached) {
        ltx_gemma_ane_mlp_free(mlp);
        return NULL;
    }
    return cached;
}

static void gemma_ane_cache_drop(ltx_gemma_encoder *encoder,
                                 ltx_gemma_ane_mlp *mlp) {
    if (!encoder || !mlp) return;
    for (uint32_t index = 0; index < encoder->ane_cache_count; index++) {
        if (encoder->ane_cache[index].mlp != mlp) continue;
        ltx_gemma_ane_mlp_free(mlp);
        encoder->ane_cache[index] = encoder->ane_cache[
            --encoder->ane_cache_count];
        return;
    }
}

static int gemma_ane_manifest_for_layer(
        const ltx_gemma_encoder *encoder, uint32_t layer,
        char *path, size_t path_size) {
    if (!encoder || !encoder->ane_manifest || !path || !path_size)
        return 0;
    if (gemma_regular_file(encoder->ane_manifest)) {
        if (layer != 0u) return 0;
        int length = snprintf(path, path_size, "%s", encoder->ane_manifest);
        return length >= 0 && (size_t)length < path_size;
    }
    if (!gemma_directory(encoder->ane_manifest)) return 0;
    static const char *patterns[] = {
        "%s/block-%02u/manifest.json",
        "%s/block-%u/manifest.json",
        "%s/%02u/manifest.json",
        "%s/%u/manifest.json",
        "%s/block-%02u.json",
        "%s/block-%u.json",
    };
    for (size_t index = 0; index < sizeof(patterns) / sizeof(patterns[0]);
         index++) {
        int length = snprintf(path, path_size, patterns[index],
                              encoder->ane_manifest, layer);
        if (length >= 0 && (size_t)length < path_size &&
            gemma_regular_file(path)) return 1;
    }
    return 0;
}

typedef struct {
    ltx_gemma_encoder *encoder;
    uint32_t layer;
    uint32_t rows;
    char path[PATH_MAX];
    ltx_gemma_ane_mlp *mlp;
    char error[1024];
} gemma_ane_preload_job;

typedef struct {
    gemma_ane_preload_job *jobs;
    uint32_t count;
    uint32_t next;
    pthread_mutex_t mutex;
} gemma_ane_preload_queue;

static uint32_t gemma_ane_load_workers(uint32_t jobs) {
    uint32_t workers = 4u;
    const char *configured = getenv(
        "TURBOCIDER_LTX_GEMMA_ANE_LOAD_WORKERS");
    if (configured && *configured) {
        char *end = NULL;
        unsigned long value = strtoul(configured, &end, 10);
        if (end && *end == '\0' && value >= 1u && value <= 16u)
            workers = (uint32_t)value;
    }
    return workers < jobs ? workers : jobs;
}

static int gemma_ane_cache_has(
        const ltx_gemma_encoder *encoder, uint32_t layer, uint32_t rows,
        const char *path) {
    if (!encoder || !path) return 0;
    for (uint32_t index = 0; index < encoder->ane_cache_count; index++) {
        const gemma_ane_cache_entry *entry = &encoder->ane_cache[index];
        if (entry->layer == layer && entry->requested_rows == rows &&
            strcmp(entry->path, path) == 0) return 1;
    }
    return 0;
}

static void *gemma_ane_preload_worker(void *opaque) {
    gemma_ane_preload_queue *queue = opaque;
    for (;;) {
        pthread_mutex_lock(&queue->mutex);
        uint32_t index = queue->next++;
        pthread_mutex_unlock(&queue->mutex);
        if (index >= queue->count) break;
        gemma_ane_preload_job *job = &queue->jobs[index];
        job->mlp = ltx_gemma_ane_mlp_create(
            job->encoder->gpu, job->path, job->encoder->header.path,
            job->layer, job->rows, job->error, sizeof(job->error));
    }
    return NULL;
}

static void gemma_ane_preload(ltx_gemma_encoder *encoder, uint32_t rows) {
    if (!encoder || !encoder->ane_manifest || !rows) return;
    if (encoder->ane_preload_attempted &&
        encoder->ane_preload_rows == rows) return;
    encoder->ane_preload_attempted = 1;
    encoder->ane_preload_rows = rows;
    if (!encoder->ane_cache_capacity)
        encoder->ane_cache_capacity = gemma_ane_cache_capacity();
    gemma_ane_preload_job jobs[GEMMA_LAYERS] = {};
    uint32_t count = 0;
    for (uint32_t layer = 0;
         layer < GEMMA_LAYERS && count < encoder->ane_cache_capacity;
         layer++) {
        char path[PATH_MAX] = {};
        if (!gemma_ane_manifest_for_layer(
                encoder, layer, path, sizeof(path)) ||
            gemma_ane_cache_has(encoder, layer, rows, path)) continue;
        uint32_t execution_rows = 0;
        uint32_t minimum_profitable_rows = 0;
        char plan_error[1024] = {};
        if (!ltx_gemma_ane_mlp_plan(
                path, rows, &execution_rows, &minimum_profitable_rows,
                plan_error, sizeof(plan_error))) continue;
        const int exact = rows == execution_rows;
        const int measured = rows >= minimum_profitable_rows;
        const int override = gemma_ane_padding_allowed(rows, execution_rows);
        if (!exact && !measured && !override) continue;
        jobs[count].encoder = encoder;
        jobs[count].layer = layer;
        jobs[count].rows = rows;
        snprintf(jobs[count].path, sizeof(jobs[count].path), "%s", path);
        count++;
    }
    if (!count) return;
    uint32_t workers = gemma_ane_load_workers(count);
    gemma_ane_preload_queue queue = {
        .jobs = jobs, .count = count, .next = 0u,
    };
    if (pthread_mutex_init(&queue.mutex, NULL) != 0) return;
    pthread_t threads[15];
    uint32_t background = workers > 1u ? workers - 1u : 0u;
    uint32_t started = 0;
    double preload_started = gemma_probe_now();
    for (; started < background; started++)
        if (pthread_create(&threads[started], NULL,
                           gemma_ane_preload_worker, &queue) != 0) break;
    gemma_ane_preload_worker(&queue);
    for (uint32_t index = 0; index < started; index++)
        pthread_join(threads[index], NULL);
    pthread_mutex_destroy(&queue.mutex);
    uint32_t loaded = 0;
    for (uint32_t index = 0; index < count; index++) {
        gemma_ane_preload_job *job = &jobs[index];
        if (!job->mlp) continue;
        char cache_error[1024] = {};
        if (gemma_ane_cache_put(
                encoder, job->layer, job->rows, job->path, job->mlp,
                cache_error, sizeof(cache_error))) {
            loaded++;
        } else {
            ltx_gemma_ane_mlp_free(job->mlp);
        }
    }
    encoder->ane_preload_models_session_total += loaded;
    /* The caller participates in the queue, so a partial pthread_create
     * failure still loads every job.  Report the workers that actually ran,
     * rather than the requested pool size. */
    encoder->ane_preload_workers = started + 1u;
    encoder->ane_preload_seconds_session_total +=
        gemma_probe_now() - preload_started;
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
    if (!encoder || !tensor || tensor->data_end < tensor->data_begin) {
        gemma_fail(error, error_size, "invalid Gemma tensor load range");
        return NULL;
    }
    const uint64_t range = tensor->data_end - tensor->data_begin;
    if (!range || range > SIZE_MAX) {
        gemma_fail(error, error_size, "Gemma tensor load range is too large");
        return NULL;
    }
    size_t cache_index = SIZE_MAX;
    if (encoder->resident_weight_cache) {
        for (size_t index = 0; index < encoder->resident_weight_cache_count;
             index++) {
            if (&encoder->header.tensors[index] == tensor) {
                cache_index = index;
                break;
            }
        }
        if (cache_index == SIZE_MAX) {
            gemma_fail(error, error_size,
                       "Gemma tensor is outside the resident cache index");
            return NULL;
        }
        ltx_gpu_buffer *cached = encoder->resident_weight_cache[cache_index];
        if (cached) {
            encoder->telemetry.resident_weight_cache_hits++;
            return ltx_gpu_buffer_retain(cached);
        }
        encoder->telemetry.resident_weight_cache_misses++;
    }
    /* Do not dereference the multi-gigabyte safetensors mmap for every layer
     * load.  A repeated encode otherwise leaves file-backed weight pages
     * resident while the same bytes are copied into a temporary Metal
     * buffer, inflating RSS and making warm samples sensitive to VM pressure.
     * Read the bounded tensor range directly into the shared Metal buffer;
     * the mapping remains available for the large embedding lookup. */
    const size_t bytes = (size_t)range;
    ltx_gpu_buffer *buffer = ltx_gpu_buffer_new_classified(
        encoder->gpu, bytes, LTX_GPU_MEMORY_WEIGHTS,
        "ltx_gemma_weight", error, error_size);
    if (!buffer) return NULL;
    if (!ltx_st_read_mapped_data(
            &encoder->mapping, tensor, ltx_gpu_buffer_contents(buffer),
            bytes, error, error_size)) {
        ltx_gpu_buffer_free(buffer);
        return NULL;
    }
    if (cache_index != SIZE_MAX) {
        encoder->resident_weight_cache[cache_index] =
            ltx_gpu_buffer_retain(buffer);
        if (!encoder->resident_weight_cache[cache_index]) {
            ltx_gpu_buffer_free(buffer);
            gemma_fail(error, error_size,
                       "cannot retain Gemma resident weight buffer");
            return NULL;
        }
        encoder->resident_weight_bytes += bytes;
        encoder->telemetry.resident_weight_bytes =
            encoder->resident_weight_bytes;
    }
    return buffer;
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

static int gemma_apply_linear(ltx_gemma_encoder *encoder,
                              ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              const gemma_linear_weights *linear,
                              uint32_t rows, char *error,
                              size_t error_size);

static int gemma_load_mlp_linears(
        ltx_gemma_encoder *encoder, uint32_t layer,
        gemma_linear_weights *gate, gemma_linear_weights *up,
        gemma_linear_weights *down, char *error, size_t error_size) {
    char name[512];
    snprintf(name, sizeof(name), "model.layers.%u.mlp.gate_proj", layer);
    if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN,
                           GEMMA_INTERMEDIATE, gate, error, error_size))
        return 0;
    snprintf(name, sizeof(name), "model.layers.%u.mlp.up_proj", layer);
    if (!gemma_load_linear(encoder, name, GEMMA_HIDDEN,
                           GEMMA_INTERMEDIATE, up, error, error_size)) {
        gemma_free_linear(gate);
        return 0;
    }
    snprintf(name, sizeof(name), "model.layers.%u.mlp.down_proj", layer);
    if (!gemma_load_linear(encoder, name, GEMMA_INTERMEDIATE,
                           GEMMA_HIDDEN, down, error, error_size)) {
        gemma_free_linear(up);
        gemma_free_linear(gate);
        return 0;
    }
    return 1;
}

static int gemma_run_gpu_mlp(
        ltx_gemma_encoder *encoder, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input, const gemma_linear_weights *gate,
        const gemma_linear_weights *up, const gemma_linear_weights *down,
        ltx_gpu_buffer *zero, ltx_gpu_buffer *gate_raw,
        ltx_gpu_buffer *up_raw, ltx_gpu_buffer *gelu,
        ltx_gpu_buffer *product, uint32_t rows, int fused,
        char *error, size_t error_size) {
    if (fused)
        return ltx_gpu_gated_mlp_int8_convrot_mps_bf16(
            encoder->gpu, output, input,
            gate->weight, gate->scale, up->weight, up->scale,
            down->weight, down->scale, rows, GEMMA_HIDDEN,
            GEMMA_INTERMEDIATE, GEMMA_HIDDEN, GEMMA_CONVROT_GROUP,
            error, error_size);
    if (!ltx_gpu_batch_begin(encoder->gpu, error, error_size)) return 0;
    int ok = gemma_apply_linear(encoder, gate_raw, input, gate, rows,
                                error, error_size) &&
        gemma_apply_linear(encoder, up_raw, input, up, rows,
                           error, error_size) &&
        ltx_gpu_gelu_tanh_bf16(encoder->gpu, gelu, gate_raw,
                               rows * GEMMA_INTERMEDIATE, error, error_size) &&
        ltx_gpu_residual_gate_bf16(encoder->gpu, product, zero, gelu,
                                   up_raw, rows, GEMMA_INTERMEDIATE, rows,
                                   error, error_size) &&
        gemma_apply_linear(encoder, output, product, down, rows,
                           error, error_size);
    if (!ltx_gpu_batch_end(encoder->gpu, error, error_size)) ok = 0;
    return ok;
}

static int gemma_run_ane_mlp(
        ltx_gemma_encoder *encoder, ltx_gemma_ane_mlp *mlp,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, ltx_gemma_ane_mlp_timing *timing,
        char *error, size_t error_size) {
    const ltx_gemma_ane_mlp_shape *shape =
        ltx_gemma_ane_mlp_get_shape(mlp);
    if (!shape || shape->hidden != GEMMA_HIDDEN || shape->rows < rows)
        return gemma_fail(error, error_size,
                          "invalid Gemma ANE MLP execution shape");
    const uint32_t execution_rows = shape->rows;
    if (execution_rows == rows)
        return ltx_gemma_ane_mlp_eval(mlp, encoder->gpu, output, input,
                                      timing, error, error_size);
    const size_t input_bytes = (size_t)execution_rows * GEMMA_HIDDEN *
                               sizeof(uint16_t);
    const size_t real_bytes = (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t);
    ltx_gpu_buffer *padded_input = ltx_gpu_buffer_new(
        encoder->gpu, input_bytes, error, error_size);
    ltx_gpu_buffer *padded_output = ltx_gpu_buffer_new(
        encoder->gpu, input_bytes, error, error_size);
    uint16_t *host = calloc((size_t)execution_rows * GEMMA_HIDDEN,
                            sizeof(*host));
    int ok = padded_input && padded_output && host &&
        ltx_gpu_buffer_read(input, host, real_bytes, error, error_size) &&
        ltx_gpu_buffer_write(padded_input, host, input_bytes,
                             error, error_size) &&
        ltx_gemma_ane_mlp_eval(mlp, encoder->gpu, padded_output,
                               padded_input, timing, error, error_size) &&
        ltx_gpu_buffer_copy(encoder->gpu, output, 0u, padded_output, 0u,
                            real_bytes, error, error_size);
    free(host);
    ltx_gpu_buffer_free(padded_output);
    ltx_gpu_buffer_free(padded_input);
    return ok;
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

static double gemma_probe_now(void) {
    struct timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int gemma_probe_mlp_run(
        ltx_gemma_encoder *encoder, ltx_gpu_buffer *input,
        const gemma_linear_weights *gate, const gemma_linear_weights *up,
        const gemma_linear_weights *down, ltx_gpu_buffer *zero,
        ltx_gpu_buffer *gate_raw, ltx_gpu_buffer *up_raw,
        ltx_gpu_buffer *gelu, ltx_gpu_buffer *product,
        ltx_gpu_buffer *output, uint32_t rows, char *error,
        size_t error_size) {
    if (getenv("TURBOCIDER_LTX_GEMMA_FUSED_MLP")) {
        return ltx_gpu_gated_mlp_int8_convrot_mps_bf16(
            encoder->gpu, output, input,
            gate->weight, gate->scale, up->weight, up->scale,
            down->weight, down->scale, rows, GEMMA_HIDDEN,
            GEMMA_INTERMEDIATE, GEMMA_HIDDEN, GEMMA_CONVROT_GROUP,
            error, error_size);
    }
    int batch = 0;
    if (!ltx_gpu_batch_begin(encoder->gpu, error, error_size)) return 0;
    batch = 1;
#define GEMMA_PROBE_CALL(LABEL, EXPR) do { \
        if (error && error_size) error[0] = '\0'; \
        if (!(EXPR)) { \
            if (error && error_size && !error[0]) \
                snprintf(error, error_size, "Gemma MLP %s failed", LABEL); \
            goto failed; \
        } \
    } while (0)
    GEMMA_PROBE_CALL("gate projection", gemma_apply_linear(
        encoder, gate_raw, input, gate, rows, error, error_size));
    GEMMA_PROBE_CALL("up projection", gemma_apply_linear(
        encoder, up_raw, input, up, rows, error, error_size));
    GEMMA_PROBE_CALL("GELU", ltx_gpu_gelu_tanh_bf16(
        encoder->gpu, gelu, gate_raw, rows * GEMMA_INTERMEDIATE,
        error, error_size));
    GEMMA_PROBE_CALL("gated product", ltx_gpu_residual_gate_bf16(
        encoder->gpu, product, zero, gelu, up_raw, rows,
        GEMMA_INTERMEDIATE, rows, error, error_size));
    GEMMA_PROBE_CALL("down projection", gemma_apply_linear(
        encoder, output, product, down, rows, error, error_size));
#undef GEMMA_PROBE_CALL
    if (!ltx_gpu_batch_end(encoder->gpu, error, error_size)) return 0;
    return 1;
failed:
#undef GEMMA_PROBE_CALL
    if (batch) {
        char cleanup_error[512] = {};
        ltx_gpu_batch_end(encoder->gpu, cleanup_error,
                          sizeof(cleanup_error));
    }
    return 0;
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

int ltx_gemma_mlp_gpu_probe(
    const char *checkpoint, const char *shader_source,
    uint32_t layer, uint32_t rows, uint32_t warm_runs,
    const uint16_t *input, size_t input_elements,
    uint16_t *output, size_t output_elements,
    ltx_gemma_mlp_probe_result *result,
    char *error, size_t error_size) {
    const size_t input_count = (size_t)rows * GEMMA_HIDDEN;
    if (!checkpoint || !shader_source || !input || !output || !result ||
        !rows || rows > 1024u || layer >= GEMMA_LAYERS ||
        !warm_runs || warm_runs > LTX_GEMMA_MLP_PROBE_MAX_RUNS ||
        input_elements < input_count || output_elements < input_count)
        return gemma_fail(error, error_size,
                          "invalid Gemma MLP probe arguments");
    if (error && error_size) error[0] = '\0';
    memset(result, 0, sizeof(*result));
    result->layer = layer;
    result->rows = rows;
    result->warm_runs = warm_runs;
    result->hidden = GEMMA_HIDDEN;
    result->intermediate = GEMMA_INTERMEDIATE;

    int ok = 0;
    ltx_gemma_encoder encoder = {};
    gemma_linear_weights gate = {}, up = {}, down = {};
    ltx_gpu_buffer *input_gpu = NULL;
    ltx_gpu_buffer *zero = NULL;
    ltx_gpu_buffer *gate_raw = NULL;
    ltx_gpu_buffer *up_raw = NULL;
    ltx_gpu_buffer *gelu = NULL;
    ltx_gpu_buffer *product = NULL;
    ltx_gpu_buffer *output_gpu = NULL;
    char name[512];
    double started = gemma_probe_now();

    ltx_gemma_checkpoint_info info;
    int validated = ltx_gemma_checkpoint_inspect(
        checkpoint, &info, error, error_size);
    if (validated)
        validated = ltx_gemma_checkpoint_validate(&info, error, error_size);
    result->checkpoint_validation_seconds = gemma_probe_now() - started;
    if (!validated) goto cleanup;

    started = gemma_probe_now();
    if (!ltx_st_read_header(checkpoint, &encoder.header, error, error_size) ||
        !ltx_st_map_open(&encoder.header, &encoder.mapping,
                         error, error_size)) {
        result->checkpoint_open_seconds = gemma_probe_now() - started;
        goto cleanup;
    }
    encoder.mapping_open = 1;
    result->checkpoint_open_seconds = gemma_probe_now() - started;

    started = gemma_probe_now();
    encoder.gpu = ltx_gpu_create(shader_source, error, error_size);
    result->gpu_setup_seconds = gemma_probe_now() - started;
    if (!encoder.gpu) goto cleanup;

    started = gemma_probe_now();
    snprintf(name, sizeof(name), "model.layers.%u.mlp.gate_proj", layer);
    if (!gemma_load_linear(&encoder, name, GEMMA_HIDDEN,
                           GEMMA_INTERMEDIATE, &gate, error, error_size))
        goto cleanup;
    snprintf(name, sizeof(name), "model.layers.%u.mlp.up_proj", layer);
    if (!gemma_load_linear(&encoder, name, GEMMA_HIDDEN,
                           GEMMA_INTERMEDIATE, &up, error, error_size))
        goto cleanup;
    snprintf(name, sizeof(name), "model.layers.%u.mlp.down_proj", layer);
    if (!gemma_load_linear(&encoder, name, GEMMA_INTERMEDIATE,
                           GEMMA_HIDDEN, &down, error, error_size))
        goto cleanup;
    if (!ltx_st_map_discard(&encoder.mapping, error, error_size))
        goto cleanup;
    result->weight_load_seconds = gemma_probe_now() - started;
    result->resident_weight_bytes =
        ltx_gpu_buffer_bytes(gate.weight) + ltx_gpu_buffer_bytes(gate.scale) +
        ltx_gpu_buffer_bytes(up.weight) + ltx_gpu_buffer_bytes(up.scale) +
        ltx_gpu_buffer_bytes(down.weight) + ltx_gpu_buffer_bytes(down.scale);
    ltx_st_map_close(&encoder.mapping);
    encoder.mapping_open = 0;
    ltx_st_free_header(&encoder.header);

    started = gemma_probe_now();
    const size_t hidden_bytes = input_count * sizeof(uint16_t);
    const size_t intermediate_bytes =
        (size_t)rows * GEMMA_INTERMEDIATE * sizeof(uint16_t);
    input_gpu = ltx_gpu_buffer_new_copy(
        encoder.gpu, input, hidden_bytes, error, error_size);
    zero = ltx_gpu_buffer_new(
        encoder.gpu, intermediate_bytes, error, error_size);
    gate_raw = ltx_gpu_buffer_new(
        encoder.gpu, intermediate_bytes, error, error_size);
    up_raw = ltx_gpu_buffer_new(
        encoder.gpu, intermediate_bytes, error, error_size);
    gelu = ltx_gpu_buffer_new(
        encoder.gpu, intermediate_bytes, error, error_size);
    product = ltx_gpu_buffer_new(
        encoder.gpu, intermediate_bytes, error, error_size);
    output_gpu = ltx_gpu_buffer_new(
        encoder.gpu, hidden_bytes, error, error_size);
    if (!input_gpu || !zero || !gate_raw || !up_raw || !gelu ||
        !product || !output_gpu)
        goto cleanup;
    memset(ltx_gpu_buffer_contents(zero), 0, intermediate_bytes);
    result->workspace_bytes =
        ltx_gpu_buffer_bytes(input_gpu) + ltx_gpu_buffer_bytes(zero) +
        ltx_gpu_buffer_bytes(gate_raw) + ltx_gpu_buffer_bytes(up_raw) +
        ltx_gpu_buffer_bytes(gelu) + ltx_gpu_buffer_bytes(product) +
        ltx_gpu_buffer_bytes(output_gpu);
    result->workspace_setup_seconds = gemma_probe_now() - started;

    started = gemma_probe_now();
    if (!gemma_probe_mlp_run(
            &encoder, input_gpu, &gate, &up, &down, zero, gate_raw, up_raw,
            gelu, product, output_gpu, rows, error, error_size))
        goto cleanup;
    result->first_seconds = gemma_probe_now() - started;
    for (uint32_t run = 0; run < warm_runs; run++) {
        started = gemma_probe_now();
        if (!gemma_probe_mlp_run(
                &encoder, input_gpu, &gate, &up, &down, zero, gate_raw,
                up_raw, gelu, product, output_gpu, rows,
                error, error_size))
            goto cleanup;
        result->warm_seconds[run] = gemma_probe_now() - started;
    }
    if (!ltx_gpu_buffer_read(output_gpu, output, hidden_bytes,
                             error, error_size))
        goto cleanup;
    ok = 1;

cleanup:
    gemma_free_buffer(&output_gpu);
    gemma_free_buffer(&product);
    gemma_free_buffer(&gelu);
    gemma_free_buffer(&up_raw);
    gemma_free_buffer(&gate_raw);
    gemma_free_buffer(&zero);
    gemma_free_buffer(&input_gpu);
    gemma_free_linear(&down);
    gemma_free_linear(&up);
    gemma_free_linear(&gate);
    ltx_gpu_free(encoder.gpu);
    if (encoder.mapping_open) ltx_st_map_close(&encoder.mapping);
    ltx_st_free_header(&encoder.header);
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
    if (options->memory_hooks) {
        const size_t async_size = offsetof(ltx_gpu_memory_hooks, complete) +
            sizeof(options->memory_hooks->complete);
        if (options->memory_hooks->version < 2u ||
            options->memory_hooks->struct_size < async_size ||
            !options->memory_hooks->retire ||
            !options->memory_hooks->complete ||
            !ltx_gpu_set_memory_hooks_for_queue(
                encoder->gpu, options->memory_hooks,
                options->memory_allocator_domain,
                options->memory_generation,
                LTX_GPU_MEMORY_QUEUE_TEXT, error, error_size)) {
            if (error && error_size && !error[0])
                gemma_fail(error, error_size,
                           "constrained LTX Gemma encoder requires version-2 "
                           "asynchronous memory hooks");
            ltx_gemma_encoder_free(encoder);
            return NULL;
        }
    }
    encoder->max_tokens = options->max_tokens ? options->max_tokens : 1024u;
    const char *resident_weights = getenv(
        "TURBOCIDER_LTX_GEMMA_RESIDENT_WEIGHTS");
    if (resident_weights && strcmp(resident_weights, "1") == 0) {
        encoder->resident_weight_cache_count = encoder->header.tensor_count;
        encoder->resident_weight_cache = calloc(
            encoder->resident_weight_cache_count,
            sizeof(*encoder->resident_weight_cache));
        if (!encoder->resident_weight_cache) {
            gemma_fail(error, error_size,
                       "out of memory creating Gemma resident weight cache");
            ltx_gemma_encoder_free(encoder);
            return NULL;
        }
    }
    encoder->ane_manifest = gemma_copy_string(options->ane_manifest);
    if (options->ane_manifest && *options->ane_manifest &&
        !encoder->ane_manifest) {
        gemma_fail(error, error_size,
                   "out of memory copying Gemma ANE manifest path");
        ltx_gemma_encoder_free(encoder);
        return NULL;
    }
    return encoder;
}

void ltx_gemma_encoder_free(ltx_gemma_encoder *encoder) {
    if (!encoder) return;
    gemma_ane_cache_clear(encoder);
    for (size_t index = 0; index < encoder->resident_weight_cache_count;
         index++)
        ltx_gpu_buffer_free(encoder->resident_weight_cache[index]);
    free(encoder->resident_weight_cache);
    ltx_gpu_free(encoder->gpu);
    ltx_gemma_tokenizer_free(encoder->tokenizer);
    if (encoder->mapping_open) ltx_st_map_close(&encoder->mapping);
    ltx_st_free_header(&encoder->header);
    free(encoder->ane_manifest);
    free(encoder);
}

int ltx_gemma_encoder_get_telemetry(
        const ltx_gemma_encoder *encoder,
        ltx_gemma_encoder_telemetry *telemetry) {
    if (!encoder || !telemetry) return 0;
    *telemetry = encoder->telemetry;
    return 1;
}

int ltx_gemma_encoder_prepare_prompt(
        ltx_gemma_encoder *encoder, const char *prompt,
        char *error, size_t error_size) {
    if (!encoder || !prompt)
        return gemma_fail(error, error_size,
                          "invalid Gemma encoder preparation arguments");
    if (error && error_size) error[0] = '\0';
    if (!encoder->ane_manifest) return 1;
    uint32_t *ids = NULL;
    uint8_t *mask = NULL;
    size_t token_count = 0;
    if (!ltx_gemma_tokenizer_encode(encoder->tokenizer, prompt,
                                    encoder->max_tokens,
                                    &ids, &mask, &token_count,
                                    error, error_size)) return 0;
    size_t first = 0;
    while (first < token_count && !mask[first]) first++;
    size_t real_count = token_count - first;
    ltx_gemma_tokenizer_ids_free(ids, mask);
    if (!real_count || real_count > UINT32_MAX)
        return gemma_fail(error, error_size,
                          "Gemma prompt token count is out of range");
    gemma_ane_preload(encoder, (uint32_t)real_count);
    return 1;
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
    memset(&encoder->telemetry, 0, sizeof(encoder->telemetry));
    encoder->telemetry.ane_requested = encoder->ane_manifest != NULL;
    encoder->telemetry.ane_output_backing_used = 0;
    encoder->telemetry.resident_weights_enabled =
        encoder->resident_weight_cache != NULL;
    encoder->telemetry.resident_weight_bytes =
        encoder->resident_weight_bytes;
    encoder->telemetry.ane_preload_models_session_total =
        encoder->ane_preload_models_session_total;
    encoder->telemetry.ane_preload_workers = encoder->ane_preload_workers;
    encoder->telemetry.ane_preload_seconds_session_total =
        encoder->ane_preload_seconds_session_total;
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
    gemma_ane_preload(encoder, rows);
    encoder->telemetry.ane_preload_models_session_total =
        encoder->ane_preload_models_session_total;
    encoder->telemetry.ane_preload_workers = encoder->ane_preload_workers;
    encoder->telemetry.ane_preload_seconds_session_total =
        encoder->ane_preload_seconds_session_total;
    encoder->telemetry.rows = rows;
    snprintf(encoder->telemetry.ane_plan_reason,
             sizeof(encoder->telemetry.ane_plan_reason), "%s",
             encoder->ane_manifest ? "no_compatible_artifact" :
                                     "no_ane_manifest");
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
    ltx_gpu_buffer *video_flat_gpu = NULL;
    ltx_gpu_buffer *audio_flat_gpu = NULL;
    char name[512];
    const int fused_mlp = getenv("TURBOCIDER_LTX_GEMMA_FUSED_MLP") != NULL;
    const int gpu_taps = getenv("TURBOCIDER_LTX_GEMMA_GPU_TAPS") != NULL;

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
    /* The embedding rows have been copied to the resident Metal input. Drop
     * clean file-backed pages before streaming the per-layer tensors with
     * bounded pread calls.  A later encode can fault the selected rows back
     * without remapping the checkpoint. */
    {
        char discard_error[256] = {};
        (void)ltx_st_map_discard(&encoder->mapping,
                                 discard_error, sizeof(discard_error));
    }
    if (gpu_taps) {
        size_t flat_bytes = (size_t)rows * GEMMA_PROJECTION_INPUT *
                            sizeof(uint16_t);
        video_flat_gpu = ltx_gpu_buffer_new(
            encoder->gpu, flat_bytes, error, error_size);
        audio_flat_gpu = ltx_gpu_buffer_new(
            encoder->gpu, flat_bytes, error, error_size);
        if (!video_flat_gpu || !audio_flat_gpu) goto cleanup;
    } else {
        states = calloc((size_t)(GEMMA_LAYERS + 1u) * rows * GEMMA_HIDDEN,
                        sizeof(*states));
        if (!states) {
            gemma_fail(error, error_size,
                       "out of memory for Gemma hidden states");
            goto cleanup;
        }
        if (!ltx_gpu_buffer_read(
                x, states, (size_t)rows * GEMMA_HIDDEN * sizeof(*states),
                error, error_size)) goto cleanup;
    }
    if (!fused_mlp) {
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
    }

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
        ltx_gemma_ane_mlp *ane_mlp = NULL;
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
        /* Fixed-shape ANE procedures are qualified only at exact buckets by
         * default.  Tail padding is a benchmark-only override until a padded
         * full-encoder crossover is separately qualified. */
        char ane_path[PATH_MAX] = {};
        uint32_t ane_bucket = 0;
        uint32_t ane_minimum_profitable_rows = 0;
        char ane_error[2048] = {};
        int ane_plan_valid = 0;
        int ane_plan_allowed = 0;
        int ane_manifest_found = gemma_ane_manifest_for_layer(
            encoder, layer, ane_path, sizeof(ane_path));
        if (ane_manifest_found) {
            ane_plan_valid = ltx_gemma_ane_mlp_plan(
                ane_path, rows, &ane_bucket, &ane_minimum_profitable_rows,
                ane_error, sizeof(ane_error));
            if (ane_plan_valid) {
                const int exact = rows == ane_bucket;
                const int measured = rows >= ane_minimum_profitable_rows;
                const int override = gemma_ane_padding_allowed(rows, ane_bucket);
                ane_plan_allowed = exact || measured || override;
                if (!encoder->telemetry.ane_selected_bucket) {
                    encoder->telemetry.ane_selected_bucket = ane_bucket;
                    encoder->telemetry.ane_padding_rows = ane_bucket - rows;
                    encoder->telemetry.ane_minimum_profitable_rows =
                        ane_minimum_profitable_rows;
                    snprintf(encoder->telemetry.ane_plan_reason,
                             sizeof(encoder->telemetry.ane_plan_reason), "%s",
                             exact ? "exact_bucket" :
                             measured ? "measured_tail_padding" :
                             override ? "padding_override" :
                                        "below_min_profitable_rows");
                }
            }
        }
        if (ane_manifest_found && !ane_plan_valid &&
            getenv("TURBOCIDER_LTX_GEMMA_ANE_DEBUG"))
            fprintf(stderr, "Gemma layer %u ANE plan skipped: %s\n",
                    layer, ane_error[0] ? ane_error : "unknown error");
        if (ane_plan_valid && ane_plan_allowed) {
            encoder->telemetry.ane_layers_available++;
            encoder->telemetry.ane_layers_attempted++;
            ane_mlp = gemma_ane_cache_get(
                encoder, layer, rows, ane_path,
                ane_error, sizeof(ane_error), NULL);
            const ltx_gemma_ane_mlp_shape *ane_shape =
                ltx_gemma_ane_mlp_get_shape(ane_mlp);
            if (ane_mlp && (!ane_shape || ane_shape->rows != ane_bucket ||
                    ane_shape->minimum_profitable_rows !=
                        ane_minimum_profitable_rows ||
                    (rows != ane_shape->rows &&
                     rows < ane_shape->minimum_profitable_rows &&
                     !gemma_ane_padding_allowed(rows, ane_shape->rows)))) {
                snprintf(ane_error, sizeof(ane_error),
                         "Gemma ANE execution rows are not qualified: %u -> %u",
                         rows, ane_shape ? ane_shape->rows : 0u);
                gemma_ane_cache_drop(encoder, ane_mlp);
                ane_mlp = NULL;
            }
            if (!ane_mlp) {
                encoder->telemetry.ane_layers_fallback++;
                if (getenv("TURBOCIDER_LTX_GEMMA_ANE_DEBUG"))
                    fprintf(stderr, "Gemma layer %u ANE load fallback: %s\n",
                            layer, ane_error[0] ? ane_error : "unknown error");
            }
        }
        if (!ane_mlp) {
            layer_stage = "load MLP projections";
            if (!gemma_load_mlp_linears(
                    encoder, layer, &gate, &up, &down,
                    error, error_size)) goto layer_cleanup;
        }

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
        if (!fused_mlp) {
            GEMMA_NEW(gate_raw, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
            GEMMA_NEW(up_raw, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
            GEMMA_NEW(gelu, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
            GEMMA_NEW(product, (size_t)rows * GEMMA_INTERMEDIATE * 2u);
        }
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
        if (gpu_taps && layer == 0u)
            GEMMA_CALL("embedding projection tap",
                ltx_gpu_gemma_projection_tap_bf16(
                    encoder->gpu, video_flat_gpu, audio_flat_gpu, x,
                    rows, GEMMA_HIDDEN, 0u, GEMMA_LAYERS + 1u,
                    sqrtf((float)GEMMA_VIDEO_DIM / (float)GEMMA_HIDDEN),
                    sqrtf((float)GEMMA_AUDIO_DIM / (float)GEMMA_HIDDEN),
                    error, error_size));
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
        if (ane_mlp) {
            if (error && error_size) error[0] = '\0';
            int pre_mlp_ok = ltx_gpu_batch_end(
                encoder->gpu, error, error_size);
            batch = 0;
            if (!pre_mlp_ok) goto layer_cleanup;
            ltx_gemma_ane_mlp_timing timing = {};
            char ane_error[2048] = {};
            int ane_ok = gemma_run_ane_mlp(
                encoder, ane_mlp, down_raw, input_norm, rows, &timing,
                ane_error, sizeof(ane_error));
            if (ane_ok) {
                const ltx_gemma_ane_mlp_shape *shape =
                    ltx_gemma_ane_mlp_get_shape(ane_mlp);
                encoder->telemetry.ane_layers_succeeded++;
                encoder->telemetry.ane_execution_rows += shape->rows;
                encoder->telemetry.ane_total_seconds +=
                    timing.total_ms / 1000.0;
                encoder->telemetry.ane_used = 1;
                encoder->telemetry.ane_output_backing_used =
                    encoder->telemetry.ane_layers_succeeded == 1u ?
                    timing.ane_output_backing_used :
                    encoder->telemetry.ane_output_backing_used &&
                    timing.ane_output_backing_used;
            } else {
                encoder->telemetry.ane_layers_fallback++;
                if (getenv("TURBOCIDER_LTX_GEMMA_ANE_DEBUG"))
                    fprintf(stderr, "Gemma layer %u ANE execution fallback: %s\n",
                            layer, ane_error[0] ? ane_error : "unknown error");
                gemma_ane_cache_drop(encoder, ane_mlp);
                ane_mlp = NULL;
                if (error && error_size) error[0] = '\0';
                layer_stage = "load GPU MLP fallback";
                if (!gemma_load_mlp_linears(
                        encoder, layer, &gate, &up, &down,
                        error, error_size) ||
                    !gemma_run_gpu_mlp(
                        encoder, down_raw, input_norm, &gate, &up, &down,
                        zero, gate_raw, up_raw, gelu, product, rows, fused_mlp,
                        error, error_size)) goto layer_cleanup;
            }
            layer_stage = "resume GPU batch after ANE MLP";
            if (!ltx_gpu_batch_begin(encoder->gpu, error, error_size))
                goto layer_cleanup;
            batch = 1;
        } else if (fused_mlp) {
            GEMMA_CALL("fused gated MLP",
                ltx_gpu_gated_mlp_int8_convrot_mps_bf16(
                    encoder->gpu, down_raw, input_norm,
                    gate.weight, gate.scale, up.weight, up.scale,
                    down.weight, down.scale, rows, GEMMA_HIDDEN,
                    GEMMA_INTERMEDIATE, GEMMA_HIDDEN, GEMMA_CONVROT_GROUP,
                    error, error_size));
        } else {
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
        }
        GEMMA_CALL("post-feedforward norm", gemma_norm(encoder, post_ff,
                        down_raw, post_ff_weight, rows, GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("feedforward residual", ltx_gpu_add_bf16(encoder->gpu,
                        residual, residual, post_ff, rows * GEMMA_HIDDEN,
                        error, error_size));
        GEMMA_CALL("layer affine", ltx_gpu_affine_bf16(
                encoder->gpu, next, residual, layer_scale, layer_shift,
                rows, GEMMA_HIDDEN, 1u, error, error_size));
        if (gpu_taps && layer + 1u < GEMMA_LAYERS)
            GEMMA_CALL("hidden-state projection tap",
                ltx_gpu_gemma_projection_tap_bf16(
                    encoder->gpu, video_flat_gpu, audio_flat_gpu, next,
                    rows, GEMMA_HIDDEN, layer + 1u, GEMMA_LAYERS + 1u,
                    sqrtf((float)GEMMA_VIDEO_DIM / (float)GEMMA_HIDDEN),
                    sqrtf((float)GEMMA_AUDIO_DIM / (float)GEMMA_HIDDEN),
                    error, error_size));
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
        if (!gpu_taps) {
            layer_stage = "read hidden state";
            if (!ltx_gpu_buffer_read(
                    next, states + (size_t)(layer + 1u) * rows * GEMMA_HIDDEN,
                    (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t),
                    error, error_size)) goto layer_cleanup;
        }
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
    int final_ok = final_gpu != NULL;
    int final_batch = 0;
    if (final_ok && gpu_taps) {
        final_ok = ltx_gpu_batch_begin(encoder->gpu, error, error_size);
        final_batch = final_ok;
        if (final_ok)
            final_ok = gemma_norm(
                encoder, final_gpu, x, final_norm, rows, GEMMA_HIDDEN,
                error, error_size);
        if (final_ok)
            final_ok = ltx_gpu_gemma_projection_tap_bf16(
                encoder->gpu, video_flat_gpu, audio_flat_gpu, final_gpu,
                rows, GEMMA_HIDDEN, GEMMA_LAYERS, GEMMA_LAYERS + 1u,
                sqrtf((float)GEMMA_VIDEO_DIM / (float)GEMMA_HIDDEN),
                sqrtf((float)GEMMA_AUDIO_DIM / (float)GEMMA_HIDDEN),
                error, error_size);
        if (final_ok) {
            final_ok = ltx_gpu_batch_end(encoder->gpu, error, error_size);
            final_batch = 0;
        }
    } else if (final_ok) {
        final_ok = gemma_norm(
            encoder, final_gpu, x, final_norm, rows, GEMMA_HIDDEN,
            error, error_size) &&
            ltx_gpu_buffer_read(
                final_gpu,
                states + (size_t)GEMMA_LAYERS * rows * GEMMA_HIDDEN,
                (size_t)rows * GEMMA_HIDDEN * sizeof(uint16_t),
                error, error_size);
    }
    if (final_batch) {
        char cleanup_error[512] = {};
        ltx_gpu_batch_end(
            encoder->gpu, cleanup_error, sizeof(cleanup_error));
    }
    if (!final_ok) {
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

    if (gpu_taps) {
        if (!gemma_project(
                encoder, video_flat_gpu, rows, GEMMA_VIDEO_DIM,
                "text_embedding_projection.video_aggregate_embed.weight",
                "text_embedding_projection.video_aggregate_embed.bias",
                video_output, video_count, error, error_size) ||
            !gemma_project(
                encoder, audio_flat_gpu, rows, GEMMA_AUDIO_DIM,
                "text_embedding_projection.audio_aggregate_embed.weight",
                "text_embedding_projection.audio_aggregate_embed.bias",
                audio_output, audio_count, error, error_size))
            goto cleanup;
    } else {
        uint16_t *flat = malloc((size_t)rows * GEMMA_PROJECTION_INPUT *
                                sizeof(*flat));
        if (!flat || !gemma_build_projection_input(
                states, rows, flat, GEMMA_VIDEO_DIM)) {
            free(flat);
            gemma_fail(error, error_size,
                       "failed to build Gemma projection input");
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
    }
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
    gemma_free_buffer(&video_flat_gpu);
    gemma_free_buffer(&audio_flat_gpu);
    free(embedding);
    free(states);
    free(cosine_host);
    free(sine_host);
    ltx_gemma_tokenizer_ids_free(ids, mask);
    return ok;
}
