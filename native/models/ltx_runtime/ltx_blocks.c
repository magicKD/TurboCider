#include "ltx_runtime_config.h"
#include "ltx_native.h"
#include "ltx.h"
#include "ltx_conditioning.h"
#include "ltx_connector.h"
#include "ltx_gpu.h"
#include "ltx_latent_stats.h"
#include "ltx_rng.h"
#include "ltx_safetensors.h"
#include "ltx_transformer_io.h"
#include "ltx_upsampler.h"
#include "ltx_video_vae.h"
#include "ltx_weights.h"
#include "ltx_mlx_upsampler.h"

#ifdef LTX_ENABLE_ANE_MLP
#include "ltx_ane_mlp.h"
#endif

#ifdef LTX_ENABLE_ANE_V2A
#include "ltx_ane_v2a.h"
#endif

#ifdef LTX_ENABLE_ANE_KV
#include "ltx_ane_kv.h"
#endif

#ifdef LTX_ENABLE_ANE_QKV
#include "ltx_ane_qkv.h"
#endif

#ifdef LTX_ENABLE_MLX_MEDIA
#include "ltx_mlx_video_vae.h"
#endif

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Timing is reported through TurboCider events/results, never stdout. */
#define printf(...) ((int)0)
extern char **environ;

#ifdef LTX_ENABLE_MLX_MEDIA
typedef struct {
    pid_t process;
} mlx_vae_worker;
#endif

/* Experimental second Metal context used to measure whether the two
 * mathematically independent AV cross-attention directions can overlap.
 * Each context owns a separate command queue and MPSGraph cache; buffers are
 * still shared through the same unified-memory Metal device. */
static ltx_native_progress native_progress;
static void *native_opaque;
static int report_progress(const char *phase,int current,int total,char *error,size_t size) {
 if(native_progress&&native_progress(phase,current,total,native_opaque)){snprintf(error,size,"generation cancelled");return 1;}return 0;
}
static ltx_gpu *av_parallel_audio_gpu;
static int av_parallel_streams;
static int av_parallel_cross;
static int av_parallel_ffn;
static int av_persistent_worker_enabled;
static int video_text_kv_prefetch_stage1;
static int video_text_kv_prefetch_stage2;
static int ane_video_text_kv_enabled;
static int video_attention_batch;
/* Image-to-video uses a contiguous first-frame token prefix. Transformer
 * modulation is selected per row while attention/MLP geometry stays fixed. */
static uint32_t active_video_conditioned_prefix_rows;
#ifdef LTX_ENABLE_ANE_MLP
static int ane_mlp_fused_residual;
static int ane_mlp_fused_adaln_pack;
#endif
#ifdef LTX_ENABLE_ANE_QKV
static int ane_video_self_qkv_enabled;
static uint32_t ane_video_self_qkv_prefix_rows;
#endif

typedef struct {
    uint32_t requested_width;
    uint32_t requested_height;
    uint32_t latent_frames;
    uint32_t stage1_height;
    uint32_t stage1_width;
    uint32_t stage2_height;
    uint32_t stage2_width;
    uint32_t stage1_rows;
    uint32_t stage2_rows;
    uint32_t output_frames;
    uint32_t output_height;
    uint32_t output_width;
} generation_geometry;

static generation_geometry active_geometry = {
    .requested_width = 704u,
    .requested_height = 480u,
    .latent_frames = 13u,
    .stage1_height = 7u,
    .stage1_width = 11u,
    .stage2_height = 14u,
    .stage2_width = 22u,
    .stage1_rows = 1001u,
    .stage2_rows = 4004u,
    .output_frames = 97u,
    .output_height = 448u,
    .output_width = 704u,
};

typedef struct {
    int stage1_enabled;
    int stage2_enabled;
    int batch_commands;
    uint32_t dense_edge_blocks;
    uint32_t dense_edge_steps;
    float tau;
    float step_taus[32];
    size_t step_tau_count;
    size_t active_step;
    size_t active_step_count;
} sol_video_self_options;

static sol_video_self_options sol_video_self = {
    .stage1_enabled = 0,
    .stage2_enabled = 0,
    .batch_commands = 0,
    .dense_edge_blocks = 1u,
    .dense_edge_steps = 1u,
    .tau = 0.5f,
    .step_tau_count = 0u,
    .active_step = SIZE_MAX,
    .active_step_count = 0u,
};

enum {
    LTX_MAX_TABLE_ROWS = 9,
    LTX_MAX_CUSTOM_SIGMAS = 32,
};

typedef struct {
    ltx_gpu_buffer *weight;
    ltx_gpu_buffer *scale;
    ltx_gpu_buffer *bias;
    uint32_t input_dim;
    uint32_t output_dim;
    size_t weight_bytes;
} gpu_linear;

typedef struct {
    gpu_linear query;
    gpu_linear key;
    gpu_linear value;
    gpu_linear output;
    ltx_gpu_buffer *query_norm;
    ltx_gpu_buffer *key_norm;
    ltx_gpu_buffer *gate_weight;
    ltx_gpu_buffer *gate_bias;
    uint32_t query_dim;
    uint32_t key_value_dim;
    uint32_t inner_dim;
    uint32_t output_dim;
    uint32_t heads;
    uint32_t head_dim;
} attention_weights;

typedef struct {
    gpu_linear fc1;
    gpu_linear fc2;
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t output_dim;
} mlp_weights;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint16_t *base_values;
    ltx_gpu_buffer *row[LTX_MAX_TABLE_ROWS];
} parameter_table;

typedef struct {
    attention_weights video_self;
    attention_weights audio_self;
    attention_weights video_text;
    attention_weights audio_text;
    attention_weights audio_to_video;
    attention_weights video_to_audio;
    mlp_weights video_mlp;
    mlp_weights audio_mlp;
    parameter_table video_adaln;
    parameter_table video_adaln_conditioned;
    parameter_table audio_adaln;
    parameter_table video_prompt;
    parameter_table audio_prompt;
    parameter_table av_video;
    parameter_table av_video_conditioned;
    parameter_table av_audio;
#ifdef LTX_ENABLE_ANE_MLP
    ltx_ane_mlp *ane_video_mlp[2];
#endif
#ifdef LTX_ENABLE_ANE_V2A
    ltx_ane_v2a *ane_v2a[2];
#endif
#ifdef LTX_ENABLE_ANE_KV
    ltx_ane_kv *ane_video_text_kv;
#endif
#ifdef LTX_ENABLE_ANE_QKV
    ltx_ane_qkv *ane_video_self_qkv[2];
#endif
} block_weights;

typedef struct {
    ltx_gpu_buffer *cosine;
    ltx_gpu_buffer *sine;
} rope_pair;

typedef struct {
    rope_pair video_self;
    rope_pair audio_self;
    rope_pair video_cross;
    rope_pair audio_cross;
} block_rope;

typedef struct {
    ltx_gpu_buffer *video_state[2];
    ltx_gpu_buffer *audio_state[2];
    ltx_gpu_buffer *video_normed;
    ltx_gpu_buffer *audio_normed;
    ltx_gpu_buffer *video_branch;
    ltx_gpu_buffer *audio_branch;
    ltx_gpu_buffer *video_text_scaled;
    ltx_gpu_buffer *audio_text_scaled;
    ltx_gpu_buffer *video_text_key;
    ltx_gpu_buffer *video_text_value;
    ltx_gpu_buffer *video_norm3;
    ltx_gpu_buffer *audio_norm3;
    ltx_gpu_buffer *video_a2v;
    ltx_gpu_buffer *audio_a2v;
    ltx_gpu_buffer *video_v2a;
    ltx_gpu_buffer *audio_v2a;
    ltx_gpu_buffer *video_sol_query;
    ltx_gpu_buffer *video_sol_key;
    ltx_gpu_buffer *video_sol_value;
    ltx_gpu_buffer *video_sol_gate_logits;
    ltx_gpu_buffer *video_sol_gate;
    ltx_gpu_buffer *video_sol_core;
    ltx_gpu_buffer *video_sol_rotated;
    ltx_gpu_buffer *video_sol_packed_query;
    ltx_gpu_buffer *video_sol_packed_key;
    ltx_gpu_buffer *video_sol_packed_value;
    ltx_gpu_buffer *video_sol_packed_output;
    ltx_gpu_buffer *video_sol_query_centroids;
    ltx_gpu_buffer *video_sol_key_centroids;
    ltx_gpu_buffer *video_sol_value_sums;
    ltx_gpu_buffer *video_sol_thresholds;
    ltx_gpu_buffer *video_sol_routes;
#ifdef LTX_ENABLE_ANE_QKV
    ltx_gpu_buffer *video_qkv_query_prefix;
    ltx_gpu_buffer *video_qkv_key_prefix;
    ltx_gpu_buffer *video_qkv_value_prefix;
    ltx_gpu_buffer *video_qkv_query;
    ltx_gpu_buffer *video_qkv_key;
    ltx_gpu_buffer *video_qkv_value;
    ltx_gpu_buffer *video_qkv_gate_logits;
    ltx_gpu_buffer *video_qkv_gate;
    ltx_gpu_buffer *video_qkv_core;
    ltx_gpu_buffer *video_qkv_rotated;
#endif
} block_workspace;

typedef struct {
    double video_self;
    double audio_self;
    double video_text;
    double audio_text;
    double av_norm_modulation;
    double audio_to_video;
    double video_to_audio;
    double av_residual;
    double video_ffn;
    double audio_ffn;
    double ane_mlp_pack;
    double ane_mlp_overlap;
    double ane_mlp_gpu;
    double ane_mlp_ane;
    double ane_mlp_join;
    double av_parallel_stream_wall;
    double av_parallel_cross_wall;
    double av_parallel_ffn_wall;
    double video_text_kv;
    double video_text_query;
    double video_text_prefetch_wall;
    double ane_kv_pack;
    double ane_kv_compute;
    double ane_kv_unpack;
    double ane_kv_overlap;
    double ane_v2a_pack;
    double ane_v2a_compute;
    double ane_v2a_unpack;
    double ane_v2a_overlap;
#ifdef LTX_ENABLE_ANE_QKV
    double ane_qkv_pack;
    double ane_qkv_compute;
    double ane_qkv_gpu;
    double ane_qkv_concat;
    double ane_qkv_overlap;
#endif
} block_timing;

typedef struct {
    double rel_l2;
    double cosine;
    double max_abs;
    double reference_rms;
    double candidate_rms;
    uint64_t nonfinite;
} comparison_metrics;

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

typedef void *(*av_worker_function)(void *opaque);

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t wake;
    pthread_cond_t done;
    av_worker_function function;
    void *opaque;
    int initialized;
    int stop;
    int active;
    int pending;
    int completed;
} av_persistent_worker;

typedef struct {
    pthread_t thread;
    int started;
    int persistent;
} av_task_handle;

static av_persistent_worker av_audio_worker = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .wake = PTHREAD_COND_INITIALIZER,
    .done = PTHREAD_COND_INITIALIZER,
};

static void *av_persistent_worker_main(void *opaque) {
    av_persistent_worker *worker = opaque;
    for (;;) {
        pthread_mutex_lock(&worker->mutex);
        while (!worker->pending && !worker->stop)
            pthread_cond_wait(&worker->wake, &worker->mutex);
        if (worker->stop && !worker->pending) {
            pthread_mutex_unlock(&worker->mutex);
            return NULL;
        }
        av_worker_function function = worker->function;
        void *context = worker->opaque;
        worker->pending = 0;
        worker->active = 1;
        pthread_mutex_unlock(&worker->mutex);

        function(context);

        pthread_mutex_lock(&worker->mutex);
        worker->active = 0;
        worker->completed = 1;
        pthread_cond_signal(&worker->done);
        pthread_mutex_unlock(&worker->mutex);
    }
}

static int av_persistent_worker_start(char *error, size_t error_size) {
    av_persistent_worker *worker = &av_audio_worker;
    pthread_mutex_lock(&worker->mutex);
    worker->stop = 0;
    worker->active = 0;
    worker->pending = 0;
    worker->completed = 0;
    worker->function = NULL;
    worker->opaque = NULL;
    pthread_mutex_unlock(&worker->mutex);
    if (pthread_create(
            &worker->thread, NULL, av_persistent_worker_main, worker) != 0) {
        snprintf(error, error_size,
                 "cannot create persistent Audio worker");
        return 0;
    }
    pthread_mutex_lock(&worker->mutex);
    worker->initialized = 1;
    pthread_mutex_unlock(&worker->mutex);
    return 1;
}

static void av_persistent_worker_stop(void) {
    av_persistent_worker *worker = &av_audio_worker;
    pthread_mutex_lock(&worker->mutex);
    int initialized = worker->initialized;
    if (initialized) {
        worker->stop = 1;
        pthread_cond_signal(&worker->wake);
    }
    pthread_mutex_unlock(&worker->mutex);
    if (!initialized) return;
    pthread_join(worker->thread, NULL);
    pthread_mutex_lock(&worker->mutex);
    worker->initialized = 0;
    worker->stop = 0;
    worker->active = 0;
    worker->pending = 0;
    worker->completed = 0;
    worker->function = NULL;
    worker->opaque = NULL;
    pthread_mutex_unlock(&worker->mutex);
}

static int av_task_start(av_task_handle *handle,
                         av_worker_function function, void *opaque,
                         const char *label,
                         char *error, size_t error_size) {
    memset(handle, 0, sizeof(*handle));
    if (av_persistent_worker_enabled) {
        av_persistent_worker *worker = &av_audio_worker;
        pthread_mutex_lock(&worker->mutex);
        if (!worker->initialized || worker->stop || worker->active ||
            worker->pending || worker->completed) {
            pthread_mutex_unlock(&worker->mutex);
            snprintf(error, error_size,
                     "persistent Audio worker is busy for %s", label);
            return 0;
        }
        worker->function = function;
        worker->opaque = opaque;
        worker->pending = 1;
        pthread_cond_signal(&worker->wake);
        pthread_mutex_unlock(&worker->mutex);
        handle->started = 1;
        handle->persistent = 1;
        return 1;
    }
    if (pthread_create(&handle->thread, NULL, function, opaque) != 0) {
        snprintf(error, error_size, "cannot create %s thread", label);
        return 0;
    }
    handle->started = 1;
    return 1;
}

static int av_task_wait(av_task_handle *handle) {
    if (!handle || !handle->started) return 0;
    if (!handle->persistent) {
        int ok = pthread_join(handle->thread, NULL) == 0;
        handle->started = 0;
        return ok;
    }
    av_persistent_worker *worker = &av_audio_worker;
    pthread_mutex_lock(&worker->mutex);
    while (!worker->completed && worker->initialized)
        pthread_cond_wait(&worker->done, &worker->mutex);
    int ok = worker->completed;
    worker->completed = 0;
    worker->function = NULL;
    worker->opaque = NULL;
    pthread_mutex_unlock(&worker->mutex);
    handle->started = 0;
    return ok;
}

static int environment_flag(const char *name, int default_value) {
    const char *value = ltx_runtime_getenv(name);
    if (!value || !value[0]) return default_value;
    return strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "no") != 0;
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
        fprintf(stderr, "bench_block: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void configure_generation_geometry(void) {
    const char *width_text = ltx_runtime_getenv("LTX_OUTPUT_WIDTH");
    const char *height_text = ltx_runtime_getenv("LTX_OUTPUT_HEIGHT");
    uint32_t requested_width = width_text && width_text[0] ?
        parse_u32(width_text, "LTX_OUTPUT_WIDTH") : 704u;
    uint32_t requested_height = height_text && height_text[0] ?
        parse_u32(height_text, "LTX_OUTPUT_HEIGHT") : 480u;
    uint32_t stage1_width = requested_width / 64u;
    uint32_t stage1_height = requested_height / 64u;
    if (!stage1_width || !stage1_height ||
        stage1_width > UINT32_MAX / 2u ||
        stage1_height > UINT32_MAX / 2u) {
        fprintf(stderr,
                "bench_block: output size must be at least 64x64: %ux%u\n",
                requested_width, requested_height);
        exit(2);
    }
    uint32_t stage2_width = stage1_width * 2u;
    uint32_t stage2_height = stage1_height * 2u;
    uint64_t stage1_rows =
        (uint64_t)active_geometry.latent_frames *
        stage1_height * stage1_width;
    uint64_t stage2_rows =
        (uint64_t)active_geometry.latent_frames *
        stage2_height * stage2_width;
    if (stage1_rows > UINT32_MAX || stage2_rows > UINT32_MAX) {
        fprintf(stderr,
                "bench_block: output geometry is too large: %ux%u\n",
                requested_width, requested_height);
        exit(2);
    }
    active_geometry.requested_width = requested_width;
    active_geometry.requested_height = requested_height;
    active_geometry.stage1_height = stage1_height;
    active_geometry.stage1_width = stage1_width;
    active_geometry.stage2_height = stage2_height;
    active_geometry.stage2_width = stage2_width;
    active_geometry.stage1_rows = (uint32_t)stage1_rows;
    active_geometry.stage2_rows = (uint32_t)stage2_rows;
    active_geometry.output_height = stage2_height * 32u;
    active_geometry.output_width = stage2_width * 32u;
}

static uint32_t parse_index(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > UINT32_MAX) {
        fprintf(stderr, "bench_block: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static uint64_t parse_u64(const char *text, const char *label) {
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (!text[0] || !end || *end) {
        fprintf(stderr, "bench_block: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint64_t)value;
}

static float parse_f32(const char *text, const char *label) {
    char *end = NULL;
    float value = strtof(text, &end);
    if (!text[0] || !end || *end || !isfinite(value)) {
        fprintf(stderr, "bench_block: invalid %s: %s\n", label, text);
        exit(2);
    }
    return value;
}

static uint32_t environment_index(const char *name,
                                  uint32_t default_value) {
    const char *value = ltx_runtime_getenv(name);
    return value && value[0] ? parse_index(value, name) : default_value;
}

static int parse_float_list(const char *name, const char *text,
                            float *values, size_t capacity, size_t *count,
                            char *error, size_t error_size) {
    if (!text || !text[0]) {
        *count = 0u;
        return 1;
    }
    const char *cursor = text;
    size_t parsed = 0u;
    while (*cursor) {
        char *end = NULL;
        errno = 0;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value) ||
            parsed >= capacity) {
            snprintf(error, error_size, "invalid %s list: %s", name, text);
            return 0;
        }
        values[parsed++] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (!*cursor) break;
        if (*cursor != ',') {
            snprintf(error, error_size, "invalid %s separator: %s",
                     name, text);
            return 0;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (!*cursor) {
            snprintf(error, error_size, "trailing comma in %s: %s",
                     name, text);
            return 0;
        }
    }
    *count = parsed;
    return 1;
}

static int sol_video_self_stage_enabled(uint32_t rows) {
    if (rows == active_geometry.stage1_rows)
        return sol_video_self.stage1_enabled;
    if (rows == active_geometry.stage2_rows)
        return sol_video_self.stage2_enabled;
    return 0;
}

static float sol_video_self_current_tau(void) {
    if (sol_video_self.step_tau_count &&
        sol_video_self.active_step < sol_video_self.step_tau_count)
        return sol_video_self.step_taus[sol_video_self.active_step];
    return sol_video_self.tau;
}

static int sol_video_self_should_run(uint32_t rows,
                                     uint32_t block_index) {
    if (!sol_video_self_stage_enabled(rows) || block_index >= 48u)
        return 0;
    uint32_t edge_blocks = sol_video_self.dense_edge_blocks;
    if (edge_blocks >= 24u || block_index < edge_blocks ||
        block_index >= 48u - edge_blocks) return 0;
    if (sol_video_self.active_step_count) {
        size_t edge_steps = sol_video_self.dense_edge_steps;
        if (edge_steps * 2u >= sol_video_self.active_step_count ||
            sol_video_self.active_step < edge_steps ||
            sol_video_self.active_step >=
                sol_video_self.active_step_count - edge_steps) return 0;
    }
    return 1;
}

static int resolve_sigma_schedule(
        const char *environment_name,
        const float *default_sigmas, size_t default_count,
        float *storage, size_t storage_capacity,
        const float **sigmas, size_t *sigma_count,
        char *error, size_t error_size) {
    const char *text = ltx_runtime_getenv(environment_name);
    if (!text || !text[0]) {
        *sigmas = default_sigmas;
        *sigma_count = default_count;
        return 1;
    }
    const char *cursor = text;
    size_t count = 0;
    while (*cursor) {
        char *end = NULL;
        errno = 0;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value) ||
            value < 0.0f || value > 1.0f ||
            count >= storage_capacity) {
            snprintf(error, error_size, "invalid %s sigma list: %s",
                     environment_name, text);
            return 0;
        }
        storage[count++] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (!*cursor) break;
        if (*cursor != ',') {
            snprintf(error, error_size, "invalid %s separator: %s",
                     environment_name, text);
            return 0;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (!*cursor) {
            snprintf(error, error_size, "trailing comma in %s: %s",
                     environment_name, text);
            return 0;
        }
    }
    if (count < 2u || storage[0] <= 0.0f || storage[count - 1u] != 0.0f) {
        snprintf(error, error_size,
                 "%s must start above zero and end at zero",
                 environment_name);
        return 0;
    }
    for (size_t index = 1; index < count; index++) {
        if (storage[index] >= storage[index - 1u]) {
            snprintf(error, error_size,
                     "%s sigmas must be strictly descending",
                     environment_name);
            return 0;
        }
    }
    *sigmas = storage;
    *sigma_count = count;
    return 1;
}

static int checked_bytes(uint64_t elements, size_t element_size,
                         size_t *bytes) {
    if (!bytes || elements > SIZE_MAX / element_size) return 0;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int fixture_path(char *path, size_t path_size,
                        const char *directory, const char *name,
                        char *error, size_t error_size) {
    int length = snprintf(path, path_size, "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= path_size) {
        snprintf(error, error_size, "fixture path is too long");
        return 0;
    }
    return 1;
}

static int file_size_exact(const char *path, size_t *bytes,
                           char *error, size_t error_size) {
    struct stat info;
    if (stat(path, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_size < 0 || (uint64_t)info.st_size > SIZE_MAX) {
        snprintf(error, error_size, "cannot stat fixture file %s", path);
        return 0;
    }
    *bytes = (size_t)info.st_size;
    return 1;
}

static int read_file_exact(const char *path, void *data, size_t bytes,
                           char *error, size_t error_size) {
    size_t actual = 0;
    if (!file_size_exact(path, &actual, error, error_size)) return 0;
    if (actual != bytes) {
        snprintf(error, error_size,
                 "fixture file %s has %zu bytes, expected %zu",
                 path, actual, bytes);
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "cannot open fixture file %s", path);
        return 0;
    }
    size_t count = fread(data, 1u, bytes, file);
    int close_result = fclose(file);
    int ok = count == bytes && close_result == 0;
    if (!ok)
        snprintf(error, error_size, "cannot read fixture file %s", path);
    return ok;
}

static int write_file_exact(const char *path, const void *data, size_t bytes,
                            char *error, size_t error_size) {
    if (!path || (!data && bytes)) {
        snprintf(error, error_size, "invalid output file arguments");
        return 0;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        snprintf(error, error_size, "cannot open output file %s", path);
        return 0;
    }
    size_t count = fwrite(data, 1u, bytes, file);
    int close_result = fclose(file);
    if (count != bytes || close_result != 0) {
        snprintf(error, error_size, "cannot write output file %s", path);
        return 0;
    }
    return 1;
}

static int ensure_directory(const char *path,
                            char *error, size_t error_size) {
    struct stat info;
    if (stat(path, &info) == 0) {
        if (S_ISDIR(info.st_mode)) return 1;
        snprintf(error, error_size, "output path is not a directory: %s",
                 path);
        return 0;
    }
    if (errno != ENOENT || mkdir(path, 0755) != 0) {
        snprintf(error, error_size, "cannot create output directory %s",
                 path);
        return 0;
    }
    return 1;
}

static int read_fixture_file(const char *directory, const char *name,
                             void *data, size_t bytes,
                             char *error, size_t error_size) {
    char path[4096];
    return fixture_path(path, sizeof(path), directory, name,
                        error, error_size) &&
        read_file_exact(path, data, bytes, error, error_size);
}

static int read_fixture_file_prefix(const char *directory, const char *name,
                                    void *data, size_t bytes,
                                    char *error, size_t error_size) {
    char path[4096];
    size_t actual = 0;
    if (!fixture_path(path, sizeof(path), directory, name,
                      error, error_size) ||
        !file_size_exact(path, &actual, error, error_size)) return 0;
    if (actual < bytes) {
        snprintf(error, error_size,
                 "fixture file %s has %zu bytes, needs prefix %zu",
                 path, actual, bytes);
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "cannot open fixture file %s", path);
        return 0;
    }
    size_t count = fread(data, 1u, bytes, file);
    int close_result = fclose(file);
    if (count != bytes || close_result != 0) {
        snprintf(error, error_size,
                 "cannot read fixture prefix from %s", path);
        return 0;
    }
    return 1;
}

static int fixture_rows(const char *directory, const char *name,
                        size_t row_bytes, uint32_t *rows,
                        char *error, size_t error_size) {
    char path[4096];
    size_t bytes = 0;
    if (!row_bytes ||
        !fixture_path(path, sizeof(path), directory, name,
                      error, error_size) ||
        !file_size_exact(path, &bytes, error, error_size) ||
        !bytes || bytes % row_bytes || bytes / row_bytes > UINT32_MAX) {
        if (!error[0])
            snprintf(error, error_size,
                     "invalid row geometry for fixture file %s", name);
        return 0;
    }
    *rows = (uint32_t)(bytes / row_bytes);
    return 1;
}

static int conditioning_rows(const char *directory,
                             uint32_t video_dim, uint32_t audio_dim,
                             uint32_t *rows,
                             char *error, size_t error_size) {
    size_t video_row_bytes = 0;
    size_t audio_bytes = 0;
    size_t mask_bytes = 0;
    size_t actual_audio_bytes = 0;
    size_t actual_mask_bytes = 0;
    char path[4096];
    if (!directory || !rows ||
        !checked_bytes(video_dim, sizeof(uint16_t), &video_row_bytes) ||
        !fixture_rows(directory, "video_context.bf16", video_row_bytes,
                      rows, error, error_size) ||
        !checked_bytes((uint64_t)*rows * audio_dim, sizeof(uint16_t),
                       &audio_bytes) ||
        !checked_bytes(*rows, sizeof(uint16_t), &mask_bytes) ||
        !fixture_path(path, sizeof(path), directory,
                      "audio_context.bf16", error, error_size) ||
        !file_size_exact(path, &actual_audio_bytes, error, error_size) ||
        actual_audio_bytes != audio_bytes) {
        if (!error[0])
            snprintf(error, error_size,
                     "invalid audio context geometry in %s", directory);
        return 0;
    }
    if (!fixture_path(path, sizeof(path), directory,
                      "text_mask.bf16", error, error_size) ||
        !file_size_exact(path, &actual_mask_bytes, error, error_size) ||
        actual_mask_bytes != mask_bytes) {
        if (!error[0])
            snprintf(error, error_size,
                     "invalid text mask geometry in %s", directory);
        return 0;
    }
    return 1;
}

static int write_gpu_buffer_file(
        const ltx_gpu_buffer *buffer, size_t bytes,
        const char *directory, const char *name,
        char *error, size_t error_size) {
    char path[4096];
    void *host = malloc(bytes);
    if (!host) {
        snprintf(error, error_size,
                 "out of memory exporting generation artifact");
        return 0;
    }
    int ok = fixture_path(path, sizeof(path), directory, name,
                          error, error_size) &&
        ltx_gpu_buffer_read(buffer, host, bytes, error, error_size) &&
        write_file_exact(path, host, bytes, error, error_size);
    free(host);
    return ok;
}

static int write_generation_outputs(
        const char *directory,
        const ltx_gpu_buffer *video_latent,
        const ltx_gpu_buffer *audio_latent,
        const ltx_gpu_buffer *video_pixels,
        uint32_t video_rows, uint32_t video_channels,
        uint32_t audio_rows, uint32_t audio_channels,
        uint32_t text_rows, uint64_t seed, double decode_seconds,
        char *error, size_t error_size) {
    size_t video_bytes = 0;
    size_t audio_bytes = 0;
    size_t pixel_bytes = 0;
    char metadata_path[4096];
    if (!directory || !video_latent || !audio_latent ||
        !checked_bytes((uint64_t)video_rows * video_channels,
                       sizeof(uint16_t), &video_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_channels,
                       sizeof(uint16_t), &audio_bytes) ||
        (video_pixels &&
         !checked_bytes((uint64_t)3u * active_geometry.output_frames *
                        active_geometry.output_height *
                        active_geometry.output_width,
                        sizeof(uint16_t), &pixel_bytes)) ||
        !ensure_directory(directory, error, error_size) ||
        !write_gpu_buffer_file(
            video_latent, video_bytes, directory, "video_latent.bf16",
            error, error_size) ||
        !write_gpu_buffer_file(
            audio_latent, audio_bytes, directory, "audio_latent.bf16",
            error, error_size) ||
        (video_pixels &&
         !write_gpu_buffer_file(
             video_pixels, pixel_bytes, directory, "video_pixels.bf16",
             error, error_size)) ||
        !fixture_path(metadata_path, sizeof(metadata_path), directory,
                      "generation.json", error, error_size)) return 0;

    FILE *metadata = fopen(metadata_path, "w");
    if (!metadata) {
        snprintf(error, error_size, "cannot open output file %s",
                 metadata_path);
        return 0;
    }
    int count = fprintf(
        metadata,
        "{\n"
        "  \"format\": \"ltx-mac-generation-v2\",\n"
        "  \"dtype\": \"bfloat16\",\n"
        "  \"seed\": %llu,\n"
        "  \"text_rows\": %u,\n"
        "  \"requested_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"decoded_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"video\": {\"file\": \"video_latent.bf16\", "
        "\"layout\": \"BFHWC-token-major\", "
        "\"shape\": [1, %u, %u, %u, %u], \"rows\": %u},\n"
        "  \"audio\": {\"file\": \"audio_latent.bf16\", "
        "\"layout\": \"BLC-token-major\", "
        "\"shape\": [1, %u, %u]},\n"
        "  \"native_video\": {\"file\": \"video_pixels.bf16\", "
        "\"layout\": \"BCFHW\", \"dtype\": \"bfloat16\", "
        "\"shape\": [1, 3, %u, %u, %u], "
        "\"decode_seconds\": %.6f}\n"
        "}\n",
        (unsigned long long)seed, text_rows,
        active_geometry.requested_width, active_geometry.requested_height,
        active_geometry.output_frames,
        active_geometry.output_width, active_geometry.output_height,
        active_geometry.output_frames,
        active_geometry.latent_frames, active_geometry.stage2_height,
        active_geometry.stage2_width, video_channels, video_rows,
        audio_rows, audio_channels,
        active_geometry.output_frames, active_geometry.output_height,
        active_geometry.output_width,
        decode_seconds);
    int close_result = fclose(metadata);
    if (count < 0 || close_result != 0) {
        snprintf(error, error_size, "cannot write output file %s",
                 metadata_path);
        return 0;
    }
    return 1;
}

#ifdef LTX_ENABLE_MLX_MEDIA
static int write_generation_outputs_host(
        const char *directory,
        const uint16_t *video_latent,
        const uint16_t *audio_latent,
        const uint16_t *video_pixels,
        uint32_t video_rows, uint32_t video_channels,
        uint32_t audio_rows, uint32_t audio_channels,
        uint32_t text_rows, uint64_t seed, double decode_seconds,
        char *error, size_t error_size) {
    size_t video_bytes = 0;
    size_t audio_bytes = 0;
    size_t pixel_bytes = 0;
    char video_path[4096];
    char audio_path[4096];
    char pixel_path[4096];
    char metadata_path[4096];
    if (!directory || !video_latent || !audio_latent || !video_pixels ||
        !checked_bytes((uint64_t)video_rows * video_channels,
                       sizeof(uint16_t), &video_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_channels,
                       sizeof(uint16_t), &audio_bytes) ||
        !checked_bytes((uint64_t)3u * active_geometry.output_frames *
                       active_geometry.output_height *
                       active_geometry.output_width,
                       sizeof(uint16_t), &pixel_bytes) ||
        !ensure_directory(directory, error, error_size) ||
        !fixture_path(video_path, sizeof(video_path), directory,
                      "video_latent.bf16", error, error_size) ||
        !fixture_path(audio_path, sizeof(audio_path), directory,
                      "audio_latent.bf16", error, error_size) ||
        !fixture_path(pixel_path, sizeof(pixel_path), directory,
                      "video_pixels.bf16", error, error_size) ||
        !fixture_path(metadata_path, sizeof(metadata_path), directory,
                      "generation.json", error, error_size) ||
        !write_file_exact(video_path, video_latent, video_bytes,
                          error, error_size) ||
        !write_file_exact(audio_path, audio_latent, audio_bytes,
                          error, error_size) ||
        !write_file_exact(pixel_path, video_pixels, pixel_bytes,
                          error, error_size)) return 0;

    FILE *metadata = fopen(metadata_path, "w");
    if (!metadata) {
        snprintf(error, error_size, "cannot open output file %s",
                 metadata_path);
        return 0;
    }
    int count = fprintf(
        metadata,
        "{\n"
        "  \"format\": \"ltx-mac-generation-v2\",\n"
        "  \"dtype\": \"bfloat16\",\n"
        "  \"seed\": %llu,\n"
        "  \"text_rows\": %u,\n"
        "  \"requested_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"decoded_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"video\": {\"file\": \"video_latent.bf16\", "
        "\"layout\": \"BFHWC-token-major\", "
        "\"shape\": [1, %u, %u, %u, %u], \"rows\": %u},\n"
        "  \"audio\": {\"file\": \"audio_latent.bf16\", "
        "\"layout\": \"BLC-token-major\", "
        "\"shape\": [1, %u, %u]},\n"
        "  \"native_video\": {\"file\": \"video_pixels.bf16\", "
        "\"layout\": \"BCFHW\", \"dtype\": \"bfloat16\", "
        "\"shape\": [1, 3, %u, %u, %u], "
        "\"decode_seconds\": %.6f}\n"
        "}\n",
        (unsigned long long)seed, text_rows,
        active_geometry.requested_width, active_geometry.requested_height,
        active_geometry.output_frames,
        active_geometry.output_width, active_geometry.output_height,
        active_geometry.output_frames,
        active_geometry.latent_frames, active_geometry.stage2_height,
        active_geometry.stage2_width, video_channels, video_rows,
        audio_rows, audio_channels,
        active_geometry.output_frames, active_geometry.output_height,
        active_geometry.output_width,
        decode_seconds);
    int close_result = fclose(metadata);
    if (count < 0 || close_result != 0) {
        snprintf(error, error_size, "cannot write output file %s",
                 metadata_path);
        return 0;
    }
    return 1;
}

static int write_generation_latents_host(
        const char *directory,
        const uint16_t *video_latent,
        const uint16_t *audio_latent,
        uint32_t video_rows, uint32_t video_channels,
        uint32_t audio_rows, uint32_t audio_channels,
        char *error, size_t error_size) {
    size_t video_bytes = 0;
    size_t audio_bytes = 0;
    char video_path[4096];
    char audio_path[4096];
    return directory && video_latent && audio_latent &&
        checked_bytes((uint64_t)video_rows * video_channels,
                      sizeof(uint16_t), &video_bytes) &&
        checked_bytes((uint64_t)audio_rows * audio_channels,
                      sizeof(uint16_t), &audio_bytes) &&
        ensure_directory(directory, error, error_size) &&
        fixture_path(video_path, sizeof(video_path), directory,
                     "video_latent.bf16", error, error_size) &&
        fixture_path(audio_path, sizeof(audio_path), directory,
                     "audio_latent.bf16", error, error_size) &&
        write_file_exact(video_path, video_latent, video_bytes,
                         error, error_size) &&
        write_file_exact(audio_path, audio_latent, audio_bytes,
                         error, error_size);
}

static int write_generation_metadata_only(
        const char *directory,
        uint32_t video_rows, uint32_t video_channels,
        uint32_t audio_rows, uint32_t audio_channels,
        uint32_t text_rows, uint64_t seed, double decode_seconds,
        char *error, size_t error_size) {
    char metadata_path[4096];
    if (!fixture_path(metadata_path, sizeof(metadata_path), directory,
                      "generation.json", error, error_size)) return 0;
    FILE *metadata = fopen(metadata_path, "w");
    if (!metadata) {
        snprintf(error, error_size, "cannot open output file %s",
                 metadata_path);
        return 0;
    }
    int count = fprintf(
        metadata,
        "{\n"
        "  \"format\": \"ltx-mac-generation-v2\",\n"
        "  \"dtype\": \"bfloat16\",\n"
        "  \"seed\": %llu,\n"
        "  \"text_rows\": %u,\n"
        "  \"requested_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"decoded_video\": {\"width\": %u, \"height\": %u, "
        "\"frames\": %u, \"fps\": 24},\n"
        "  \"video\": {\"file\": \"video_latent.bf16\", "
        "\"layout\": \"BFHWC-token-major\", "
        "\"shape\": [1, %u, %u, %u, %u], \"rows\": %u},\n"
        "  \"audio\": {\"file\": \"audio_latent.bf16\", "
        "\"layout\": \"BLC-token-major\", "
        "\"shape\": [1, %u, %u]},\n"
        "  \"native_video\": {\"file\": \"video_pixels.bf16\", "
        "\"layout\": \"BCFHW\", \"dtype\": \"bfloat16\", "
        "\"shape\": [1, 3, %u, %u, %u], "
        "\"decode_seconds\": %.6f}\n"
        "}\n",
        (unsigned long long)seed, text_rows,
        active_geometry.requested_width, active_geometry.requested_height,
        active_geometry.output_frames,
        active_geometry.output_width, active_geometry.output_height,
        active_geometry.output_frames,
        active_geometry.latent_frames, active_geometry.stage2_height,
        active_geometry.stage2_width, video_channels, video_rows,
        audio_rows, audio_channels,
        active_geometry.output_frames, active_geometry.output_height,
        active_geometry.output_width,
        decode_seconds);
    int close_result = fclose(metadata);
    if (count < 0 || close_result != 0) {
        snprintf(error, error_size, "cannot write output file %s",
                 metadata_path);
        return 0;
    }
    return 1;
}

static int run_mlx_video_vae_helper(
        const char *helper,
        const char *checkpoint,
        const char *directory,
        char *error, size_t error_size) {
    char input_path[4096];
    char output_path[4096];
    char frames_text[16];
    char height_text[16];
    char width_text[16];
    if (!helper || !helper[0] || !checkpoint || !checkpoint[0] ||
        !fixture_path(input_path, sizeof(input_path), directory,
                      "video_latent.bf16", error, error_size) ||
        !fixture_path(output_path, sizeof(output_path), directory,
                      "video_pixels.bf16", error, error_size) ||
        snprintf(frames_text, sizeof(frames_text), "%u",
                 active_geometry.latent_frames) < 0 ||
        snprintf(height_text, sizeof(height_text), "%u",
                 active_geometry.stage2_height) < 0 ||
        snprintf(width_text, sizeof(width_text), "%u",
                 active_geometry.stage2_width) < 0) return 0;
    char *const arguments[] = {
        (char *)helper,
        (char *)checkpoint,
        frames_text,
        height_text,
        width_text,
        (char *)"0",
        input_path,
        output_path,
        NULL,
    };
    pid_t process = 0;
    int spawn_error = posix_spawn(
        &process, helper, NULL, NULL, arguments, environ);
    if (spawn_error != 0) {
        snprintf(error, error_size, "spawn MLX VAE helper: %s",
                 strerror(spawn_error));
        return 0;
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(process, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != process) {
        snprintf(error, error_size, "wait for MLX VAE helper: %s",
                 strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        snprintf(error, error_size,
                 "MLX VAE helper failed with status %d", status);
        return 0;
    }
    return 1;
}

static int start_mlx_video_vae_worker(
        mlx_vae_worker *worker,
        const char *helper,
        const char *checkpoint,
        const char *directory,
        char *error, size_t error_size) {
    char input_path[4096];
    char output_path[4096];
    char frames_text[16];
    char height_text[16];
    char width_text[16];
    if (!worker || worker->process > 0 || !helper || !helper[0] ||
        !checkpoint || !checkpoint[0] ||
        !fixture_path(input_path, sizeof(input_path), directory,
                      "video_latent.bf16", error, error_size) ||
        !fixture_path(output_path, sizeof(output_path), directory,
                      "video_pixels.bf16", error, error_size) ||
        snprintf(frames_text, sizeof(frames_text), "%u",
                 active_geometry.latent_frames) < 0 ||
        snprintf(height_text, sizeof(height_text), "%u",
                 active_geometry.stage2_height) < 0 ||
        snprintf(width_text, sizeof(width_text), "%u",
                 active_geometry.stage2_width) < 0) return 0;
    char *const arguments[] = {
        (char *)helper,
        (char *)"--resident-worker",
        (char *)checkpoint,
        frames_text,
        height_text,
        width_text,
        input_path,
        output_path,
        NULL,
    };
    int spawn_error = posix_spawn(
        &worker->process, helper, NULL, NULL, arguments, environ);
    if (spawn_error != 0) {
        worker->process = 0;
        snprintf(error, error_size, "spawn resident MLX VAE worker: %s",
                 strerror(spawn_error));
        return 0;
    }
    printf("mlx_vae_worker_spawn pid=%d mode=prewarm-and-stop\n",
           (int)worker->process);
    return 1;
}

static int finish_mlx_video_vae_worker(
        mlx_vae_worker *worker, double *decode_seconds,
        char *error, size_t error_size) {
    if (!worker || worker->process <= 0) {
        snprintf(error, error_size, "resident MLX VAE worker is not running");
        return 0;
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(worker->process, &status, WUNTRACED);
    } while (waited < 0 && errno == EINTR);
    if (waited != worker->process) {
        snprintf(error, error_size, "wait for resident MLX VAE worker: %s",
                 strerror(errno));
        return 0;
    }
    if (!WIFSTOPPED(status) || WSTOPSIG(status) != SIGSTOP) {
        if (WIFEXITED(status))
            snprintf(error, error_size,
                     "resident MLX VAE worker exited before decode: %d",
                     WEXITSTATUS(status));
        else
            snprintf(error, error_size,
                     "resident MLX VAE worker did not reach ready state");
        worker->process = 0;
        return 0;
    }
    double started = now_seconds();
    if (kill(worker->process, SIGCONT) != 0) {
        snprintf(error, error_size, "resume resident MLX VAE worker: %s",
                 strerror(errno));
        return 0;
    }
    do {
        waited = waitpid(worker->process, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (decode_seconds) *decode_seconds = now_seconds() - started;
    worker->process = 0;
    if (waited <= 0) {
        snprintf(error, error_size, "wait for resident MLX VAE decode: %s",
                 strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        snprintf(error, error_size,
                 "resident MLX VAE decode failed with status %d", status);
        return 0;
    }
    return 1;
}

static void stop_mlx_video_vae_worker(mlx_vae_worker *worker) {
    if (!worker || worker->process <= 0) return;
    (void)kill(worker->process, SIGKILL);
    int status = 0;
    while (waitpid(worker->process, &status, 0) < 0 && errno == EINTR) {}
    worker->process = 0;
}

static int exec_mlx_video_vae_helper(
        const char *helper,
        const char *checkpoint,
        const char *directory,
        uint64_t seed,
        uint32_t text_rows,
        double two_stage_elapsed_before_exec,
        double boundary_seconds,
        char *error, size_t error_size) {
    char input_path[4096];
    char output_path[4096];
    char seed_text[32];
    char rows_text[32];
    char elapsed_text[64];
    char boundary_text[64];
    char frames_text[16];
    char height_text[16];
    char width_text[16];
    char requested_width_text[16];
    char requested_height_text[16];
    if (!helper || !helper[0] || !checkpoint || !checkpoint[0] ||
        !fixture_path(input_path, sizeof(input_path), directory,
                      "video_latent.bf16", error, error_size) ||
        !fixture_path(output_path, sizeof(output_path), directory,
                      "video_pixels.bf16", error, error_size) ||
        snprintf(seed_text, sizeof(seed_text), "%llu",
                 (unsigned long long)seed) < 0 ||
        snprintf(rows_text, sizeof(rows_text), "%u", text_rows) < 0 ||
        snprintf(elapsed_text, sizeof(elapsed_text), "%.9f",
                 two_stage_elapsed_before_exec) < 0 ||
        snprintf(boundary_text, sizeof(boundary_text), "%.9f",
                 boundary_seconds) < 0 ||
        snprintf(frames_text, sizeof(frames_text), "%u",
                 active_geometry.latent_frames) < 0 ||
        snprintf(height_text, sizeof(height_text), "%u",
                 active_geometry.stage2_height) < 0 ||
        snprintf(width_text, sizeof(width_text), "%u",
                 active_geometry.stage2_width) < 0 ||
        snprintf(requested_width_text, sizeof(requested_width_text), "%u",
                 active_geometry.requested_width) < 0 ||
        snprintf(requested_height_text, sizeof(requested_height_text), "%u",
                 active_geometry.requested_height) < 0) return 0;
    if (setenv("LTX_MLX_FINALIZE_DIR", directory, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_SEED", seed_text, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_TEXT_ROWS", rows_text, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_PREDECODE_SECONDS", elapsed_text, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_BOUNDARY_SECONDS", boundary_text, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_REQUESTED_WIDTH",
               requested_width_text, 1) != 0 ||
        setenv("LTX_MLX_FINALIZE_REQUESTED_HEIGHT",
               requested_height_text, 1) != 0) {
        snprintf(error, error_size, "set MLX finalize environment: %s",
                 strerror(errno));
        return 0;
    }
    char *const arguments[] = {
        (char *)helper,
        (char *)checkpoint,
        frames_text,
        height_text,
        width_text,
        (char *)"0",
        input_path,
        output_path,
        NULL,
    };
    fflush(NULL);
    execve(helper, arguments, environ);
    snprintf(error, error_size, "exec MLX VAE helper: %s", strerror(errno));
    return 0;
}
#endif

static comparison_metrics compare_bf16_f32(
        const uint16_t *candidate, const float *reference,
        uint64_t elements) {
    comparison_metrics metrics = {0};
    double difference2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    for (uint64_t index = 0; index < elements; index++) {
        double expected = reference[index];
        double actual = bf16_to_f32(candidate[index]);
        if (!isfinite(actual)) {
            metrics.nonfinite++;
            continue;
        }
        double difference = actual - expected;
        double absolute = fabs(difference);
        if (absolute > metrics.max_abs) metrics.max_abs = absolute;
        difference2 += difference * difference;
        reference2 += expected * expected;
        candidate2 += actual * actual;
        dot += expected * actual;
    }
    metrics.rel_l2 = reference2 > 0.0 ?
        sqrt(difference2 / reference2) : 0.0;
    metrics.cosine = reference2 > 0.0 && candidate2 > 0.0 ?
        dot / sqrt(reference2 * candidate2) : 0.0;
    metrics.reference_rms = elements ?
        sqrt(reference2 / (double)elements) : 0.0;
    metrics.candidate_rms = elements ?
        sqrt(candidate2 / (double)elements) : 0.0;
    return metrics;
}

static int make_name(char *name, size_t name_size,
                     const char *prefix, const char *suffix,
                     char *error, size_t error_size) {
    int length = snprintf(name, name_size, "%s.%s", prefix, suffix);
    if (length < 0 || (size_t)length >= name_size) {
        snprintf(error, error_size, "tensor name is too long");
        return 0;
    }
    return 1;
}

static const ltx_st_tensor *find_tensor(
        const ltx_st_header *header, const char *prefix, const char *suffix,
        char *error, size_t error_size) {
    char name[1024];
    if (!make_name(name, sizeof(name), prefix, suffix,
                   error, error_size)) return NULL;
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor) snprintf(error, error_size, "missing %s", name);
    return tensor;
}

static void free_linear(gpu_linear *linear) {
    ltx_gpu_buffer_free(linear->weight);
    ltx_gpu_buffer_free(linear->scale);
    ltx_gpu_buffer_free(linear->bias);
    memset(linear, 0, sizeof(*linear));
}

static int load_linear(const ltx_st_header *header,
                       const ltx_st_mapping *mapping,
                       ltx_gpu *gpu, const char *prefix,
                       gpu_linear *linear,
                       char *error, size_t error_size) {
    memset(linear, 0, sizeof(*linear));
    ltx_linear_weight_info info;
    if (!ltx_linear_weight_resolve(header, mapping, prefix, &info,
                                   error, error_size)) return 0;
    if (!info.quantized_int8 || !info.convrot ||
        info.convrot_group_size != 256u ||
        !info.weight_scale || info.weight->dtype != LTX_DTYPE_I8 ||
        info.weight_scale->dtype != LTX_DTYPE_F32 ||
        (info.bias && info.bias->dtype != LTX_DTYPE_BF16)) {
        snprintf(error, error_size,
                 "%s is not a supported ConvRot INT8 linear", prefix);
        return 0;
    }
    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
    size_t bias_bytes = 0;
    const void *weight = ltx_st_map_tensor(
        mapping, info.weight, &weight_bytes, error, error_size);
    const void *scale = ltx_st_map_tensor(
        mapping, info.weight_scale, &scale_bytes, error, error_size);
    const void *bias = info.bias ? ltx_st_map_tensor(
        mapping, info.bias, &bias_bytes, error, error_size) : NULL;
    if (!weight || !scale || (info.bias && !bias)) return 0;
    linear->weight = ltx_gpu_buffer_new_copy(
        gpu, weight, weight_bytes, error, error_size);
    linear->scale = ltx_gpu_buffer_new_copy(
        gpu, scale, scale_bytes, error, error_size);
    if (bias)
        linear->bias = ltx_gpu_buffer_new_copy(
            gpu, bias, bias_bytes, error, error_size);
    linear->input_dim = info.input_dim;
    linear->output_dim = info.output_dim;
    linear->weight_bytes = weight_bytes;
    if (!linear->weight || !linear->scale || (bias && !linear->bias)) {
        free_linear(linear);
        return 0;
    }
    return 1;
}

static void free_attention(attention_weights *attention) {
    free_linear(&attention->query);
    free_linear(&attention->key);
    free_linear(&attention->value);
    free_linear(&attention->output);
    ltx_gpu_buffer_free(attention->query_norm);
    ltx_gpu_buffer_free(attention->key_norm);
    ltx_gpu_buffer_free(attention->gate_weight);
    ltx_gpu_buffer_free(attention->gate_bias);
    memset(attention, 0, sizeof(*attention));
}

static ltx_gpu_buffer *upload_tensor(
        const ltx_st_mapping *mapping, const ltx_st_tensor *tensor,
        ltx_gpu *gpu, char *error, size_t error_size) {
    size_t bytes = 0;
    const void *data = ltx_st_map_tensor(
        mapping, tensor, &bytes, error, error_size);
    return data ? ltx_gpu_buffer_new_copy(
        gpu, data, bytes, error, error_size) : NULL;
}

static int load_attention(const ltx_st_header *header,
                          const ltx_st_mapping *mapping,
                          ltx_gpu *gpu, const char *prefix,
                          attention_weights *attention,
                          char *error, size_t error_size) {
    memset(attention, 0, sizeof(*attention));
    char linear_prefix[1024];
#define LTX_LOAD_PROJECTION(FIELD, SUFFIX) \
    (make_name(linear_prefix, sizeof(linear_prefix), prefix, (SUFFIX), \
               error, error_size) && \
     load_linear(header, mapping, gpu, linear_prefix, \
                 &(attention)->FIELD, error, error_size))
    if (!LTX_LOAD_PROJECTION(query, "to_q") ||
        !LTX_LOAD_PROJECTION(key, "to_k") ||
        !LTX_LOAD_PROJECTION(value, "to_v") ||
        !LTX_LOAD_PROJECTION(output, "to_out.0")) {
#undef LTX_LOAD_PROJECTION
        free_attention(attention);
        return 0;
    }
#undef LTX_LOAD_PROJECTION
    attention->query_dim = attention->query.input_dim;
    attention->key_value_dim = attention->key.input_dim;
    attention->inner_dim = attention->query.output_dim;
    attention->output_dim = attention->output.output_dim;
    if (attention->key.output_dim != attention->inner_dim ||
        attention->value.input_dim != attention->key_value_dim ||
        attention->value.output_dim != attention->inner_dim ||
        attention->output.input_dim != attention->inner_dim) {
        snprintf(error, error_size, "%s has incompatible projections", prefix);
        free_attention(attention);
        return 0;
    }
    const ltx_st_tensor *query_norm = find_tensor(
        header, prefix, "q_norm.weight", error, error_size);
    const ltx_st_tensor *key_norm = find_tensor(
        header, prefix, "k_norm.weight", error, error_size);
    const ltx_st_tensor *gate_weight = find_tensor(
        header, prefix, "to_gate_logits.weight", error, error_size);
    const ltx_st_tensor *gate_bias = find_tensor(
        header, prefix, "to_gate_logits.bias", error, error_size);
    if (!query_norm || !key_norm || !gate_weight || !gate_bias ||
        query_norm->dtype != LTX_DTYPE_BF16 || query_norm->ndim != 1u ||
        query_norm->shape[0] != attention->inner_dim ||
        key_norm->dtype != LTX_DTYPE_BF16 || key_norm->ndim != 1u ||
        key_norm->shape[0] != attention->inner_dim ||
        gate_weight->dtype != LTX_DTYPE_BF16 || gate_weight->ndim != 2u ||
        gate_weight->shape[1] != attention->query_dim ||
        !gate_weight->shape[0] || gate_weight->shape[0] > UINT32_MAX ||
        gate_bias->dtype != LTX_DTYPE_BF16 || gate_bias->ndim != 1u ||
        gate_bias->shape[0] != gate_weight->shape[0]) {
        snprintf(error, error_size, "%s has invalid norm/gate tensors", prefix);
        free_attention(attention);
        return 0;
    }
    attention->heads = (uint32_t)gate_weight->shape[0];
    if (attention->inner_dim % attention->heads) {
        snprintf(error, error_size, "%s has invalid head geometry", prefix);
        free_attention(attention);
        return 0;
    }
    attention->head_dim = attention->inner_dim / attention->heads;
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
        free_attention(attention);
        return 0;
    }
    return 1;
}

static void free_mlp(mlp_weights *mlp) {
    free_linear(&mlp->fc1);
    free_linear(&mlp->fc2);
    memset(mlp, 0, sizeof(*mlp));
}

static int load_mlp(const ltx_st_header *header,
                    const ltx_st_mapping *mapping,
                    ltx_gpu *gpu, const char *prefix,
                    mlp_weights *mlp,
                    char *error, size_t error_size) {
    memset(mlp, 0, sizeof(*mlp));
    char fc1_prefix[1024];
    char fc2_prefix[1024];
    if (!make_name(fc1_prefix, sizeof(fc1_prefix), prefix, "net.0.proj",
                   error, error_size) ||
        !make_name(fc2_prefix, sizeof(fc2_prefix), prefix, "net.2",
                   error, error_size) ||
        !load_linear(header, mapping, gpu, fc1_prefix, &mlp->fc1,
                     error, error_size) ||
        !load_linear(header, mapping, gpu, fc2_prefix, &mlp->fc2,
                     error, error_size)) {
        free_mlp(mlp);
        return 0;
    }
    mlp->input_dim = mlp->fc1.input_dim;
    mlp->hidden_dim = mlp->fc1.output_dim;
    mlp->output_dim = mlp->fc2.output_dim;
    if (mlp->fc2.input_dim != mlp->hidden_dim ||
        mlp->output_dim != mlp->input_dim) {
        snprintf(error, error_size, "%s has incompatible MLP geometry", prefix);
        free_mlp(mlp);
        return 0;
    }
    return 1;
}

static void free_table(parameter_table *table) {
    for (uint32_t row = 0; row < table->rows; row++)
        ltx_gpu_buffer_free(table->row[row]);
    free(table->base_values);
    memset(table, 0, sizeof(*table));
}

static int load_table(const ltx_st_header *header,
                      const ltx_st_mapping *mapping,
                      ltx_gpu *gpu, const char *prefix,
                      const char *suffix, uint32_t expected_rows,
                      uint32_t expected_columns,
                      parameter_table *table,
                      char *error, size_t error_size) {
    memset(table, 0, sizeof(*table));
    const ltx_st_tensor *tensor = find_tensor(
        header, prefix, suffix, error, error_size);
    if (!tensor || expected_rows > LTX_MAX_TABLE_ROWS ||
        tensor->dtype != LTX_DTYPE_F32 || tensor->ndim != 2u ||
        tensor->shape[0] != expected_rows ||
        tensor->shape[1] != expected_columns) {
        snprintf(error, error_size, "invalid %s.%s", prefix, suffix);
        return 0;
    }
    size_t mapped_bytes = 0;
    const float *values = ltx_st_map_tensor(
        mapping, tensor, &mapped_bytes, error, error_size);
    size_t expected_bytes = 0;
    if (!values ||
        !checked_bytes((uint64_t)expected_rows * expected_columns,
                       sizeof(float), &expected_bytes) ||
        mapped_bytes != expected_bytes) return 0;
    size_t converted_bytes = 0;
    if (!checked_bytes((uint64_t)expected_rows * expected_columns,
                       sizeof(uint16_t), &converted_bytes)) {
        snprintf(error, error_size, "AdaLN table conversion overflow");
        return 0;
    }
    uint16_t *converted = malloc(converted_bytes);
    if (!converted) {
        snprintf(error, error_size, "out of memory converting AdaLN table");
        return 0;
    }
    table->rows = expected_rows;
    table->columns = expected_columns;
    table->base_values = converted;
    for (uint32_t row = 0; row < expected_rows; row++) {
        for (uint32_t column = 0; column < expected_columns; column++)
            converted[(uint64_t)row * expected_columns + column] =
                f32_to_bf16(
                values[(uint64_t)row * expected_columns + column]);
        table->row[row] = ltx_gpu_buffer_new_copy(
            gpu, converted + (uint64_t)row * expected_columns,
            (size_t)expected_columns * sizeof(uint16_t),
            error, error_size);
        if (!table->row[row]) {
            free_table(table);
            return 0;
        }
    }
    return 1;
}

static void free_block_weights(block_weights *weights) {
#ifdef LTX_ENABLE_ANE_MLP
    ltx_ane_mlp *stage1_mlp = weights->ane_video_mlp[0];
    ltx_ane_mlp *stage2_mlp = weights->ane_video_mlp[1];
    weights->ane_video_mlp[0] = NULL;
    weights->ane_video_mlp[1] = NULL;
    ltx_ane_mlp_free(stage1_mlp);
    if (stage2_mlp != stage1_mlp) ltx_ane_mlp_free(stage2_mlp);
#endif
#ifdef LTX_ENABLE_ANE_V2A
    ltx_ane_v2a_free(weights->ane_v2a[0]);
    ltx_ane_v2a_free(weights->ane_v2a[1]);
    weights->ane_v2a[0] = NULL;
    weights->ane_v2a[1] = NULL;
#endif
#ifdef LTX_ENABLE_ANE_KV
    ltx_ane_kv_free(weights->ane_video_text_kv);
    weights->ane_video_text_kv = NULL;
#endif
#ifdef LTX_ENABLE_ANE_QKV
    ltx_ane_qkv *stage1_qkv = weights->ane_video_self_qkv[0];
    ltx_ane_qkv *stage2_qkv = weights->ane_video_self_qkv[1];
    weights->ane_video_self_qkv[0] = NULL;
    weights->ane_video_self_qkv[1] = NULL;
    ltx_ane_qkv_free(stage1_qkv);
    if (stage2_qkv != stage1_qkv) ltx_ane_qkv_free(stage2_qkv);
#endif
    free_attention(&weights->video_self);
    free_attention(&weights->audio_self);
    free_attention(&weights->video_text);
    free_attention(&weights->audio_text);
    free_attention(&weights->audio_to_video);
    free_attention(&weights->video_to_audio);
    free_mlp(&weights->video_mlp);
    free_mlp(&weights->audio_mlp);
    free_table(&weights->video_adaln);
    free_table(&weights->video_adaln_conditioned);
    free_table(&weights->audio_adaln);
    free_table(&weights->video_prompt);
    free_table(&weights->audio_prompt);
    free_table(&weights->av_video);
    free_table(&weights->av_video_conditioned);
    free_table(&weights->av_audio);
}

static int load_block_weights(const ltx_st_header *header,
                              const ltx_st_mapping *mapping,
                              ltx_gpu *gpu, uint32_t block,
                              block_weights *weights,
                              char *error, size_t error_size) {
    memset(weights, 0, sizeof(*weights));
    char prefix[1024];
    int length = snprintf(prefix, sizeof(prefix),
        "model.diffusion_model.transformer_blocks.%u", block);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        snprintf(error, error_size, "block prefix is too long");
        return 0;
    }
    char module[1024];
#define LTX_LOAD_ATTENTION(FIELD, SUFFIX) \
    (make_name(module, sizeof(module), prefix, (SUFFIX), \
               error, error_size) && \
     load_attention(header, mapping, gpu, module, \
                    &(weights)->FIELD, error, error_size))
#define LTX_LOAD_MLP(FIELD, SUFFIX) \
    (make_name(module, sizeof(module), prefix, (SUFFIX), \
               error, error_size) && \
     load_mlp(header, mapping, gpu, module, \
              &(weights)->FIELD, error, error_size))
    if (!LTX_LOAD_ATTENTION(video_self, "attn1") ||
        !LTX_LOAD_ATTENTION(audio_self, "audio_attn1") ||
        !LTX_LOAD_ATTENTION(video_text, "attn2") ||
        !LTX_LOAD_ATTENTION(audio_text, "audio_attn2") ||
        !LTX_LOAD_ATTENTION(audio_to_video, "audio_to_video_attn") ||
        !LTX_LOAD_ATTENTION(video_to_audio, "video_to_audio_attn") ||
        !LTX_LOAD_MLP(video_mlp, "ff") ||
        !LTX_LOAD_MLP(audio_mlp, "audio_ff")) {
#undef LTX_LOAD_ATTENTION
#undef LTX_LOAD_MLP
        free_block_weights(weights);
        return 0;
    }
#undef LTX_LOAD_ATTENTION
#undef LTX_LOAD_MLP
    uint32_t video_dim = weights->video_self.query_dim;
    uint32_t audio_dim = weights->audio_self.query_dim;
    if (weights->video_self.key_value_dim != video_dim ||
        weights->video_self.output_dim != video_dim ||
        weights->video_text.query_dim != video_dim ||
        weights->video_text.key_value_dim != video_dim ||
        weights->video_text.output_dim != video_dim ||
        weights->audio_self.key_value_dim != audio_dim ||
        weights->audio_self.output_dim != audio_dim ||
        weights->audio_text.query_dim != audio_dim ||
        weights->audio_text.key_value_dim != audio_dim ||
        weights->audio_text.output_dim != audio_dim ||
        weights->audio_to_video.query_dim != video_dim ||
        weights->audio_to_video.key_value_dim != audio_dim ||
        weights->audio_to_video.output_dim != video_dim ||
        weights->video_to_audio.query_dim != audio_dim ||
        weights->video_to_audio.key_value_dim != video_dim ||
        weights->video_to_audio.output_dim != audio_dim ||
        weights->video_mlp.input_dim != video_dim ||
        weights->audio_mlp.input_dim != audio_dim) {
        snprintf(error, error_size, "block %u has incompatible geometry", block);
        free_block_weights(weights);
        return 0;
    }
    if (!load_table(header, mapping, gpu, prefix, "scale_shift_table",
                    9u, video_dim, &weights->video_adaln,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix, "scale_shift_table",
                    9u, video_dim, &weights->video_adaln_conditioned,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix, "audio_scale_shift_table",
                    9u, audio_dim, &weights->audio_adaln,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix, "prompt_scale_shift_table",
                    2u, video_dim, &weights->video_prompt,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix,
                    "audio_prompt_scale_shift_table",
                    2u, audio_dim, &weights->audio_prompt,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix,
                    "scale_shift_table_a2v_ca_video",
                    5u, video_dim, &weights->av_video,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix,
                    "scale_shift_table_a2v_ca_video",
                    5u, video_dim, &weights->av_video_conditioned,
                    error, error_size) ||
        !load_table(header, mapping, gpu, prefix,
                    "scale_shift_table_a2v_ca_audio",
                    5u, audio_dim, &weights->av_audio,
                    error, error_size)) {
        free_block_weights(weights);
        return 0;
    }
    return 1;
}

static int add_parameters_to_table(
        parameter_table *table, uint32_t first_row, uint32_t row_count,
        const uint16_t *parameters,
        char *error, size_t error_size) {
    if (!table || !parameters || !row_count ||
        first_row > table->rows || row_count > table->rows - first_row) {
        snprintf(error, error_size, "invalid AdaLN table update");
        return 0;
    }
    size_t row_bytes = (size_t)table->columns * sizeof(uint16_t);
    uint16_t *row_values = malloc(row_bytes);
    if (!row_values) {
        snprintf(error, error_size, "out of memory updating AdaLN table");
        return 0;
    }
    int ok = 1;
    for (uint32_t row = 0; ok && row < row_count; row++) {
        const uint16_t *base = table->base_values +
            (uint64_t)(first_row + row) * table->columns;
        for (uint32_t column = 0; ok && column < table->columns; column++) {
            uint64_t index = (uint64_t)row * table->columns + column;
            row_values[column] = f32_to_bf16(
                bf16_to_f32(base[column]) +
                bf16_to_f32(parameters[index]));
        }
        ok = ltx_gpu_buffer_write(
            table->row[first_row + row], row_values,
            row_bytes, error, error_size);
    }
    free(row_values);
    return ok;
}

static int apply_scalar_conditioning(
        ltx_transformer_conditioning *conditioning,
        block_weights *weights, uint32_t block_count, float sigma,
        ltx_transformer_conditioning_values *values,
        char *error, size_t error_size) {
    uint32_t video_dim = weights[0].video_self.query_dim;
    uint32_t audio_dim = weights[0].audio_self.query_dim;
    int ok = ltx_transformer_conditioning_eval_scalar(
        conditioning, sigma, values, error, error_size);
    if (ok && (values->video_dim != video_dim ||
               values->audio_dim != audio_dim)) {
        snprintf(error, error_size,
                 "Transformer conditioning dimensions differ from blocks");
        ok = 0;
    }
    for (uint32_t block = 0; ok && block < block_count; block++)
        ok = add_parameters_to_table(
                &weights[block].video_adaln, 0u, 9u,
                values->video_adaln, error, error_size) &&
            add_parameters_to_table(
                &weights[block].audio_adaln, 0u, 9u,
                values->audio_adaln, error, error_size) &&
            add_parameters_to_table(
                &weights[block].video_prompt, 0u, 2u,
                values->video_prompt, error, error_size) &&
            add_parameters_to_table(
                &weights[block].audio_prompt, 0u, 2u,
                values->audio_prompt, error, error_size) &&
            add_parameters_to_table(
                &weights[block].av_video, 0u, 4u,
                values->av_video, error, error_size) &&
            add_parameters_to_table(
                &weights[block].av_video, 4u, 1u,
                values->a2v_gate, error, error_size) &&
            add_parameters_to_table(
                &weights[block].av_audio, 0u, 4u,
                values->av_audio, error, error_size) &&
            add_parameters_to_table(
                &weights[block].av_audio, 4u, 1u,
                values->v2a_gate, error, error_size);
    return ok;
}

static int apply_video_split_conditioning(
        ltx_transformer_conditioning *conditioning,
        block_weights *weights, uint32_t block_count, float sigma,
        float conditioned_sigma,
        ltx_gpu_buffer *conditioned_video_embedded,
        ltx_transformer_conditioning_values *values,
        char *error, size_t error_size) {
    if (!active_video_conditioned_prefix_rows)
        return apply_scalar_conditioning(
            conditioning, weights, block_count, sigma, values,
            error, error_size);
    if (!conditioned_video_embedded || !isfinite(conditioned_sigma) ||
        conditioned_sigma < 0.0f || conditioned_sigma > sigma) {
        snprintf(error, error_size,
                 "invalid split video conditioning arguments");
        return 0;
    }
    ltx_transformer_conditioning_values conditioned = {0};
    int ok = ltx_transformer_conditioning_eval_scalar(
        conditioning, conditioned_sigma, &conditioned, error, error_size);
    uint32_t video_dim = weights[0].video_self.query_dim;
    if (ok && conditioned.video_dim != video_dim) {
        snprintf(error, error_size,
                 "conditioned video dimension differs from blocks");
        ok = 0;
    }
    for (uint32_t block = 0; ok && block < block_count; block++)
        ok = add_parameters_to_table(
                &weights[block].video_adaln_conditioned, 0u, 9u,
                conditioned.video_adaln, error, error_size) &&
            add_parameters_to_table(
                &weights[block].av_video_conditioned, 0u, 4u,
                conditioned.av_video, error, error_size);
    size_t embedded_bytes = (size_t)video_dim * sizeof(uint16_t);
    uint16_t *embedded_host = ok ? malloc(embedded_bytes) : NULL;
    if (ok && !embedded_host) {
        snprintf(error, error_size,
                 "out of memory copying conditioned video embedding");
        ok = 0;
    }
    if (ok)
        ok = ltx_gpu_buffer_read(
                conditioned.video_embedded, embedded_host, embedded_bytes,
                error, error_size) &&
            ltx_gpu_buffer_write(
                conditioned_video_embedded, embedded_host, embedded_bytes,
                error, error_size);
    free(embedded_host);
    return ok && apply_scalar_conditioning(
        conditioning, weights, block_count, sigma, values,
        error, error_size);
}

static int video_adaln(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input,
        const parameter_table *generated,
        const parameter_table *conditioned,
        uint32_t scale_row, uint32_t shift_row,
        uint32_t rows, uint32_t columns,
        char *error, size_t error_size) {
    if (!active_video_conditioned_prefix_rows)
        return ltx_gpu_adaln_bf16(
            gpu, output, input, generated->row[scale_row],
            generated->row[shift_row], rows, columns, 1u, 1e-6f,
            error, error_size);
    return ltx_gpu_adaln_bf16_split(
        gpu, output, input, generated->row[scale_row],
        generated->row[shift_row], conditioned->row[scale_row],
        conditioned->row[shift_row], rows, columns,
        active_video_conditioned_prefix_rows, 1e-6f, error, error_size);
}

static int video_affine(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input,
        const parameter_table *generated,
        const parameter_table *conditioned,
        uint32_t scale_row, uint32_t shift_row,
        uint32_t rows, uint32_t columns,
        char *error, size_t error_size) {
    if (!active_video_conditioned_prefix_rows)
        return ltx_gpu_affine_bf16(
            gpu, output, input, generated->row[scale_row],
            generated->row[shift_row], rows, columns, 1u,
            error, error_size);
    return ltx_gpu_affine_bf16_split(
        gpu, output, input, generated->row[scale_row],
        generated->row[shift_row], conditioned->row[scale_row],
        conditioned->row[shift_row], rows, columns,
        active_video_conditioned_prefix_rows, error, error_size);
}

static int video_residual_gate(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *residual,
        const ltx_gpu_buffer *branch,
        const parameter_table *generated,
        const parameter_table *conditioned,
        uint32_t gate_row, uint32_t rows, uint32_t columns,
        char *error, size_t error_size) {
    if (!active_video_conditioned_prefix_rows)
        return ltx_gpu_residual_gate_bf16(
            gpu, output, residual, branch, generated->row[gate_row],
            rows, columns, 1u, error, error_size);
    return ltx_gpu_residual_gate_bf16_split(
        gpu, output, residual, branch, generated->row[gate_row],
        conditioned->row[gate_row], rows, columns,
        active_video_conditioned_prefix_rows, error, error_size);
}

static int video_geometry(uint32_t rows, uint32_t *frames,
                          uint32_t *height, uint32_t *width) {
    if (rows == active_geometry.stage1_rows) {
        *frames = active_geometry.latent_frames;
        *height = active_geometry.stage1_height;
        *width = active_geometry.stage1_width;
        return 1;
    }
    if (rows == active_geometry.stage2_rows) {
        *frames = active_geometry.latent_frames;
        *height = active_geometry.stage2_height;
        *width = active_geometry.stage2_width;
        return 1;
    }
    if (rows == 1001u) {
        *frames = 13u; *height = 7u; *width = 11u;
        return 1;
    }
    if (rows == 4004u) {
        *frames = 13u; *height = 14u; *width = 22u;
        return 1;
    }
    return 0;
}

static void free_rope_pair(rope_pair *rope) {
    ltx_gpu_buffer_free(rope->cosine);
    ltx_gpu_buffer_free(rope->sine);
    memset(rope, 0, sizeof(*rope));
}

static int upload_rope(ltx_gpu *gpu, const float *positions,
                       uint32_t rows, uint32_t axes,
                       uint32_t heads, uint32_t head_dim,
                       const float *max_positions, rope_pair *rope,
                       char *error, size_t error_size) {
    uint64_t elements = (uint64_t)heads * rows * (head_dim / 2u);
    size_t bytes = 0;
    if (!checked_bytes(elements, sizeof(uint16_t), &bytes)) {
        snprintf(error, error_size, "RoPE allocation overflow");
        return 0;
    }
    uint16_t *cosine = malloc(bytes);
    uint16_t *sine = malloc(bytes);
    if (!cosine || !sine || !ltx_compute_rope_split_bf16(
            cosine, sine, elements, positions, rows, axes,
            heads, head_dim, 10000.0, max_positions, 1,
            error, error_size)) {
        free(cosine);
        free(sine);
        return 0;
    }
    rope->cosine = ltx_gpu_buffer_new_copy(
        gpu, cosine, bytes, error, error_size);
    rope->sine = ltx_gpu_buffer_new_copy(
        gpu, sine, bytes, error, error_size);
    free(cosine);
    free(sine);
    if (!rope->cosine || !rope->sine) {
        free_rope_pair(rope);
        return 0;
    }
    return 1;
}

static void free_block_rope(block_rope *rope) {
    free_rope_pair(&rope->video_self);
    free_rope_pair(&rope->audio_self);
    free_rope_pair(&rope->video_cross);
    free_rope_pair(&rope->audio_cross);
}

static int create_block_rope_from_positions(
        ltx_gpu *gpu, const block_weights *weights,
        const float *video_positions, const float *audio_positions,
        uint32_t video_rows, uint32_t audio_rows,
        block_rope *rope, char *error, size_t error_size) {
    memset(rope, 0, sizeof(*rope));
    float *video_temporal = malloc((size_t)video_rows * sizeof(float));
    if (!video_temporal) {
        snprintf(error, error_size, "out of memory creating cross RoPE");
        return 0;
    }
    for (uint32_t row = 0; row < video_rows; row++)
        video_temporal[row] = video_positions[(uint64_t)row * 3u];
    float video_max[3] = {20.0f, 2048.0f, 2048.0f};
    float temporal_max[1] = {20.0f};
    int ok = upload_rope(
            gpu, video_positions, video_rows, 3u,
            weights->video_self.heads, weights->video_self.head_dim,
            video_max, &rope->video_self, error, error_size) &&
        upload_rope(
            gpu, audio_positions, audio_rows, 1u,
            weights->audio_self.heads, weights->audio_self.head_dim,
            temporal_max, &rope->audio_self, error, error_size) &&
        upload_rope(
            gpu, video_temporal, video_rows, 1u,
            weights->audio_to_video.heads,
            weights->audio_to_video.head_dim,
            temporal_max, &rope->video_cross, error, error_size) &&
        upload_rope(
            gpu, audio_positions, audio_rows, 1u,
            weights->audio_to_video.heads,
            weights->audio_to_video.head_dim,
            temporal_max, &rope->audio_cross, error, error_size);
    free(video_temporal);
    if (!ok) free_block_rope(rope);
    return ok;
}

static int create_block_rope(ltx_gpu *gpu, const block_weights *weights,
                             uint32_t video_rows, uint32_t audio_rows,
                             block_rope *rope,
                             char *error, size_t error_size) {
    uint32_t frames = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    if (!video_geometry(video_rows, &frames, &height, &width)) {
        snprintf(error, error_size,
                 "video rows %u do not match configured geometries %u/%u",
                 video_rows, active_geometry.stage1_rows,
                 active_geometry.stage2_rows);
        return 0;
    }
    float *video_positions = malloc(
        (size_t)video_rows * 3u * sizeof(float));
    float *audio_positions = malloc((size_t)audio_rows * sizeof(float));
    if (!video_positions || !audio_positions ||
        !ltx_compute_video_positions(
            video_positions, (uint64_t)video_rows * 3u,
            frames, height, width, 24.0f, error, error_size) ||
        !ltx_compute_audio_positions(
            audio_positions, audio_rows, audio_rows, error, error_size)) {
        free(video_positions);
        free(audio_positions);
        return 0;
    }
    int ok = create_block_rope_from_positions(
        gpu, weights, video_positions, audio_positions,
        video_rows, audio_rows, rope, error, error_size);
    free(video_positions);
    free(audio_positions);
    return ok;
}

static ltx_gpu_buffer *new_tensor(ltx_gpu *gpu,
                                  uint32_t rows, uint32_t columns,
                                  char *error, size_t error_size) {
    size_t bytes = 0;
    if (!checked_bytes((uint64_t)rows * columns,
                       sizeof(uint16_t), &bytes)) return NULL;
    return ltx_gpu_buffer_new(gpu, bytes, error, error_size);
}

static ltx_gpu_buffer *new_elements(ltx_gpu *gpu, uint64_t elements,
                                    size_t element_size,
                                    char *error, size_t error_size) {
    size_t bytes = 0;
    if (!checked_bytes(elements, element_size, &bytes)) {
        snprintf(error, error_size,
                 "Sol attention workspace size overflow");
        return NULL;
    }
    return ltx_gpu_buffer_new(gpu, bytes, error, error_size);
}

static int create_sol_video_self_workspace(
        ltx_gpu *gpu, uint32_t rows, uint32_t hidden,
        uint32_t heads, uint32_t head_dim,
        block_workspace *workspace,
        char *error, size_t error_size) {
    if (!sol_video_self_stage_enabled(rows)) return 1;
    uint32_t blocks = (rows + 63u) / 64u;
    if (blocks > 64u) {
        snprintf(error, error_size,
                 "tiled Sol attention supports at most 4096 rows, got %u",
                 rows);
        return 0;
    }
    uint64_t tensor_elements = (uint64_t)rows * hidden;
    uint64_t gate_elements = (uint64_t)rows * heads;
    uint64_t summary_elements = (uint64_t)heads * blocks * head_dim;
    uint64_t threshold_elements = (uint64_t)heads * blocks;
    uint64_t route_elements = threshold_elements * blocks;
#define LTX_NEW_SOL_TENSOR(FIELD) \
    ((workspace)->FIELD = new_elements( \
        gpu, tensor_elements, sizeof(uint16_t), error, error_size))
    int ok = LTX_NEW_SOL_TENSOR(video_sol_query) &&
        LTX_NEW_SOL_TENSOR(video_sol_key) &&
        LTX_NEW_SOL_TENSOR(video_sol_value) &&
        ((workspace->video_sol_gate_logits = new_elements(
              gpu, gate_elements, sizeof(uint16_t),
              error, error_size)) != NULL) &&
        ((workspace->video_sol_gate = new_elements(
              gpu, gate_elements, sizeof(uint16_t),
              error, error_size)) != NULL) &&
        LTX_NEW_SOL_TENSOR(video_sol_core) &&
        LTX_NEW_SOL_TENSOR(video_sol_rotated) &&
        LTX_NEW_SOL_TENSOR(video_sol_packed_query) &&
        LTX_NEW_SOL_TENSOR(video_sol_packed_key) &&
        LTX_NEW_SOL_TENSOR(video_sol_packed_value) &&
        LTX_NEW_SOL_TENSOR(video_sol_packed_output) &&
        ((workspace->video_sol_query_centroids = new_elements(
              gpu, summary_elements, sizeof(float),
              error, error_size)) != NULL) &&
        ((workspace->video_sol_key_centroids = new_elements(
              gpu, summary_elements, sizeof(uint16_t),
              error, error_size)) != NULL) &&
        ((workspace->video_sol_value_sums = new_elements(
              gpu, summary_elements, sizeof(uint16_t),
              error, error_size)) != NULL) &&
        ((workspace->video_sol_thresholds = new_elements(
              gpu, threshold_elements, sizeof(float),
              error, error_size)) != NULL) &&
        ((workspace->video_sol_routes = new_elements(
              gpu, route_elements, sizeof(float),
              error, error_size)) != NULL);
#undef LTX_NEW_SOL_TENSOR
    return ok;
}

static void free_workspace(block_workspace *workspace) {
    ltx_gpu_buffer_free(workspace->video_state[0]);
    ltx_gpu_buffer_free(workspace->video_state[1]);
    ltx_gpu_buffer_free(workspace->audio_state[0]);
    ltx_gpu_buffer_free(workspace->audio_state[1]);
    ltx_gpu_buffer_free(workspace->video_normed);
    ltx_gpu_buffer_free(workspace->audio_normed);
    ltx_gpu_buffer_free(workspace->video_branch);
    ltx_gpu_buffer_free(workspace->audio_branch);
    ltx_gpu_buffer_free(workspace->video_text_scaled);
    ltx_gpu_buffer_free(workspace->audio_text_scaled);
    ltx_gpu_buffer_free(workspace->video_text_key);
    ltx_gpu_buffer_free(workspace->video_text_value);
    ltx_gpu_buffer_free(workspace->video_norm3);
    ltx_gpu_buffer_free(workspace->audio_norm3);
    ltx_gpu_buffer_free(workspace->video_a2v);
    ltx_gpu_buffer_free(workspace->audio_a2v);
    ltx_gpu_buffer_free(workspace->video_v2a);
    ltx_gpu_buffer_free(workspace->audio_v2a);
    ltx_gpu_buffer_free(workspace->video_sol_query);
    ltx_gpu_buffer_free(workspace->video_sol_key);
    ltx_gpu_buffer_free(workspace->video_sol_value);
    ltx_gpu_buffer_free(workspace->video_sol_gate_logits);
    ltx_gpu_buffer_free(workspace->video_sol_gate);
    ltx_gpu_buffer_free(workspace->video_sol_core);
    ltx_gpu_buffer_free(workspace->video_sol_rotated);
    ltx_gpu_buffer_free(workspace->video_sol_packed_query);
    ltx_gpu_buffer_free(workspace->video_sol_packed_key);
    ltx_gpu_buffer_free(workspace->video_sol_packed_value);
    ltx_gpu_buffer_free(workspace->video_sol_packed_output);
    ltx_gpu_buffer_free(workspace->video_sol_query_centroids);
    ltx_gpu_buffer_free(workspace->video_sol_key_centroids);
    ltx_gpu_buffer_free(workspace->video_sol_value_sums);
    ltx_gpu_buffer_free(workspace->video_sol_thresholds);
    ltx_gpu_buffer_free(workspace->video_sol_routes);
#ifdef LTX_ENABLE_ANE_QKV
    ltx_gpu_buffer_free(workspace->video_qkv_query_prefix);
    ltx_gpu_buffer_free(workspace->video_qkv_key_prefix);
    ltx_gpu_buffer_free(workspace->video_qkv_value_prefix);
    ltx_gpu_buffer_free(workspace->video_qkv_query);
    ltx_gpu_buffer_free(workspace->video_qkv_key);
    ltx_gpu_buffer_free(workspace->video_qkv_value);
    ltx_gpu_buffer_free(workspace->video_qkv_gate_logits);
    ltx_gpu_buffer_free(workspace->video_qkv_gate);
    ltx_gpu_buffer_free(workspace->video_qkv_core);
    ltx_gpu_buffer_free(workspace->video_qkv_rotated);
#endif
    memset(workspace, 0, sizeof(*workspace));
}

static int create_workspace(ltx_gpu *gpu,
                            uint32_t video_rows, uint32_t video_dim,
                            uint32_t video_heads,
                            uint32_t audio_rows, uint32_t audio_dim,
                            uint32_t text_rows,
                            block_workspace *workspace,
                            char *error, size_t error_size) {
    memset(workspace, 0, sizeof(*workspace));
#define LTX_NEW_VIDEO(FIELD) \
    ((workspace)->FIELD = new_tensor( \
        gpu, video_rows, video_dim, error, error_size))
#define LTX_NEW_AUDIO(FIELD) \
    ((workspace)->FIELD = new_tensor( \
        gpu, audio_rows, audio_dim, error, error_size))
    int ok = LTX_NEW_VIDEO(video_state[0]) &&
        LTX_NEW_VIDEO(video_state[1]) &&
        LTX_NEW_AUDIO(audio_state[0]) &&
        LTX_NEW_AUDIO(audio_state[1]) &&
        LTX_NEW_VIDEO(video_normed) && LTX_NEW_AUDIO(audio_normed) &&
        LTX_NEW_VIDEO(video_branch) && LTX_NEW_AUDIO(audio_branch) &&
        ((workspace->video_text_scaled = new_tensor(
            gpu, text_rows, video_dim, error, error_size)) != NULL) &&
        ((workspace->audio_text_scaled = new_tensor(
            gpu, text_rows, audio_dim, error, error_size)) != NULL) &&
        ((!video_text_kv_prefetch_stage1 &&
          !video_text_kv_prefetch_stage2 &&
          !ane_video_text_kv_enabled) ||
         (((workspace->video_text_key = new_tensor(
               gpu, text_rows, video_dim, error, error_size)) != NULL) &&
          ((workspace->video_text_value = new_tensor(
               gpu, text_rows, video_dim, error, error_size)) != NULL))) &&
        LTX_NEW_VIDEO(video_norm3) && LTX_NEW_AUDIO(audio_norm3) &&
        LTX_NEW_VIDEO(video_a2v) && LTX_NEW_AUDIO(audio_a2v) &&
        LTX_NEW_VIDEO(video_v2a) && LTX_NEW_AUDIO(audio_v2a) &&
        video_heads && video_dim % video_heads == 0u &&
        create_sol_video_self_workspace(
            gpu, video_rows, video_dim, video_heads,
            video_dim / video_heads, workspace, error, error_size)
#ifdef LTX_ENABLE_ANE_QKV
        && (!ane_video_self_qkv_enabled ||
            (ane_video_self_qkv_prefix_rows &&
             ((workspace->video_qkv_query_prefix = new_tensor(
                   gpu, ane_video_self_qkv_prefix_rows, video_dim,
                   error, error_size)) != NULL) &&
             ((workspace->video_qkv_key_prefix = new_tensor(
                   gpu, ane_video_self_qkv_prefix_rows, video_dim,
                   error, error_size)) != NULL) &&
             ((workspace->video_qkv_value_prefix = new_tensor(
                   gpu, ane_video_self_qkv_prefix_rows, video_dim,
                   error, error_size)) != NULL) &&
             LTX_NEW_VIDEO(video_qkv_query) &&
             LTX_NEW_VIDEO(video_qkv_key) &&
             LTX_NEW_VIDEO(video_qkv_value) &&
             ((workspace->video_qkv_gate_logits = new_tensor(
                   gpu, video_rows, video_heads,
                   error, error_size)) != NULL) &&
             ((workspace->video_qkv_gate = new_tensor(
                   gpu, video_rows, video_heads,
                   error, error_size)) != NULL) &&
             LTX_NEW_VIDEO(video_qkv_core) &&
             LTX_NEW_VIDEO(video_qkv_rotated)))
#endif
        ;
#undef LTX_NEW_VIDEO
#undef LTX_NEW_AUDIO
    if (!ok) free_workspace(workspace);
    return ok;
}

static int run_self_attention(ltx_gpu *gpu, ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              const attention_weights *weights,
                              const rope_pair *rope, uint32_t rows,
                              char *error, size_t error_size) {
    return ltx_gpu_self_attention_int8_mps_bf16(
        gpu, output, input,
        weights->query.weight, weights->query.scale, weights->query.bias,
        weights->key.weight, weights->key.scale, weights->key.bias,
        weights->value.weight, weights->value.scale, weights->value.bias,
        weights->query_norm, weights->key_norm,
        weights->gate_weight, weights->gate_bias,
        weights->output.weight, weights->output.scale, weights->output.bias,
        rope->cosine, rope->sine,
        rows, weights->heads, weights->head_dim, 256u, 1e-6f,
        error, error_size);
}

static int run_sol_video_self_attention(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input,
        const attention_weights *weights,
        const rope_pair *rope, block_workspace *workspace,
        uint32_t rows, char *error, size_t error_size) {
    uint32_t inner_dim = weights->inner_dim;
    if (!workspace->video_sol_query || !workspace->video_sol_key ||
        !workspace->video_sol_value ||
        !workspace->video_sol_gate_logits || !workspace->video_sol_gate ||
        !workspace->video_sol_core || !workspace->video_sol_rotated ||
        !workspace->video_sol_packed_query ||
        !workspace->video_sol_packed_key ||
        !workspace->video_sol_packed_value ||
        !workspace->video_sol_packed_output ||
        !workspace->video_sol_query_centroids ||
        !workspace->video_sol_key_centroids ||
        !workspace->video_sol_value_sums ||
        !workspace->video_sol_thresholds ||
        !workspace->video_sol_routes) {
        snprintf(error, error_size,
                 "Sol Video self-attention workspace is incomplete");
        return 0;
    }
    if (sol_video_self.batch_commands &&
        !ltx_gpu_batch_begin(gpu, error, error_size)) return 0;
    int ok = ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu,
            workspace->video_sol_query,
            workspace->video_sol_key,
            workspace->video_sol_value,
            input,
            weights->query.weight, weights->query.scale,
            weights->query.bias,
            weights->key.weight, weights->key.scale,
            weights->key.bias,
            weights->value.weight, weights->value.scale,
            weights->value.bias,
            weights->query_norm, weights->key_norm,
            rows, weights->query_dim, inner_dim,
            256u, 1e-6f, error, error_size) &&
        ltx_gpu_linear_bf16(
            gpu, workspace->video_sol_gate_logits, input,
            weights->gate_weight, weights->gate_bias,
            rows, weights->query_dim, weights->heads,
            error, error_size) &&
        ltx_gpu_sigmoid2_bf16(
            gpu, workspace->video_sol_gate,
            workspace->video_sol_gate_logits,
            rows * weights->heads, error, error_size) &&
        ltx_gpu_self_attention_core_sol_bf16(
            gpu, workspace->video_sol_core,
            workspace->video_sol_query,
            workspace->video_sol_key,
            workspace->video_sol_value,
            rope->cosine, rope->sine,
            workspace->video_sol_gate,
            workspace->video_sol_packed_query,
            workspace->video_sol_packed_key,
            workspace->video_sol_packed_value,
            workspace->video_sol_packed_output,
            workspace->video_sol_query_centroids,
            workspace->video_sol_key_centroids,
            workspace->video_sol_value_sums,
            workspace->video_sol_thresholds,
            workspace->video_sol_routes,
            rows, weights->heads, weights->head_dim,
            1.0f / sqrtf((float)weights->head_dim),
            sol_video_self_current_tau(),
            0u, 0u, 0u, 0u, error, error_size) &&
        ltx_gpu_convrot_bf16(
            gpu, workspace->video_sol_rotated,
            workspace->video_sol_core,
            rows, inner_dim, 256u, error, error_size) &&
        ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, output, workspace->video_sol_rotated,
            weights->output.weight, weights->output.scale,
            weights->output.bias,
            rows, inner_dim, weights->output_dim,
            error, error_size);
    if (!sol_video_self.batch_commands) return ok;
    char batch_error[1024] = {0};
    int batch_ok = ltx_gpu_batch_end(
        gpu, batch_error, sizeof(batch_error));
    if (!batch_ok && ok)
        snprintf(error, error_size, "%s", batch_error[0] ?
                 batch_error : "Sol attention command batch failed");
    return ok && batch_ok;
}

#ifdef LTX_ENABLE_ANE_QKV
static int run_video_self_attention_ane(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input, const block_weights *block,
        const rope_pair *rope, block_workspace *workspace,
        uint32_t rows, block_timing *timing,
        char *error, size_t error_size) {
    unsigned slot = rows == active_geometry.stage1_rows ? 0u :
        rows == active_geometry.stage2_rows ? 1u : 2u;
    ltx_ane_qkv *ane_qkv = slot < 2u ?
        block->ane_video_self_qkv[slot] : NULL;
    const attention_weights *weights = &block->video_self;
    if (!ane_qkv)
        return run_self_attention(
            gpu, output, input, weights, rope, rows, error, error_size);

    const ltx_ane_qkv_shape *shape = ltx_ane_qkv_get_shape(ane_qkv);
    if (!shape || shape->rows >= rows || shape->hidden != weights->inner_dim) {
        snprintf(error, error_size,
                 "ANE Video self QKV shape does not match %u rows", rows);
        return 0;
    }
    uint32_t prefix_rows = rows - shape->rows;
    if (prefix_rows != ane_video_self_qkv_prefix_rows ||
        !workspace->video_qkv_query_prefix ||
        !workspace->video_qkv_key_prefix ||
        !workspace->video_qkv_value_prefix ||
        !workspace->video_qkv_query || !workspace->video_qkv_key ||
        !workspace->video_qkv_value ||
        !workspace->video_qkv_gate_logits || !workspace->video_qkv_gate ||
        !workspace->video_qkv_core || !workspace->video_qkv_rotated) {
        snprintf(error, error_size,
                 "ANE Video self QKV workspace is incomplete");
        return 0;
    }

    if (!ltx_ane_qkv_start(
            ane_qkv, gpu, input, rows, prefix_rows,
            error, error_size)) return 0;
    double gpu_start = now_seconds();
    if (!ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu,
            workspace->video_qkv_query_prefix,
            workspace->video_qkv_key_prefix,
            workspace->video_qkv_value_prefix,
            input,
            weights->query.weight, weights->query.scale, weights->query.bias,
            weights->key.weight, weights->key.scale, weights->key.bias,
            weights->value.weight, weights->value.scale, weights->value.bias,
            weights->query_norm, weights->key_norm,
            prefix_rows, weights->query_dim, weights->inner_dim,
            256u, 1e-6f, error, error_size)) return 0;
    double gpu_seconds = now_seconds() - gpu_start;

    if (!ltx_gpu_linear_bf16(
            gpu, workspace->video_qkv_gate_logits, input,
            weights->gate_weight, weights->gate_bias,
            rows, weights->query_dim, weights->heads,
            error, error_size) ||
        !ltx_gpu_sigmoid2_bf16(
            gpu, workspace->video_qkv_gate,
            workspace->video_qkv_gate_logits,
            rows * weights->heads, error, error_size)) return 0;

    ltx_ane_qkv_timing qkv_timing = {0};
    if (!ltx_ane_qkv_wait(
            ane_qkv, gpu,
            workspace->video_qkv_query,
            workspace->video_qkv_key,
            workspace->video_qkv_value,
            workspace->video_qkv_query_prefix,
            workspace->video_qkv_key_prefix,
            workspace->video_qkv_value_prefix,
            prefix_rows, &qkv_timing, error, error_size) ||
        !ltx_gpu_self_attention_core_mps_bf16(
            gpu, workspace->video_qkv_core,
            workspace->video_qkv_query,
            workspace->video_qkv_key,
            workspace->video_qkv_value,
            rope->cosine, rope->sine,
            workspace->video_qkv_gate,
            rows, weights->heads, weights->head_dim,
            1.0f / sqrtf((float)weights->head_dim),
            error, error_size) ||
        !ltx_gpu_convrot_bf16(
            gpu, workspace->video_qkv_rotated,
            workspace->video_qkv_core,
            rows, weights->inner_dim, 256u, error, error_size) ||
        !ltx_gpu_linear_int8_weight_mps_bf16(
            gpu, output, workspace->video_qkv_rotated,
            weights->output.weight, weights->output.scale,
            weights->output.bias,
            rows, weights->inner_dim, weights->output_dim,
            error, error_size)) return 0;
    if (timing) {
        timing->ane_qkv_pack += qkv_timing.pack_ms * 1e-3;
        timing->ane_qkv_compute += qkv_timing.ane_ms * 1e-3;
        timing->ane_qkv_gpu += gpu_seconds;
        timing->ane_qkv_concat += qkv_timing.unpack_ms * 1e-3;
        timing->ane_qkv_overlap += qkv_timing.total_ms * 1e-3;
    }
    return 1;
}
#endif

static int run_video_self_attention(
        ltx_gpu *gpu, ltx_gpu_buffer *output,
        const ltx_gpu_buffer *input, const block_weights *block,
        const rope_pair *rope, block_workspace *workspace,
        uint32_t rows, uint32_t block_index, block_timing *timing,
        char *error, size_t error_size) {
#ifdef LTX_ENABLE_ANE_QKV
    unsigned slot = rows == active_geometry.stage1_rows ? 0u :
        rows == active_geometry.stage2_rows ? 1u : 2u;
    if (slot < 2u && block->ane_video_self_qkv[slot])
        return run_video_self_attention_ane(
            gpu, output, input, block, rope, workspace, rows, timing,
            error, error_size);
#else
    (void)timing;
#endif
    if (sol_video_self_should_run(rows, block_index))
        return run_sol_video_self_attention(
            gpu, output, input, &block->video_self,
            rope, workspace, rows, error, error_size);
    return run_self_attention(
        gpu, output, input, &block->video_self,
        rope, rows, error, error_size);
}

static int run_cross_attention(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *query_input,
                               const ltx_gpu_buffer *key_value_input,
                               const attention_weights *weights,
                               const rope_pair *query_rope,
                               const rope_pair *key_rope,
                               const ltx_gpu_buffer *attention_mask,
                               uint32_t attention_mask_rows,
                               uint32_t query_rows,
                               uint32_t key_value_rows,
                               char *error, size_t error_size) {
    return ltx_gpu_cross_attention_int8_mps_bf16_masked(
        gpu, output, query_input, key_value_input,
        weights->query.weight, weights->query.scale, weights->query.bias,
        weights->key.weight, weights->key.scale, weights->key.bias,
        weights->value.weight, weights->value.scale, weights->value.bias,
        weights->query_norm, weights->key_norm,
        weights->gate_weight, weights->gate_bias,
        weights->output.weight, weights->output.scale, weights->output.bias,
        query_rope ? query_rope->cosine : NULL,
        query_rope ? query_rope->sine : NULL,
        key_rope ? key_rope->cosine : NULL,
        key_rope ? key_rope->sine : NULL,
        attention_mask, attention_mask_rows,
        query_rows, key_value_rows,
        weights->query_dim, weights->key_value_dim,
        weights->heads, weights->head_dim, weights->output_dim,
        256u, 1e-6f, error, error_size);
}

typedef struct {
    ltx_gpu *gpu;
    ltx_gpu_buffer *output;
    const ltx_gpu_buffer *query_input;
    const ltx_gpu_buffer *key_value_input;
    const attention_weights *weights;
    const rope_pair *query_rope;
    const rope_pair *key_rope;
    uint32_t query_rows;
    uint32_t key_value_rows;
    int ok;
    double elapsed_seconds;
    char error[1024];
} cross_attention_task;

static void *run_cross_attention_thread(void *opaque) {
    cross_attention_task *task = opaque;
    double start = now_seconds();
    task->ok = run_cross_attention(
        task->gpu, task->output, task->query_input,
        task->key_value_input, task->weights,
        task->query_rope, task->key_rope,
        NULL, 0u, task->query_rows, task->key_value_rows,
        task->error, sizeof(task->error));
    task->elapsed_seconds = now_seconds() - start;
    return NULL;
}

static int run_video_self_pre_cross(
        ltx_gpu *gpu, const block_weights *weights,
        const block_rope *rope, block_workspace *workspace,
        uint32_t video_rows, uint32_t block_index,
        block_timing *timing, char *error, size_t error_size) {
    uint32_t video_dim = weights->video_self.query_dim;
    double start = now_seconds();
    if (!video_adaln(
            gpu, workspace->video_normed, workspace->video_state[0],
            &weights->video_adaln, &weights->video_adaln_conditioned,
            1u, 0u, video_rows, video_dim, error, error_size) ||
        !run_video_self_attention(
            gpu, workspace->video_branch, workspace->video_normed,
            weights, &rope->video_self, workspace, video_rows,
            block_index, timing,
            error, error_size) ||
        !video_residual_gate(
            gpu, workspace->video_state[1], workspace->video_state[0],
            workspace->video_branch, &weights->video_adaln,
            &weights->video_adaln_conditioned, 2u,
            video_rows, video_dim, error, error_size)) return 0;
    if (timing) timing->video_self += now_seconds() - start;
    return 1;
}

static int run_video_text_pre_cross(
        ltx_gpu *gpu, const block_weights *weights,
        block_workspace *workspace,
        const ltx_gpu_buffer *video_text,
        const ltx_gpu_buffer *text_mask,
        uint32_t video_rows, uint32_t text_rows,
        block_timing *timing, char *error, size_t error_size) {
    uint32_t video_dim = weights->video_self.query_dim;
    double start = now_seconds();
    if (!video_adaln(
            gpu, workspace->video_normed, workspace->video_state[1],
            &weights->video_adaln, &weights->video_adaln_conditioned,
            7u, 6u, video_rows, video_dim, error, error_size) ||
        !ltx_gpu_affine_bf16(
            gpu, workspace->video_text_scaled, video_text,
            weights->video_prompt.row[1], weights->video_prompt.row[0],
            text_rows, video_dim, 1u, error, error_size) ||
        !run_cross_attention(
            gpu, workspace->video_branch, workspace->video_normed,
            workspace->video_text_scaled, &weights->video_text,
            NULL, NULL, text_mask, text_mask ? 1u : 0u,
            video_rows, text_rows, error, error_size) ||
        !video_residual_gate(
            gpu, workspace->video_state[0], workspace->video_state[1],
            workspace->video_branch, &weights->video_adaln,
            &weights->video_adaln_conditioned, 8u,
            video_rows, video_dim, error, error_size)) return 0;
    if (timing) timing->video_text += now_seconds() - start;
    return 1;
}

static int run_video_pre_cross(
        ltx_gpu *gpu, const block_weights *weights,
        const block_rope *rope, block_workspace *workspace,
        const ltx_gpu_buffer *video_text,
        const ltx_gpu_buffer *text_mask,
        uint32_t video_rows, uint32_t text_rows, uint32_t block_index,
        block_timing *timing, char *error, size_t error_size) {
    if (!video_attention_batch)
        return run_video_self_pre_cross(
            gpu, weights, rope, workspace, video_rows, block_index,
            timing, error, error_size) &&
            run_video_text_pre_cross(
            gpu, weights, workspace, video_text, text_mask,
            video_rows, text_rows, timing, error, error_size);

    if (!ltx_gpu_batch_begin(gpu, error, error_size)) return 0;
    int ok = run_video_self_pre_cross(
            gpu, weights, rope, workspace, video_rows, block_index,
            timing, error, error_size) &&
        run_video_text_pre_cross(
            gpu, weights, workspace, video_text, text_mask,
            video_rows, text_rows, timing, error, error_size);
    char batch_error[1024] = {0};
    int batch_ok = ltx_gpu_batch_end(
        gpu, batch_error, sizeof(batch_error));
    if (!batch_ok && ok)
        snprintf(error, error_size, "%s",
                 batch_error[0] ? batch_error :
                 "batched Video attention failed");
    return ok && batch_ok;
}

static int run_video_text_kv_to(
        ltx_gpu *gpu, const block_weights *weights,
        block_workspace *workspace,
        ltx_gpu_buffer *key_output,
        ltx_gpu_buffer *value_output,
        const ltx_gpu_buffer *video_text,
        uint32_t text_rows, char *error, size_t error_size) {
    uint32_t video_dim = weights->video_text.key_value_dim;
    return ltx_gpu_affine_bf16(
            gpu, workspace->video_text_scaled, video_text,
            weights->video_prompt.row[1], weights->video_prompt.row[0],
            text_rows, video_dim, 1u, error, error_size) &&
        ltx_gpu_cross_attention_kv_int8_mps_bf16(
            gpu, key_output, value_output,
            workspace->video_text_scaled,
            weights->video_text.key.weight,
            weights->video_text.key.scale,
            weights->video_text.key.bias,
            weights->video_text.value.weight,
            weights->video_text.value.scale,
            weights->video_text.value.bias,
            weights->video_text.key_norm,
            text_rows, video_dim,
            weights->video_text.heads, weights->video_text.head_dim,
            256u, 1e-6f, error, error_size);
}

static int run_video_text_kv(
        ltx_gpu *gpu, const block_weights *weights,
        block_workspace *workspace,
        const ltx_gpu_buffer *video_text,
        uint32_t text_rows, char *error, size_t error_size) {
    return run_video_text_kv_to(
        gpu, weights, workspace,
        workspace->video_text_key, workspace->video_text_value,
        video_text, text_rows, error, error_size);
}

static int run_video_text_query_with_kv(
        ltx_gpu *gpu, const block_weights *weights,
        block_workspace *workspace,
        const ltx_gpu_buffer *key,
        const ltx_gpu_buffer *value,
        const ltx_gpu_buffer *text_mask,
        uint32_t video_rows, uint32_t text_rows,
        block_timing *timing, char *error, size_t error_size) {
    uint32_t video_dim = weights->video_self.query_dim;
    double start = now_seconds();
    int ok = video_adaln(
            gpu, workspace->video_normed, workspace->video_state[1],
            &weights->video_adaln, &weights->video_adaln_conditioned,
            7u, 6u, video_rows, video_dim, error, error_size) &&
        ltx_gpu_cross_attention_query_int8_mps_bf16_masked(
            gpu, workspace->video_branch, workspace->video_normed,
            key, value,
            weights->video_text.query.weight,
            weights->video_text.query.scale,
            weights->video_text.query.bias,
            weights->video_text.query_norm,
            weights->video_text.gate_weight,
            weights->video_text.gate_bias,
            weights->video_text.output.weight,
            weights->video_text.output.scale,
            weights->video_text.output.bias,
            text_mask, text_mask ? 1u : 0u,
            video_rows, text_rows, video_dim,
            weights->video_text.heads, weights->video_text.head_dim,
            weights->video_text.output_dim,
            256u, 1e-6f, error, error_size) &&
        video_residual_gate(
            gpu, workspace->video_state[0], workspace->video_state[1],
            workspace->video_branch, &weights->video_adaln,
            &weights->video_adaln_conditioned, 8u,
            video_rows, video_dim, error, error_size);
    if (timing) timing->video_text_query += now_seconds() - start;
    return ok;
}

static int run_video_text_query(
        ltx_gpu *gpu, const block_weights *weights,
        block_workspace *workspace,
        const ltx_gpu_buffer *text_mask,
        uint32_t video_rows, uint32_t text_rows,
        block_timing *timing, char *error, size_t error_size) {
    return run_video_text_query_with_kv(
        gpu, weights, workspace,
        workspace->video_text_key, workspace->video_text_value,
        text_mask, video_rows, text_rows, timing, error, error_size);
}

typedef struct {
    ltx_gpu *gpu;
    const block_weights *weights;
    block_workspace *workspace;
    const ltx_gpu_buffer *video_text;
    uint32_t text_rows;
    int ok;
    double elapsed_seconds;
    char error[1024];
} video_text_kv_task;

static void *run_video_text_kv_thread(void *opaque) {
    video_text_kv_task *task = opaque;
    double start = now_seconds();
    task->ok = run_video_text_kv(
        task->gpu, task->weights, task->workspace,
        task->video_text, task->text_rows,
        task->error, sizeof(task->error));
    task->elapsed_seconds = now_seconds() - start;
    return NULL;
}

static int run_audio_pre_cross(
        ltx_gpu *gpu, const block_weights *weights,
        const block_rope *rope, block_workspace *workspace,
        const ltx_gpu_buffer *audio_text,
        const ltx_gpu_buffer *text_mask,
        uint32_t audio_rows, uint32_t text_rows,
        block_timing *timing, char *error, size_t error_size) {
    uint32_t audio_dim = weights->audio_self.query_dim;
    double start = now_seconds();
    if (!ltx_gpu_adaln_bf16(
            gpu, workspace->audio_normed, workspace->audio_state[0],
            weights->audio_adaln.row[1], weights->audio_adaln.row[0],
            audio_rows, audio_dim, 1u, 1e-6f, error, error_size) ||
        !run_self_attention(
            gpu, workspace->audio_branch, workspace->audio_normed,
            &weights->audio_self, &rope->audio_self, audio_rows,
            error, error_size) ||
        !ltx_gpu_residual_gate_bf16(
            gpu, workspace->audio_state[1], workspace->audio_state[0],
            workspace->audio_branch, weights->audio_adaln.row[2],
            audio_rows, audio_dim, 1u, error, error_size)) return 0;
    if (timing) timing->audio_self += now_seconds() - start;

    start = now_seconds();
    if (!ltx_gpu_adaln_bf16(
            gpu, workspace->audio_normed, workspace->audio_state[1],
            weights->audio_adaln.row[7], weights->audio_adaln.row[6],
            audio_rows, audio_dim, 1u, 1e-6f, error, error_size) ||
        !ltx_gpu_affine_bf16(
            gpu, workspace->audio_text_scaled, audio_text,
            weights->audio_prompt.row[1], weights->audio_prompt.row[0],
            text_rows, audio_dim, 1u, error, error_size) ||
        !run_cross_attention(
            gpu, workspace->audio_branch, workspace->audio_normed,
            workspace->audio_text_scaled, &weights->audio_text,
            NULL, NULL, text_mask, text_mask ? 1u : 0u,
            audio_rows, text_rows, error, error_size) ||
        !ltx_gpu_residual_gate_bf16(
            gpu, workspace->audio_state[0], workspace->audio_state[1],
            workspace->audio_branch, weights->audio_adaln.row[8],
            audio_rows, audio_dim, 1u, error, error_size)) return 0;
    if (timing) timing->audio_text += now_seconds() - start;
    return 1;
}

typedef struct {
    ltx_gpu *gpu;
    const block_weights *weights;
    const block_rope *rope;
    block_workspace *workspace;
    const ltx_gpu_buffer *audio_text;
    const ltx_gpu_buffer *text_mask;
    uint32_t audio_rows;
    uint32_t text_rows;
    block_timing timing;
    int ok;
    char error[1024];
} audio_stream_task;

static void *run_audio_stream_thread(void *opaque) {
    audio_stream_task *task = opaque;
    task->ok = run_audio_pre_cross(
        task->gpu, task->weights, task->rope, task->workspace,
        task->audio_text, task->text_mask,
        task->audio_rows, task->text_rows, &task->timing,
        task->error, sizeof(task->error));
    return NULL;
}

static int run_mlp(ltx_gpu *gpu, ltx_gpu_buffer *output,
                   const ltx_gpu_buffer *input,
                   const mlp_weights *weights, uint32_t rows,
                   char *error, size_t error_size) {
    return ltx_gpu_mlp_int8_convrot_mps_bf16(
        gpu, output, input,
        weights->fc1.weight, weights->fc1.scale, weights->fc1.bias,
        weights->fc2.weight, weights->fc2.scale, weights->fc2.bias,
        rows, weights->input_dim, weights->hidden_dim,
        weights->output_dim, 256u, error, error_size);
}

typedef struct {
    ltx_gpu *gpu;
    const block_weights *weights;
    block_workspace *workspace;
    ltx_gpu_buffer *audio_hidden;
    ltx_gpu_buffer *audio_next;
    uint32_t audio_rows;
    int ok;
    double elapsed_seconds;
    char error[1024];
} audio_ffn_task;

static void *run_audio_ffn_thread(void *opaque) {
    audio_ffn_task *task = opaque;
    uint32_t audio_dim = task->weights->audio_self.query_dim;
    double start = now_seconds();
    task->ok = ltx_gpu_adaln_bf16(
            task->gpu, task->workspace->audio_normed,
            task->audio_hidden,
            task->weights->audio_adaln.row[4],
            task->weights->audio_adaln.row[3],
            task->audio_rows, audio_dim, 1u, 1e-6f,
            task->error, sizeof(task->error)) &&
        run_mlp(
            task->gpu, task->workspace->audio_branch,
            task->workspace->audio_normed,
            &task->weights->audio_mlp, task->audio_rows,
            task->error, sizeof(task->error)) &&
        ltx_gpu_residual_gate_bf16(
            task->gpu, task->audio_next, task->audio_hidden,
            task->workspace->audio_branch,
            task->weights->audio_adaln.row[5],
            task->audio_rows, audio_dim, 1u,
            task->error, sizeof(task->error));
    task->elapsed_seconds = now_seconds() - start;
    return NULL;
}

static void swap_buffers(ltx_gpu_buffer **left, ltx_gpu_buffer **right) {
    ltx_gpu_buffer *temporary = *left;
    *left = *right;
    *right = temporary;
}

static int run_block(ltx_gpu *gpu, block_weights *weights,
                     uint32_t block_index,
                     const block_rope *rope, block_workspace *workspace,
                     const ltx_gpu_buffer *video_text,
                     const ltx_gpu_buffer *audio_text,
                     const ltx_gpu_buffer *text_mask,
                     uint32_t video_rows, uint32_t audio_rows,
                     uint32_t text_rows, block_timing *timing,
                     char *error, size_t error_size) {
    uint32_t video_dim = weights->video_self.query_dim;
    uint32_t audio_dim = weights->audio_self.query_dim;
    ltx_gpu_buffer *video_hidden = workspace->video_state[0];
    ltx_gpu_buffer *video_next = workspace->video_state[1];
    ltx_gpu_buffer *audio_hidden = workspace->audio_state[0];
    ltx_gpu_buffer *audio_next = workspace->audio_state[1];
    double start = 0.0;

    int prefetch_video_text_kv =
        (video_rows == active_geometry.stage1_rows &&
         video_text_kv_prefetch_stage1) ||
        (video_rows == active_geometry.stage2_rows &&
         video_text_kv_prefetch_stage2);
#ifdef LTX_ENABLE_ANE_KV
    if (weights->ane_video_text_kv) {
        uint32_t video_text_dim = weights->video_text.key_value_dim;
        double parallel_start = now_seconds();
        double affine_start = parallel_start;
        if (!ltx_gpu_affine_bf16(
                gpu, workspace->video_text_scaled, video_text,
                weights->video_prompt.row[1], weights->video_prompt.row[0],
                text_rows, video_text_dim, 1u,
                error, error_size)) return 0;
        double affine_seconds = now_seconds() - affine_start;

        audio_stream_task audio_task = {
            .gpu = av_parallel_audio_gpu,
            .weights = weights,
            .rope = rope,
            .workspace = workspace,
            .audio_text = audio_text,
            .text_mask = text_mask,
            .audio_rows = audio_rows,
            .text_rows = text_rows,
        };
        av_task_handle audio_handle = {0};
        int audio_task_started = 0;
        double stream_start = 0.0;
        if (av_parallel_audio_gpu && av_parallel_streams) {
            stream_start = now_seconds();
            if (!av_task_start(
                    &audio_handle, run_audio_stream_thread, &audio_task,
                    "parallel Audio stream", error, error_size)) {
                return 0;
            }
            audio_task_started = 1;
        }

        if (!ltx_ane_kv_start(
                weights->ane_video_text_kv, gpu,
                workspace->video_text_scaled, error, error_size)) {
            if (audio_task_started) av_task_wait(&audio_handle);
            return 0;
        }
        int video_ok = run_video_self_pre_cross(
            gpu, weights, rope, workspace, video_rows, block_index,
            timing, error, error_size);
        int audio_ok = 1;
        if (!audio_task_started && video_ok)
            audio_ok = run_audio_pre_cross(
                gpu, weights, rope, workspace, audio_text, text_mask,
                audio_rows, text_rows, timing, error, error_size);

        ltx_ane_kv_timing ane_timing = {0};
        char ane_error[1024] = {0};
        int kv_ok = ltx_ane_kv_wait(
            weights->ane_video_text_kv, gpu,
            workspace->video_text_key, workspace->video_text_value,
            &ane_timing, ane_error, sizeof(ane_error));
        int join_ok = 1;
        if (audio_task_started) {
            join_ok = av_task_wait(&audio_handle);
            audio_ok = join_ok && audio_task.ok;
            if (timing) {
                timing->audio_self += audio_task.timing.audio_self;
                timing->audio_text += audio_task.timing.audio_text;
                timing->av_parallel_stream_wall +=
                    now_seconds() - stream_start;
            }
        }
        if (timing) {
            timing->video_text_kv +=
                affine_seconds + ane_timing.total_ms * 1e-3;
            timing->video_text_prefetch_wall +=
                now_seconds() - parallel_start;
            timing->ane_kv_pack += ane_timing.pack_ms * 1e-3;
            timing->ane_kv_compute += ane_timing.ane_ms * 1e-3;
            timing->ane_kv_unpack += ane_timing.unpack_ms * 1e-3;
            timing->ane_kv_overlap += now_seconds() - parallel_start;
        }
        if (!join_ok) {
            snprintf(error, error_size,
                     "cannot join parallel audio stream thread");
            return 0;
        }
        if (!video_ok || !audio_ok) {
            if (audio_task_started && !audio_task.ok)
                snprintf(error, error_size, "%s",
                         audio_task.error[0] ? audio_task.error :
                         "parallel audio stream failed");
            return 0;
        }
        if (!kv_ok) {
            snprintf(error, error_size, "%s",
                     ane_error[0] ? ane_error :
                     "ANE Video-text K/V failed");
            return 0;
        }
        double query_before = timing ? timing->video_text_query : 0.0;
        if (!run_video_text_query(
                gpu, weights, workspace, text_mask,
                video_rows, text_rows, timing,
                error, error_size)) return 0;
        if (timing)
            timing->video_text += affine_seconds +
                ane_timing.total_ms * 1e-3 +
                (timing->video_text_query - query_before);
    } else
#endif
    if (prefetch_video_text_kv) {
        video_text_kv_task kv_task = {
            .gpu = av_parallel_audio_gpu,
            .weights = weights,
            .workspace = workspace,
            .video_text = video_text,
            .text_rows = text_rows,
        };
        pthread_t kv_thread;
        double parallel_start = now_seconds();
        if (!av_parallel_audio_gpu || pthread_create(
                &kv_thread, NULL, run_video_text_kv_thread,
                &kv_task) != 0) {
            snprintf(error, error_size,
                     "cannot create Video-text K/V prefetch thread");
            return 0;
        }
        int video_ok = run_video_self_pre_cross(
            gpu, weights, rope, workspace, video_rows, block_index,
            timing, error, error_size);
        int audio_ok = video_ok && run_audio_pre_cross(
            gpu, weights, rope, workspace, audio_text, text_mask,
            audio_rows, text_rows, timing, error, error_size);
        int join_ok = pthread_join(kv_thread, NULL) == 0;
        if (timing) {
            timing->video_text_kv += kv_task.elapsed_seconds;
            timing->video_text_prefetch_wall +=
                now_seconds() - parallel_start;
        }
        if (!join_ok) {
            snprintf(error, error_size,
                     "cannot join Video-text K/V prefetch thread");
            return 0;
        }
        if (!video_ok || !audio_ok) return 0;
        if (!kv_task.ok) {
            snprintf(error, error_size, "%s",
                     kv_task.error[0] ? kv_task.error :
                     "Video-text K/V prefetch failed");
            return 0;
        }
        double query_before = timing ? timing->video_text_query : 0.0;
        if (!run_video_text_query(
                gpu, weights, workspace, text_mask,
                video_rows, text_rows, timing,
                error, error_size)) return 0;
        if (timing)
            timing->video_text += kv_task.elapsed_seconds +
                (timing->video_text_query - query_before);
    } else if (av_parallel_audio_gpu && av_parallel_streams) {
        audio_stream_task audio_task = {
            .gpu = av_parallel_audio_gpu,
            .weights = weights,
            .rope = rope,
            .workspace = workspace,
            .audio_text = audio_text,
            .text_mask = text_mask,
            .audio_rows = audio_rows,
            .text_rows = text_rows,
        };
        av_task_handle audio_handle = {0};
        double parallel_start = now_seconds();
        if (!av_task_start(
                &audio_handle, run_audio_stream_thread, &audio_task,
                "parallel Audio stream", error, error_size)) {
            return 0;
        }
        int video_ok = run_video_pre_cross(
            gpu, weights, rope, workspace, video_text, text_mask,
            video_rows, text_rows, block_index,
            timing, error, error_size);
        int join_ok = av_task_wait(&audio_handle);
        if (timing) {
            timing->audio_self += audio_task.timing.audio_self;
            timing->audio_text += audio_task.timing.audio_text;
            timing->av_parallel_stream_wall +=
                now_seconds() - parallel_start;
        }
        if (!join_ok) {
            snprintf(error, error_size,
                     "cannot join parallel audio stream thread");
            return 0;
        }
        if (!video_ok) return 0;
        if (!audio_task.ok) {
            snprintf(error, error_size, "%s",
                     audio_task.error[0] ? audio_task.error :
                     "parallel audio stream failed");
            return 0;
        }
    } else {
        if (!run_video_pre_cross(
                gpu, weights, rope, workspace, video_text, text_mask,
                video_rows, text_rows, block_index,
                timing, error, error_size) ||
            !run_audio_pre_cross(
                gpu, weights, rope, workspace, audio_text, text_mask,
                audio_rows, text_rows, timing, error, error_size)) return 0;
    }

    start = now_seconds();
    if (!ltx_gpu_rms_norm_bf16(
            gpu, workspace->video_norm3, video_hidden,
            video_rows, video_dim, 1e-6f, error, error_size) ||
        !ltx_gpu_rms_norm_bf16(
            gpu, workspace->audio_norm3, audio_hidden,
            audio_rows, audio_dim, 1e-6f, error, error_size) ||
        !video_affine(
            gpu, workspace->video_a2v, workspace->video_norm3,
            &weights->av_video, &weights->av_video_conditioned,
            0u, 1u, video_rows, video_dim, error, error_size) ||
        !ltx_gpu_affine_bf16(
            gpu, workspace->audio_a2v, workspace->audio_norm3,
            weights->av_audio.row[0], weights->av_audio.row[1],
            audio_rows, audio_dim, 1u, error, error_size) ||
        !video_affine(
            gpu, workspace->video_v2a, workspace->video_norm3,
            &weights->av_video, &weights->av_video_conditioned,
            2u, 3u, video_rows, video_dim, error, error_size) ||
        !ltx_gpu_affine_bf16(
            gpu, workspace->audio_v2a, workspace->audio_norm3,
            weights->av_audio.row[2], weights->av_audio.row[3],
            audio_rows, audio_dim, 1u, error, error_size)) return 0;
    if (timing) timing->av_norm_modulation += now_seconds() - start;

#ifdef LTX_ENABLE_ANE_V2A
    ltx_ane_v2a *ane_v2a = video_rows == active_geometry.stage1_rows ?
        weights->ane_v2a[0] : video_rows == active_geometry.stage2_rows ?
        weights->ane_v2a[1] : NULL;
    if (ane_v2a) {
        ltx_ane_v2a_timing ane_timing = {0};
        double overlap_start = now_seconds();
        if (!ltx_ane_v2a_start(
                ane_v2a, gpu, workspace->audio_v2a,
                workspace->video_v2a, error, error_size)) return 0;
        start = now_seconds();
        int video_ok = run_cross_attention(
            gpu, workspace->video_branch,
            workspace->video_a2v, workspace->audio_a2v,
            &weights->audio_to_video,
            &rope->video_cross, &rope->audio_cross,
            NULL, 0u, video_rows, audio_rows, error, error_size);
        double video_seconds = now_seconds() - start;
        char ane_error[1024] = {0};
        int audio_ok = ltx_ane_v2a_wait(
            ane_v2a, gpu, workspace->audio_branch, &ane_timing,
            ane_error, sizeof(ane_error));
        if (timing) {
            timing->audio_to_video += video_seconds;
            timing->video_to_audio += ane_timing.total_ms * 1e-3;
            timing->ane_v2a_pack += ane_timing.pack_ms * 1e-3;
            timing->ane_v2a_compute += ane_timing.ane_ms * 1e-3;
            timing->ane_v2a_unpack += ane_timing.unpack_ms * 1e-3;
            timing->ane_v2a_overlap += now_seconds() - overlap_start;
        }
        if (!video_ok) return 0;
        if (!audio_ok) {
            snprintf(error, error_size, "%s",
                     ane_error[0] ? ane_error :
                     "ANE V-to-A attention failed");
            return 0;
        }
    } else
#endif
    if (av_parallel_audio_gpu && av_parallel_cross) {
        cross_attention_task audio_task = {
            .gpu = av_parallel_audio_gpu,
            .output = workspace->audio_branch,
            .query_input = workspace->audio_v2a,
            .key_value_input = workspace->video_v2a,
            .weights = &weights->video_to_audio,
            .query_rope = &rope->audio_cross,
            .key_rope = &rope->video_cross,
            .query_rows = audio_rows,
            .key_value_rows = video_rows,
        };
        av_task_handle audio_handle = {0};
        double parallel_start = now_seconds();
        if (!av_task_start(
                &audio_handle, run_cross_attention_thread, &audio_task,
                "parallel V-to-A attention", error, error_size)) {
            return 0;
        }
        start = now_seconds();
        int video_ok = run_cross_attention(
            gpu, workspace->video_branch,
            workspace->video_a2v, workspace->audio_a2v,
            &weights->audio_to_video,
            &rope->video_cross, &rope->audio_cross,
            NULL, 0u, video_rows, audio_rows, error, error_size);
        double video_seconds = now_seconds() - start;
        int join_ok = av_task_wait(&audio_handle);
        if (timing) {
            timing->audio_to_video += video_seconds;
            timing->video_to_audio += audio_task.elapsed_seconds;
            timing->av_parallel_cross_wall +=
                now_seconds() - parallel_start;
        }
        if (!join_ok) {
            snprintf(error, error_size,
                     "cannot join parallel V-to-A attention thread");
            return 0;
        }
        if (!video_ok) return 0;
        if (!audio_task.ok) {
            snprintf(error, error_size, "%s",
                     audio_task.error[0] ? audio_task.error :
                     "parallel V-to-A attention failed");
            return 0;
        }
    } else {
        start = now_seconds();
        if (!run_cross_attention(
                gpu, workspace->video_branch,
                workspace->video_a2v, workspace->audio_a2v,
                &weights->audio_to_video,
                &rope->video_cross, &rope->audio_cross,
                NULL, 0u,
                video_rows, audio_rows, error, error_size)) return 0;
        if (timing) timing->audio_to_video += now_seconds() - start;

        start = now_seconds();
        if (!run_cross_attention(
                gpu, workspace->audio_branch,
                workspace->audio_v2a, workspace->video_v2a,
                &weights->video_to_audio,
                &rope->audio_cross, &rope->video_cross,
                NULL, 0u,
                audio_rows, video_rows, error, error_size)) return 0;
        if (timing) timing->video_to_audio += now_seconds() - start;
    }

    start = now_seconds();
    if (!ltx_gpu_residual_gate_bf16(
            gpu, video_next, video_hidden, workspace->video_branch,
            weights->av_video.row[4], video_rows, video_dim, 1u,
            error, error_size) ||
        !ltx_gpu_residual_gate_bf16(
            gpu, audio_next, audio_hidden, workspace->audio_branch,
            weights->av_audio.row[4], audio_rows, audio_dim, 1u,
            error, error_size)) return 0;
    if (timing) timing->av_residual += now_seconds() - start;
    swap_buffers(&video_hidden, &video_next);
    swap_buffers(&audio_hidden, &audio_next);

    audio_ffn_task audio_task = {
        .gpu = av_parallel_audio_gpu,
        .weights = weights,
        .workspace = workspace,
        .audio_hidden = audio_hidden,
        .audio_next = audio_next,
        .audio_rows = audio_rows,
    };
    av_task_handle audio_handle = {0};
    int audio_task_started = 0;
    double ffn_parallel_start = 0.0;
    if (av_parallel_audio_gpu && av_parallel_ffn) {
        ffn_parallel_start = now_seconds();
        if (!av_task_start(
                &audio_handle, run_audio_ffn_thread, &audio_task,
                "parallel Audio FFN", error, error_size)) {
            return 0;
        }
        audio_task_started = 1;
    }

    start = now_seconds();
#ifdef LTX_ENABLE_ANE_MLP
    ltx_ane_mlp *ane_video_mlp = video_rows == active_geometry.stage1_rows ?
        weights->ane_video_mlp[0] : video_rows == active_geometry.stage2_rows ?
        weights->ane_video_mlp[1] : NULL;
    ltx_ane_mlp_timing ane_mlp_timing = {0};
    int ane_mlp_shape_ok = !ane_video_mlp || ltx_ane_mlp_set_rows(
        ane_video_mlp, gpu, video_rows, error, error_size);
#endif
#ifdef LTX_ENABLE_ANE_MLP
    int video_ffn_ok = ane_mlp_shape_ok;
    if (video_ffn_ok && ane_video_mlp && ane_mlp_fused_adaln_pack &&
        !active_video_conditioned_prefix_rows) {
        video_ffn_ok = ltx_ane_mlp_eval_adaln(
            ane_video_mlp, gpu, workspace->video_branch,
            workspace->video_normed, video_hidden,
            weights->video_adaln.row[4], weights->video_adaln.row[3],
            video_rows, video_dim, 1u, 1e-6f, &ane_mlp_timing,
            error, error_size) &&
            ltx_gpu_residual_gate_bf16(
                gpu, video_next, video_hidden, workspace->video_branch,
                weights->video_adaln.row[5], video_rows, video_dim, 1u,
                error, error_size);
    } else {
        if (video_ffn_ok)
            video_ffn_ok = video_adaln(
                gpu, workspace->video_normed, video_hidden,
                &weights->video_adaln, &weights->video_adaln_conditioned,
                4u, 3u, video_rows, video_dim, error, error_size);
        if (video_ffn_ok && ane_video_mlp && ane_mlp_fused_residual &&
            !active_video_conditioned_prefix_rows) {
            video_ffn_ok = ltx_ane_mlp_eval_residual(
                ane_video_mlp, gpu, video_next, video_hidden,
                workspace->video_normed, weights->video_adaln.row[5],
                video_rows, video_dim, 1u, &ane_mlp_timing,
                error, error_size);
        } else if (video_ffn_ok) {
            video_ffn_ok = (ane_video_mlp ?
                ltx_ane_mlp_eval(
                    ane_video_mlp, gpu, workspace->video_branch,
                    workspace->video_normed, &ane_mlp_timing,
                    error, error_size) :
                run_mlp(gpu, workspace->video_branch,
                        workspace->video_normed, &weights->video_mlp,
                        video_rows, error, error_size)) &&
                video_residual_gate(
                    gpu, video_next, video_hidden, workspace->video_branch,
                    &weights->video_adaln,
                    &weights->video_adaln_conditioned, 5u,
                    video_rows, video_dim, error, error_size);
        }
    }
#else
    int video_ffn_ok = video_adaln(
            gpu, workspace->video_normed, video_hidden,
            &weights->video_adaln, &weights->video_adaln_conditioned,
            4u, 3u, video_rows, video_dim, error, error_size);
    if (video_ffn_ok)
        video_ffn_ok = run_mlp(
            gpu, workspace->video_branch, workspace->video_normed,
            &weights->video_mlp, video_rows, error, error_size) &&
            video_residual_gate(
            gpu, video_next, video_hidden, workspace->video_branch,
            &weights->video_adaln, &weights->video_adaln_conditioned, 5u,
            video_rows, video_dim, error, error_size);
#endif
    if (timing) {
        timing->video_ffn += now_seconds() - start;
#ifdef LTX_ENABLE_ANE_MLP
        timing->ane_mlp_pack += ane_mlp_timing.pack_ms * 1e-3;
        timing->ane_mlp_overlap += ane_mlp_timing.overlap_ms * 1e-3;
        timing->ane_mlp_gpu += ane_mlp_timing.gpu_ms * 1e-3;
        timing->ane_mlp_ane += ane_mlp_timing.ane_ms * 1e-3;
        timing->ane_mlp_join += ane_mlp_timing.join_ms * 1e-3;
#endif
    }
    if (audio_task_started) {
        int join_ok = av_task_wait(&audio_handle);
        if (timing) {
            timing->audio_ffn += audio_task.elapsed_seconds;
            timing->av_parallel_ffn_wall +=
                now_seconds() - ffn_parallel_start;
        }
        if (!join_ok) {
            snprintf(error, error_size,
                     "cannot join parallel audio FFN thread");
            return 0;
        }
        if (!video_ffn_ok) return 0;
        if (!audio_task.ok) {
            snprintf(error, error_size, "%s",
                     audio_task.error[0] ? audio_task.error :
                     "parallel audio FFN failed");
            return 0;
        }
    } else if (!video_ffn_ok) {
        return 0;
    }
    swap_buffers(&video_hidden, &video_next);

    if (!audio_task_started) {
        start = now_seconds();
        if (!ltx_gpu_adaln_bf16(
                gpu, workspace->audio_normed, audio_hidden,
                weights->audio_adaln.row[4], weights->audio_adaln.row[3],
                audio_rows, audio_dim, 1u, 1e-6f,
                error, error_size) ||
            !run_mlp(
                gpu, workspace->audio_branch, workspace->audio_normed,
                &weights->audio_mlp, audio_rows, error, error_size) ||
            !ltx_gpu_residual_gate_bf16(
                gpu, audio_next, audio_hidden, workspace->audio_branch,
                weights->audio_adaln.row[5], audio_rows, audio_dim, 1u,
                error, error_size)) return 0;
        if (timing) timing->audio_ffn += now_seconds() - start;
    }
    swap_buffers(&audio_hidden, &audio_next);

    if (video_hidden != workspace->video_state[0] ||
        audio_hidden != workspace->audio_state[0]) {
        snprintf(error, error_size, "internal block state parity failed");
        return 0;
    }
    return 1;
}

static int run_blocks(ltx_gpu *gpu, block_weights *weights,
                      uint32_t first_block, uint32_t block_count,
                      const block_rope *rope, block_workspace *workspace,
                      const ltx_gpu_buffer *video_text,
                      const ltx_gpu_buffer *audio_text,
                      const ltx_gpu_buffer *text_mask,
                      uint32_t video_rows, uint32_t audio_rows,
                      uint32_t text_rows, block_timing *timing,
                      char *error, size_t error_size) {
    for (uint32_t index = 0; index < block_count; index++)
        if (report_progress("ltx_block",(int)index,(int)block_count,error,error_size) || !run_block(
                gpu, &weights[index], first_block + index,
                rope, workspace,
                video_text, audio_text, text_mask,
                video_rows, audio_rows, text_rows, timing,
                error, error_size)) {
            char detail[1024];
            snprintf(detail, sizeof(detail), "%s",
                     error[0] ? error : "unknown error");
            snprintf(error, error_size, "block %u: %s",
                     first_block + index, detail);
            return 0;
        }
    return 1;
}

static size_t block_weight_bytes(const block_weights *weights) {
    return
        weights->video_self.query.weight_bytes +
        weights->video_self.key.weight_bytes +
        weights->video_self.value.weight_bytes +
        weights->video_self.output.weight_bytes +
        weights->audio_self.query.weight_bytes +
        weights->audio_self.key.weight_bytes +
        weights->audio_self.value.weight_bytes +
        weights->audio_self.output.weight_bytes +
        weights->video_text.query.weight_bytes +
        weights->video_text.key.weight_bytes +
        weights->video_text.value.weight_bytes +
        weights->video_text.output.weight_bytes +
        weights->audio_text.query.weight_bytes +
        weights->audio_text.key.weight_bytes +
        weights->audio_text.value.weight_bytes +
        weights->audio_text.output.weight_bytes +
        weights->audio_to_video.query.weight_bytes +
        weights->audio_to_video.key.weight_bytes +
        weights->audio_to_video.value.weight_bytes +
        weights->audio_to_video.output.weight_bytes +
        weights->video_to_audio.query.weight_bytes +
        weights->video_to_audio.key.weight_bytes +
        weights->video_to_audio.value.weight_bytes +
        weights->video_to_audio.output.weight_bytes +
        weights->video_mlp.fc1.weight_bytes +
        weights->video_mlp.fc2.weight_bytes +
        weights->audio_mlp.fc1.weight_bytes +
        weights->audio_mlp.fc2.weight_bytes;
}

static int run_blocks_and_release(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        const block_rope *rope, block_workspace *workspace,
        const ltx_gpu_buffer *video_text,
        const ltx_gpu_buffer *audio_text,
        const ltx_gpu_buffer *text_mask,
        uint32_t video_rows, uint32_t audio_rows,
        uint32_t text_rows, block_timing *timing,
        size_t *released_bytes,
        char *error, size_t error_size) {
    size_t bytes = 0;
    for (uint32_t index = 0; index < block_count; index++) {
        if (!run_block(
                gpu, &weights[index], first_block + index,
                rope, workspace,
                video_text, audio_text, text_mask,
                video_rows, audio_rows, text_rows, timing,
                error, error_size)) {
            char detail[1024];
            snprintf(detail, sizeof(detail), "%s",
                     error[0] ? error : "unknown error");
            snprintf(error, error_size, "block %u: %s",
                     first_block + index, detail);
            return 0;
        }
        bytes += block_weight_bytes(&weights[index]);
        free_block_weights(&weights[index]);
    }
    if (released_bytes) *released_bytes = bytes;
    return 1;
}

#ifdef LTX_ENABLE_ANE_MLP
static int attach_ane_video_mlp(
        ltx_gpu *gpu, block_weights *weights, uint32_t block,
        uint32_t expected_rows, const char *manifest, const char *variant,
        char *error, size_t error_size) {
    unsigned slot = expected_rows == active_geometry.stage1_rows ? 0u :
        expected_rows == active_geometry.stage2_rows ? 1u : 2u;
    if (slot >= 2u) {
        snprintf(error, error_size, "invalid ANE MLP stage");
        return 0;
    }
    ltx_ane_mlp *existing = weights->ane_video_mlp[slot];
    if (existing) {
        const ltx_ane_mlp_shape *shape = ltx_ane_mlp_get_shape(existing);
        if (!shape || shape->block_index != block ||
            shape->hidden != 4096u ||
            shape->full_intermediate != 16384u ||
            !ltx_ane_mlp_supports_rows(existing, expected_rows) ||
            !ltx_ane_mlp_set_rows(
                existing, gpu, expected_rows, error, error_size)) {
            if (!error[0])
                snprintf(error, error_size,
                         "existing ANE artifact does not support block %u "
                         "at %u rows", block, expected_rows);
            return 0;
        }
        return 1;
    }
    ltx_ane_mlp *model = ltx_ane_mlp_create(
        gpu, manifest, variant, error, error_size);
    const ltx_ane_mlp_shape *shape = ltx_ane_mlp_get_shape(model);
    if (!model || !shape || shape->block_index != block ||
        shape->hidden != 4096u || shape->full_intermediate != 16384u ||
        !ltx_ane_mlp_supports_rows(model, expected_rows) ||
        !ltx_ane_mlp_set_rows(
            model, gpu, expected_rows, error, error_size)) {
        if (!error[0])
            snprintf(error, error_size,
                     "ANE artifact does not match block %u at %u rows",
                     block, expected_rows);
        ltx_ane_mlp_free(model);
        return 0;
    }
    weights->ane_video_mlp[slot] = model;
    unsigned other_slot = slot ^ 1u;
    uint32_t other_rows = other_slot == 0u ?
        active_geometry.stage1_rows : active_geometry.stage2_rows;
    if (!weights->ane_video_mlp[other_slot] &&
        ltx_ane_mlp_supports_rows(model, other_rows))
        weights->ane_video_mlp[other_slot] = model;
    return 1;
}

static int attach_ane_directory(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        uint32_t rows, const char *directory, const char *variant,
        uint32_t *attached, char *error, size_t error_size) {
    if (!directory || !directory[0]) return 1;
    for (uint32_t index = 0; index < block_count; index++) {
        uint32_t block = first_block + index;
        char manifest[4096];
        int length = snprintf(manifest, sizeof(manifest),
                              "%s/block-%u/manifest.json",
                              directory, block);
        struct stat info;
        if (length < 0 || (size_t)length >= sizeof(manifest)) {
            snprintf(error, error_size, "ANE artifact path is too long");
            return 0;
        }
        if (stat(manifest, &info) != 0) {
            if (errno == ENOENT) continue;
            snprintf(error, error_size, "cannot stat ANE artifact %s",
                     manifest);
            return 0;
        }
        if (!S_ISREG(info.st_mode) ||
            !attach_ane_video_mlp(
                gpu, &weights[index], block, rows, manifest, variant,
                error, error_size)) return 0;
        (*attached)++;
    }
    return 1;
}

static uint32_t detach_ane_stage(
        block_weights *weights, uint32_t block_count, uint32_t rows) {
    unsigned slot = rows == active_geometry.stage1_rows ? 0u :
        rows == active_geometry.stage2_rows ? 1u : 2u;
    uint32_t detached = 0;
    if (!weights || slot >= 2u) return 0;
    for (uint32_t index = 0; index < block_count; index++) {
        ltx_ane_mlp *model = weights[index].ane_video_mlp[slot];
        if (!model) continue;
        weights[index].ane_video_mlp[slot] = NULL;
        if (weights[index].ane_video_mlp[slot ^ 1u] == model) continue;
        ltx_ane_mlp_free(model);
        detached++;
    }
    return detached;
}

static int ane_mlp_directory_complete(
        const char *directory, uint32_t first_block,
        uint32_t block_count) {
    if (!directory || !directory[0]) return 0;
    for (uint32_t index = 0; index < block_count; index++) {
        char manifest[4096];
        int length = snprintf(
            manifest, sizeof(manifest), "%s/block-%u/manifest.json",
            directory, first_block + index);
        struct stat info;
        if (length < 0 || (size_t)length >= sizeof(manifest) ||
            stat(manifest, &info) != 0 || !S_ISREG(info.st_mode)) return 0;
    }
    return 1;
}
#endif

#ifdef LTX_ENABLE_ANE_V2A
static int attach_ane_v2a(
        ltx_gpu *gpu, block_weights *weights, uint32_t block,
        uint32_t expected_video_rows, const char *manifest,
        const char *variant, char *error, size_t error_size) {
    unsigned slot = expected_video_rows == active_geometry.stage1_rows ? 0u :
        expected_video_rows == active_geometry.stage2_rows ? 1u : 2u;
    if (slot >= 2u) {
        snprintf(error, error_size,
                 "invalid ANE V-to-A stage");
        return 0;
    }
    ltx_ane_v2a *existing = weights->ane_v2a[slot];
    if (existing) {
        const ltx_ane_v2a_shape *shape = ltx_ane_v2a_get_shape(existing);
        if (!shape || shape->block_index != block ||
            shape->video_rows != expected_video_rows ||
            shape->audio_rows != 101u || shape->audio_dim != 2048u ||
            shape->video_dim != 4096u || shape->heads != 32u ||
            shape->head_dim != 64u) {
            snprintf(error, error_size,
                     "existing ANE V-to-A artifact does not match block %u "
                     "at %u rows", block, expected_video_rows);
            return 0;
        }
        return 1;
    }
    ltx_ane_v2a *attention = ltx_ane_v2a_create(
        gpu, manifest, variant, error, error_size);
    const ltx_ane_v2a_shape *shape =
        ltx_ane_v2a_get_shape(attention);
    if (!attention || !shape || shape->block_index != block ||
        shape->video_rows != expected_video_rows ||
        shape->audio_rows != 101u || shape->audio_dim != 2048u ||
        shape->video_dim != 4096u || shape->heads != 32u ||
        shape->head_dim != 64u) {
        if (!error[0])
            snprintf(error, error_size,
                     "ANE V-to-A artifact does not match block %u at %u rows",
                     block, expected_video_rows);
        ltx_ane_v2a_free(attention);
        return 0;
    }
    weights->ane_v2a[slot] = attention;
    return 1;
}

static int attach_ane_v2a_directory(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        uint32_t video_rows, const char *directory, const char *variant,
        uint32_t *attached, char *error, size_t error_size) {
    if (!directory || !directory[0]) return 1;
    for (uint32_t index = 0; index < block_count; index++) {
        uint32_t block = first_block + index;
        char manifest[4096];
        int length = snprintf(manifest, sizeof(manifest),
                              "%s/block-%u/manifest.json",
                              directory, block);
        struct stat info;
        if (length < 0 || (size_t)length >= sizeof(manifest)) {
            snprintf(error, error_size,
                     "ANE V-to-A artifact path is too long");
            return 0;
        }
        if (stat(manifest, &info) != 0) {
            if (errno == ENOENT) continue;
            snprintf(error, error_size,
                     "cannot stat ANE V-to-A artifact %s", manifest);
            return 0;
        }
        if (!S_ISREG(info.st_mode) ||
            !attach_ane_v2a(
                gpu, &weights[index], block, video_rows,
                manifest, variant, error, error_size)) return 0;
        (*attached)++;
    }
    return 1;
}

static uint32_t detach_ane_v2a_stage(
        block_weights *weights, uint32_t block_count,
        uint32_t video_rows) {
    unsigned slot = video_rows == active_geometry.stage1_rows ? 0u :
        video_rows == active_geometry.stage2_rows ? 1u : 2u;
    uint32_t detached = 0;
    if (!weights || slot >= 2u) return 0;
    for (uint32_t index = 0; index < block_count; index++) {
        if (!weights[index].ane_v2a[slot]) continue;
        ltx_ane_v2a_free(weights[index].ane_v2a[slot]);
        weights[index].ane_v2a[slot] = NULL;
        detached++;
    }
    return detached;
}
#endif

#ifdef LTX_ENABLE_ANE_KV
static int attach_ane_video_text_kv(
        ltx_gpu *gpu, block_weights *weights, uint32_t block,
        const char *manifest, const char *variant,
        char *error, size_t error_size) {
    if (weights->ane_video_text_kv) {
        const ltx_ane_kv_shape *shape = ltx_ane_kv_get_shape(
            weights->ane_video_text_kv);
        if (!shape || shape->block_index != block ||
            shape->hidden != 4096u || shape->heads != 32u ||
            shape->head_dim != 128u) {
            snprintf(error, error_size,
                     "existing ANE Video-text K/V artifact does not match "
                     "block %u", block);
            return 0;
        }
        return 1;
    }
    ltx_ane_kv *model = ltx_ane_kv_create(
        gpu, manifest, variant, error, error_size);
    const ltx_ane_kv_shape *shape = ltx_ane_kv_get_shape(model);
    if (!model || !shape || shape->block_index != block ||
        shape->hidden != 4096u ||
        shape->heads != 32u || shape->head_dim != 128u) {
        if (!error[0])
            snprintf(error, error_size,
                     "ANE Video-text K/V artifact does not match block %u",
                     block);
        ltx_ane_kv_free(model);
        return 0;
    }
    weights->ane_video_text_kv = model;
    return 1;
}

static int attach_ane_kv_directory(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        const char *directory, const char *variant,
        uint32_t *attached, char *error, size_t error_size) {
    if (!directory || !directory[0]) return 1;
    for (uint32_t index = 0; index < block_count; index++) {
        uint32_t block = first_block + index;
        char manifest[4096];
        int length = snprintf(manifest, sizeof(manifest),
                              "%s/block-%u/manifest.json",
                              directory, block);
        struct stat info;
        if (length < 0 || (size_t)length >= sizeof(manifest)) {
            snprintf(error, error_size,
                     "ANE Video-text K/V path is too long");
            return 0;
        }
        if (stat(manifest, &info) != 0) {
            if (errno == ENOENT) continue;
            snprintf(error, error_size,
                     "cannot stat ANE Video-text K/V artifact %s",
                     manifest);
            return 0;
        }
        if (!S_ISREG(info.st_mode) ||
            !attach_ane_video_text_kv(
                gpu, &weights[index], block, manifest, variant,
                error, error_size)) return 0;
        (*attached)++;
    }
    return 1;
}

static uint32_t detach_ane_kv_models(
        block_weights *weights, uint32_t block_count) {
    uint32_t detached = 0;
    if (!weights) return 0;
    for (uint32_t index = 0; index < block_count; index++) {
        if (!weights[index].ane_video_text_kv) continue;
        ltx_ane_kv_free(weights[index].ane_video_text_kv);
        weights[index].ane_video_text_kv = NULL;
        detached++;
    }
    return detached;
}

static int validate_ane_kv_text_rows(
        const block_weights *weights, uint32_t block_count,
        uint32_t expected_rows, char *error, size_t error_size) {
    if (!weights || !expected_rows) return 0;
    for (uint32_t index = 0; index < block_count; index++) {
        if (!weights[index].ane_video_text_kv) continue;
        const ltx_ane_kv_shape *shape = ltx_ane_kv_get_shape(
            weights[index].ane_video_text_kv);
        if (!shape || shape->text_rows != expected_rows) {
            snprintf(error, error_size,
                     "ANE Video-text K/V block %u has %u rows; expected %u",
                     index, shape ? shape->text_rows : 0u, expected_rows);
            return 0;
        }
    }
    return 1;
}
#endif

#ifdef LTX_ENABLE_ANE_QKV
static int attach_ane_video_self_qkv(
        ltx_gpu *gpu, block_weights *weights, uint32_t block,
        uint32_t expected_rows, const char *manifest, const char *variant,
        char *error, size_t error_size) {
    unsigned slot = expected_rows == active_geometry.stage1_rows ? 0u :
        expected_rows == active_geometry.stage2_rows ? 1u : 2u;
    if (slot >= 2u) {
        snprintf(error, error_size,
                 "invalid ANE Video self QKV stage");
        return 0;
    }
    ltx_ane_qkv *existing = weights->ane_video_self_qkv[slot];
    if (existing) {
        const ltx_ane_qkv_shape *shape = ltx_ane_qkv_get_shape(existing);
        if (!shape || shape->block_index != block ||
            shape->hidden != 4096u || shape->rows >= expected_rows) {
            snprintf(error, error_size,
                     "existing ANE Video self QKV artifact does not match "
                     "block %u at %u rows", block, expected_rows);
            return 0;
        }
        uint32_t prefix_rows = expected_rows - shape->rows;
        if (ane_video_self_qkv_prefix_rows &&
            ane_video_self_qkv_prefix_rows != prefix_rows) {
            snprintf(error, error_size,
                     "ANE Video self QKV artifacts use inconsistent row splits");
            return 0;
        }
        ane_video_self_qkv_prefix_rows = prefix_rows;
        return 1;
    }
    ltx_ane_qkv *model = ltx_ane_qkv_create(
        gpu, manifest, variant, error, error_size);
    const ltx_ane_qkv_shape *shape = ltx_ane_qkv_get_shape(model);
    if (!model || !shape || shape->block_index != block ||
        shape->hidden != 4096u || shape->rows >= expected_rows) {
        if (!error[0])
            snprintf(error, error_size,
                     "ANE Video self QKV does not match block %u at %u rows",
                     block, expected_rows);
        ltx_ane_qkv_free(model);
        return 0;
    }
    uint32_t prefix_rows = expected_rows - shape->rows;
    if (ane_video_self_qkv_prefix_rows &&
        ane_video_self_qkv_prefix_rows != prefix_rows) {
        snprintf(error, error_size,
                 "ANE Video self QKV artifacts use inconsistent row splits");
        ltx_ane_qkv_free(model);
        return 0;
    }
    ane_video_self_qkv_prefix_rows = prefix_rows;
    weights->ane_video_self_qkv[slot] = model;
    return 1;
}

static int attach_ane_qkv_directory(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        uint32_t rows, const char *directory, const char *variant,
        uint32_t *attached, char *error, size_t error_size) {
    if (!directory || !directory[0]) return 1;
    for (uint32_t index = 0; index < block_count; index++) {
        uint32_t block = first_block + index;
        char manifest[4096];
        int length = snprintf(manifest, sizeof(manifest),
                              "%s/block-%u/manifest.json",
                              directory, block);
        struct stat info;
        if (length < 0 || (size_t)length >= sizeof(manifest)) {
            snprintf(error, error_size,
                     "ANE Video self QKV path is too long");
            return 0;
        }
        if (stat(manifest, &info) != 0) {
            if (errno == ENOENT) continue;
            snprintf(error, error_size,
                     "cannot stat ANE Video self QKV artifact %s",
                     manifest);
            return 0;
        }
        if (!S_ISREG(info.st_mode) ||
            !attach_ane_video_self_qkv(
                gpu, &weights[index], block, rows, manifest, variant,
                error, error_size)) return 0;
        (*attached)++;
    }
    return 1;
}

static uint32_t detach_ane_qkv_stage(
        block_weights *weights, uint32_t block_count, uint32_t rows) {
    unsigned slot = rows == active_geometry.stage1_rows ? 0u :
        rows == active_geometry.stage2_rows ? 1u : 2u;
    uint32_t detached = 0u;
    if (!weights || slot >= 2u) return 0u;
    for (uint32_t index = 0u; index < block_count; index++) {
        ltx_ane_qkv *model = weights[index].ane_video_self_qkv[slot];
        if (!model) continue;
        weights[index].ane_video_self_qkv[slot] = NULL;
        ltx_ane_qkv_free(model);
        detached++;
    }
    ane_video_self_qkv_enabled = 0;
    ane_video_self_qkv_prefix_rows = 0u;
    return detached;
}
#endif

static int run_block_fixture(
                             const ltx_st_header *header,
                             const ltx_st_mapping *mapping,
                             ltx_gpu *gpu, block_weights *weights,
                             const char *fixture_directory,
                             char *error, size_t error_size) {
    uint32_t video_rows = 0;
    uint32_t audio_rows = 0;
    uint32_t text_rows = 0;
    uint32_t video_dim = weights->video_self.query_dim;
    uint32_t audio_dim = weights->audio_self.query_dim;
    size_t video_bytes = 0;
    size_t audio_bytes = 0;
    size_t video_text_bytes = 0;
    size_t audio_text_bytes = 0;
    size_t mask_bytes = 0;
    size_t video_position_bytes = 0;
    size_t audio_position_bytes = 0;
    size_t video_reference_bytes = 0;
    size_t audio_reference_bytes = 0;
    size_t video_patch_input_bytes = 0;
    size_t audio_patch_input_bytes = 0;
    size_t video_head_reference_bytes = 0;
    size_t audio_head_reference_bytes = 0;
    uint16_t *host_video = NULL;
    uint16_t *host_audio = NULL;
    uint16_t *host_video_patch = NULL;
    uint16_t *host_audio_patch = NULL;
    uint16_t *host_video_text = NULL;
    uint16_t *host_audio_text = NULL;
    uint16_t *host_mask = NULL;
    uint16_t *actual_video_patch = NULL;
    uint16_t *actual_audio_patch = NULL;
    uint16_t *actual_video = NULL;
    uint16_t *actual_audio = NULL;
    uint16_t *actual_video_head = NULL;
    uint16_t *actual_audio_head = NULL;
    float *video_positions = NULL;
    float *audio_positions = NULL;
    float *reference_video_patch = NULL;
    float *reference_audio_patch = NULL;
    float *reference_video = NULL;
    float *reference_audio = NULL;
    float *reference_video_head = NULL;
    float *reference_audio_head = NULL;
    ltx_transformer_io *io = NULL;
    ltx_transformer_conditioning *conditioning = NULL;
    ltx_transformer_conditioning_values conditioning_values = {0};
    ltx_gpu_buffer *video_patch_input = NULL;
    ltx_gpu_buffer *audio_patch_input = NULL;
    ltx_gpu_buffer *video_embedded = NULL;
    ltx_gpu_buffer *audio_embedded = NULL;
    ltx_gpu_buffer *video_head_workspace = NULL;
    ltx_gpu_buffer *audio_head_workspace = NULL;
    ltx_gpu_buffer *video_head_output = NULL;
    ltx_gpu_buffer *audio_head_output = NULL;
    ltx_gpu_buffer *video_text = NULL;
    ltx_gpu_buffer *audio_text = NULL;
    ltx_gpu_buffer *text_mask = NULL;
    block_rope rope = {0};
    block_workspace workspace = {0};
    int ok = 0;

    io = ltx_transformer_io_load(
        header, mapping, gpu, "model.diffusion_model",
        error, error_size);
    conditioning = ltx_transformer_conditioning_load(
        header, mapping, gpu, "model.diffusion_model",
        error, error_size);
    if (!io || !conditioning ||
        ltx_transformer_io_video_hidden_dim(io) != video_dim ||
        ltx_transformer_io_audio_hidden_dim(io) != audio_dim) {
        if (!error[0])
            snprintf(error, error_size,
                     "incompatible Transformer I/O fixture geometry");
        goto cleanup;
    }
    uint32_t video_patch_dim =
        ltx_transformer_io_video_patch_dim(io);
    uint32_t audio_patch_dim =
        ltx_transformer_io_audio_patch_dim(io);
    if (!fixture_rows(fixture_directory, "video_positions.f32",
                      3u * sizeof(float), &video_rows,
                      error, error_size) ||
        !fixture_rows(fixture_directory, "audio_positions.f32",
                      sizeof(float), &audio_rows,
                      error, error_size) ||
        !fixture_rows(fixture_directory, "text_mask.bf16",
                      sizeof(uint16_t), &text_rows,
                      error, error_size) ||
        !checked_bytes((uint64_t)video_rows * video_dim,
                       sizeof(uint16_t), &video_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_dim,
                       sizeof(uint16_t), &audio_bytes) ||
        !checked_bytes((uint64_t)text_rows * video_dim,
                       sizeof(uint16_t), &video_text_bytes) ||
        !checked_bytes((uint64_t)text_rows * audio_dim,
                       sizeof(uint16_t), &audio_text_bytes) ||
        !checked_bytes(text_rows, sizeof(uint16_t), &mask_bytes) ||
        !checked_bytes((uint64_t)video_rows * 3u,
                       sizeof(float), &video_position_bytes) ||
        !checked_bytes(audio_rows, sizeof(float), &audio_position_bytes) ||
        !checked_bytes((uint64_t)video_rows * video_dim,
                       sizeof(float), &video_reference_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_dim,
                       sizeof(float), &audio_reference_bytes) ||
        !checked_bytes((uint64_t)video_rows * video_patch_dim,
                       sizeof(uint16_t), &video_patch_input_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_patch_dim,
                       sizeof(uint16_t), &audio_patch_input_bytes) ||
        !checked_bytes((uint64_t)video_rows * video_patch_dim,
                       sizeof(float), &video_head_reference_bytes) ||
        !checked_bytes((uint64_t)audio_rows * audio_patch_dim,
                       sizeof(float), &audio_head_reference_bytes)) {
        if (!error[0])
            snprintf(error, error_size, "invalid fixture geometry");
        return 0;
    }

    host_video = malloc(video_bytes);
    host_audio = malloc(audio_bytes);
    host_video_patch = malloc(video_patch_input_bytes);
    host_audio_patch = malloc(audio_patch_input_bytes);
    host_video_text = malloc(video_text_bytes);
    host_audio_text = malloc(audio_text_bytes);
    host_mask = malloc(mask_bytes);
    actual_video_patch = malloc(video_bytes);
    actual_audio_patch = malloc(audio_bytes);
    actual_video = malloc(video_bytes);
    actual_audio = malloc(audio_bytes);
    actual_video_head = malloc(
        (size_t)video_rows * video_patch_dim * sizeof(uint16_t));
    actual_audio_head = malloc(
        (size_t)audio_rows * audio_patch_dim * sizeof(uint16_t));
    video_positions = malloc(video_position_bytes);
    audio_positions = malloc(audio_position_bytes);
    reference_video_patch = malloc(video_reference_bytes);
    reference_audio_patch = malloc(audio_reference_bytes);
    reference_video = malloc(video_reference_bytes);
    reference_audio = malloc(audio_reference_bytes);
    reference_video_head = malloc(video_head_reference_bytes);
    reference_audio_head = malloc(audio_head_reference_bytes);
    if (!host_video || !host_audio || !host_video_patch ||
        !host_audio_patch || !host_video_text ||
        !host_audio_text || !host_mask ||
        !actual_video_patch || !actual_audio_patch ||
        !actual_video || !actual_audio ||
        !actual_video_head || !actual_audio_head ||
        !video_positions || !audio_positions ||
        !reference_video_patch || !reference_audio_patch ||
        !reference_video || !reference_audio ||
        !reference_video_head || !reference_audio_head) {
        snprintf(error, error_size, "out of memory reading block fixture");
        goto cleanup;
    }

    if (!read_fixture_file(fixture_directory, "video_input.bf16",
                           host_video, video_bytes, error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_input.bf16",
                           host_audio, audio_bytes, error, error_size) ||
        !read_fixture_file(fixture_directory, "video_patch_input.bf16",
                           host_video_patch, video_patch_input_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_patch_input.bf16",
                           host_audio_patch, audio_patch_input_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "video_patch_output.f32",
                           reference_video_patch, video_reference_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_patch_output.f32",
                           reference_audio_patch, audio_reference_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "video_context.bf16",
                           host_video_text, video_text_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_context.bf16",
                           host_audio_text, audio_text_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "text_mask.bf16",
                           host_mask, mask_bytes, error, error_size) ||
        !read_fixture_file(fixture_directory, "video_positions.f32",
                           video_positions, video_position_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_positions.f32",
                           audio_positions, audio_position_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "video_output.f32",
                           reference_video, video_reference_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_output.f32",
                           reference_audio, audio_reference_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "video_head_output.f32",
                           reference_video_head,
                           video_head_reference_bytes,
                           error, error_size) ||
        !read_fixture_file(fixture_directory, "audio_head_output.f32",
                           reference_audio_head,
                           audio_head_reference_bytes,
                           error, error_size) ||
        !apply_scalar_conditioning(
            conditioning, weights, 1u, 0.9f, &conditioning_values,
            error, error_size) ||
        !apply_scalar_conditioning(
            conditioning, weights, 1u, 0.5f, &conditioning_values,
            error, error_size)) goto cleanup;
    video_embedded = conditioning_values.video_embedded;
    audio_embedded = conditioning_values.audio_embedded;

    if (!create_block_rope_from_positions(
            gpu, weights, video_positions, audio_positions,
            video_rows, audio_rows, &rope, error, error_size) ||
        !create_workspace(
            gpu, video_rows, video_dim, weights->video_self.heads,
            audio_rows, audio_dim,
            text_rows, &workspace, error, error_size)) goto cleanup;
    video_patch_input = ltx_gpu_buffer_new_copy(
        gpu, host_video_patch, video_patch_input_bytes,
        error, error_size);
    audio_patch_input = ltx_gpu_buffer_new_copy(
        gpu, host_audio_patch, audio_patch_input_bytes,
        error, error_size);
    if (!video_patch_input || !audio_patch_input ||
        !video_embedded || !audio_embedded) goto cleanup;
    video_head_workspace = ltx_gpu_buffer_new(
        gpu, video_bytes, error, error_size);
    audio_head_workspace = ltx_gpu_buffer_new(
        gpu, audio_bytes, error, error_size);
    video_head_output = ltx_gpu_buffer_new(
        gpu, video_patch_input_bytes, error, error_size);
    audio_head_output = ltx_gpu_buffer_new(
        gpu, audio_patch_input_bytes, error, error_size);
    video_text = ltx_gpu_buffer_new_copy(
        gpu, host_video_text, video_text_bytes, error, error_size);
    audio_text = ltx_gpu_buffer_new_copy(
        gpu, host_audio_text, audio_text_bytes, error, error_size);
    text_mask = ltx_gpu_buffer_new_copy(
        gpu, host_mask, mask_bytes, error, error_size);
    if (!video_head_workspace || !audio_head_workspace ||
        !video_head_output || !audio_head_output ||
        !video_text || !audio_text || !text_mask) goto cleanup;

    if (!ltx_transformer_io_patchify_video(
            io, workspace.video_state[0], video_patch_input,
            video_rows, error, error_size) ||
        !ltx_transformer_io_patchify_audio(
            io, workspace.audio_state[0], audio_patch_input,
            audio_rows, error, error_size) ||
        !ltx_gpu_buffer_read(
            workspace.video_state[0], actual_video_patch,
            video_bytes, error, error_size) ||
        !ltx_gpu_buffer_read(
            workspace.audio_state[0], actual_audio_patch,
            audio_bytes, error, error_size)) goto cleanup;
    comparison_metrics video_patch_metrics = compare_bf16_f32(
        actual_video_patch, reference_video_patch,
        (uint64_t)video_rows * video_dim);
    comparison_metrics audio_patch_metrics = compare_bf16_f32(
        actual_audio_patch, reference_audio_patch,
        (uint64_t)audio_rows * audio_dim);

#define LTX_FIXTURE_RESET_AND_RUN(TIMING) \
    (ltx_transformer_io_patchify_video( \
         io, workspace.video_state[0], video_patch_input, \
         video_rows, error, error_size) && \
     ltx_transformer_io_patchify_audio( \
         io, workspace.audio_state[0], audio_patch_input, \
         audio_rows, error, error_size) && \
     run_block(gpu, weights, 0u, &rope, &workspace, video_text, audio_text, \
               text_mask, video_rows, audio_rows, text_rows, (TIMING), \
               error, error_size) && \
     ltx_transformer_io_output_video( \
         io, video_head_output, video_head_workspace, \
         workspace.video_state[0], video_embedded, video_rows, \
         error, error_size) && \
     ltx_transformer_io_output_audio( \
         io, audio_head_output, audio_head_workspace, \
         workspace.audio_state[0], audio_embedded, audio_rows, \
         error, error_size))
    if (!LTX_FIXTURE_RESET_AND_RUN(NULL)) goto cleanup;
    block_timing timing = {0};
    double start = now_seconds();
    if (!LTX_FIXTURE_RESET_AND_RUN(&timing)) goto cleanup;
    double elapsed = now_seconds() - start;
    if (!ltx_gpu_buffer_read(workspace.video_state[0], actual_video,
                             video_bytes, error, error_size) ||
        !ltx_gpu_buffer_read(workspace.audio_state[0], actual_audio,
                             audio_bytes, error, error_size) ||
        !ltx_gpu_buffer_read(video_head_output, actual_video_head,
                             video_patch_input_bytes,
                             error, error_size) ||
        !ltx_gpu_buffer_read(audio_head_output, actual_audio_head,
                             audio_patch_input_bytes,
                             error, error_size)) goto cleanup;
#undef LTX_FIXTURE_RESET_AND_RUN

    comparison_metrics video_metrics = compare_bf16_f32(
        actual_video, reference_video, (uint64_t)video_rows * video_dim);
    comparison_metrics audio_metrics = compare_bf16_f32(
        actual_audio, reference_audio, (uint64_t)audio_rows * audio_dim);
    comparison_metrics video_head_metrics = compare_bf16_f32(
        actual_video_head, reference_video_head,
        (uint64_t)video_rows * video_patch_dim);
    comparison_metrics audio_head_metrics = compare_bf16_f32(
        actual_audio_head, reference_audio_head,
        (uint64_t)audio_rows * audio_patch_dim);
    printf("fixture=%s video_rows=%u audio_rows=%u text_rows=%u\n",
           fixture_directory, video_rows, audio_rows, text_rows);
    printf("fixture_step_ms=%.3f path=patchify-block-output-head "
           "text_mask=additive-bf16 semantics=reference-pre-cross\n",
           elapsed * 1000.0);
    printf("video_patch rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "nonfinite=%llu\n",
           video_patch_metrics.rel_l2, video_patch_metrics.cosine,
           video_patch_metrics.max_abs,
           (unsigned long long)video_patch_metrics.nonfinite);
    printf("audio_patch rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "nonfinite=%llu\n",
           audio_patch_metrics.rel_l2, audio_patch_metrics.cosine,
           audio_patch_metrics.max_abs,
           (unsigned long long)audio_patch_metrics.nonfinite);
    printf("video rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "reference_rms=%.9g candidate_rms=%.9g nonfinite=%llu\n",
           video_metrics.rel_l2, video_metrics.cosine,
           video_metrics.max_abs, video_metrics.reference_rms,
           video_metrics.candidate_rms,
           (unsigned long long)video_metrics.nonfinite);
    printf("audio rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "reference_rms=%.9g candidate_rms=%.9g nonfinite=%llu\n",
           audio_metrics.rel_l2, audio_metrics.cosine,
           audio_metrics.max_abs, audio_metrics.reference_rms,
           audio_metrics.candidate_rms,
           (unsigned long long)audio_metrics.nonfinite);
    printf("video_head rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "reference_rms=%.9g candidate_rms=%.9g nonfinite=%llu\n",
           video_head_metrics.rel_l2, video_head_metrics.cosine,
           video_head_metrics.max_abs, video_head_metrics.reference_rms,
           video_head_metrics.candidate_rms,
           (unsigned long long)video_head_metrics.nonfinite);
    printf("audio_head rel_l2=%.9g cosine=%.9g max_abs=%.9g "
           "reference_rms=%.9g candidate_rms=%.9g nonfinite=%llu\n",
           audio_head_metrics.rel_l2, audio_head_metrics.cosine,
           audio_head_metrics.max_abs, audio_head_metrics.reference_rms,
           audio_head_metrics.candidate_rms,
           (unsigned long long)audio_head_metrics.nonfinite);
    printf("profile_ms video_sa=%.3f audio_sa=%.3f "
           "video_text=%.3f audio_text=%.3f av_norm_mod=%.3f "
           "a2v=%.3f v2a=%.3f av_residual=%.3f "
           "video_ffn=%.3f audio_ffn=%.3f\n",
           timing.video_self * 1000.0, timing.audio_self * 1000.0,
           timing.video_text * 1000.0, timing.audio_text * 1000.0,
           timing.av_norm_modulation * 1000.0,
           timing.audio_to_video * 1000.0,
           timing.video_to_audio * 1000.0,
           timing.av_residual * 1000.0,
           timing.video_ffn * 1000.0, timing.audio_ffn * 1000.0);

    if (video_patch_metrics.nonfinite || audio_patch_metrics.nonfinite ||
        video_metrics.nonfinite || audio_metrics.nonfinite ||
        video_head_metrics.nonfinite || audio_head_metrics.nonfinite ||
        video_patch_metrics.rel_l2 > 0.05 ||
        audio_patch_metrics.rel_l2 > 0.05 ||
        video_metrics.rel_l2 > 0.15 || audio_metrics.rel_l2 > 0.15 ||
        video_head_metrics.rel_l2 > 0.15 ||
        audio_head_metrics.rel_l2 > 0.15 ||
        video_patch_metrics.cosine < 0.999 ||
        audio_patch_metrics.cosine < 0.999 ||
        video_metrics.cosine < 0.99 || audio_metrics.cosine < 0.99 ||
        video_head_metrics.cosine < 0.99 ||
        audio_head_metrics.cosine < 0.99) {
        snprintf(error, error_size,
                 "block fixture parity outside tolerance");
        goto cleanup;
    }
    ok = 1;

cleanup:
    ltx_gpu_buffer_free(video_text);
    ltx_gpu_buffer_free(audio_text);
    ltx_gpu_buffer_free(text_mask);
    ltx_gpu_buffer_free(video_head_output);
    ltx_gpu_buffer_free(audio_head_output);
    ltx_gpu_buffer_free(video_head_workspace);
    ltx_gpu_buffer_free(audio_head_workspace);
    ltx_gpu_buffer_free(video_patch_input);
    ltx_gpu_buffer_free(audio_patch_input);
    ltx_transformer_conditioning_free(conditioning);
    ltx_transformer_io_free(io);
    free_workspace(&workspace);
    free_block_rope(&rope);
    free(host_video);
    free(host_audio);
    free(host_video_patch);
    free(host_audio_patch);
    free(host_video_text);
    free(host_audio_text);
    free(host_mask);
    free(actual_video_patch);
    free(actual_audio_patch);
    free(actual_video);
    free(actual_audio);
    free(actual_video_head);
    free(actual_audio_head);
    free(video_positions);
    free(audio_positions);
    free(reference_video_patch);
    free(reference_audio_patch);
    free(reference_video);
    free(reference_audio);
    free(reference_video_head);
    free(reference_audio_head);
    return ok;
}

static void fill_schedule_noise(float *values, uint32_t elements,
                                size_t step, float phase) {
    for (uint32_t index = 0; index < elements; index++) {
        float x = (float)(index % 8191u);
        values[index] =
            sinf(x * 0.017f + (float)step * 0.73f + phase) * 0.8f +
            cosf(x * 0.007f + (float)step * 0.31f + phase) * 0.2f;
    }
}

static int run_denoise_schedule(
        ltx_gpu *gpu, block_weights *weights,
        uint32_t first_block, uint32_t block_count,
        const block_rope *rope, block_workspace *workspace,
        ltx_transformer_io *io,
        ltx_transformer_conditioning *conditioning,
        const ltx_gpu_buffer *video_text,
        const ltx_gpu_buffer *audio_text,
        const ltx_gpu_buffer *text_mask,
        ltx_gpu_buffer *video_latent,
        ltx_gpu_buffer *audio_latent,
        const ltx_gpu_buffer *video_clean_prefix,
        uint32_t conditioned_prefix_rows,
        float conditioning_strength,
        ltx_gpu_buffer *video_head_workspace,
        ltx_gpu_buffer *audio_head_workspace,
        ltx_gpu_buffer *video_velocity,
        ltx_gpu_buffer *audio_velocity,
        uint32_t video_rows, uint32_t audio_rows, uint32_t text_rows,
        uint32_t video_patch_dim, uint32_t audio_patch_dim,
        const float *sigmas, size_t sigma_count, int ancestral,
        ltx_rng *video_noise_rng, ltx_rng *audio_noise_rng,
        int release_blocks_on_final_step,
        char *error, size_t error_size) {
    uint64_t video_elements64 = (uint64_t)video_rows * video_patch_dim;
    uint64_t audio_elements64 = (uint64_t)audio_rows * audio_patch_dim;
    uint64_t conditioned_elements64 =
        (uint64_t)conditioned_prefix_rows * video_patch_dim;
    int has_video_conditioning = conditioned_prefix_rows != 0u;
    if (!gpu || !weights || !rope || !workspace || !io || !conditioning ||
        !video_text || !audio_text || !video_latent || !audio_latent ||
        !video_head_workspace || !audio_head_workspace ||
        !video_velocity || !audio_velocity || !sigmas ||
        sigma_count < 2u || video_elements64 > UINT32_MAX ||
        audio_elements64 > UINT32_MAX || conditioned_elements64 > UINT32_MAX ||
        conditioned_prefix_rows > video_rows ||
        (has_video_conditioning && !video_clean_prefix) ||
        !isfinite(conditioning_strength) || conditioning_strength < 0.0f ||
        conditioning_strength > 1.0f) {
        snprintf(error, error_size, "invalid denoise schedule arguments");
        return 0;
    }
    if (sol_video_self_stage_enabled(video_rows) &&
        sol_video_self.step_tau_count &&
        sol_video_self.step_tau_count != sigma_count - 1u) {
        snprintf(error, error_size,
                 "LTX_SOL_TAUS has %zu values, but this stage has %zu steps",
                 sol_video_self.step_tau_count, sigma_count - 1u);
        return 0;
    }
    uint32_t video_elements = (uint32_t)video_elements64;
    uint32_t audio_elements = (uint32_t)audio_elements64;
    uint32_t conditioned_elements = (uint32_t)conditioned_elements64;
    float conditioning_mask = 1.0f - conditioning_strength;
    size_t video_bf16_bytes = (size_t)video_elements * sizeof(uint16_t);
    size_t audio_bf16_bytes = (size_t)audio_elements * sizeof(uint16_t);
    size_t video_f32_bytes = (size_t)video_elements * sizeof(float);
    size_t audio_f32_bytes = (size_t)audio_elements * sizeof(float);

    ltx_gpu_buffer *video_x0 = ltx_gpu_buffer_new(
        gpu, video_bf16_bytes, error, error_size);
    ltx_gpu_buffer *audio_x0 = ltx_gpu_buffer_new(
        gpu, audio_bf16_bytes, error, error_size);
    ltx_gpu_buffer *video_scratch = ltx_gpu_buffer_new(
        gpu, video_bf16_bytes, error, error_size);
    ltx_gpu_buffer *audio_scratch = ltx_gpu_buffer_new(
        gpu, audio_bf16_bytes, error, error_size);
    ltx_gpu_buffer *video_noise = ltx_gpu_buffer_new(
        gpu, video_f32_bytes, error, error_size);
    ltx_gpu_buffer *audio_noise = ltx_gpu_buffer_new(
        gpu, audio_f32_bytes, error, error_size);
    ltx_gpu_buffer *conditioned_video_embedded = has_video_conditioning ?
        ltx_gpu_buffer_new(
            gpu, (size_t)weights[0].video_self.query_dim * sizeof(uint16_t),
            error, error_size) : NULL;
    float *video_noise_host = malloc(video_f32_bytes);
    float *audio_noise_host = malloc(audio_f32_bytes);
    uint16_t *video_final_host = malloc(video_bf16_bytes);
    uint16_t *audio_final_host = malloc(audio_bf16_bytes);
    int ok = video_x0 && audio_x0 && video_scratch && audio_scratch &&
        video_noise && audio_noise && video_noise_host && audio_noise_host &&
        video_final_host && audio_final_host &&
        (!has_video_conditioning || conditioned_video_embedded);
    if (!ok) {
        if (!error[0])
            snprintf(error, error_size,
                     "out of memory creating denoise schedule workspace");
        goto cleanup;
    }

    active_video_conditioned_prefix_rows = conditioned_prefix_rows;

    if (!ancestral) {
        if (video_noise_rng)
            ltx_rng_fill_normal_f32(
                video_noise_rng, video_noise_host, video_elements);
        else
            fill_schedule_noise(
                video_noise_host, video_elements, 0u, 0.25f);
        if (audio_noise_rng)
            ltx_rng_fill_normal_f32(
                audio_noise_rng, audio_noise_host, audio_elements);
        else
            fill_schedule_noise(
                audio_noise_host, audio_elements, 0u, 1.25f);
        for (uint32_t index = 0; index < video_elements; index++)
            video_final_host[index] =
                f32_to_bf16(video_noise_host[index]);
        for (uint32_t index = 0; index < audio_elements; index++)
            audio_final_host[index] =
                f32_to_bf16(audio_noise_host[index]);
        ok = ltx_gpu_buffer_write(
                video_scratch, video_final_host, video_bf16_bytes,
                error, error_size) &&
            ltx_gpu_buffer_write(
                audio_scratch, audio_final_host, audio_bf16_bytes,
                error, error_size) &&
            ltx_gpu_renoise_bf16(
                gpu, video_latent, video_latent, video_scratch,
                video_elements, sigmas[0], error, error_size) &&
            ltx_gpu_renoise_bf16(
                gpu, audio_latent, audio_latent, audio_scratch,
                audio_elements, sigmas[0], error, error_size);
        if (!ok) goto cleanup;
    }
    /* Upstream applies first-frame conditioning after constructing the
     * initial noised state.  The conditioned prefix therefore starts from
     * the exact clean latent for every strength; strength only controls its
     * per-token timestep and the clean/x0 blend during denoising. */
    if (has_video_conditioning &&
        !ltx_gpu_condition_prefix_bf16(
            gpu, video_latent, video_clean_prefix, conditioned_elements,
            0.0f, error, error_size)) {
        ok = 0;
        goto cleanup;
    }

    ltx_gpu_buffer *video_current = video_latent;
    ltx_gpu_buffer *audio_current = audio_latent;
    ltx_gpu_buffer *video_next = video_scratch;
    ltx_gpu_buffer *audio_next = audio_scratch;
    double schedule_start = now_seconds();
    for (size_t step = 0; step + 1u < sigma_count; step++) {
        if(report_progress("ltx_step",(int)step,(int)sigma_count-1,error,error_size)){ok=0;goto cleanup;}
        sol_video_self.active_step = step;
        sol_video_self.active_step_count = sigma_count - 1u;
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1u];
        ltx_transformer_conditioning_values values = {0};
        double conditioning_start = now_seconds();
        if (!apply_video_split_conditioning(
                conditioning, weights, block_count, sigma,
                sigma * conditioning_mask, conditioned_video_embedded,
                &values,
                error, error_size)) {
            char detail[1024];
            snprintf(detail, sizeof(detail), "%s",
                     error[0] ? error : "unknown conditioning error");
            snprintf(error, error_size, "step %zu conditioning: %s",
                     step, detail);
            ok = 0;
            goto cleanup;
        }
        double conditioning_seconds = now_seconds() - conditioning_start;

        double model_start = now_seconds();
        int release_blocks = release_blocks_on_final_step &&
            step + 2u == sigma_count;
        size_t released_block_bytes = 0;
        double release_start = release_blocks ? now_seconds() : 0.0;
        if (!ltx_transformer_io_patchify_video(
                io, workspace->video_state[0], video_current,
                video_rows, error, error_size) ||
            !ltx_transformer_io_patchify_audio(
                io, workspace->audio_state[0], audio_current,
                audio_rows, error, error_size) ||
            !(release_blocks ?
              run_blocks_and_release(
                  gpu, weights, first_block, block_count, rope, workspace,
                  video_text, audio_text, text_mask,
                  video_rows, audio_rows, text_rows, NULL,
                  &released_block_bytes, error, error_size) :
              run_blocks(
                  gpu, weights, first_block, block_count, rope, workspace,
                  video_text, audio_text, text_mask,
                  video_rows, audio_rows, text_rows, NULL,
                  error, error_size)) ||
            !(has_video_conditioning ?
              ltx_transformer_io_output_video_split(
                  io, video_velocity, video_head_workspace,
                  workspace->video_state[0], values.video_embedded,
                  conditioned_video_embedded, video_rows,
                  conditioned_prefix_rows, error, error_size) :
              ltx_transformer_io_output_video(
                  io, video_velocity, video_head_workspace,
                  workspace->video_state[0], values.video_embedded,
                  video_rows, error, error_size)) ||
            !ltx_transformer_io_output_audio(
                io, audio_velocity, audio_head_workspace,
                workspace->audio_state[0], values.audio_embedded,
                audio_rows, error, error_size)) {
            char detail[1024];
            snprintf(detail, sizeof(detail), "%s",
                     error[0] ? error : "unknown Transformer error");
            snprintf(error, error_size, "step %zu Transformer: %s",
                     step, detail);
            ok = 0;
            goto cleanup;
        }
        if (release_blocks)
            printf("final_step_block_release blocks=%u bytes=%zu "
                   "elapsed_ms=%.3f\n",
                   block_count, released_block_bytes,
                   (now_seconds() - release_start) * 1000.0);
        double model_seconds = now_seconds() - model_start;

        double update_start = now_seconds();
        if (!(has_video_conditioning ?
              ltx_gpu_velocity_to_denoised_bf16_split(
                  gpu, video_x0, video_current, video_velocity,
                  video_rows, video_patch_dim, conditioned_prefix_rows,
                  sigma, sigma * conditioning_mask,
                  error, error_size) :
              ltx_gpu_velocity_to_denoised_bf16(
                  gpu, video_x0, video_current, video_velocity,
                  video_elements, sigma, error, error_size)) ||
            !ltx_gpu_velocity_to_denoised_bf16(
                gpu, audio_x0, audio_current, audio_velocity,
                audio_elements, sigma, error, error_size)) {
            ok = 0;
            goto cleanup;
        }
        if (has_video_conditioning &&
            !ltx_gpu_condition_prefix_bf16(
                gpu, video_x0, video_clean_prefix, conditioned_elements,
                conditioning_mask, error, error_size)) {
            ok = 0;
            goto cleanup;
        }
        if (ancestral) {
            if (sigma_next != 0.0f) {
                if (video_noise_rng)
                    ltx_rng_fill_normal_f32(
                        video_noise_rng, video_noise_host, video_elements);
                else
                    fill_schedule_noise(
                        video_noise_host, video_elements, step, 0.25f);
                if (audio_noise_rng)
                    ltx_rng_fill_normal_f32(
                        audio_noise_rng, audio_noise_host, audio_elements);
                else
                    fill_schedule_noise(
                        audio_noise_host, audio_elements, step, 1.25f);
                if (!ltx_gpu_buffer_write(
                        video_noise, video_noise_host, video_f32_bytes,
                        error, error_size) ||
                    !ltx_gpu_buffer_write(
                        audio_noise, audio_noise_host, audio_f32_bytes,
                        error, error_size)) {
                    ok = 0;
                    goto cleanup;
                }
            }
            if (!ltx_gpu_euler_ancestral_step_bf16(
                    gpu, video_next, video_current, video_x0,
                    sigma_next == 0.0f ? NULL : video_noise,
                    video_elements, sigma, sigma_next, 1.0f, 1.0f,
                    error, error_size) ||
                !ltx_gpu_euler_ancestral_step_bf16(
                    gpu, audio_next, audio_current, audio_x0,
                    sigma_next == 0.0f ? NULL : audio_noise,
                    audio_elements, sigma, sigma_next, 1.0f, 1.0f,
                    error, error_size)) {
                ok = 0;
                goto cleanup;
            }
            if (has_video_conditioning && sigma_next != 0.0f &&
                !ltx_gpu_condition_prefix_bf16(
                    gpu, video_next, video_clean_prefix,
                    conditioned_elements, conditioning_mask,
                    error, error_size)) {
                ok = 0;
                goto cleanup;
            }
        } else if (!ltx_gpu_euler_step_bf16(
                       gpu, video_next, video_current, video_x0,
                       video_elements, sigma, sigma_next,
                       error, error_size) ||
                   !ltx_gpu_euler_step_bf16(
                       gpu, audio_next, audio_current, audio_x0,
                       audio_elements, sigma, sigma_next,
                       error, error_size)) {
            ok = 0;
            goto cleanup;
        }
        double update_seconds = now_seconds() - update_start;
        swap_buffers(&video_current, &video_next);
        swap_buffers(&audio_current, &audio_next);
        printf("schedule_step=%zu sigma=%.9g sigma_next=%.9g "
               "conditioning_ms=%.3f transformer_ms=%.3f update_ms=%.3f\n",
               step, sigma, sigma_next,
               conditioning_seconds * 1000.0,
               model_seconds * 1000.0,
               update_seconds * 1000.0);
    }

    ok = ltx_gpu_buffer_read(
            video_current, video_final_host, video_bf16_bytes,
            error, error_size) &&
        ltx_gpu_buffer_read(
            audio_current, audio_final_host, audio_bf16_bytes,
            error, error_size);
    if (!ok) goto cleanup;
    uint64_t nonfinite = 0;
    double video_square = 0.0;
    double audio_square = 0.0;
    for (uint32_t index = 0; index < video_elements; index++) {
        double value = bf16_to_f32(video_final_host[index]);
        if (!isfinite(value)) nonfinite++;
        video_square += value * value;
    }
    for (uint32_t index = 0; index < audio_elements; index++) {
        double value = bf16_to_f32(audio_final_host[index]);
        if (!isfinite(value)) nonfinite++;
        audio_square += value * value;
    }
    printf("schedule=%s steps=%zu blocks=%u video_rows=%u audio_rows=%u "
           "elapsed_seconds=%.6f video_rms=%.9g audio_rms=%.9g "
           "nonfinite=%llu noise=%s\n",
           ancestral ? "ltx2.5-stage1-ancestral" :
                       "ltx2.5-stage2-deterministic",
           sigma_count - 1u, block_count, video_rows, audio_rows,
           now_seconds() - schedule_start,
           sqrt(video_square / video_elements),
           sqrt(audio_square / audio_elements),
           (unsigned long long)nonfinite,
           video_noise_rng || audio_noise_rng ?
               "seeded-pcg32-gaussian" : "deterministic-synthetic");
    if (nonfinite) {
        snprintf(error, error_size,
                 "denoise schedule produced non-finite values");
        ok = 0;
    } else {
        if (video_current != video_latent)
            ok = ltx_gpu_buffer_write(
                video_latent, video_final_host, video_bf16_bytes,
                error, error_size);
        if (ok && audio_current != audio_latent)
            ok = ltx_gpu_buffer_write(
                audio_latent, audio_final_host, audio_bf16_bytes,
                error, error_size);
    }

cleanup:
    active_video_conditioned_prefix_rows = 0u;
    sol_video_self.active_step = SIZE_MAX;
    sol_video_self.active_step_count = 0u;
    ltx_gpu_buffer_free(video_x0);
    ltx_gpu_buffer_free(audio_x0);
    ltx_gpu_buffer_free(video_scratch);
    ltx_gpu_buffer_free(audio_scratch);
    ltx_gpu_buffer_free(video_noise);
    ltx_gpu_buffer_free(audio_noise);
    ltx_gpu_buffer_free(conditioned_video_embedded);
    free(video_noise_host);
    free(audio_noise_host);
    free(video_final_host);
    free(audio_final_host);
    return ok;
}


/* TurboCider typed library boundary. The benchmark CLI above was removed;
 * graph construction receives tensors and explicit options, with no argv,
 * raw conditioning files, generated-media files or child process boundary. */
struct ltx_native_denoiser {
    ltx_native_options options;
    char *owned_strings[9];
    ltx_workload workload;
    ltx_gpu *gpu;
    ltx_gpu *audio_gpu;
    block_weights weights[48];
    ltx_transformer_io *io;
    ltx_transformer_conditioning *conditioning;
    int full_gpu_mlp_released;
};

static void ltx_native_activate_geometry(ltx_native_denoiser *ctx) {
    ltx_workload *g = &ctx->workload;
    active_geometry = (generation_geometry){
        g->output_width, g->output_height, g->latent_frames,
        g->stage1_latent_height, g->stage1_latent_width,
        g->stage2_latent_height, g->stage2_latent_width,
        (uint32_t)g->stage1_video_tokens,
        (uint32_t)g->stage2_video_tokens,
        g->frames, g->output_height, g->output_width
    };
}

#ifdef LTX_ENABLE_ANE_MLP
static int ltx_native_release_full_gpu_mlp(
        ltx_native_denoiser *ctx, char *error, size_t error_size) {
    if (!ctx->options.release_full_gpu_mlp ||
        ctx->full_gpu_mlp_released) return 1;
    for (uint32_t index = 0; index < 48u; index++) {
        if (!ctx->weights[index].ane_video_mlp[0] ||
            !ctx->weights[index].ane_video_mlp[1]) {
            snprintf(error, error_size,
                     "releasing full GPU Video MLP requires complete "
                     "Stage-1 and Stage-2 ANE coverage");
            return 0;
        }
    }
    for (uint32_t index = 0; index < 48u; index++)
        free_mlp(&ctx->weights[index].video_mlp);
    ctx->full_gpu_mlp_released = 1;
    return 1;
}
#endif

static void ltx_native_detach_ane_stage(
        ltx_native_denoiser *ctx, int stage, uint32_t rows) {
    int detach = stage == 1 ? ctx->options.detach_ane_stage1 :
                             ctx->options.detach_ane_stage2;
    if (!detach) return;
#ifdef LTX_ENABLE_ANE_MLP
    detach_ane_stage(ctx->weights, 48u, rows);
#endif
#ifdef LTX_ENABLE_ANE_V2A
    detach_ane_v2a_stage(ctx->weights, 48u, rows);
#endif
#ifdef LTX_ENABLE_ANE_KV
    detach_ane_kv_models(ctx->weights, 48u);
#endif
#ifdef LTX_ENABLE_ANE_QKV
    detach_ane_qkv_stage(ctx->weights, 48u, rows);
#endif
}

void ltx_native_free(ltx_native_denoiser *ctx) {
    if(!ctx)return;
    for(unsigned i=0;i<48;++i)free_block_weights(&ctx->weights[i]);
    ltx_transformer_conditioning_free(ctx->conditioning);
    ltx_transformer_io_free(ctx->io);ltx_gpu_free(ctx->audio_gpu);ltx_gpu_free(ctx->gpu);
    for(unsigned i=0;i<9;++i)free(ctx->owned_strings[i]);
    free(ctx);
}
ltx_native_denoiser *ltx_native_create(const ltx_native_options *options,
    ltx_native_progress progress,void *opaque,char *error,size_t error_size) {
    if(!options||!options->checkpoint||options->fps!=24){snprintf(error,error_size,"LTX requires checkpoint and 24 fps");return NULL;}
#ifdef LTX_ENABLE_ANE_MLP
    if (options->release_full_gpu_mlp &&
        (!options->preload_ane_stage2 || !options->mlp_directories[0] ||
         !options->mlp_directories[1] ||
         !ane_mlp_directory_complete(options->mlp_directories[0], 0u, 48u) ||
         !ane_mlp_directory_complete(options->mlp_directories[1], 0u, 48u))) {
        snprintf(error, error_size,
                 "full GPU Video MLP release requires complete preloaded "
                 "Stage-1 and Stage-2 ANE directories");
        return NULL;
    }
    if (options->ane_mlp_fused_residual &&
        options->ane_mlp_fused_adaln_pack) {
        snprintf(error, error_size,
                 "ANE MLP residual and AdaLN-pack fusion cannot both be enabled");
        return NULL;
    }
#endif
    ltx_native_denoiser *ctx=calloc(1,sizeof(*ctx));if(!ctx){snprintf(error,error_size,"allocation failed");return NULL;}
    ctx->options=*options;
    ltx_st_header header={0};ltx_st_mapping mapping={0};
    const char **fields[]={&ctx->options.checkpoint,&ctx->options.shader_source,&ctx->options.mlp_directories[0],&ctx->options.mlp_directories[1],&ctx->options.v2a_directories[0],&ctx->options.v2a_directories[1],&ctx->options.kv_directory,&ctx->options.qkv_directories[0],&ctx->options.qkv_directories[1]};
    for(unsigned i=0;i<9;++i)if(*fields[i]){ctx->owned_strings[i]=strdup(*fields[i]);if(!ctx->owned_strings[i]){snprintf(error,error_size,"option allocation failed");goto failed;}*fields[i]=ctx->owned_strings[i];}
    if(!ltx_workload_init(&ctx->workload,options->width,options->height,options->frames,options->fps,error,error_size))goto failed;
    if(ctx->workload.output_width!=options->width||ctx->workload.output_height!=options->height||ctx->workload.frames!=options->frames){snprintf(error,error_size,"LTX dimensions must already satisfy the two-stage geometry");goto failed;}
    const char *shader_source=ctx->options.shader_source?ctx->options.shader_source:"ltx_shaders.metal";
    ctx->gpu=ltx_gpu_create(shader_source,error,error_size);if(!ctx->gpu)goto failed;
    if(options->parallel_av){ctx->audio_gpu=ltx_gpu_create(shader_source,error,error_size);if(!ctx->audio_gpu)goto failed;}
    if(!ltx_st_read_header(options->checkpoint,&header,error,error_size)||!ltx_st_map_open(&header,&mapping,error,error_size))goto failed;
    for(unsigned i=0;i<48;++i){
        if(progress&&progress("ltx_load_block",i,48,opaque)){snprintf(error,error_size,"generation cancelled");goto failed;}
        if(!load_block_weights(&header,&mapping,ctx->gpu,i,&ctx->weights[i],error,error_size))goto failed;
    }
    ctx->io=ltx_transformer_io_load(&header,&mapping,ctx->gpu,"model.diffusion_model",error,error_size);
    ctx->conditioning=ltx_transformer_conditioning_load(&header,&mapping,ctx->gpu,"model.diffusion_model",error,error_size);
    if(!ctx->io||!ctx->conditioning)goto failed;
    ltx_native_activate_geometry(ctx);
#ifdef LTX_ENABLE_ANE_MLP
    if (ctx->options.preload_ane_stage2 && ctx->options.mlp_directories[1]) {
        uint32_t attached = 0u;
        if (progress && progress("ltx_preload_ane_stage2", 0, 1, opaque)) {
            snprintf(error,error_size,"generation cancelled");goto failed;
        }
        if (!attach_ane_directory(
                ctx->gpu, ctx->weights, 0u, 48u,
                active_geometry.stage2_rows,
                ctx->options.mlp_directories[1], "int8_pc", &attached,
                error, error_size) || attached != 48u) {
            if (!error[0])
                snprintf(error, error_size,
                         "Stage-2 ANE MLP preload attached %u of 48 blocks",
                         attached);
            goto failed;
        }
        if (progress && progress("ltx_preload_ane_stage2", 1, 1, opaque)) {
            snprintf(error,error_size,"generation cancelled");goto failed;
        }
    }
#endif
#ifdef LTX_ENABLE_ANE_V2A
    if (ctx->options.preload_ane_stage2 && ctx->options.v2a_directories[1]) {
        uint32_t attached = 0u;
        if (!attach_ane_v2a_directory(
                ctx->gpu, ctx->weights, 0u, 48u,
                active_geometry.stage2_rows,
                ctx->options.v2a_directories[1], "int8_pc", &attached,
                error, error_size) || attached != 48u) goto failed;
    }
#endif
#ifdef LTX_ENABLE_ANE_QKV
    if (ctx->options.preload_ane_stage2 && ctx->options.qkv_directories[1]) {
        uint32_t attached = 0u;
        if (!attach_ane_qkv_directory(
                ctx->gpu, ctx->weights, 0u, 48u,
                active_geometry.stage2_rows,
                ctx->options.qkv_directories[1], "int8_pc", &attached,
                error, error_size) || attached != 48u) goto failed;
    }
#endif
    ltx_st_map_close(&mapping);ltx_st_free_header(&header);return ctx;
failed:
    ltx_st_map_close(&mapping);ltx_st_free_header(&header);ltx_native_free(ctx);return NULL;
}
int ltx_native_run(ltx_native_denoiser *ctx,int stage,uint64_t seed,
    uint16_t *video,size_t video_elements,uint16_t *audio,size_t audio_elements,
    const uint16_t *video_text_host,const uint16_t *audio_text_host,const uint16_t *mask_host,
    uint32_t text_rows,const uint16_t *first_frame,float strength,
    ltx_native_progress progress,void *opaque,char *error,size_t error_size) {
    if(!ctx||stage<1||stage>2||!video||!audio||!video_text_host||!audio_text_host||text_rows==0||text_rows>4096){snprintf(error,error_size,"invalid LTX stage input");return 0;}
    ltx_workload *g=&ctx->workload;
    uint32_t rows=(uint32_t)(stage==1?g->stage1_video_tokens:g->stage2_video_tokens);
    uint32_t audio_rows=g->audio_tokens,vd=ltx_transformer_io_video_hidden_dim(ctx->io),ad=ltx_transformer_io_audio_hidden_dim(ctx->io);
    uint32_t vp=ltx_transformer_io_video_patch_dim(ctx->io),ap=ltx_transformer_io_audio_patch_dim(ctx->io);
    if(video_elements!=(size_t)rows*vp||audio_elements!=(size_t)audio_rows*ap){snprintf(error,error_size,"LTX latent tensor size mismatch");return 0;}
    ltx_native_activate_geometry(ctx);
    native_progress=progress;native_opaque=opaque;
    av_parallel_audio_gpu=ctx->audio_gpu;av_parallel_streams=ctx->options.parallel_av;
    av_parallel_cross=ctx->options.parallel_av;av_parallel_ffn=ctx->options.parallel_av;
    av_persistent_worker_enabled=0;video_attention_batch=1;
#ifdef LTX_ENABLE_ANE_MLP
    ane_mlp_fused_residual=ctx->options.ane_mlp_fused_residual;
    ane_mlp_fused_adaln_pack=ctx->options.ane_mlp_fused_adaln_pack;
#endif
#ifdef LTX_ENABLE_ANE_KV
    ane_video_text_kv_enabled=0;
#endif
#ifdef LTX_ENABLE_ANE_QKV
    ane_video_self_qkv_enabled=0;
    ane_video_self_qkv_prefix_rows=0u;
#endif
    ltx_gpu *gpu=ctx->gpu;block_rope rope={0};block_workspace workspace={0};
    ltx_gpu_buffer *buffers[9]={0};int ok=0;
    buffers[0]=ltx_gpu_buffer_new_copy(gpu,video,video_elements*2,error,error_size);
    buffers[1]=ltx_gpu_buffer_new_copy(gpu,audio,audio_elements*2,error,error_size);
    buffers[2]=ltx_gpu_buffer_new_copy(gpu,video_text_host,(size_t)text_rows*vd*2,error,error_size);
    buffers[3]=ltx_gpu_buffer_new_copy(gpu,audio_text_host,(size_t)text_rows*ad*2,error,error_size);
    buffers[4]=mask_host?ltx_gpu_buffer_new_copy(gpu,mask_host,text_rows*2,error,error_size):NULL;
    buffers[5]=ltx_gpu_buffer_new(gpu,(size_t)rows*vd*2,error,error_size);
    buffers[6]=ltx_gpu_buffer_new(gpu,(size_t)audio_rows*ad*2,error,error_size);
    buffers[7]=ltx_gpu_buffer_new(gpu,video_elements*2,error,error_size);
    buffers[8]=ltx_gpu_buffer_new(gpu,audio_elements*2,error,error_size);
    ltx_gpu_buffer *clean=NULL;
    uint32_t prefix=first_frame?(stage==1?g->stage1_latent_height*g->stage1_latent_width:g->stage2_latent_height*g->stage2_latent_width):0;
    if(first_frame)clean=ltx_gpu_buffer_new_copy(gpu,first_frame,(size_t)prefix*vp*2,error,error_size);
    for(unsigned i=0;i<9;++i)if((i!=4||mask_host)&&!buffers[i])goto cleanup;
    if(first_frame&&!clean)goto cleanup;
#ifdef LTX_ENABLE_ANE_MLP
    uint32_t loaded=0;
    if(ctx->options.mlp_directories[stage-1]&&!attach_ane_directory(gpu,ctx->weights,0,48,rows,ctx->options.mlp_directories[stage-1],"int8_pc",&loaded,error,error_size))goto cleanup;
    if (ctx->options.mlp_directories[stage-1] && loaded != 48u) {
        snprintf(error, error_size,
                 "LTX ANE MLP Stage %d attached %u of 48 blocks",
                 stage, loaded);
        goto cleanup;
    }
    if (!ltx_native_release_full_gpu_mlp(ctx, error, error_size)) goto cleanup;
#endif
#ifdef LTX_ENABLE_ANE_V2A
    uint32_t loaded_v2a=0;
    if(ctx->options.v2a_directories[stage-1]&&!attach_ane_v2a_directory(gpu,ctx->weights,0,48,rows,ctx->options.v2a_directories[stage-1],"int8_pc",&loaded_v2a,error,error_size))goto cleanup;
    if (ctx->options.v2a_directories[stage-1] && loaded_v2a != 48u) {
        snprintf(error, error_size,
                 "LTX ANE V-to-A Stage %d attached %u of 48 blocks",
                 stage, loaded_v2a);
        goto cleanup;
    }
#endif
#ifdef LTX_ENABLE_ANE_KV
    uint32_t loaded_kv=0;
    int use_kv = (ctx->options.ane_kv_stage_mask & (1u << (stage - 1))) != 0u;
    if(use_kv&&ctx->options.kv_directory&&!attach_ane_kv_directory(gpu,ctx->weights,0,48,ctx->options.kv_directory,"int8_pc",&loaded_kv,error,error_size))goto cleanup;
    if (use_kv && ctx->options.kv_directory && loaded_kv != 48u) {
        snprintf(error, error_size,
                 "LTX ANE text K/V Stage %d attached %u of 48 blocks",
                 stage, loaded_kv);
        goto cleanup;
    }
    if(loaded_kv && !validate_ane_kv_text_rows(
            ctx->weights, 48u, text_rows, error, error_size)) goto cleanup;
    ane_video_text_kv_enabled=loaded_kv!=0;
#endif
#ifdef LTX_ENABLE_ANE_QKV
    uint32_t loaded_qkv=0;
    if(ctx->options.qkv_directories[stage-1]&&!attach_ane_qkv_directory(gpu,ctx->weights,0,48,rows,ctx->options.qkv_directories[stage-1],"int8_pc",&loaded_qkv,error,error_size))goto cleanup;
    if (ctx->options.qkv_directories[stage-1] && loaded_qkv != 48u) {
        snprintf(error, error_size,
                 "LTX ANE Video self QKV Stage %d attached %u of 48 blocks",
                 stage, loaded_qkv);
        goto cleanup;
    }
    ane_video_self_qkv_enabled=loaded_qkv!=0;
#endif
    /* Workspace shape depends on the active ANE K/V and QKV partitions, so
     * attach and validate those artifacts before allocating stage buffers. */
    if(!create_block_rope(gpu,&ctx->weights[0],rows,audio_rows,&rope,error,error_size)||!create_workspace(gpu,rows,vd,ctx->weights[0].video_self.heads,audio_rows,ad,text_rows,&workspace,error,error_size))goto cleanup;
    size_t count=0;const float *sigmas=stage==1?ltx_distilled_stage1_sigmas(&count):ltx_distilled_stage2_sigmas(&count);
    ltx_rng video_rng,audio_rng;ltx_rng_seed(&video_rng,seed+(stage==1?10000u:2u),0);ltx_rng_seed(&audio_rng,seed+2u,0);
    ok=run_denoise_schedule(gpu,ctx->weights,0,48,&rope,&workspace,ctx->io,ctx->conditioning,buffers[2],buffers[3],buffers[4],buffers[0],buffers[1],clean,prefix,strength,buffers[5],buffers[6],buffers[7],buffers[8],rows,audio_rows,text_rows,vp,ap,sigmas,count,stage==1,&video_rng,stage==1?&video_rng:&audio_rng,stage==2&&ctx->options.release_blocks_final_step,error,error_size);
    if(ok)ok=ltx_gpu_buffer_read(buffers[0],video,video_elements*2,error,error_size)&&ltx_gpu_buffer_read(buffers[1],audio,audio_elements*2,error,error_size);
cleanup:
    free_workspace(&workspace);free_block_rope(&rope);ltx_gpu_buffer_free(clean);
    for(unsigned i=0;i<9;++i)ltx_gpu_buffer_free(buffers[i]);
#ifdef LTX_ENABLE_ANE_KV
    ane_video_text_kv_enabled=0;
#endif
#ifdef LTX_ENABLE_ANE_QKV
    ane_video_self_qkv_enabled=0;
    ane_video_self_qkv_prefix_rows=0u;
#endif
    ltx_native_detach_ane_stage(ctx, stage, rows);
#ifdef LTX_ENABLE_ANE_MLP
    ane_mlp_fused_residual=0;
    ane_mlp_fused_adaln_pack=0;
#endif
    native_progress=NULL;native_opaque=NULL;av_parallel_audio_gpu=NULL;return ok;
}

int ltx_native_upsample_stage2(
    ltx_native_denoiser *ctx,const char *upsampler_checkpoint,
    const char *video_vae_checkpoint,
    uint16_t *output,size_t output_elements,
    const uint16_t *input,size_t input_elements,
    char *error,size_t error_size) {
    if(!ctx||!upsampler_checkpoint||!video_vae_checkpoint||!output||!input){snprintf(error,error_size,"invalid LTX stage boundary input");return 0;}
    const size_t expected_input=(size_t)ctx->workload.stage1_video_tokens*LTX_VIDEO_CHANNELS;
    const size_t expected_output=(size_t)ctx->workload.stage2_video_tokens*LTX_VIDEO_CHANNELS;
    if(input_elements!=expected_input||output_elements!=expected_output){snprintf(error,error_size,"LTX stage boundary tensor size mismatch");return 0;}
    ltx_latent_stats *stats=ltx_latent_stats_load(ctx->gpu,video_vae_checkpoint,error,error_size);
    ltx_mlx_upsampler *upsampler=NULL;
    ltx_gpu_buffer *stage1=NULL,*stage2=NULL;
    uint16_t *denormalized=NULL;int ok=0;
    if(!stats||ltx_latent_stats_channels(stats)!=LTX_VIDEO_CHANNELS)goto cleanup;
    stage1=ltx_gpu_buffer_new_copy(ctx->gpu,input,input_elements*sizeof(uint16_t),error,error_size);
    stage2=ltx_gpu_buffer_new(ctx->gpu,output_elements*sizeof(uint16_t),error,error_size);
    denormalized=malloc(input_elements*sizeof(uint16_t));
    if(!stage1||!stage2||!denormalized){if(error&&error_size&&!error[0])snprintf(error,error_size,"out of memory at LTX stage boundary");goto cleanup;}
    if(!ltx_latent_denormalize_tokens_bf16(stats,stage1,stage1,(uint32_t)ctx->workload.stage1_video_tokens,error,error_size)||
       !ltx_gpu_buffer_read(stage1,denormalized,input_elements*sizeof(uint16_t),error,error_size))goto cleanup;
    upsampler=ltx_mlx_upsampler_create(upsampler_checkpoint,error,error_size);if(!upsampler)goto cleanup;
    if(!ltx_mlx_upsampler_run_tokens_bf16(upsampler,output,output_elements,denormalized,input_elements,1u,ctx->workload.latent_frames,ctx->workload.stage1_latent_height,ctx->workload.stage1_latent_width,error,error_size)||
       !ltx_gpu_buffer_write(stage2,output,output_elements*sizeof(uint16_t),error,error_size)||
       !ltx_latent_normalize_tokens_bf16(stats,stage2,stage2,(uint32_t)ctx->workload.stage2_video_tokens,error,error_size)||
       !ltx_gpu_buffer_read(stage2,output,output_elements*sizeof(uint16_t),error,error_size))goto cleanup;
    ok=1;
cleanup:
    ltx_mlx_upsampler_free(upsampler);ltx_gpu_buffer_free(stage2);ltx_gpu_buffer_free(stage1);ltx_latent_stats_free(stats);free(denormalized);ltx_mlx_clear_cache();return ok;
}

int ltx_native_connect_conditioning(
    ltx_native_denoiser *ctx,
    uint16_t *video_output,size_t video_output_elements,
    uint16_t *audio_output,size_t audio_output_elements,
    uint16_t *mask_output,size_t mask_output_elements,
    uint32_t output_rows,
    const uint16_t *video_input,size_t video_input_elements,
    const uint16_t *audio_input,size_t audio_input_elements,
    const uint16_t *mask_input,size_t mask_input_elements,
    uint32_t input_rows,
    char *error,size_t error_size) {
    if(!ctx||!video_output||!audio_output||!mask_output||!video_input||
       !audio_input||!mask_input||!input_rows||!output_rows){
        snprintf(error,error_size,"invalid LTX connector input");return 0;
    }
    ltx_st_header header={0};ltx_st_mapping mapping={0};
    ltx_connector *connector=NULL;
    ltx_gpu_buffer *video_in=NULL,*audio_in=NULL,*video_out=NULL,*audio_out=NULL;
    int ok=0;
    if(!ltx_st_read_header(ctx->options.checkpoint,&header,error,error_size)||
       !ltx_st_map_open(&header,&mapping,error,error_size))goto cleanup;
    connector=ltx_connector_load(&header,&mapping,ctx->gpu,
        "model.diffusion_model",error,error_size);
    if(!connector)goto cleanup;
    const uint32_t video_dim=ltx_connector_video_dim(connector);
    const uint32_t audio_dim=ltx_connector_audio_dim(connector);
    const uint32_t expected_rows=ltx_connector_output_rows(connector,input_rows);
    size_t video_input_bytes=0,audio_input_bytes=0;
    size_t video_output_bytes=0,audio_output_bytes=0,mask_output_bytes=0;
    if(expected_rows!=output_rows||mask_input_elements!=input_rows||
       mask_output_elements!=output_rows||
       video_input_elements!=(size_t)input_rows*video_dim||
       audio_input_elements!=(size_t)input_rows*audio_dim||
       video_output_elements!=(size_t)output_rows*video_dim||
       audio_output_elements!=(size_t)output_rows*audio_dim||
       !checked_bytes(video_input_elements,sizeof(uint16_t),&video_input_bytes)||
       !checked_bytes(audio_input_elements,sizeof(uint16_t),&audio_input_bytes)||
       !checked_bytes(video_output_elements,sizeof(uint16_t),&video_output_bytes)||
       !checked_bytes(audio_output_elements,sizeof(uint16_t),&audio_output_bytes)||
       !checked_bytes(mask_output_elements,sizeof(uint16_t),&mask_output_bytes)){
        snprintf(error,error_size,"LTX connector tensor geometry mismatch");
        goto cleanup;
    }
    video_in=ltx_gpu_buffer_new_copy(ctx->gpu,video_input,video_input_bytes,
        error,error_size);
    audio_in=ltx_gpu_buffer_new_copy(ctx->gpu,audio_input,audio_input_bytes,
        error,error_size);
    video_out=ltx_gpu_buffer_new(ctx->gpu,video_output_bytes,error,error_size);
    audio_out=ltx_gpu_buffer_new(ctx->gpu,audio_output_bytes,error,error_size);
    if(!video_in||!audio_in||!video_out||!audio_out||
       !ltx_connector_run_bf16(connector,video_out,audio_out,video_in,audio_in,
            input_rows,error,error_size)||
       !ltx_gpu_buffer_read(video_out,video_output,video_output_bytes,
            error,error_size)||
       !ltx_gpu_buffer_read(audio_out,audio_output,audio_output_bytes,
            error,error_size))goto cleanup;
    memcpy(mask_output,mask_input,input_rows*sizeof(uint16_t));
    memset(mask_output+input_rows,0,mask_output_bytes-input_rows*sizeof(uint16_t));
    ok=1;
cleanup:
    ltx_gpu_buffer_free(audio_out);ltx_gpu_buffer_free(video_out);
    ltx_gpu_buffer_free(audio_in);ltx_gpu_buffer_free(video_in);
    ltx_connector_free(connector);ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return ok;
}
