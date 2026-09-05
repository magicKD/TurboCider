#include "ltx_latent_stats.h"

#include "ltx_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ltx_latent_stats {
    ltx_gpu *gpu;
    ltx_gpu_buffer *mean;
    ltx_gpu_buffer *standard_deviation;
    uint32_t channels;
};

static const ltx_st_tensor *find_first(
        const ltx_st_header *header,
        const char *const *names, size_t count) {
    for (size_t index = 0; index < count; index++) {
        const ltx_st_tensor *tensor = ltx_st_find(header, names[index]);
        if (tensor) return tensor;
    }
    return NULL;
}

ltx_latent_stats *ltx_latent_stats_load(ltx_gpu *gpu,
                                        const char *video_vae_path,
                                        char *error, size_t error_size) {
    static const char *const mean_names[] = {
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics._mean_of_means",
        "vae_encoder.per_channel_statistics._mean_of_means",
    };
    static const char *const std_names[] = {
        "per_channel_statistics.std-of-means",
        "per_channel_statistics._std_of_means",
        "vae_encoder.per_channel_statistics._std_of_means",
    };
    if (error && error_size) error[0] = '\0';
    if (!gpu || !video_vae_path || !video_vae_path[0]) {
        if (error && error_size)
            snprintf(error, error_size,
                     "missing GPU context/video VAE path");
        return NULL;
    }

    ltx_st_header header;
    ltx_st_mapping mapping;
    memset(&header, 0, sizeof(header));
    memset(&mapping, 0, sizeof(mapping));
    if (!ltx_st_read_header(video_vae_path, &header, error, error_size))
        return NULL;
    const ltx_st_tensor *mean = find_first(
        &header, mean_names, sizeof(mean_names) / sizeof(mean_names[0]));
    const ltx_st_tensor *std = find_first(
        &header, std_names, sizeof(std_names) / sizeof(std_names[0]));
    if (!mean || !std || mean->dtype != LTX_DTYPE_BF16 ||
        std->dtype != LTX_DTYPE_BF16 ||
        mean->ndim != 1u || std->ndim != 1u ||
        !mean->shape[0] || mean->shape[0] > UINT32_MAX ||
        std->shape[0] != mean->shape[0]) {
        if (error && error_size)
            snprintf(error, error_size,
                     "video VAE has invalid/missing latent statistics");
        ltx_st_free_header(&header);
        return NULL;
    }
    if (!ltx_st_map_open(&header, &mapping, error, error_size)) {
        ltx_st_free_header(&header);
        return NULL;
    }
    size_t mean_bytes = 0;
    size_t std_bytes = 0;
    const void *mean_data = ltx_st_map_tensor(
        &mapping, mean, &mean_bytes, error, error_size);
    const void *std_data = ltx_st_map_tensor(
        &mapping, std, &std_bytes, error, error_size);
    ltx_latent_stats *stats = NULL;
    if (mean_data && std_data) {
        stats = calloc(1, sizeof(*stats));
        if (!stats && error && error_size)
            snprintf(error, error_size,
                     "out of memory creating latent statistics");
    }
    if (stats) {
        stats->gpu = gpu;
        stats->channels = (uint32_t)mean->shape[0];
        stats->mean = ltx_gpu_buffer_new_copy(
            gpu, mean_data, mean_bytes, error, error_size);
        stats->standard_deviation = ltx_gpu_buffer_new_copy(
            gpu, std_data, std_bytes, error, error_size);
        if (!stats->mean || !stats->standard_deviation) {
            ltx_latent_stats_free(stats);
            stats = NULL;
        }
    }
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return stats;
}

void ltx_latent_stats_free(ltx_latent_stats *stats) {
    if (!stats) return;
    ltx_gpu_buffer_free(stats->mean);
    ltx_gpu_buffer_free(stats->standard_deviation);
    free(stats);
}

uint32_t ltx_latent_stats_channels(const ltx_latent_stats *stats) {
    return stats ? stats->channels : 0u;
}

static int apply_stats(const ltx_latent_stats *stats,
                       ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       uint32_t rows, int normalize,
                       char *error, size_t error_size) {
    if (!stats)
        return 0;
    return ltx_gpu_latent_stats_bf16(
        stats->gpu, output, input, stats->mean,
        stats->standard_deviation, rows, stats->channels, normalize,
        error, error_size);
}

int ltx_latent_denormalize_tokens_bf16(
        const ltx_latent_stats *stats,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, char *error, size_t error_size) {
    return apply_stats(
        stats, output, input, rows, 0, error, error_size);
}

int ltx_latent_normalize_tokens_bf16(
        const ltx_latent_stats *stats,
        ltx_gpu_buffer *output, const ltx_gpu_buffer *input,
        uint32_t rows, char *error, size_t error_size) {
    return apply_stats(
        stats, output, input, rows, 1, error, error_size);
}
