#include "h3_dit.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static uint64_t fnv1a(const void *data, size_t bytes, uint64_t hash) {
    const uint8_t *cursor = data;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= cursor[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed == 0)
        fprintf(stderr, "%s %d/%d\n", phase, completed, total);
}

static int parse_u64(const char *text, uint64_t *value) {
    if (!text || !*text || !value || *text == '-') return 0;
    errno = 0;
    char *tail = NULL;
    unsigned long long parsed = strtoull(text, &tail, 10);
    if (errno || tail == text || *tail) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s TRANSFORMER_DIR MEMORY_BUDGET_BYTES OUTPUT_F32\n",
                argv[0]);
        return 2;
    }
    uint64_t budget = 0;
    if (!parse_u64(argv[2], &budget)) {
        fprintf(stderr, "invalid memory budget: %s\n", argv[2]);
        return 2;
    }

    enum { text_tokens = 32, text_width = H3_TEXT_HIDDEN_SIZE };
    h3_text_embedding text = {0};
    text.tokens = text_tokens;
    text.width = text_width;
    text.values = calloc((size_t)text_tokens * text_width,
                         sizeof(*text.values));
    if (!text.values) {
        fprintf(stderr, "cannot allocate synthetic text embedding\n");
        return 2;
    }
    for (size_t index = 0; index < (size_t)text_tokens * text_width; index++) {
        int centered = (int)(index % 17) - 8;
        text.values[index] = bf16((float)centered / 1024.0f);
    }

    h3_temporal_shape temporal = h3_temporal(22);
    h3_layout_spec spec = {
        text_tokens, temporal.video_t, 16, 16, temporal.audio_t,
        temporal.frame_count, NULL, 0, NULL, 0
    };
    h3_layout layout = {0};
    h3_sigma_schedule sigmas = {0};
    char error[1024] = {0};
    if (!h3_layout_build(&spec, &layout, error, sizeof(error)) ||
        !h3_serving_schedule_build_shifts(
            4, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT, &sigmas)) {
        fprintf(stderr, "cannot prepare probe geometry: %s\n", error);
        free(text.values);
        h3_layout_free(&layout);
        return 2;
    }

    double total_started = now_seconds();
    double load_started = total_started;
    h3_dit *dit = h3_dit_load_t2va(
        argv[1], "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 1, 0, budget, 1.0f,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        progress, NULL, error, sizeof(error));
    double load_seconds = now_seconds() - load_started;
    free(text.values);
    h3_layout_free(&layout);
    if (!dit) {
        fprintf(stderr, "cannot load H3 DiT: %s\n", error);
        return 2;
    }

    size_t video_elements = h3_dit_video_elements(dit);
    size_t audio_elements = h3_dit_audio_elements(dit);
    float *video = malloc(video_elements * sizeof(*video));
    float *audio = malloc(audio_elements * sizeof(*audio));
    if (!video || !audio) {
        fprintf(stderr, "cannot allocate probe latents\n");
        free(video);
        free(audio);
        h3_dit_free(dit);
        return 2;
    }
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, 42);
    h3_rng_seed(&audio_rng, 42);
    h3_rng_fill_normal(&video_rng, video, video_elements);
    h3_rng_fill_normal(&audio_rng, audio, audio_elements);

    double denoise_started = now_seconds();
    int ok = h3_dit_denoise_euler(
        dit, video, audio, 1, progress, NULL, error, sizeof(error));
    double denoise_seconds = now_seconds() - denoise_started;
    h3_dit_streaming_info streaming = {0};
    h3_gpu_stats gpu = {0};
    (void)h3_dit_get_streaming_info(dit, &streaming);
    (void)h3_dit_get_gpu_stats(dit, &gpu);
    if (!ok) {
        fprintf(stderr, "H3 DiT probe failed: %s\n", error);
        free(video);
        free(audio);
        h3_dit_free(dit);
        return 2;
    }

    int finite = 1;
    for (size_t index = 0; index < video_elements; index++)
        if (!isfinite(video[index])) finite = 0;
    for (size_t index = 0; index < audio_elements; index++)
        if (!isfinite(audio[index])) finite = 0;
    FILE *output = fopen(argv[3], "wb");
    int write_ok = output != NULL;
    if (write_ok &&
        fwrite(video, sizeof(*video), video_elements, output) != video_elements)
        write_ok = 0;
    if (write_ok &&
        fwrite(audio, sizeof(*audio), audio_elements, output) != audio_elements)
        write_ok = 0;
    if (output && fclose(output)) write_ok = 0;
    if (!write_ok) {
        fprintf(stderr, "cannot write probe output: %s\n", strerror(errno));
        free(video);
        free(audio);
        h3_dit_free(dit);
        return 2;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = fnv1a(video, video_elements * sizeof(*video), hash);
    hash = fnv1a(audio, audio_elements * sizeof(*audio), hash);
    double total_seconds = now_seconds() - total_started;

    printf("{\"schema\":\"turbocider-h3-dit-streaming-probe-v1\","
           "\"memory_budget_bytes\":%llu,\"finite\":%s,"
           "\"video_elements\":%zu,\"audio_elements\":%zu,"
           "\"output_fnv1a64\":\"%016llx\","
           "\"load_seconds\":%.9f,\"denoise_seconds\":%.9f,"
           "\"total_seconds\":%.9f,\"pinned_blocks\":%u,"
           "\"streamed_blocks\":%u,\"block_bytes\":%llu,"
           "\"activation_reserve_bytes\":%llu,\"bytes_read\":%llu,"
           "\"read_seconds\":%.9f,\"wait_seconds\":%.9f,"
           "\"gpu_seconds\":%.9f,\"peak_live_bytes\":%llu}\n",
           (unsigned long long)budget, finite ? "true" : "false",
           video_elements, audio_elements, (unsigned long long)hash,
           load_seconds, denoise_seconds, total_seconds,
           streaming.pinned_blocks, streaming.streamed_blocks,
           (unsigned long long)streaming.block_bytes,
           (unsigned long long)streaming.activation_reserve_bytes,
           (unsigned long long)streaming.bytes_read,
           streaming.read_seconds, streaming.wait_seconds, gpu.gpu_seconds,
           (unsigned long long)gpu.peak_live_bytes);

    free(video);
    free(audio);
    h3_dit_free(dit);
    return finite ? 0 : 1;
}
