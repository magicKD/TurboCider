#include "ltx_gemma_encoder.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum {
    MAX_ROWS = 1024,
    VIDEO_DIM = 4096,
    AUDIO_DIM = 2048
};

static int write_exact(const char *directory, const char *name,
                       const void *data, size_t bytes) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= sizeof(path)) return 0;
    FILE *stream = fopen(path, "wb");
    if (!stream) return 0;
    size_t written = fwrite(data, 1u, bytes, stream);
    int close_status = fclose(stream);
    return written == bytes && close_status == 0;
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) +
        (double)(end->tv_nsec - start->tv_nsec) / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 8) {
        fprintf(stderr,
                "usage: ltx-gemma-encode CHECKPOINT TOKENIZER SHADER "
                "PROMPT OUTPUT_DIR [WARM_RUNS] [ANE_MANIFEST]\n");
        return 2;
    }
    uint32_t warm_runs = 0u;
    if (argc >= 7) {
        char *end = NULL;
        unsigned long parsed = strtoul(argv[6], &end, 10);
        if (!end || *end || parsed == 0u || parsed > 50u) {
            fprintf(stderr, "WARM_RUNS must be an integer in 1...50\n");
            return 2;
        }
        warm_runs = (uint32_t)parsed;
    }
    if (mkdir(argv[5], 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create output directory: %s\n", argv[5]);
        return 1;
    }
    size_t video_capacity = (size_t)MAX_ROWS * VIDEO_DIM;
    size_t audio_capacity = (size_t)MAX_ROWS * AUDIO_DIM;
    uint16_t *video = calloc(video_capacity, sizeof(*video));
    uint16_t *audio = calloc(audio_capacity, sizeof(*audio));
    uint16_t *mask = calloc(MAX_ROWS, sizeof(*mask));
    if (!video || !audio || !mask) {
        fprintf(stderr, "out of memory for Gemma outputs\n");
        free(video); free(audio); free(mask);
        return 1;
    }
    char error[2048] = {};
    ltx_gemma_encoder_options options = {
        .checkpoint = argv[1],
        .tokenizer_json = argv[2],
        .shader_source = argv[3],
        .max_tokens = MAX_ROWS,
        .ane_manifest = argc == 8 ? argv[7] : NULL,
    };
    struct timespec start = {}, end = {};
    clock_gettime(CLOCK_MONOTONIC, &start);
    ltx_gemma_encoder *encoder = ltx_gemma_encoder_create(
        &options, error, sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &end);
    double create_seconds = elapsed_seconds(&start, &end);
    clock_gettime(CLOCK_MONOTONIC, &start);
    int prepared = encoder && ltx_gemma_encoder_prepare_prompt(
        encoder, argv[4], error, sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &end);
    double prepare_seconds = elapsed_seconds(&start, &end);
    uint32_t rows = 0;
    double first_seconds = 0.0;
    double warm_seconds[50] = {};
    double total_encode_seconds = 0.0;
    int fused_mlp = getenv("TURBOCIDER_LTX_GEMMA_FUSED_MLP") != NULL;
    int gpu_taps = getenv("TURBOCIDER_LTX_GEMMA_GPU_TAPS") != NULL;
    int ok = prepared;
    for (uint32_t run = 0; ok && run <= warm_runs; run++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        ok = ltx_gemma_encoder_encode(
            encoder, argv[4], video, video_capacity, audio, audio_capacity,
            mask, MAX_ROWS, &rows, NULL, NULL, error, sizeof(error));
        clock_gettime(CLOCK_MONOTONIC, &end);
        double seconds = elapsed_seconds(&start, &end);
        total_encode_seconds += seconds;
        if (run == 0u) first_seconds = seconds;
        else warm_seconds[run - 1u] = seconds;
    }
    ltx_gemma_encoder_telemetry telemetry = {};
    if (encoder) ltx_gemma_encoder_get_telemetry(encoder, &telemetry);
    ltx_gemma_encoder_free(encoder);
    if (!ok) {
        fprintf(stderr, "%s\n", error[0] ? error : "Gemma encode failed");
        free(video); free(audio); free(mask);
        return 1;
    }
    size_t video_elements = (size_t)rows * VIDEO_DIM;
    size_t audio_elements = (size_t)rows * AUDIO_DIM;
    ok = write_exact(argv[5], "raw_video_context.bf16", video,
                     video_elements * sizeof(*video)) &&
        write_exact(argv[5], "raw_audio_context.bf16", audio,
                    audio_elements * sizeof(*audio)) &&
        write_exact(argv[5], "raw_text_mask.bf16", mask,
                    (size_t)rows * sizeof(*mask));
    free(video); free(audio); free(mask);
    if (!ok) {
        fprintf(stderr, "cannot write Gemma output tensors\n");
        return 1;
    }
    printf("{\"format\":\"turbocider-native-gemma-raw-candidate-v1\","
           "\"rows\":%u,\"video_dim\":%u,\"audio_dim\":%u,"
           "\"fused_mlp\":%s,\"gpu_taps\":%s,"
           "\"runs\":%u,\"warm_runs\":%u,"
           "\"create_seconds\":%.9f,\"prepare_seconds\":%.9f,"
           "\"first_seconds\":%.9f,"
           "\"seconds\":%.9f,"
           "\"warm_seconds\":[", rows, VIDEO_DIM, AUDIO_DIM,
           fused_mlp ? "true" : "false", gpu_taps ? "true" : "false",
           warm_runs + 1u, warm_runs, create_seconds, prepare_seconds,
           first_seconds, create_seconds + prepare_seconds +
               total_encode_seconds);
    for (uint32_t run = 1u; run <= warm_runs; run++) {
        if (run > 1u) fputc(',', stdout);
        printf("%.9f", warm_seconds[run - 1u]);
    }
    printf("],\"ane_requested\":%s,\"ane_used\":%s,"
           "\"resident_weights_enabled\":%s,"
           "\"resident_weight_cache_hits\":%u,"
           "\"resident_weight_cache_misses\":%u,"
           "\"resident_weight_bytes\":%llu,"
           "\"ane_layers_available\":%u,\"ane_layers_attempted\":%u,"
           "\"ane_layers_succeeded\":%u,\"ane_layers_fallback\":%u,"
           "\"ane_cache_hits\":%u,\"ane_cache_misses\":%u,"
           "\"ane_preload_models_session_total\":%u,"
           "\"ane_preload_workers\":%u,"
           "\"ane_preload_seconds_session_total\":%.9f,"
           "\"ane_selected_bucket\":%u,\"ane_padding_rows\":%u,"
           "\"ane_minimum_profitable_rows\":%u,"
           "\"ane_execution_rows\":%llu,\"ane_total_seconds\":%.9f,"
           "\"ane_output_backing_used\":%s,\"ane_plan_reason\":\"%s\"}\n",
           telemetry.ane_requested ? "true" : "false",
           telemetry.ane_used ? "true" : "false",
           telemetry.resident_weights_enabled ? "true" : "false",
           telemetry.resident_weight_cache_hits,
           telemetry.resident_weight_cache_misses,
           (unsigned long long)telemetry.resident_weight_bytes,
           telemetry.ane_layers_available, telemetry.ane_layers_attempted,
           telemetry.ane_layers_succeeded, telemetry.ane_layers_fallback,
           telemetry.ane_cache_hits, telemetry.ane_cache_misses,
           telemetry.ane_preload_models_session_total,
           telemetry.ane_preload_workers,
           telemetry.ane_preload_seconds_session_total,
           telemetry.ane_selected_bucket, telemetry.ane_padding_rows,
           telemetry.ane_minimum_profitable_rows,
           (unsigned long long)telemetry.ane_execution_rows,
           telemetry.ane_total_seconds,
           telemetry.ane_output_backing_used ? "true" : "false",
           telemetry.ane_plan_reason);
    return 0;
}
