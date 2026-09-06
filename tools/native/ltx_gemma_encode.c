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
    if (argc != 6) {
        fprintf(stderr,
                "usage: ltx-gemma-encode CHECKPOINT TOKENIZER SHADER "
                "PROMPT OUTPUT_DIR\n");
        return 2;
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
    };
    struct timespec start = {}, end = {};
    clock_gettime(CLOCK_MONOTONIC, &start);
    ltx_gemma_encoder *encoder = ltx_gemma_encoder_create(
        &options, error, sizeof(error));
    uint32_t rows = 0;
    int ok = encoder && ltx_gemma_encoder_encode(
        encoder, argv[4], video, video_capacity, audio, audio_capacity,
        mask, MAX_ROWS, &rows, NULL, NULL, error, sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &end);
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
           "\"seconds\":%.9f}\n",
           rows, VIDEO_DIM, AUDIO_DIM, elapsed_seconds(&start, &end));
    return 0;
}
