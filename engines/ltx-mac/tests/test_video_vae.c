#include "ltx_gpu.h"
#include "ltx_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int read_exact(const char *path, void *data, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    int ok = fread(data, 1u, bytes, file) == bytes;
    ok = ok && fgetc(file) == EOF;
    fclose(file);
    return ok;
}

int main(void) {
    uint32_t output_frames = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    if (!ltx_video_vae_output_shape(13u, 14u, 22u,
                                    &output_frames, &output_height,
                                    &output_width) ||
        output_frames != 97u || output_height != 448u ||
        output_width != 704u) {
        fputs("test_video_vae: output shape test failed\n", stderr);
        return 1;
    }

    const char *model_path = getenv("LTX_VIDEO_VAE_MODEL");
    const char *input_path = getenv("LTX_VIDEO_VAE_INPUT");
    const char *output_path = getenv("LTX_VIDEO_VAE_OUTPUT");
    if (!model_path || !input_path || !output_path) {
        puts("test_video_vae: PASS shape; SKIP parity "
             "(set LTX_VIDEO_VAE_MODEL/INPUT/OUTPUT)");
        return 0;
    }

    enum { batch = 1, channels = 128, frames = 2, height = 2, width = 3 };
    const uint32_t decoded_frames = frames * 8u - 7u;
    const uint32_t decoded_height = height * 32u;
    const uint32_t decoded_width = width * 32u;
    const size_t input_count =
        (size_t)batch * frames * height * width * channels;
    const size_t output_count =
        (size_t)batch * 3u * decoded_frames *
        decoded_height * decoded_width;
    uint16_t *input = malloc(input_count * sizeof(*input));
    uint16_t *actual = malloc(output_count * sizeof(*actual));
    float *expected = malloc(output_count * sizeof(*expected));
    if (!input || !actual || !expected) {
        fputs("test_video_vae: host allocation failed\n", stderr);
        free(input);
        free(actual);
        free(expected);
        return 1;
    }
    if (!read_exact(input_path, input, input_count * sizeof(*input)) ||
        !read_exact(output_path, expected,
                    output_count * sizeof(*expected))) {
        fputs("test_video_vae: fixture read failed\n", stderr);
        free(input);
        free(actual);
        free(expected);
        return 1;
    }

    char error[2048] = {0};
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal",
                                  error, sizeof(error));
    ltx_video_vae *vae = gpu ?
        ltx_video_vae_create(gpu, model_path,
                             error, sizeof(error)) : NULL;
    ltx_gpu_buffer *input_buffer = vae ?
        ltx_gpu_buffer_new_copy(gpu, input,
            input_count * sizeof(*input), error, sizeof(error)) : NULL;
    ltx_gpu_buffer *output_buffer = input_buffer ?
        ltx_gpu_buffer_new(gpu, output_count * sizeof(*actual),
                           error, sizeof(error)) : NULL;
    int ok = output_buffer &&
        ltx_video_vae_decode_tokens_bf16(
            vae, output_buffer, input_buffer,
            batch, frames, height, width, error, sizeof(error)) &&
        ltx_gpu_buffer_read(output_buffer, actual,
            output_count * sizeof(*actual), error, sizeof(error));

    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double error_norm = 0.0;
    double max_error = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; ok && index < output_count; index++) {
        double actual_value = bf16_to_f32(actual[index]);
        double expected_value = expected[index];
        double difference = actual_value - expected_value;
        if (!isfinite(actual_value) || !isfinite(expected_value))
            nonfinite++;
        dot += actual_value * expected_value;
        actual_norm += actual_value * actual_value;
        expected_norm += expected_value * expected_value;
        error_norm += difference * difference;
        if (fabs(difference) > max_error) max_error = fabs(difference);
    }
    double cosine = ok && actual_norm > 0.0 && expected_norm > 0.0 ?
        dot / sqrt(actual_norm * expected_norm) : 0.0;
    double rel_l2 = ok && expected_norm > 0.0 ?
        sqrt(error_norm / expected_norm) : INFINITY;
    printf("test_video_vae: cosine=%.9f rel-L2=%.7f "
           "max=%.7g nonfinite=%zu\n",
           cosine, rel_l2, max_error, nonfinite);
    if (ok && (nonfinite || cosine < 0.995 || rel_l2 > 0.10)) {
        snprintf(error, sizeof(error),
                 "video VAE parity failed: cosine %.9f rel-L2 %.7f",
                 cosine, rel_l2);
        ok = 0;
    }
    if (!ok) fprintf(stderr, "test_video_vae: FAIL: %s\n", error);
    else puts("test_video_vae: PASS");

    ltx_gpu_buffer_free(output_buffer);
    ltx_gpu_buffer_free(input_buffer);
    ltx_video_vae_free(vae);
    ltx_gpu_free(gpu);
    free(input);
    free(actual);
    free(expected);
    return ok ? 0 : 1;
}
