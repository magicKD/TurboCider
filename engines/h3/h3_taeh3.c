#include "h3_taeh3.h"

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TAE_CHANNELS = 24,
    STAGE0_CHANNELS = 256,
    STAGE1_CHANNELS = 128,
    STAGE2_CHANNELS = 64,
    MEMBLOCKS = 3
};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} tae_conv;

typedef struct {
    tae_conv conv[3];
} tae_memblock;

struct h3_taeh3_decoder {
    h3_gpu *gpu;
    h3_weight_store *weights;
    tae_conv input;
    tae_memblock stage0[MEMBLOCKS];
    tae_conv tgrow0;
    tae_conv transition0;
    tae_memblock stage1[MEMBLOCKS];
    tae_conv tgrow1;
    tae_conv transition1;
    tae_memblock stage2[MEMBLOCKS];
    tae_conv tgrow2;
    tae_conv transition2;
    tae_conv output;
    int latent_height;
    int latent_width;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0");
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void free_conv(tae_conv *conv) {
    free_tensor(&conv->weight);
    free_tensor(&conv->bias);
}

static void free_memblock(tae_memblock *block) {
    for (int index = 0; index < 3; index++) free_conv(&block->conv[index]);
}

static int load_conv(h3_taeh3_decoder *decoder, tae_conv *conv,
                     const char *prefix, uint64_t output_channels,
                     uint64_t input_channels, uint64_t kernel, int has_bias,
                     char *error, size_t error_size) {
    char name[128];
    uint64_t weight_shape[] = {
        output_channels, input_channels, kernel, kernel
    };
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = h3_weight_load_bf16(
        decoder->weights, decoder->gpu, name, 4, weight_shape,
        error, error_size);
    if (!conv->weight) return 0;
    if (has_bias) {
        uint64_t bias_shape[] = {output_channels};
        snprintf(name, sizeof(name), "%s.bias", prefix);
        conv->bias = h3_weight_load_bf16(
            decoder->weights, decoder->gpu, name, 1, bias_shape,
            error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_memblock(h3_taeh3_decoder *decoder, tae_memblock *block,
                         int layer, uint64_t channels,
                         char *error, size_t error_size) {
    static const int conv_indices[] = {0, 2, 4};
    for (int index = 0; index < 3; index++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.%d.conv.%d", layer,
                 conv_indices[index]);
        uint64_t input_channels = index ? channels : channels * 2;
        if (!load_conv(decoder, &block->conv[index], prefix, channels,
                       input_channels, 3, 1, error, error_size)) return 0;
    }
    return 1;
}

static int gpu_ok(h3_taeh3_decoder *decoder, int result,
                  const char *operation, char *error, size_t error_size) {
    if (result) return 1;
    fail(error, error_size, "%s: %s", operation,
         h3_gpu_error(decoder->gpu));
    return 0;
}

static int submit(h3_taeh3_decoder *decoder, const char *label,
                  char *error, size_t error_size) {
    return gpu_ok(decoder, h3_gpu_submit(decoder->gpu), label,
                  error, error_size);
}

static h3_gpu_tensor *new_bf16(h3_taeh3_decoder *decoder, size_t elements,
                               const char *label, char *error,
                               size_t error_size) {
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(decoder->gpu, elements);
    if (!result) fail(error, error_size, "cannot allocate %s: %s", label,
                      h3_gpu_error(decoder->gpu));
    return result;
}

static int conv2d(h3_taeh3_decoder *decoder, h3_gpu_tensor *output,
                  const h3_gpu_tensor *input, const tae_conv *conv,
                  uint32_t time, uint32_t height, uint32_t width,
                  uint32_t input_channels, uint32_t output_channels,
                  uint32_t kernel, int relu, const char *label,
                  char *error, size_t error_size) {
    return gpu_ok(decoder, h3_gpu_conv2d_bf16(
        decoder->gpu, output, input, conv->weight, conv->bias,
        time, height, width, input_channels, output_channels, kernel,
        kernel / 2, relu), label, error, error_size);
}

static int run_mem_stage(h3_taeh3_decoder *decoder,
                         const tae_memblock blocks[MEMBLOCKS],
                         h3_gpu_tensor **activation, uint32_t time,
                         uint32_t height, uint32_t width, uint32_t channels,
                         char *error, size_t error_size) {
    size_t elements = (size_t)time * height * width * channels;
    h3_gpu_tensor *branch = new_bf16(
        decoder, elements, "TAEH3 branch", error, error_size);
    h3_gpu_tensor *scratch = new_bf16(
        decoder, elements, "TAEH3 scratch", error, error_size);
    h3_gpu_tensor *concat = new_bf16(
        decoder, elements * 2, "TAEH3 temporal concat", error, error_size);
    if (!branch || !scratch || !concat) {
        free_tensor(&branch); free_tensor(&scratch); free_tensor(&concat);
        return 0;
    }
    double started = now_seconds();
    int ok = gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                    "begin TAEH3 memory stage", error, error_size);
    h3_gpu_tensor *current = *activation;
    for (int index = 0; index < MEMBLOCKS && ok; index++) {
        ok = gpu_ok(decoder, h3_gpu_taeh3_temporal_concat_bf16(
            decoder->gpu, concat, current, time, height, width, channels),
            "TAEH3 temporal memory concat", error, error_size) &&
            conv2d(decoder, branch, concat, &blocks[index].conv[0],
                   time, height, width, channels * 2, channels, 3, 1,
                   "TAEH3 memory conv 0", error, error_size) &&
            conv2d(decoder, scratch, branch, &blocks[index].conv[1],
                   time, height, width, channels, channels, 3, 1,
                   "TAEH3 memory conv 1", error, error_size) &&
            conv2d(decoder, branch, scratch, &blocks[index].conv[2],
                   time, height, width, channels, channels, 3, 0,
                   "TAEH3 memory conv 2", error, error_size) &&
            gpu_ok(decoder, h3_gpu_taeh3_add_relu_bf16(
                decoder->gpu, scratch, current, branch, (uint32_t)elements),
                "TAEH3 memory residual", error, error_size);
        if (ok) {
            h3_gpu_tensor *old = current;
            current = scratch;
            scratch = old;
        }
    }
    if (ok) ok = submit(decoder, "submit TAEH3 memory stage",
                         error, error_size);
    if (ok) {
        *activation = current;
        if (profile_enabled())
            fprintf(stderr,
                    "h3 profile: TAEH3 mem T%u %ux%u C%u wall=%.3fs\n",
                    time, width, height, channels, now_seconds() - started);
    }
    if (branch != *activation) free_tensor(&branch);
    if (scratch != *activation) free_tensor(&scratch);
    if (current != *activation) free_tensor(&current);
    free_tensor(&concat);
    return ok;
}

static int run_transition(h3_taeh3_decoder *decoder,
                          h3_gpu_tensor **activation,
                          const tae_conv *tgrow,
                          const tae_conv *transition,
                          uint32_t *time, uint32_t *height, uint32_t width,
                          uint32_t input_channels,
                          uint32_t output_channels, uint32_t stride,
                          char *error, size_t error_size) {
    size_t upsample_elements =
        (size_t)(*time) * (*height * 2) * (width * 2) * input_channels;
    size_t grown_elements = upsample_elements * stride;
    h3_gpu_tensor *upsampled = new_bf16(
        decoder, upsample_elements, "TAEH3 spatial upsample", error,
        error_size);
    h3_gpu_tensor *grown = new_bf16(
        decoder, grown_elements, "TAEH3 temporal projection", error,
        error_size);
    if (!upsampled || !grown) {
        free_tensor(&upsampled); free_tensor(&grown);
        return 0;
    }
    int ok = gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                    "begin TAEH3 upsample/TGrow", error, error_size) &&
        gpu_ok(decoder, h3_gpu_taeh3_upsample2_bf16(
            decoder->gpu, upsampled, *activation, *time, *height, width,
            input_channels), "TAEH3 spatial upsample", error, error_size) &&
        conv2d(decoder, grown, upsampled, tgrow,
               *time, *height * 2, width * 2, input_channels,
               input_channels * stride, 1, 0, "TAEH3 temporal projection",
               error, error_size) &&
        submit(decoder, "submit TAEH3 upsample/TGrow", error, error_size);
    free_tensor(activation);
    free_tensor(&upsampled);
    if (!ok) {
        free_tensor(&grown);
        return 0;
    }
    h3_gpu_tensor *ordered = grown;
    if (stride == 2) {
        ordered = new_bf16(decoder, grown_elements,
                           "TAEH3 temporal reorder", error, error_size);
        if (!ordered ||
            !gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                    "begin TAEH3 temporal reorder", error, error_size) ||
            !gpu_ok(decoder, h3_gpu_taeh3_tgrow_bf16(
                decoder->gpu, ordered, grown, *time, *height * 2, width * 2,
                input_channels, stride), "TAEH3 temporal reorder",
                error, error_size) ||
            !submit(decoder, "submit TAEH3 temporal reorder",
                    error, error_size)) {
            free_tensor(&ordered);
            free_tensor(&grown);
            return 0;
        }
        free_tensor(&grown);
    }
    *time *= stride;
    *height *= 2;
    size_t next_elements =
        (size_t)(*time) * (*height) * (width * 2) * output_channels;
    h3_gpu_tensor *next = new_bf16(
        decoder, next_elements, "TAEH3 transition output", error, error_size);
    if (!next ||
        !gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                "begin TAEH3 transition", error, error_size) ||
        !conv2d(decoder, next, ordered, transition,
                *time, *height, width * 2, input_channels, output_channels,
                3, 0, "TAEH3 transition conv", error, error_size) ||
        !submit(decoder, "submit TAEH3 transition", error, error_size)) {
        free_tensor(&next);
        free_tensor(&ordered);
        return 0;
    }
    free_tensor(&ordered);
    *activation = next;
    return 1;
}

h3_taeh3_decoder *h3_taeh3_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > 256 || latent_width > 256) {
        fail(error, error_size, "invalid TAEH3 decoder arguments");
        return NULL;
    }
    h3_taeh3_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating TAEH3 decoder");
        return NULL;
    }
    decoder->latent_height = latent_height;
    decoder->latent_width = latent_width;
    decoder->weights = h3_weight_store_open(weight_directory,
                                            error, error_size);
    if (decoder->weights)
        decoder->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (decoder->gpu)
        h3_gpu_profile_set_label(decoder->gpu, "TAEH3 decoder");
    int ok = decoder->weights && decoder->gpu &&
        load_conv(decoder, &decoder->input, "decoder.1", 256, 24, 3, 1,
                  error, error_size);
    static const int stage0_layers[] = {3, 4, 5};
    static const int stage1_layers[] = {9, 10, 11};
    static const int stage2_layers[] = {15, 16, 17};
    for (int index = 0; index < MEMBLOCKS && ok; index++)
        ok = load_memblock(decoder, &decoder->stage0[index],
                           stage0_layers[index], 256, error, error_size);
    if (ok) ok = load_conv(decoder, &decoder->tgrow0, "decoder.7.conv",
                           256, 256, 1, 0, error, error_size) &&
                 load_conv(decoder, &decoder->transition0, "decoder.8",
                           128, 256, 3, 0, error, error_size);
    for (int index = 0; index < MEMBLOCKS && ok; index++)
        ok = load_memblock(decoder, &decoder->stage1[index],
                           stage1_layers[index], 128, error, error_size);
    if (ok) ok = load_conv(decoder, &decoder->tgrow1, "decoder.13.conv",
                           256, 128, 1, 0, error, error_size) &&
                 load_conv(decoder, &decoder->transition1, "decoder.14",
                           64, 128, 3, 0, error, error_size);
    for (int index = 0; index < MEMBLOCKS && ok; index++)
        ok = load_memblock(decoder, &decoder->stage2[index],
                           stage2_layers[index], 64, error, error_size);
    if (ok) ok = load_conv(decoder, &decoder->tgrow2, "decoder.19.conv",
                           128, 64, 1, 0, error, error_size) &&
                 load_conv(decoder, &decoder->transition2, "decoder.20",
                           64, 64, 3, 0, error, error_size) &&
                 load_conv(decoder, &decoder->output, "decoder.22",
                           12, 64, 3, 1, error, error_size);
    if (!ok) {
        h3_taeh3_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

int h3_taeh3_decoder_decode(h3_taeh3_decoder *decoder,
                            const float *normalized_latent,
                            int latent_time, h3_video_frames *output,
                            char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || latent_time < 1 ||
        (latent_time - 2) % 5) {
        fail(error, error_size, "invalid TAEH3 decode arguments");
        return 0;
    }
    uint32_t time = (uint32_t)latent_time;
    uint32_t height = (uint32_t)decoder->latent_height;
    uint32_t width = (uint32_t)decoder->latent_width;
    size_t latent_elements =
        (size_t)time * height * width * TAE_CHANNELS;
    h3_gpu_tensor *input_f32 = h3_gpu_tensor_from_f32(
        decoder->gpu, normalized_latent, latent_elements);
    h3_gpu_tensor *input_bf16 = new_bf16(
        decoder, latent_elements, "TAEH3 normalized input", error, error_size);
    size_t stage0_elements =
        (size_t)time * height * width * STAGE0_CHANNELS;
    h3_gpu_tensor *activation = new_bf16(
        decoder, stage0_elements, "TAEH3 stage 0", error, error_size);
    if (!input_f32 || !input_bf16 || !activation) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate TAEH3 input: %s",
                 h3_gpu_error(decoder->gpu));
        free_tensor(&input_f32); free_tensor(&input_bf16);
        free_tensor(&activation);
        return 0;
    }
    double started = now_seconds();
    int ok = gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                    "begin TAEH3 input", error, error_size) &&
        gpu_ok(decoder, h3_gpu_taeh3_input_bf16(
            decoder->gpu, input_bf16, input_f32, time, height, width,
            TAE_CHANNELS), "TAEH3 input clamp/layout", error, error_size) &&
        conv2d(decoder, activation, input_bf16, &decoder->input,
               time, height, width, TAE_CHANNELS, STAGE0_CHANNELS, 3, 1,
               "TAEH3 input conv", error, error_size) &&
        submit(decoder, "submit TAEH3 input", error, error_size);
    free_tensor(&input_f32);
    free_tensor(&input_bf16);
    if (ok) ok = run_mem_stage(decoder, decoder->stage0, &activation,
                               time, height, width, STAGE0_CHANNELS,
                               error, error_size);
    if (ok) ok = run_transition(decoder, &activation,
                                &decoder->tgrow0, &decoder->transition0,
                                &time, &height, width, STAGE0_CHANNELS,
                                STAGE1_CHANNELS, 1, error, error_size);
    width *= 2;
    if (ok) ok = run_mem_stage(decoder, decoder->stage1, &activation,
                               time, height, width, STAGE1_CHANNELS,
                               error, error_size);
    if (ok) ok = run_transition(decoder, &activation,
                                &decoder->tgrow1, &decoder->transition1,
                                &time, &height, width, STAGE1_CHANNELS,
                                STAGE2_CHANNELS, 2, error, error_size);
    width *= 2;
    if (ok) ok = run_mem_stage(decoder, decoder->stage2, &activation,
                               time, height, width, STAGE2_CHANNELS,
                               error, error_size);

    size_t upsample_elements =
        (size_t)time * height * 2 * width * 2 * STAGE2_CHANNELS;
    size_t grown_elements = upsample_elements * 2;
    h3_gpu_tensor *upsampled = NULL;
    h3_gpu_tensor *grown = NULL;
    h3_gpu_tensor *ordered = NULL;
    h3_gpu_tensor *transitioned = NULL;
    h3_gpu_tensor *patches = NULL;
    h3_gpu_tensor *rgb = NULL;
    if (ok) {
        upsampled = new_bf16(decoder, upsample_elements,
                             "TAEH3 final upsample", error, error_size);
        grown = new_bf16(decoder, grown_elements,
                         "TAEH3 final TGrow", error, error_size);
        ok = upsampled && grown &&
            gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                   "begin TAEH3 final upsample", error, error_size) &&
            gpu_ok(decoder, h3_gpu_taeh3_upsample2_bf16(
                decoder->gpu, upsampled, activation, time, height, width,
                STAGE2_CHANNELS), "TAEH3 final spatial upsample",
                error, error_size) &&
            conv2d(decoder, grown, upsampled, &decoder->tgrow2,
                   time, height * 2, width * 2, STAGE2_CHANNELS,
                   STAGE2_CHANNELS * 2, 1, 0,
                   "TAEH3 final temporal projection", error, error_size) &&
            submit(decoder, "submit TAEH3 final upsample",
                   error, error_size);
    }
    free_tensor(&activation);
    free_tensor(&upsampled);
    if (ok) {
        ordered = new_bf16(decoder, grown_elements,
                           "TAEH3 final temporal reorder", error, error_size);
        ok = ordered &&
            gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                   "begin TAEH3 final reorder", error, error_size) &&
            gpu_ok(decoder, h3_gpu_taeh3_tgrow_bf16(
                decoder->gpu, ordered, grown, time, height * 2, width * 2,
                STAGE2_CHANNELS, 2), "TAEH3 final temporal reorder",
                error, error_size) &&
            submit(decoder, "submit TAEH3 final reorder", error, error_size);
    }
    free_tensor(&grown);
    time *= 2;
    height *= 2;
    width *= 2;
    size_t transition_elements =
        (size_t)time * height * width * STAGE2_CHANNELS;
    size_t patch_elements = (size_t)time * height * width * 12;
    if (ok) {
        transitioned = new_bf16(decoder, transition_elements,
                                 "TAEH3 final features", error, error_size);
        patches = new_bf16(decoder, patch_elements,
                           "TAEH3 RGB patches", error, error_size);
        ok = transitioned && patches &&
            gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                   "begin TAEH3 output conv", error, error_size) &&
            conv2d(decoder, transitioned, ordered, &decoder->transition2,
                   time, height, width, STAGE2_CHANNELS, STAGE2_CHANNELS,
                   3, 1, "TAEH3 final feature conv", error, error_size) &&
            conv2d(decoder, patches, transitioned, &decoder->output,
                   time, height, width, STAGE2_CHANNELS, 12, 3, 0,
                   "TAEH3 RGB conv", error, error_size) &&
            submit(decoder, "submit TAEH3 output conv", error, error_size);
    }
    free_tensor(&ordered);
    free_tensor(&transitioned);
    uint32_t frame_count = ((time + 19) / 20) * 17 - 12;
    uint32_t pixel_height = height * 2;
    uint32_t pixel_width = width * 2;
    size_t rgb_elements =
        (size_t)frame_count * pixel_height * pixel_width * 3;
    if (ok) {
        rgb = h3_gpu_tensor_new_f32(decoder->gpu, rgb_elements);
        if (!rgb) {
            fail(error, error_size, "cannot allocate TAEH3 RGB output: %s",
                 h3_gpu_error(decoder->gpu));
            ok = 0;
        }
    }
    if (ok) ok = gpu_ok(decoder, h3_gpu_begin(decoder->gpu),
                        "begin TAEH3 output layout", error, error_size) &&
        gpu_ok(decoder, h3_gpu_taeh3_output_f32(
            decoder->gpu, rgb, patches, time, height, width, frame_count),
            "TAEH3 pixel shuffle/trim", error, error_size) &&
        submit(decoder, "submit TAEH3 output layout", error, error_size);
    free_tensor(&patches);
    float *host_rgb = NULL;
    if (ok) {
        host_rgb = malloc(rgb_elements * sizeof(*host_rgb));
        if (!host_rgb || !h3_gpu_tensor_read_f32(rgb, host_rgb, rgb_elements)) {
            free(host_rgb);
            host_rgb = NULL;
            fail(error, error_size, "cannot read TAEH3 RGB output");
            ok = 0;
        }
    }
    free_tensor(&rgb);
    if (!ok) {
        free(host_rgb);
        return 0;
    }
    output->frames = (int)frame_count;
    output->height = (int)pixel_height;
    output->width = (int)pixel_width;
    output->rgb = host_rgb;
    if (profile_enabled())
        fprintf(stderr,
                "h3 profile: TAEH3 decode %dx%dx%d wall=%.3fs\n",
                output->width, output->height, output->frames,
                now_seconds() - started);
    return 1;
}

int h3_taeh3_decoder_get_gpu_stats(const h3_taeh3_decoder *decoder,
                                   h3_gpu_stats *stats) {
    return decoder && h3_gpu_get_stats(decoder->gpu, stats);
}

void h3_taeh3_decoder_free(h3_taeh3_decoder *decoder) {
    if (!decoder) return;
    free_conv(&decoder->input);
    for (int index = 0; index < MEMBLOCKS; index++)
        free_memblock(&decoder->stage0[index]);
    free_conv(&decoder->tgrow0); free_conv(&decoder->transition0);
    for (int index = 0; index < MEMBLOCKS; index++)
        free_memblock(&decoder->stage1[index]);
    free_conv(&decoder->tgrow1); free_conv(&decoder->transition1);
    for (int index = 0; index < MEMBLOCKS; index++)
        free_memblock(&decoder->stage2[index]);
    free_conv(&decoder->tgrow2); free_conv(&decoder->transition2);
    free_conv(&decoder->output);
    h3_gpu_free(decoder->gpu);
    h3_weight_store_free(decoder->weights);
    free(decoder);
}
