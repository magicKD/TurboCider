#include "ltx.h"
#include "ltx_gpu.h"
#include "ltx_weights.h"

#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s --info -d CHECKPOINT\n"
        "       %s --device [--shader PATH]\n"
        "       %s --self-test [--shader PATH]\n"
        "       %s --workload [--width N --height N --frames N --fps N]\n"
        "       %s -d CHECKPOINT --tensor NAME\n"
        "       %s -d CHECKPOINT --list-tensors PREFIX\n\n"
        "The generation pipeline is being implemented phase by phase.\n",
        program, program, program, program, program, program);
}

static uint32_t parse_u32(const char *text, const char *label) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !value || value > UINT32_MAX) {
        fprintf(stderr, "ltx: invalid %s: %s\n", label, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void print_checkpoint(const ltx_checkpoint_info *info) {
    printf("Checkpoint: %s\n", info->path);
    printf("Class: %s\n", info->model_class);
    printf("Tensors: %zu | blocks: %u/%u\n", info->tensor_count,
           info->transformer_blocks_found, info->num_layers);
    printf("Video: H=%u, heads=%u x %u, FF=%u, channels=%u\n",
           info->video_dim, info->video_heads, info->video_head_dim,
           info->video_ff_dim, info->video_channels);
    printf("Audio: H=%u, heads=%u x %u, FF=%u, channels=%u\n",
           info->audio_dim, info->audio_heads, info->audio_head_dim,
           info->audio_ff_dim, info->audio_channels);
    printf("RoPE: %s/%s | keyframe abs pos: %s\n",
           info->split_rope ? "split" : "other",
           info->double_precision_rope ? "float64 frequencies" :
                                         "default frequencies",
           info->keyframe_absolute_positions ? "yes" : "no");
    printf("FF bias: video=%s audio=%s\n",
           info->ff_bias ? "yes" : "no",
           info->audio_ff_bias ? "yes" : "no");
    printf("Weights: %s%s | filename mode: %s\n",
           info->quantized_int8 ? "INT8" : "BF16/FP16",
           info->comfy_quantized ? " (Comfy quant metadata)" : "",
           info->appears_distilled ? "distilled" : "dev/unknown");
    puts("Dtype inventory:");
    for (int dtype = LTX_DTYPE_BOOL; dtype <= LTX_DTYPE_F64; dtype++)
        if (info->dtype_counts[dtype])
            printf("  %-5s %zu\n", ltx_dtype_name((ltx_dtype)dtype),
                   info->dtype_counts[dtype]);
}

static void print_tensor(const ltx_st_tensor *tensor) {
    printf("%s  %s  [", tensor->name, ltx_dtype_name(tensor->dtype));
    for (uint32_t dimension = 0; dimension < tensor->ndim; dimension++) {
        if (dimension) putchar(',');
        printf("%" PRIu64, tensor->shape[dimension]);
    }
    printf("]  bytes=%" PRIu64 "  file_offset=%" PRIu64 "\n",
           tensor->data_end - tensor->data_begin, tensor->file_offset);
}

static int print_device(const char *shader, int self_test) {
    char error[1024];
    ltx_gpu *gpu = ltx_gpu_create(shader, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "ltx: %s\n", error);
        return 1;
    }
    ltx_gpu_info info;
    ltx_gpu_get_info(gpu, &info);
    printf("Metal device: %s\n", info.name);
    printf("Unified memory: %s\n", info.unified_memory ? "yes" : "no");
    printf("Recommended working set: %.2f GiB\n",
           (double)info.recommended_working_set_bytes /
           (1024.0 * 1024.0 * 1024.0));
    printf("Max buffer: %.2f GiB\n",
           (double)info.max_buffer_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("GPU families: Apple7=%d Apple8=%d Apple9=%d\n",
           info.supports_apple7, info.supports_apple8,
           info.supports_apple9);

    int result = 0;
    if (self_test) {
        float left[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        float right[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        float output[8] = {0};
        ltx_gpu_buffer *a = ltx_gpu_buffer_new(gpu, sizeof(left), error,
                                                sizeof(error));
        ltx_gpu_buffer *b = ltx_gpu_buffer_new(gpu, sizeof(right), error,
                                                sizeof(error));
        ltx_gpu_buffer *c = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                                sizeof(error));
        if (!a || !b || !c ||
            !ltx_gpu_buffer_write(a, left, sizeof(left), error, sizeof(error)) ||
            !ltx_gpu_buffer_write(b, right, sizeof(right), error, sizeof(error)) ||
            !ltx_gpu_add_f32(gpu, c, a, b, 8u, error, sizeof(error)) ||
            !ltx_gpu_buffer_read(c, output, sizeof(output), error,
                                 sizeof(error))) {
            fprintf(stderr, "ltx: Metal self-test failed: %s\n", error);
            result = 1;
        } else {
            for (size_t index = 0; index < 8; index++)
                if (output[index] != 8.0f) result = 1;
            puts(result ? "Metal self-test: FAILED" : "Metal self-test: PASS");
        }
        ltx_gpu_buffer_free(a);
        ltx_gpu_buffer_free(b);
        ltx_gpu_buffer_free(c);
    }
    ltx_gpu_free(gpu);
    return result;
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *shader = "ltx_shaders.metal";
    uint32_t width = 704u;
    uint32_t height = 480u;
    uint32_t frames = 97u;
    uint32_t fps = 24u;
    int mode_info = 0;
    int mode_device = 0;
    int mode_self_test = 0;
    int mode_workload = 0;
    const char *tensor_name = NULL;
    const char *tensor_prefix = NULL;

    static const struct option options[] = {
        {"model", required_argument, NULL, 'd'},
        {"info", no_argument, NULL, 1000},
        {"device", no_argument, NULL, 1001},
        {"self-test", no_argument, NULL, 1002},
        {"workload", no_argument, NULL, 1003},
        {"shader", required_argument, NULL, 1004},
        {"width", required_argument, NULL, 1005},
        {"height", required_argument, NULL, 1006},
        {"frames", required_argument, NULL, 1007},
        {"fps", required_argument, NULL, 1008},
        {"tensor", required_argument, NULL, 1009},
        {"list-tensors", required_argument, NULL, 1010},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    for (;;) {
        int option = getopt_long(argc, argv, "d:h", options, NULL);
        if (option < 0) break;
        switch (option) {
            case 'd': checkpoint = optarg; break;
            case 'h': usage(argv[0]); return 0;
            case 1000: mode_info = 1; break;
            case 1001: mode_device = 1; break;
            case 1002: mode_self_test = 1; break;
            case 1003: mode_workload = 1; break;
            case 1004: shader = optarg; break;
            case 1005: width = parse_u32(optarg, "width"); break;
            case 1006: height = parse_u32(optarg, "height"); break;
            case 1007: frames = parse_u32(optarg, "frames"); break;
            case 1008: fps = parse_u32(optarg, "fps"); break;
            case 1009: tensor_name = optarg; break;
            case 1010: tensor_prefix = optarg; break;
            default: usage(argv[0]); return 2;
        }
    }

    if (tensor_name || tensor_prefix) {
        if (!checkpoint) {
            fputs("ltx: --tensor/--list-tensors requires -d CHECKPOINT\n",
                  stderr);
            return 2;
        }
        ltx_st_header header;
        char error[1024];
        if (!ltx_st_read_header(checkpoint, &header, error, sizeof(error))) {
            fprintf(stderr, "ltx: cannot inspect tensors: %s\n", error);
            return 1;
        }
        int found = 0;
        if (tensor_name) {
            const ltx_st_tensor *tensor = ltx_st_find(&header, tensor_name);
            if (tensor) {
                print_tensor(tensor);
                found = 1;
            }
        } else {
            size_t prefix_length = strlen(tensor_prefix);
            for (size_t index = 0; index < header.tensor_count; index++) {
                const ltx_st_tensor *tensor = &header.tensors[index];
                if (!strncmp(tensor->name, tensor_prefix, prefix_length)) {
                    print_tensor(tensor);
                    found = 1;
                }
            }
        }
        ltx_st_free_header(&header);
        if (!found) {
            fprintf(stderr, "ltx: no matching tensor for %s\n",
                    tensor_name ? tensor_name : tensor_prefix);
            return 1;
        }
        return 0;
    }

    if (mode_info) {
        if (!checkpoint) {
            fputs("ltx: --info requires -d CHECKPOINT\n", stderr);
            return 2;
        }
        ltx_checkpoint_info info;
        char error[1024];
        if (!ltx_checkpoint_inspect(checkpoint, &info, error, sizeof(error)) ||
            !ltx_checkpoint_validate_transformer(&info, error, sizeof(error))) {
            fprintf(stderr, "ltx: checkpoint validation failed: %s\n", error);
            return 1;
        }
        print_checkpoint(&info);
        return 0;
    }
    if (mode_device || mode_self_test)
        return print_device(shader, mode_self_test);
    if (mode_workload) {
        ltx_workload workload;
        char error[256];
        if (!ltx_workload_init(&workload, width, height, frames, fps,
                               error, sizeof(error))) {
            fprintf(stderr, "ltx: %s\n", error);
            return 1;
        }
        printf("Requested: %ux%u, %u frames @ %u fps\n",
               width, height, frames, fps);
        printf("Two-stage output: %ux%u, %u frames\n",
               workload.output_width, workload.output_height,
               workload.frames);
        printf("Stage 1: %ux%u, latent %ux%ux%u, tokens=%" PRIu64 "\n",
               workload.stage1_width, workload.stage1_height,
               workload.latent_frames, workload.stage1_latent_height,
               workload.stage1_latent_width,
               workload.stage1_video_tokens);
        printf("Stage 2: %ux%u, latent %ux%ux%u, tokens=%" PRIu64 "\n",
               workload.output_width, workload.output_height,
               workload.latent_frames, workload.stage2_latent_height,
               workload.stage2_latent_width,
               workload.stage2_video_tokens);
        printf("Audio tokens: %u\n", workload.audio_tokens);
        return 0;
    }

    usage(argv[0]);
    return 2;
}
