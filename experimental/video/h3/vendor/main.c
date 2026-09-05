#include "h3_runtime_config.h"
#include "h3.h"
#include "h3_cli.h"
#include "h3_host.h"
#include "h3_super.h"
#include "h3_terminal.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR [options]              # interactive\n"
        "       %s -d MODEL_DIR -p PROMPT [-o OUTPUT] [options]\n"
        "       %s -d MODEL_DIR --info\n\n"
        "Options:\n"
        "  -d, --model-dir PATH   MiniMax-H3 local directory\n"
        "  -p, --prompt TEXT      Raw H3 prompt\n"
        "  -o, --output PATH      Output MP4 (default: outputs/h3.mp4)\n"
        "      --width N          Output width (default: 864)\n"
        "      --height N         Output height (default: 480)\n"
        "      --render-width N   Lower internal model width (optional)\n"
        "      --render-height N  Lower internal model height (optional)\n"
        "      --frames N         Requested frames (default: 56)\n"
        "      --seconds N        Requested duration at 24 fps (instead of --frames)\n"
        "      --steps N          Denoising passes (default: 20)\n"
        "      --video-flow-shift X  Video sigma time shift (default: 12)\n"
        "      --audio-flow-shift X  Audio sigma time shift (default: 3)\n"
        "      --reuse N          Denoiser reuse: 1 close, 2 fast, 3 aggressive\n"
        "      --layers N         DiT blocks: 50 exact, 45 fast, 40 aggressive\n"
        "      --core-reuse N     Core refresh: 1 exact, 4 fast, 6 aggressive\n"
        "      --token-reduction  Pair video tokens in middle DiT blocks\n"
        "      --ssd-streaming    Stream original BF16 DiT layers from SSD\n"
        "      --use-int8-row-fc2 Faster one-scale int8 FC2 (M5)\n"
        "      --use-reference-rope  Disable native 256 RoPE adaptation\n"
        "      --use-slower-bf16-mlp  Force close-reference BF16/MPS MLP\n"
        "      --use-slower-bf16-qkv  Force close-reference BF16 QKV\n"
        "      --use-slower-bf16-attention-output  Force BF16 attention output\n"
        "      --use-slower-row-major-attention-output  Restore SDPA transpose\n"
        "      --use-slower-unfused-int8-inputs  Keep standalone quantizers\n"
        "      --use-slower-unfused-qkv-rope  Keep separate Q/K norm/RoPE\n"
        "      --use-slower-scalar-qkv-rms  Force scalar Q/K RMS loads\n"
        "      --use-slower-uncached-int8-scales  Reread projection scales\n"
        "      --use-slower-dynamic-fc1-k  Use runtime-bound FC1 K loop\n"
        "      --use-slower-grouped-quantizer  Force 256-thread FC2 quantizer\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame conditioning image; Super reuses the exact asset\n"
        "      --last-frame PATH  Last-frame conditioning image\n"
        "      --ref-image PATH    Append an ordered Ref2VA image\n"
        "      --ref-image-size S  Image sizing: match (default) or max\n"
        "      --ref-video PATH    Append video, including embedded audio\n"
        "      --ref-silent-video PATH  Append video without its audio\n"
        "      --ref-video-audio VIDEO AUDIO  Append video + soundtrack\n"
        "      --ref-audio PATH    Append an ordered standalone audio clip\n"
        "      --frames-dir PATH  Write generated frames as PPM files\n"
        "      --show             Display a frame after every denoising step (M5)\n"
        "      --zoom N           Terminal image zoom (default: 2 for Retina)\n"
        "      --profile          Print per-phase Metal timing and allocation data\n",
        program, program, program);
    fprintf(stderr,
        "      --super            H3 Turbo Stage 1 -> local LTX-2.5 refiner\n"
        "      --super-profile P  480p, 480p-fast, 480p-quality, "
        "480p-motion, or v2\n"
        "      --super-stage1-steps N  4 (Sana), 6, or 8 H3 passes\n"
        "      --super-decoder D  taehv (default for 480p) or official\n"
        "      --super-endpoint URL  ComfyUI API (default: http://127.0.0.1:8188)\n"
        "      --super-input-dir DIR ComfyUI input directory\n"
        "      --super-output-dir DIR  ComfyUI output directory\n"
        "      --super-transformer FILE LTX transformer filename\n"
        "      --super-refiner-lora FILE  Official LTX distilled/refiner LoRA\n"
        "      --super-refiner-lora-strength X  LoRA strength (Sana: 0.8)\n"
        "      --super-text-encoder FILE LTX text encoder filename\n"
        "      --super-video-vae FILE  LTX Video VAE filename\n"
        "      --super-audio-vae FILE  LTX Audio VAE filename\n"
        "      --super-upscaler FILE   LTX x2 latent upscaler filename\n"
        "      --super-taehv FILE      TAEHV wide checkpoint filename\n"
        "      --super-taehv-dtype D   bfloat16 (default) or float16\n"
        "      --super-taehv-mode M    auto (default), parallel, or sequential\n"
        "      --super-taehv-evict     Move TAEHV back to CPU after decode\n"
        "      --super-prefetch-transformer  Overlap LTX load with H3 denoise\n"
        "      --super-prefetch-conditioning  Overlap Gemma encode with H3 denoise\n"
        "      --super-prefetch-input-models  Fold input VAE/upscaler load into conditioning prefetch\n"
        "      --super-prefetch-start-step N  Start after H3 step N (diagnostic)\n"
        "      --super-prefetch-start-block N Start during final H3 pass after block N (1-50)\n"
        "      --super-final-evict-blocks N   Release final-pass H3 weights in groups of N\n"
        "      --super-stage2-schedule S  default (3), skip-0p9 (2), or tail-0p42 (1 update)\n"
        "      --super-sol-attn        Use Sana Metal Sol-Attn in LTX layers 1-47\n"
        "      --super-dense-attn      Force original dense LTX self-attention\n"
        "      --super-fuse-denoise-decode  Fuse LTX denoise + TAEHV decode (experimental)\n"
        "      --super-bf16-handoff    Pre-resize/BF16 direct handoff (experimental)\n"
        "      --super-mp4-handoff     Diagnostic H.264/AAC handoff fallback\n"
        "      --super-negative TEXT   LTX negative prompt\n"
        "      --super-taeh3 DIR       Native Stage-1 TAEH3 weight directory\n"
        "      --super-timeout SEC     Stage-2 timeout (default: 3600)\n"
        "      --super-telemetry PATH  Write per-request JSON telemetry\n"
        "      --super-keep-stage1     Keep the intermediate handoff artifact\n"
        "      --super-face-quality-gate  Reject unstable Stage-1 faces before LTX\n"
        "      --super-person-quality-gate  Reject unstable Stage-1 face/body pose before LTX\n"
        "      --super-person-quality-preset  Lock person-strict-v1 quality recipe (recommended)\n"
        "      --super-output-quality-gate  Check refined person MP4 (also works with replay)\n"
        "      --super-min-face-area X  Require sampled face area fraction (for example 0.006)\n"
        "      --super-quality-attempts N  Try up to N consecutive seeds through final gate (1-16)\n"
        "      --super-allow-swap      Explicitly annotate that swap is accepted\n"
        "      --super-refine-input PATH  Refine an existing H3 MP4 (diagnostic)\n"
        "      --super-refine-handoff PATH  Replay an existing direct .h3sh handoff\n"
        "      --info             Inspect model/device without mapping weights\n"
        "  -h, --help             Show this help\n");
}

static double parse_double(const char *value, const char *label,
                           double minimum, double maximum) {
    char *end = NULL;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno || !end || *end || !isfinite(parsed) || parsed < minimum ||
        parsed > maximum) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return parsed;
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int executable_relative_path(const char *program, const char *relative,
                                    char *path, size_t path_size) {
    char resolved[H3_SUPER_PATH_MAX];
    if (!realpath(program, resolved)) return 0;
    char *slash = strrchr(resolved, '/');
    if (!slash) return 0;
    *slash = '\0';
    int length = snprintf(path, path_size, "%s/%s", resolved, relative);
    return length > 0 && (size_t)length < path_size;
}

static void super_memory_report(const h3_super_params *params,
                                const h3_super_result *result) {
    if (!result->swapouts_available) {
        fprintf(stderr,
                "h3: warning: system swapout telemetry is unavailable; "
                "accepting output under the speed-first policy\n");
        return;
    }
    if (!result->no_swap_gate_passed) {
        uint64_t delta = result->system_swapouts_after >=
                         result->system_swapouts_before ?
            result->system_swapouts_after - result->system_swapouts_before : 0;
        double gib = (double)delta * (double)getpagesize() /
            (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: warning: system swapouts increased by %llu pages "
                "(about %.2f GiB); accepting output because swap is an "
                "observed cost, not a hard failure%s\n",
                (unsigned long long)delta, gib,
                params->allow_swap ? " (explicitly acknowledged)" : "");
    }
}

static int parse_int(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)parsed;
}

static int frames_from_seconds(const char *value) {
    char *end = NULL;
    errno = 0;
    double seconds = strtod(value, &end);
    double frames = seconds * (double)H3_FPS;
    if (errno || !end || *end || !isfinite(seconds) || seconds <= 0.0 ||
        !isfinite(frames) || frames > (double)INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    long long rounded = llround(frames);
    if (rounded < 1 || rounded > INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    return (int)rounded;
}

static uint64_t parse_u64(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

static h3_reference *append_reference(h3_reference references[12],
                                      size_t *count) {
    if (*count >= 12) {
        fprintf(stderr, "h3: Ref2VA supports at most 12 references\n");
        exit(2);
    }
    h3_reference *reference = &references[(*count)++];
    memset(reference, 0, sizeof(*reference));
    return reference;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_component(const char *label, const h3_component_info *item) {
    printf("  %-18s %2zu files  %4zu tensors  %7.3f GiB\n",
           label, item->files, item->tensors, gib(item->tensor_bytes));
}

static void print_info(const h3_ctx *ctx) {
    const h3_device_info *device = h3_device(ctx);
    const h3_model_info *model = h3_model(ctx);
    printf("h3-metal %s\n", H3_VERSION);
    printf("Device: %s (%s)\n", device->name, device->architecture);
    printf("  physical memory       %.1f GiB\n", gib(device->physical_memory));
    printf("  recommended GPU set   %.1f GiB\n", gib(device->recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(device->max_buffer_length));
    printf("  Apple GPU family      %d\n", device->apple_gpu_family);
    printf("  Metal 4               %s\n", device->metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", device->unified_memory ? "yes" : "no");
    printf("Native checkpoint inventory (header-only):\n");
    print_component("Qwen3-VL encoder", &model->text_encoder);
    print_component("FL2VA DiT", &model->fl2va_transformer);
    print_component("Ref2VA DiT", &model->ref2va_transformer);
    print_component("video VAE", &model->video_vae);
    print_component("audio VAE", &model->audio_vae);
}

typedef struct {
    h3_super_params *params;
    const char *prompt;
    h3_super_result *result;
    pthread_t thread;
    double seconds;
    char error[2048];
    int requested;
    int attempted;
    int started;
    int success;
    int conditioning;
    int input_models;
    int input_models_attempted;
    int input_models_success;
    double input_models_seconds;
    char input_models_error[2048];
    int start_by_block;
    int started_step;
    int denoise_steps;
    int started_block;
    int dit_blocks;
} super_prefetch_task;

static void *super_prefetch_worker(void *opaque) {
    super_prefetch_task *task = opaque;
    if (task->conditioning) {
        task->success = h3_super_prefetch_conditioning(
            task->params, task->prompt, task->result, &task->seconds,
            task->error, sizeof(task->error));
        if (task->success && task->input_models) {
            task->input_models_attempted = 1;
            task->input_models_success = task->result &&
                task->result->input_model_prefetch_completed;
            if (!task->input_models_success) {
                snprintf(task->input_models_error,
                         sizeof(task->input_models_error),
                         "conditioning prompt did not retain input models");
            }
        }
    } else {
        task->success = h3_super_prefetch_transformer(
            task->params, &task->seconds, task->error, sizeof(task->error));
    }
    return NULL;
}

static const char *super_prefetch_label(const super_prefetch_task *task) {
    return task && task->conditioning ?
        "conditioning" : "Transformer";
}

static void super_prefetch_disable(super_prefetch_task *task) {
    if (!task || !task->params) return;
    task->params->prefetch_transformer = 0;
    task->params->prefetch_conditioning = 0;
    task->params->prefetch_input_models = 0;
}

static void super_prefetch_start(super_prefetch_task *task,
                                 int completed, int total, int by_block) {
    if (!task || !task->requested || task->attempted) return;
    task->attempted = 1;
    task->start_by_block = by_block;
    if (by_block) {
        task->started_block = completed;
        task->dit_blocks = total;
    } else {
        task->started_step = completed;
        task->denoise_steps = total;
    }
    int create_error = pthread_create(
        &task->thread, NULL, super_prefetch_worker, task);
    if (create_error) {
        const char *label = super_prefetch_label(task);
        super_prefetch_disable(task);
        fprintf(stderr,
                "h3: %s prefetch disabled: pthread_create: %s\n",
                label, strerror(create_error));
        return;
    }
    task->started = 1;
    if (task->result) {
        if (by_block) {
            task->result->prefetch_started_block = completed;
            task->result->prefetch_total_blocks = total;
        } else {
            task->result->prefetch_started_step = completed;
            task->result->prefetch_total_steps = total;
        }
    }
    if (by_block) {
        fprintf(stderr,
                "h3: started LTX %s prefetch during final H3 pass after "
                "block %d/%d\n",
                super_prefetch_label(task), completed, total);
    } else {
        fprintf(stderr,
                "h3: started LTX %s prefetch after H3 denoise %d/%d\n",
                super_prefetch_label(task), completed, total);
    }
}

static void super_prefetch_finish(super_prefetch_task *task,
                                  h3_super_result *result) {
    if (!task || !task->requested) return;
    if (!task->started) {
        const char *label = super_prefetch_label(task);
        super_prefetch_disable(task);
        fprintf(stderr,
                "h3: %s prefetch did not start; using staged execution\n",
                label);
        return;
    }
    int conditioning = task->conditioning;
    const char *label = conditioning ? "conditioning" : "Transformer";
    double wait_started = monotonic_seconds();
    int join_error = pthread_join(task->thread, NULL);
    if (result) {
        double wait_seconds = monotonic_seconds() - wait_started;
        if (conditioning) {
            result->conditioning_prefetch_wait_seconds = wait_seconds;
            result->conditioning_prefetch_seconds = task->seconds;
        } else {
            result->transformer_prefetch_wait_seconds = wait_seconds;
            result->transformer_prefetch_seconds = task->seconds;
        }
    }
    if (join_error || !task->success) {
        super_prefetch_disable(task);
        fprintf(stderr,
                "h3: %s prefetch failed; using staged execution: %s\n",
                label, join_error ? strerror(join_error) : task->error);
        return;
    }
    if (result) {
        if (conditioning) result->conditioning_prefetch_completed = 1;
        else result->transformer_prefetch_completed = 1;
    }
    fprintf(stderr,
            "h3: LTX %s prefetched in %.2fs; post-H3 wait %.3fs\n",
            label, task->seconds,
            result ? (conditioning ?
                result->conditioning_prefetch_wait_seconds :
                result->transformer_prefetch_wait_seconds) : 0.0);
    if (conditioning && task->input_models_attempted) {
        if (task->input_models_success) {
            if (result) {
                result->input_model_prefetch_completed = 1;
                result->input_model_prefetch_seconds =
                    task->input_models_seconds;
            }
            fprintf(stderr,
                    "h3: LTX input models prefetched in the conditioning "
                    "prompt tail; timing is folded into conditioning\n");
        } else {
            task->params->prefetch_input_models = 0;
            fprintf(stderr,
                    "h3: input-model prefetch failed; input phase will load "
                    "models normally: %s\n",
                    task->input_models_error);
        }
    }
}

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
    h3_terminal_protocol terminal;
    int display_failed;
    const char *frames_dir;
    int frame_write_failed;
    super_prefetch_task *super_prefetch;
} cli_state;

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    cli_state *state = opaque;
    if (state->super_prefetch) {
        int start_block =
            state->super_prefetch->params->prefetch_start_block;
        if (start_block && !strcmp(phase, "denoise final eviction") &&
            completed >= start_block) {
            super_prefetch_start(
                state->super_prefetch, completed, total, 1);
        } else if (!start_block && !strcmp(phase, "denoise") &&
                   completed >= state->super_prefetch->params->
                       prefetch_start_step) {
            super_prefetch_start(
                state->super_prefetch, completed, total, 0);
        }
    }
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return 0;
    if (strcmp(state->phase, phase)) {
        if (state->active) fputc('\n', stderr);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    state->completed = completed;
    state->total = total;
    state->active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!state->active) fputc('\n', stderr);
    fflush(stderr);
    return 0;
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    cli_state *state = opaque;
    int preview = frame->denoise_step >= 0;
    if (!preview && state->frames_dir && !state->frame_write_failed) {
        char path[1024];
        int length = snprintf(path, sizeof(path), "%s/frame-%04d.ppm",
                              state->frames_dir, frame->frame_index);
        FILE *output = length > 0 && (size_t)length < sizeof(path) ?
            fopen(path, "wb") : NULL;
        if (!output ||
            fprintf(output, "P6\n%d %d\n255\n", frame->width,
                    frame->height) < 0) {
            fprintf(stderr, "h3: cannot write frame %d to %s\n",
                    frame->frame_index, state->frames_dir);
            if (output) fclose(output);
            state->frame_write_failed = 1;
        } else {
            size_t row_bytes = (size_t)frame->width * 3;
            for (int row = 0; row < frame->height; row++) {
                if (fwrite(frame->rgb + (size_t)row * frame->stride, 1,
                           row_bytes, output) != row_bytes) {
                    state->frame_write_failed = 1;
                    break;
                }
            }
            if (fclose(output) != 0) state->frame_write_failed = 1;
            if (state->frame_write_failed)
                fprintf(stderr, "h3: incomplete frame %d in %s\n",
                        frame->frame_index, state->frames_dir);
        }
    }
    if (state->frame_write_failed) return 1;
    if (state->display_failed || state->terminal == H3_TERM_NONE) return 0;
    if (state->active) {
        fputc('\n', stderr);
        state->active = 0;
    }
    if (preview)
        fprintf(stderr,
                "h3: denoise preview %d/%d, video frame %d/%d via %s\n",
                frame->denoise_step + 1, frame->denoise_steps,
                frame->frame_index + 1, frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    else
        fprintf(stderr, "h3: frame %d/%d via %s\n", frame->frame_index + 1,
                frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3: terminal display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_RENDER_WIDTH, OPT_RENDER_HEIGHT,
           OPT_FRAMES, OPT_SECONDS, OPT_STEPS,
           OPT_VIDEO_FLOW_SHIFT, OPT_AUDIO_FLOW_SHIFT, OPT_REUSE,
           OPT_LAYERS,
           OPT_CORE_REUSE,
           OPT_TOKEN_REDUCTION,
           OPT_SSD_STREAMING,
           OPT_USE_INT8_ROW_FC2,
           OPT_USE_REFERENCE_ROPE,
           OPT_USE_SLOWER_BF16_MLP,
           OPT_USE_SLOWER_BF16_QKV,
           OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_UNFUSED_INT8_INPUTS,
           OPT_USE_SLOWER_UNFUSED_QKV_ROPE,
           OPT_USE_SLOWER_SCALAR_QKV_RMS,
           OPT_USE_SLOWER_UNCACHED_INT8_SCALES,
           OPT_USE_SLOWER_DYNAMIC_FC1_K,
           OPT_USE_SLOWER_GROUPED_QUANTIZER,
           OPT_SEED,
           OPT_FIRST, OPT_LAST, OPT_REF_IMAGE, OPT_REF_IMAGE_SIZE,
           OPT_REF_VIDEO, OPT_REF_SILENT_VIDEO, OPT_REF_VIDEO_AUDIO,
           OPT_REF_AUDIO, OPT_FRAMES_DIR, OPT_SHOW, OPT_ZOOM,
           OPT_PROFILE,
           OPT_SUPER, OPT_SUPER_PROFILE, OPT_SUPER_STAGE1_STEPS,
           OPT_SUPER_DECODER,
           OPT_SUPER_ENDPOINT, OPT_SUPER_INPUT_DIR,
           OPT_SUPER_OUTPUT_DIR, OPT_SUPER_TRANSFORMER,
           OPT_SUPER_REFINER_LORA, OPT_SUPER_REFINER_LORA_STRENGTH,
           OPT_SUPER_TEXT_ENCODER, OPT_SUPER_VIDEO_VAE,
           OPT_SUPER_AUDIO_VAE, OPT_SUPER_UPSCALER, OPT_SUPER_NEGATIVE,
           OPT_SUPER_TAEHV, OPT_SUPER_TAEHV_DTYPE, OPT_SUPER_TAEHV_MODE,
           OPT_SUPER_TAEHV_EVICT, OPT_SUPER_PREFETCH_TRANSFORMER,
           OPT_SUPER_PREFETCH_CONDITIONING,
           OPT_SUPER_PREFETCH_INPUT_MODELS,
           OPT_SUPER_PREFETCH_START_STEP,
           OPT_SUPER_PREFETCH_START_BLOCK,
           OPT_SUPER_FINAL_EVICT_BLOCKS,
           OPT_SUPER_STAGE2_SCHEDULE,
           OPT_SUPER_SOL_ATTN, OPT_SUPER_DENSE_ATTN,
           OPT_SUPER_FUSE_DENOISE_DECODE,
           OPT_SUPER_BF16_HANDOFF,
           OPT_SUPER_MP4_HANDOFF,
           OPT_SUPER_TAEH3, OPT_SUPER_TIMEOUT, OPT_SUPER_TELEMETRY,
           OPT_SUPER_KEEP_STAGE1, OPT_SUPER_FACE_QUALITY_GATE,
           OPT_SUPER_PERSON_QUALITY_GATE,
           OPT_SUPER_PERSON_QUALITY_PRESET,
           OPT_SUPER_OUTPUT_QUALITY_GATE,
           OPT_SUPER_MIN_FACE_AREA,
           OPT_SUPER_QUALITY_ATTEMPTS,
           OPT_SUPER_ALLOW_SWAP,
           OPT_SUPER_REFINE_INPUT, OPT_SUPER_REFINE_HANDOFF,
           OPT_INFO };
    static const struct option options[] = {
        {"model-dir", required_argument, NULL, 'd'},
        {"prompt", required_argument, NULL, 'p'},
        {"output", required_argument, NULL, 'o'},
        {"width", required_argument, NULL, OPT_WIDTH},
        {"height", required_argument, NULL, OPT_HEIGHT},
        {"render-width", required_argument, NULL, OPT_RENDER_WIDTH},
        {"render-height", required_argument, NULL, OPT_RENDER_HEIGHT},
        {"frames", required_argument, NULL, OPT_FRAMES},
        {"seconds", required_argument, NULL, OPT_SECONDS},
        {"steps", required_argument, NULL, OPT_STEPS},
        {"video-flow-shift", required_argument, NULL, OPT_VIDEO_FLOW_SHIFT},
        {"audio-flow-shift", required_argument, NULL, OPT_AUDIO_FLOW_SHIFT},
        {"reuse", required_argument, NULL, OPT_REUSE},
        {"layers", required_argument, NULL, OPT_LAYERS},
        {"core-reuse", required_argument, NULL, OPT_CORE_REUSE},
        {"token-reduction", no_argument, NULL, OPT_TOKEN_REDUCTION},
        {"ssd-streaming", no_argument, NULL, OPT_SSD_STREAMING},
        {"use-int8-row-fc2", no_argument, NULL, OPT_USE_INT8_ROW_FC2},
        {"use-reference-rope", no_argument, NULL, OPT_USE_REFERENCE_ROPE},
        {"use-slower-bf16-mlp", no_argument, NULL,
         OPT_USE_SLOWER_BF16_MLP},
        {"use-slower-bf16-qkv", no_argument, NULL,
         OPT_USE_SLOWER_BF16_QKV},
        {"use-slower-bf16-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT},
        {"use-slower-row-major-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT},
        {"use-slower-unfused-int8-inputs", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_INT8_INPUTS},
        {"use-slower-unfused-qkv-rope", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_QKV_ROPE},
        {"use-slower-scalar-qkv-rms", no_argument, NULL,
         OPT_USE_SLOWER_SCALAR_QKV_RMS},
        {"use-slower-uncached-int8-scales", no_argument, NULL,
         OPT_USE_SLOWER_UNCACHED_INT8_SCALES},
        {"use-slower-dynamic-fc1-k", no_argument, NULL,
         OPT_USE_SLOWER_DYNAMIC_FC1_K},
        {"use-slower-grouped-quantizer", no_argument, NULL,
         OPT_USE_SLOWER_GROUPED_QUANTIZER},
        {"seed", required_argument, NULL, OPT_SEED},
        {"first-frame", required_argument, NULL, OPT_FIRST},
        {"last-frame", required_argument, NULL, OPT_LAST},
        {"ref-image", required_argument, NULL, OPT_REF_IMAGE},
        {"ref-image-size", required_argument, NULL, OPT_REF_IMAGE_SIZE},
        {"ref-video", required_argument, NULL, OPT_REF_VIDEO},
        {"ref-silent-video", required_argument, NULL, OPT_REF_SILENT_VIDEO},
        {"ref-video-audio", required_argument, NULL, OPT_REF_VIDEO_AUDIO},
        {"ref-audio", required_argument, NULL, OPT_REF_AUDIO},
        {"frames-dir", required_argument, NULL, OPT_FRAMES_DIR},
        {"show", no_argument, NULL, OPT_SHOW},
        {"zoom", required_argument, NULL, OPT_ZOOM},
        {"profile", no_argument, NULL, OPT_PROFILE},
        {"super", no_argument, NULL, OPT_SUPER},
        {"super-profile", required_argument, NULL, OPT_SUPER_PROFILE},
        {"super-stage1-steps", required_argument, NULL,
         OPT_SUPER_STAGE1_STEPS},
        {"super-decoder", required_argument, NULL, OPT_SUPER_DECODER},
        {"super-endpoint", required_argument, NULL, OPT_SUPER_ENDPOINT},
        {"super-input-dir", required_argument, NULL, OPT_SUPER_INPUT_DIR},
        {"super-output-dir", required_argument, NULL, OPT_SUPER_OUTPUT_DIR},
        {"super-transformer", required_argument, NULL,
         OPT_SUPER_TRANSFORMER},
        {"super-refiner-lora", required_argument, NULL,
         OPT_SUPER_REFINER_LORA},
        {"super-refiner-lora-strength", required_argument, NULL,
         OPT_SUPER_REFINER_LORA_STRENGTH},
        {"super-text-encoder", required_argument, NULL,
         OPT_SUPER_TEXT_ENCODER},
        {"super-video-vae", required_argument, NULL, OPT_SUPER_VIDEO_VAE},
        {"super-audio-vae", required_argument, NULL, OPT_SUPER_AUDIO_VAE},
        {"super-upscaler", required_argument, NULL, OPT_SUPER_UPSCALER},
        {"super-taehv", required_argument, NULL, OPT_SUPER_TAEHV},
        {"super-taehv-dtype", required_argument, NULL,
         OPT_SUPER_TAEHV_DTYPE},
        {"super-taehv-mode", required_argument, NULL,
         OPT_SUPER_TAEHV_MODE},
        {"super-taehv-evict", no_argument, NULL, OPT_SUPER_TAEHV_EVICT},
        {"super-prefetch-transformer", no_argument, NULL,
         OPT_SUPER_PREFETCH_TRANSFORMER},
        {"super-prefetch-conditioning", no_argument, NULL,
         OPT_SUPER_PREFETCH_CONDITIONING},
        {"super-prefetch-input-models", no_argument, NULL,
         OPT_SUPER_PREFETCH_INPUT_MODELS},
        {"super-prefetch-start-step", required_argument, NULL,
         OPT_SUPER_PREFETCH_START_STEP},
        {"super-prefetch-start-block", required_argument, NULL,
         OPT_SUPER_PREFETCH_START_BLOCK},
        {"super-final-evict-blocks", required_argument, NULL,
         OPT_SUPER_FINAL_EVICT_BLOCKS},
        {"super-stage2-schedule", required_argument, NULL,
         OPT_SUPER_STAGE2_SCHEDULE},
        {"super-sol-attn", no_argument, NULL, OPT_SUPER_SOL_ATTN},
        {"super-dense-attn", no_argument, NULL, OPT_SUPER_DENSE_ATTN},
        {"super-fuse-denoise-decode", no_argument, NULL,
         OPT_SUPER_FUSE_DENOISE_DECODE},
        {"super-bf16-handoff", no_argument, NULL,
         OPT_SUPER_BF16_HANDOFF},
        {"super-mp4-handoff", no_argument, NULL, OPT_SUPER_MP4_HANDOFF},
        {"super-negative", required_argument, NULL, OPT_SUPER_NEGATIVE},
        {"super-taeh3", required_argument, NULL, OPT_SUPER_TAEH3},
        {"super-timeout", required_argument, NULL, OPT_SUPER_TIMEOUT},
        {"super-telemetry", required_argument, NULL, OPT_SUPER_TELEMETRY},
        {"super-keep-stage1", no_argument, NULL, OPT_SUPER_KEEP_STAGE1},
        {"super-face-quality-gate", no_argument, NULL,
         OPT_SUPER_FACE_QUALITY_GATE},
        {"super-person-quality-gate", no_argument, NULL,
         OPT_SUPER_PERSON_QUALITY_GATE},
        {"super-person-quality-preset", no_argument, NULL,
         OPT_SUPER_PERSON_QUALITY_PRESET},
        {"super-output-quality-gate", no_argument, NULL,
         OPT_SUPER_OUTPUT_QUALITY_GATE},
        {"super-min-face-area", required_argument, NULL,
         OPT_SUPER_MIN_FACE_AREA},
        {"super-quality-attempts", required_argument, NULL,
         OPT_SUPER_QUALITY_ATTEMPTS},
        {"super-allow-swap", no_argument, NULL, OPT_SUPER_ALLOW_SWAP},
        {"super-refine-input", required_argument, NULL,
         OPT_SUPER_REFINE_INPUT},
        {"super-refine-handoff", required_argument, NULL,
         OPT_SUPER_REFINE_HANDOFF},
        {"info", no_argument, NULL, OPT_INFO},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = "outputs/h3.mp4";
    h3_params params = H3_PARAMS_DEFAULT;
    h3_reference references[12];
    size_t reference_count = 0;
    cli_state cli = {{0}, 0, -1, -1, H3_TERM_NONE, 0, NULL, 0, NULL};
    int show = 0;
    int profile = 0;
    int info = 0;
    int frames_given = 0;
    int seconds_given = 0;
    int seed_given = 0;
    int super = 0;
    h3_super_params super_params;
    h3_super_params_init(&super_params);
    h3_super_profile super_profile = super_params.profile;
    h3_super_decoder super_decoder = super_params.decoder;
    int super_decoder_given = 0;
    int super_prefetch_start_step_given = 0;
    int super_prefetch_start_block_given = 0;
    int super_attention_override = -1;
    h3_super_result super_result = {0};
    const char *super_taeh3 = NULL;
    const char *super_refine_input = NULL;
    const char *super_refine_handoff = NULL;
    char super_default_taeh3[H3_SUPER_PATH_MAX];
    char super_default_telemetry[H3_SUPER_PATH_MAX];
    char super_stage1[H3_SUPER_PATH_MAX];
    char super_error[2048];
    super_prefetch_task super_prefetch = {0};
    int option;
    while ((option = getopt_long(argc, argv, "d:p:o:h", options, NULL)) != -1) {
        switch (option) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'h': usage(argv[0]); return 0;
            case OPT_WIDTH: params.width = parse_int(optarg, "width"); break;
            case OPT_HEIGHT: params.height = parse_int(optarg, "height"); break;
            case OPT_RENDER_WIDTH:
                params.render_width = parse_int(optarg, "render width");
                break;
            case OPT_RENDER_HEIGHT:
                params.render_height = parse_int(optarg, "render height");
                break;
            case OPT_FRAMES:
                params.frames = parse_int(optarg, "frames");
                frames_given = 1;
                break;
            case OPT_SECONDS:
                params.frames = frames_from_seconds(optarg);
                seconds_given = 1;
                break;
            case OPT_STEPS: params.steps = parse_int(optarg, "steps"); break;
            case OPT_VIDEO_FLOW_SHIFT:
                params.video_flow_shift = parse_double(
                    optarg, "video flow shift", 0.001, 1000.0);
                break;
            case OPT_AUDIO_FLOW_SHIFT:
                params.audio_flow_shift = parse_double(
                    optarg, "audio flow shift", 0.001, 1000.0);
                break;
            case OPT_REUSE:
                params.denoise_reuse = parse_int(optarg, "reuse");
                break;
            case OPT_LAYERS:
                params.dit_layers = parse_int(optarg, "layers");
                break;
            case OPT_CORE_REUSE:
                params.core_reuse = parse_int(optarg, "core reuse");
                break;
            case OPT_TOKEN_REDUCTION: params.token_reduction = 1; break;
            case OPT_SSD_STREAMING: params.ssd_streaming = 1; break;
            case OPT_USE_INT8_ROW_FC2:
                params.use_int8_row_fc2 = 1;
                break;
            case OPT_USE_REFERENCE_ROPE:
                params.use_reference_rope = 1;
                break;
            case OPT_USE_SLOWER_BF16_MLP:
                params.use_slower_bf16_mlp = 1;
                break;
            case OPT_USE_SLOWER_BF16_QKV:
                params.use_slower_bf16_qkv = 1;
                break;
            case OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT:
                params.use_slower_bf16_attention_output = 1;
                break;
            case OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT:
                params.use_slower_row_major_attention_output = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_INT8_INPUTS:
                params.use_slower_unfused_int8_inputs = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_QKV_ROPE:
                params.use_slower_unfused_qkv_rope = 1;
                break;
            case OPT_USE_SLOWER_SCALAR_QKV_RMS:
                params.use_slower_scalar_qkv_rms = 1;
                break;
            case OPT_USE_SLOWER_UNCACHED_INT8_SCALES:
                params.use_slower_uncached_int8_scales = 1;
                break;
            case OPT_USE_SLOWER_DYNAMIC_FC1_K:
                params.use_slower_dynamic_fc1_k = 1;
                break;
            case OPT_USE_SLOWER_GROUPED_QUANTIZER:
                params.use_slower_grouped_quantizer = 1;
                break;
            case OPT_SEED:
                params.seed = parse_u64(optarg, "seed");
                seed_given = 1;
                break;
            case OPT_FIRST: params.first_frame = optarg; break;
            case OPT_LAST: params.last_frame = optarg; break;
            case OPT_REF_IMAGE: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_IMAGE;
                reference->path = optarg;
                break;
            }
            case OPT_REF_IMAGE_SIZE:
                if (!strcmp(optarg, "match"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
                else if (!strcmp(optarg, "max"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MAX;
                else {
                    fprintf(stderr,
                        "h3: --ref-image-size must be match or max\n");
                    return 2;
                }
                break;
            case OPT_REF_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 1;
                break;
            }
            case OPT_REF_SILENT_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 0;
                break;
            }
            case OPT_REF_VIDEO_AUDIO: {
                if (optind >= argc) {
                    fprintf(stderr,
                        "h3: --ref-video-audio requires VIDEO and AUDIO\n");
                    return 2;
                }
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO_AUDIO;
                reference->path = optarg;
                reference->audio_path = argv[optind++];
                break;
            }
            case OPT_REF_AUDIO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_AUDIO;
                reference->path = optarg;
                break;
            }
            case OPT_FRAMES_DIR: cli.frames_dir = optarg; break;
            case OPT_SHOW: show = 1; break;
            case OPT_ZOOM:
                if (!h3_terminal_set_zoom(parse_int(optarg, "zoom"))) {
                    fprintf(stderr, "h3: --zoom must be at least 1\n");
                    return 2;
                }
                break;
            case OPT_PROFILE: profile = 1; break;
            case OPT_SUPER: super = 1; break;
            case OPT_SUPER_PROFILE:
                if (!strcmp(optarg, "480p")) {
                    super_profile = H3_SUPER_PROFILE_480P;
                } else if (!strcmp(optarg, "480p-fast")) {
                    super_profile = H3_SUPER_PROFILE_480P_FAST;
                } else if (!strcmp(optarg, "480p-quality")) {
                    super_profile = H3_SUPER_PROFILE_480P_QUALITY;
                } else if (!strcmp(optarg, "480p-motion")) {
                    super_profile = H3_SUPER_PROFILE_480P_MOTION;
                } else if (!strcmp(optarg, "v2")) {
                    super_profile = H3_SUPER_PROFILE_V2;
                } else {
                    fprintf(stderr,
                            "h3: --super-profile must be 480p, 480p-fast, "
                            "480p-quality, 480p-motion, or v2\n");
                    return 2;
                }
                break;
            case OPT_SUPER_STAGE1_STEPS:
                super_params.stage1_steps = parse_int(
                    optarg, "Super Stage-1 steps");
                if (super_params.stage1_steps != 4 &&
                    super_params.stage1_steps != 6 &&
                    super_params.stage1_steps != 8) {
                    fprintf(stderr,
                            "h3: --super-stage1-steps must be 4, 6, or 8\n");
                    return 2;
                }
                break;
            case OPT_SUPER_DECODER:
                if (!strcmp(optarg, "taehv")) {
                    super_decoder = H3_SUPER_DECODER_TAEHV;
                } else if (!strcmp(optarg, "official")) {
                    super_decoder = H3_SUPER_DECODER_OFFICIAL;
                } else {
                    fprintf(stderr,
                            "h3: --super-decoder must be taehv or official\n");
                    return 2;
                }
                super_decoder_given = 1;
                break;
            case OPT_SUPER_ENDPOINT: super_params.endpoint = optarg; break;
            case OPT_SUPER_INPUT_DIR: super_params.input_dir = optarg; break;
            case OPT_SUPER_OUTPUT_DIR: super_params.output_dir = optarg; break;
            case OPT_SUPER_TRANSFORMER:
                super_params.transformer = optarg;
                break;
            case OPT_SUPER_REFINER_LORA:
                super_params.refiner_lora = optarg;
                break;
            case OPT_SUPER_REFINER_LORA_STRENGTH:
                super_params.refiner_lora_strength = parse_double(
                    optarg, "Super refiner LoRA strength", 0.0, 100.0);
                break;
            case OPT_SUPER_TEXT_ENCODER:
                super_params.text_encoder = optarg;
                break;
            case OPT_SUPER_VIDEO_VAE: super_params.video_vae = optarg; break;
            case OPT_SUPER_AUDIO_VAE: super_params.audio_vae = optarg; break;
            case OPT_SUPER_UPSCALER:
                super_params.latent_upscaler = optarg;
                break;
            case OPT_SUPER_TAEHV:
                super_params.taehv_checkpoint = optarg;
                break;
            case OPT_SUPER_TAEHV_DTYPE:
                super_params.taehv_compute_dtype = optarg;
                break;
            case OPT_SUPER_TAEHV_MODE:
                super_params.taehv_temporal_mode = optarg;
                break;
            case OPT_SUPER_TAEHV_EVICT:
                super_params.taehv_keep_on_device = 0;
                break;
            case OPT_SUPER_PREFETCH_TRANSFORMER:
                super_params.prefetch_transformer = 1;
                break;
            case OPT_SUPER_PREFETCH_CONDITIONING:
                super_params.prefetch_conditioning = 1;
                break;
            case OPT_SUPER_PREFETCH_INPUT_MODELS:
                super_params.prefetch_input_models = 1;
                break;
            case OPT_SUPER_PREFETCH_START_STEP:
                super_params.prefetch_start_step = parse_int(
                    optarg, "Super prefetch start step");
                super_prefetch_start_step_given = 1;
                break;
            case OPT_SUPER_PREFETCH_START_BLOCK:
                super_params.prefetch_start_block = parse_int(
                    optarg, "Super prefetch start block");
                super_prefetch_start_block_given = 1;
                break;
            case OPT_SUPER_FINAL_EVICT_BLOCKS:
                super_params.final_evict_blocks = parse_int(
                    optarg, "Super final eviction group");
                break;
            case OPT_SUPER_STAGE2_SCHEDULE:
                if (!strcmp(optarg, "default")) {
                    super_params.stage2_schedule =
                        H3_SUPER_STAGE2_SCHEDULE_DEFAULT;
                } else if (!strcmp(optarg, "skip-0p9")) {
                    super_params.stage2_schedule =
                        H3_SUPER_STAGE2_SCHEDULE_SKIP_0P9;
                } else if (!strcmp(optarg, "tail-0p42")) {
                    super_params.stage2_schedule =
                        H3_SUPER_STAGE2_SCHEDULE_TAIL_0P42;
                } else {
                    fprintf(stderr,
                            "h3: --super-stage2-schedule must be default, "
                            "skip-0p9, or tail-0p42\n");
                    return 2;
                }
                break;
            case OPT_SUPER_SOL_ATTN:
                if (super_attention_override == 0) {
                    fprintf(stderr,
                            "h3: --super-sol-attn and --super-dense-attn "
                            "are mutually exclusive\n");
                    return 2;
                }
                super_attention_override = 1;
                break;
            case OPT_SUPER_DENSE_ATTN:
                if (super_attention_override == 1) {
                    fprintf(stderr,
                            "h3: --super-sol-attn and --super-dense-attn "
                            "are mutually exclusive\n");
                    return 2;
                }
                super_attention_override = 0;
                break;
            case OPT_SUPER_FUSE_DENOISE_DECODE:
                super_params.fuse_denoise_decode = 1;
                break;
            case OPT_SUPER_BF16_HANDOFF:
                super_params.prepared_handoff = 1;
                break;
            case OPT_SUPER_MP4_HANDOFF:
                super_params.direct_handoff = 0;
                break;
            case OPT_SUPER_NEGATIVE:
                super_params.negative_prompt = optarg;
                break;
            case OPT_SUPER_TAEH3: super_taeh3 = optarg; break;
            case OPT_SUPER_TIMEOUT:
                super_params.timeout_seconds = parse_double(
                    optarg, "Super timeout", 1.0, 86400.0);
                break;
            case OPT_SUPER_TELEMETRY:
                super_params.telemetry_path = optarg;
                break;
            case OPT_SUPER_KEEP_STAGE1: super_params.keep_stage1 = 1; break;
            case OPT_SUPER_FACE_QUALITY_GATE:
                super_params.face_quality_gate = 1;
                break;
            case OPT_SUPER_PERSON_QUALITY_GATE:
                super_params.person_quality_gate = 1;
                break;
            case OPT_SUPER_PERSON_QUALITY_PRESET:
                super = 1;
                super_params.strict_person_quality = 1;
                break;
            case OPT_SUPER_OUTPUT_QUALITY_GATE:
                super_params.output_quality_gate = 1;
                break;
            case OPT_SUPER_MIN_FACE_AREA:
                super_params.min_face_area = parse_double(
                    optarg, "Super minimum face area", 0.0, 0.25);
                super_params.face_quality_gate = 1;
                break;
            case OPT_SUPER_QUALITY_ATTEMPTS:
                super_params.quality_attempts = parse_int(
                    optarg, "Super quality attempts");
                break;
            case OPT_SUPER_ALLOW_SWAP: super_params.allow_swap = 1; break;
            case OPT_SUPER_REFINE_INPUT:
                super = 1;
                super_refine_input = optarg;
                break;
            case OPT_SUPER_REFINE_HANDOFF:
                super = 1;
                super_refine_handoff = optarg;
                break;
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    if (super_refine_input && super_refine_handoff) {
        fprintf(stderr,
                "h3: --super-refine-input and --super-refine-handoff are "
                "mutually exclusive\n");
        return 2;
    }
    if ((super_refine_input || super_refine_handoff) &&
        super_params.min_face_area > 0.0 &&
        super_params.face_quality_gate &&
        !super_params.person_quality_gate) {
        super_params.face_quality_gate = 0;
        super_params.output_quality_gate = 1;
    }
    if ((super_refine_input || super_refine_handoff) &&
        (super_params.face_quality_gate ||
         super_params.person_quality_gate)) {
        fprintf(stderr,
                "h3: Super quality gates require native Stage 1\n");
        return 2;
    }
    const char *super_refine_source = super_refine_handoff ?
        super_refine_handoff : super_refine_input;
    if (!model_dir && !super_refine_source) {
        usage(argv[0]);
        return 2;
    }
    if (frames_given && seconds_given) {
        fprintf(stderr, "h3: --seconds and --frames are mutually exclusive\n");
        return 2;
    }
    if (super && !prompt) {
        fprintf(stderr, "h3: --super currently requires a one-shot --prompt\n");
        return 2;
    }
    if (super && info) {
        fprintf(stderr, "h3: --super and --info are mutually exclusive\n");
        return 2;
    }
    if (super_prefetch_start_step_given &&
        super_prefetch_start_block_given) {
        fprintf(stderr,
                "h3: Super step and block prefetch deadlines are mutually "
                "exclusive\n");
        return 2;
    }
    double super_e2e_started = super ? monotonic_seconds() : 0.0;
    if (super) {
        h3_super_params_apply_prefetch_defaults(
            &super_params,
            super_prefetch_start_step_given,
            super_prefetch_start_block_given);
        if (super_params.prefetch_start_block &&
            !super_params.final_evict_blocks) {
            super_params.final_evict_blocks = 5;
        }
        h3_super_params_set_profile(&super_params, super_profile);
        /* Sana conditions both stages from the same pinned opening-frame
         * asset. Keep --first-frame as the single source of truth. */
        super_params.first_frame = params.first_frame;
        super_params.stage1_model_dir = model_dir;
        if (super_decoder_given) super_params.decoder = super_decoder;
        if (super_params.strict_person_quality) {
            if (super_refine_source) {
                fprintf(stderr,
                        "h3: --super-person-quality-preset requires native "
                        "Stage 1 and cannot be used for Stage-2 replay\n");
                return 2;
            }
            h3_super_params_apply_person_quality_preset(&super_params);
        }
        if (super_attention_override >= 0) {
            super_params.sol_attention = super_attention_override;
        }
        if (!super_refine_source) {
            if (!h3_super_stage1_schedule(
                    &super_params, &params.video_flow_shift,
                    &params.audio_flow_shift,
                    super_error, sizeof(super_error))) {
                fprintf(stderr,
                        "h3: Super Stage-1 model contract failed: %s\n",
                        super_error);
                return 1;
            }
            fprintf(stderr,
                    "h3: Super Stage 1 flow shifts video=%.6g audio=%.6g "
                    "from the pinned merge manifest\n",
                    params.video_flow_shift, params.audio_flow_shift);
        }
        if (super_refine_input) super_params.direct_handoff = 0;
        if (super_refine_handoff) super_params.direct_handoff = 1;
        if (super_refine_source &&
            (super_params.prefetch_transformer ||
             super_params.prefetch_conditioning)) {
            fprintf(stderr,
                    "h3: Super prefetch needs native Stage 1; disabling it "
                    "for Stage-2-only replay\n");
            super_params.prefetch_transformer = 0;
            super_params.prefetch_conditioning = 0;
            super_params.prefetch_input_models = 0;
            super_params.prefetch_start_step = 0;
            super_params.prefetch_start_block = 0;
            super_params.final_evict_blocks = 0;
        }
        params.width = super_params.stage1_width;
        params.height = super_params.stage1_height;
        params.render_width = 0;
        params.render_height = 0;
        params.frames = super_params.stage1_frames;
        params.steps = super_params.stage1_steps;
        params.denoise_reuse = 1;
        params.dit_layers = 50;
        params.core_reuse = 1;
        params.token_reduction = 0;
        if (!super_refine_source && !super_taeh3 && executable_relative_path(
                argv[0], "models", super_default_taeh3,
                sizeof(super_default_taeh3))) {
            super_taeh3 = super_default_taeh3;
        }
        if (!super_refine_source &&
            (!super_taeh3 || access(super_taeh3, R_OK) != 0)) {
            fprintf(stderr,
                    "h3: --super requires readable TAEH3 weights; use "
                    "--super-taeh3 PATH\n");
            return 2;
        }
        if (!super_refine_source) {
            const char *coreml_directory = h3_runtime_getenv("H3_COREML_ANE_DIR");
            const char *coreml_model = h3_runtime_getenv("H3_COREML_ANE_MODEL");
            int coreml_mlp = (coreml_directory && *coreml_directory) ||
                             (coreml_model && *coreml_model);
            if (coreml_mlp || params.use_slower_bf16_mlp) {
                unsetenv("H3_NAX_FORCE");
                unsetenv("H3_NAX");
                if (coreml_mlp)
                    fprintf(stderr,
                            "h3: Super Stage 1 uses BF16 MLP for configured "
                            "Core ML channel parallelism\n");
                else
                    fprintf(stderr,
                            "h3: Super Stage 1 honors forced BF16 MLP\n");
            } else {
                setenv("H3_NAX_FORCE", "1", 1);
                setenv("H3_NAX", "1", 1);
            }
            setenv("H3_TAEH3_WEIGHTS", super_taeh3, 1);
            if (super_params.final_evict_blocks) {
                char eviction_group[16];
                snprintf(eviction_group, sizeof(eviction_group), "%d",
                         super_params.final_evict_blocks);
                setenv("H3_FINAL_EVICT_BLOCKS", eviction_group, 1);
            }
        }
        if (!super_params.telemetry_path) {
            int length = snprintf(super_default_telemetry,
                                  sizeof(super_default_telemetry),
                                  "%s.super.json", output);
            if (length < 0 || (size_t)length >=
                              sizeof(super_default_telemetry)) {
                fprintf(stderr, "h3: Super telemetry path is too long\n");
                return 2;
            }
            super_params.telemetry_path = super_default_telemetry;
        }
        if (!super_refine_source) {
            fprintf(stderr,
                    "h3: Super %s Stage 1 at Turbo %dx%d, %d frames, "
                    "%d steps, 50 layers, TAEH3; Stage 2 %dx%d -> %dx%d "
                    "via %s\n",
                    h3_super_profile_name(super_params.profile),
                    super_params.stage1_width, super_params.stage1_height,
                    super_params.stage1_frames, super_params.stage1_steps,
                    super_params.refined_width,
                    super_params.refined_height, super_params.output_width,
                    super_params.output_height,
                    h3_super_decoder_name(super_params.decoder));
        }
        if (!h3_super_preflight(&super_params, &super_result,
                                super_error, sizeof(super_error))) {
            fprintf(stderr, "h3: Super preflight failed: %s\n", super_error);
            return 1;
        }
        if (!h3_super_make_stage1_path(&super_params, super_stage1,
                                       sizeof(super_stage1), super_error,
                                       sizeof(super_error))) {
            fprintf(stderr, "h3: cannot prepare Super handoff: %s\n",
                    super_error);
            return 1;
        }
        if (super_refine_source && !h3_super_import_stage1(
                super_refine_source, super_stage1,
                super_error, sizeof(super_error))) {
            fprintf(stderr, "h3: cannot import Super Stage-1 input: %s\n",
                    super_error);
            return 1;
        }
        if (!super_refine_source &&
            (super_params.prefetch_transformer ||
             super_params.prefetch_conditioning)) {
            super_prefetch.params = &super_params;
            super_prefetch.prompt = prompt;
            super_prefetch.result = &super_result;
            super_prefetch.conditioning =
                super_params.prefetch_conditioning;
            super_prefetch.input_models =
                super_params.prefetch_input_models;
            super_prefetch.requested = 1;
            cli.super_prefetch = &super_prefetch;
        }
    }
    if (prompt && params.steps >= 2 && params.steps <= 7 &&
        params.denoise_reuse > 1) {
        fprintf(stderr,
            "h3: warning: --reuse with only %d denoising steps leaves very "
            "few fresh model evaluations\n", params.steps);
    }
    params.references = references;
    params.reference_count = reference_count;
    if (cli.frames_dir && mkdir(cli.frames_dir, 0755) != 0 &&
        errno != EEXIST) {
        fprintf(stderr, "h3: cannot create frames directory %s: %s\n",
                cli.frames_dir, strerror(errno));
        return 1;
    }
    if (profile) setenv("H3_PROFILE", "1", 1);
    if (super_refine_source) {
        double started = super_e2e_started;
        if (!h3_super_refine_comfy(
                &super_params, super_stage1, prompt, params.seed, output,
                &super_result, super_error, sizeof(super_error))) {
            char refine_error[2048];
            snprintf(refine_error, sizeof(refine_error), "%s", super_error);
            if (super_result.output_quality_gate_applied) {
                double total_seconds = monotonic_seconds() - started;
                if (!h3_super_write_telemetry(
                        &super_params, &super_result, super_stage1, output,
                        params.seed, 0.0, total_seconds,
                        super_error, sizeof(super_error))) {
                    fprintf(stderr,
                            "h3: warning: cannot write rejected Super "
                            "telemetry: %s\n", super_error);
                }
            }
            fprintf(stderr,
                    "h3: Super Stage 2 failed: %s\n"
                    "h3: retained diagnostic Stage-1 handoff at %s\n",
                    refine_error, super_stage1);
            return 1;
        }
        double total_seconds = monotonic_seconds() - started;
        if (!h3_super_write_telemetry(
                &super_params, &super_result, super_stage1, output,
                params.seed, 0.0, total_seconds,
                super_error, sizeof(super_error))) {
            fprintf(stderr, "h3: warning: cannot write Super telemetry: %s\n",
                    super_error);
        }
        if (!super_params.keep_stage1 && unlink(super_stage1) != 0) {
                fprintf(stderr,
                        "h3: warning: cannot remove imported Stage-1 handoff "
                        "%s: %s\n",
                        super_stage1, strerror(errno));
        }
        super_memory_report(&super_params, &super_result);
        fprintf(stderr,
                "h3: Super refinement complete -> %s [Stage 2 %.2fs, "
                "total %.2fs]\n",
                output, super_result.stage2_seconds, total_seconds);
        return 0;
    }
    double initial_model_load_started = super ? monotonic_seconds() : 0.0;
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    if (super) {
        super_result.stage1_model_load_attempts++;
        super_result.stage1_model_load_seconds +=
            monotonic_seconds() - initial_model_load_started;
    }
    if (info) print_info(ctx);
    if (prompt) {
        params.output_path = super ?
            (super_params.direct_handoff ? NULL : super_stage1) : output;
        params.retain_decoded = super &&
            (super_params.direct_handoff || super_params.face_quality_gate ||
             super_params.person_quality_gate);
        params.on_progress = cli_progress;
        params.callback_opaque = &cli;
        if (cli.frames_dir) params.on_frame = cli_frame;
        if (show) {
            cli.terminal = h3_terminal_detect();
            if (cli.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3: graphical output uses %s\n",
                        h3_terminal_protocol_name(cli.terminal));
                params.on_frame = cli_frame;
                params.preview_denoise = 1;
            }
        }
        double super_started = super ? super_e2e_started : 0.0;
        double stage1_seconds = 0.0;
        h3_result *result = NULL;
        uint64_t first_seed = params.seed;
        int maximum_attempts = super ? super_params.quality_attempts : 1;
        int super_complete = 0;
        if (super) super_result.stage1_first_seed = first_seed;
        for (int attempt = 0; attempt < maximum_attempts; attempt++) {
            if (super && !ctx) {
                fprintf(stderr,
                        "h3: reloading H3 after rejected refined candidate "
                        "%d/%d\n", attempt, maximum_attempts);
                double reload_started = monotonic_seconds();
                ctx = h3_load_dir(model_dir);
                if (!ctx) {
                    fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
                    return 1;
                }
                super_result.stage1_model_load_attempts++;
                super_result.stage1_model_load_seconds +=
                    monotonic_seconds() - reload_started;
            }
            double attempt_started = monotonic_seconds();
            if (super) {
                params.seed = first_seed + (uint64_t)attempt;
                if (maximum_attempts > 1) {
                    fprintf(stderr,
                            "h3: Super full quality candidate %d/%d, "
                            "seed=%" PRIu64 "\n",
                            attempt + 1, maximum_attempts, params.seed);
                }
            }
            result = h3_generate(ctx, prompt, &params);
            if (!result) {
                if (super)
                    super_prefetch_finish(&super_prefetch, &super_result);
                if (cli.active) fputc('\n', stderr);
                fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
                h3_free(ctx);
                return 1;
            }
            if (super &&
                (result->width != super_params.stage1_width ||
                 result->height != super_params.stage1_height ||
                 result->frames != super_params.stage1_frames ||
                 result->fps != super_params.output_fps)) {
                super_prefetch_finish(&super_prefetch, &super_result);
                fprintf(stderr,
                        "h3: Super Stage 1 contract mismatch: %dx%d, "
                        "%d frames, %d fps\n", result->width,
                        result->height, result->frames, result->fps);
                h3_result_free(result);
                h3_free(ctx);
                return 1;
            }

            int quality_passed = 1;
            if (super &&
                (super_params.face_quality_gate ||
                 super_params.person_quality_gate)) {
                quality_passed = h3_super_check_stage1_face_quality(
                    &super_params, result, &super_result,
                    super_error, sizeof(super_error));
            }
            double attempt_seconds = monotonic_seconds() - attempt_started;
            if (super) {
                stage1_seconds += attempt_seconds;
                super_result.stage1_quality_attempts++;
                if (quality_passed) {
                    super_result.stage1_accepted_seed = params.seed;
                    super_result.stage1_quality_accepted_seconds +=
                        attempt_seconds;
                } else {
                    super_result.stage1_quality_rejections++;
                    super_result.stage1_quality_rejected_seconds +=
                        attempt_seconds;
                    snprintf(
                        super_result.stage1_quality_last_rejection,
                        sizeof(super_result.stage1_quality_last_rejection),
                        "%s", super_error);
                }
            }
            if (super && !quality_passed) {
                int retained = 0;
                if (super_params.keep_stage1) {
                    if (super_params.direct_handoff) {
                        char handoff_error[2048];
                        retained = h3_super_write_handoff(
                            &super_params, result, super_stage1,
                            &super_result.handoff_bytes,
                            handoff_error, sizeof(handoff_error));
                        if (!retained) {
                            fprintf(stderr,
                                    "h3: warning: cannot retain rejected "
                                    "Stage-1 handoff: %s\n", handoff_error);
                        }
                    } else {
                        retained = access(super_stage1, R_OK) == 0;
                    }
                    if (retained) {
                        fprintf(stderr,
                                "h3: retained rejected Stage-1 handoff at "
                                "%s\n", super_stage1);
                    }
                } else if (!super_params.direct_handoff &&
                           unlink(super_stage1) != 0 && errno != ENOENT) {
                    fprintf(stderr,
                            "h3: warning: cannot remove rejected Stage-1 "
                            "handoff %s: %s\n", super_stage1,
                            strerror(errno));
                }
                fprintf(stderr,
                        "h3: Super Stage 1 person quality gate rejected "
                        "seed=%" PRIu64 ": %s\n",
                        params.seed, super_error);
                h3_result_free(result);
                result = NULL;
                if (attempt + 1 < maximum_attempts) {
                    if (super_params.keep_stage1 &&
                        !h3_super_make_stage1_path(
                            &super_params, super_stage1,
                            sizeof(super_stage1), super_error,
                            sizeof(super_error))) {
                        super_prefetch_finish(
                            &super_prefetch, &super_result);
                        fprintf(stderr,
                                "h3: cannot prepare the next Super handoff: "
                                "%s\n", super_error);
                        h3_free(ctx);
                        return 1;
                    }
                    continue;
                }
                super_prefetch_finish(&super_prefetch, &super_result);
                double total_seconds =
                    monotonic_seconds() - super_started;
                if (!h3_super_write_telemetry(
                        &super_params, &super_result, super_stage1, output,
                        params.seed, stage1_seconds, total_seconds,
                        super_error, sizeof(super_error))) {
                    fprintf(stderr,
                            "h3: warning: cannot write rejected Super "
                            "telemetry: %s\n", super_error);
                }
                if (super_result.refine_quality_attempts > 0) {
                    fprintf(stderr,
                            "h3: Super exhausted %d full quality candidates; "
                            "no refined output passed\n", maximum_attempts);
                } else {
                    fprintf(stderr,
                            "h3: Super exhausted %d Stage-1 quality "
                            "attempt%s; LTX was not loaded\n",
                            maximum_attempts,
                            maximum_attempts == 1 ? "" : "s");
                }
                h3_free(ctx);
                return 1;
            }

            if (!super) break;
            if (super_params.direct_handoff) {
                double handoff_started = monotonic_seconds();
                if (!h3_super_write_handoff(
                        &super_params, result, super_stage1,
                        &super_result.handoff_bytes,
                        super_error, sizeof(super_error))) {
                    super_prefetch_finish(&super_prefetch, &super_result);
                    fprintf(stderr,
                            "h3: cannot write Super direct handoff: %s\n",
                            super_error);
                    h3_result_free(result);
                    h3_free(ctx);
                    return 1;
                }
                double handoff_seconds =
                    monotonic_seconds() - handoff_started;
                super_result.handoff_seconds += handoff_seconds;
                stage1_seconds += handoff_seconds;
            }
            h3_result_free(result);
            result = NULL;
            super_prefetch_finish(&super_prefetch, &super_result);
            h3_free(ctx);
            ctx = NULL;
            fprintf(stderr,
                    "h3: released H3 model state before LTX Stage 2\n");
            fprintf(stderr,
                    "h3: Stage 1 candidate seed=%" PRIu64
                    " complete -> %s [candidate %.2fs, cumulative %.2fs]\n",
                    params.seed, super_stage1, attempt_seconds,
                    stage1_seconds);

            double refine_started = monotonic_seconds();
            int refine_ok = h3_super_refine_comfy(
                &super_params, super_stage1, prompt, params.seed, output,
                &super_result, super_error, sizeof(super_error));
            double refine_seconds = monotonic_seconds() - refine_started;
            super_result.refine_quality_attempts++;
            super_result.refine_quality_total_seconds += refine_seconds;
            super_result.output_quality_total_seconds +=
                super_result.output_quality_seconds;
            if (refine_ok) {
                super_result.refine_quality_accepted_seconds +=
                    refine_seconds;
                super_complete = 1;
                double total_seconds =
                    monotonic_seconds() - super_started;
                if (!h3_super_write_telemetry(
                        &super_params, &super_result, super_stage1, output,
                        params.seed, stage1_seconds, total_seconds,
                        super_error, sizeof(super_error))) {
                    fprintf(stderr,
                            "h3: warning: cannot write Super telemetry: %s\n",
                            super_error);
                }
                if (!super_params.keep_stage1 &&
                    unlink(super_stage1) != 0 && errno != ENOENT) {
                    fprintf(stderr,
                            "h3: warning: cannot remove Stage-1 handoff "
                            "%s: %s\n", super_stage1, strerror(errno));
                }
                super_memory_report(&super_params, &super_result);
                fprintf(stderr,
                        "h3: Super complete -> %s [Stage 1 total %.2fs, "
                        "final Stage 2 %.2fs, E2E %.2fs, candidates %d]\n",
                        output, stage1_seconds,
                        super_result.stage2_seconds, total_seconds,
                        super_result.stage1_quality_attempts);
                break;
            }

            char refine_error[2048];
            snprintf(refine_error, sizeof(refine_error), "%s", super_error);
            int output_rejected =
                super_result.output_quality_gate_applied &&
                !super_result.output_quality_gate_passed;
            if (output_rejected) {
                super_result.refine_quality_rejections++;
                super_result.refine_quality_rejected_seconds +=
                    refine_seconds;
                snprintf(
                    super_result.refine_quality_last_rejection,
                    sizeof(super_result.refine_quality_last_rejection),
                    "%s", refine_error);
            }
            if (output_rejected && attempt + 1 < maximum_attempts) {
                fprintf(stderr,
                        "h3: refined candidate seed=%" PRIu64
                        " rejected after LTX: %s\n"
                        "h3: retrying the full pipeline with seed=%" PRIu64
                        " after backend release\n",
                        params.seed, refine_error, params.seed + 1);
                if (!super_params.keep_stage1 &&
                    unlink(super_stage1) != 0 && errno != ENOENT) {
                    fprintf(stderr,
                            "h3: warning: cannot remove intermediate "
                            "rejected handoff %s: %s\n",
                            super_stage1, strerror(errno));
                }
                if (!h3_super_make_stage1_path(
                        &super_params, super_stage1,
                        sizeof(super_stage1), super_error,
                        sizeof(super_error))) {
                    fprintf(stderr,
                            "h3: cannot prepare the next Super handoff: %s\n",
                            super_error);
                    return 1;
                }
                continue;
            }

            if (super_result.output_quality_gate_applied) {
                double total_seconds =
                    monotonic_seconds() - super_started;
                if (!h3_super_write_telemetry(
                        &super_params, &super_result, super_stage1,
                        output, params.seed, stage1_seconds,
                        total_seconds, super_error,
                        sizeof(super_error))) {
                    fprintf(stderr,
                            "h3: warning: cannot write rejected Super "
                            "telemetry: %s\n", super_error);
                }
            }
            fprintf(stderr,
                    "h3: Super Stage 2 failed: %s\n"
                    "h3: retained diagnostic Stage-1 handoff at %s\n",
                    refine_error, super_stage1);
            return 1;
        }
        if (!super) {
            h3_result_free(result);
        } else if (!super_complete) {
            if (ctx) h3_free(ctx);
            return 1;
        }
        if (!super && output && *output) {
            fprintf(stderr, "h3: wrote %s\n", output);
        }
        if (cli.frames_dir)
            fprintf(stderr, "h3: wrote frames to %s\n", cli.frames_dir);
    } else if (!info) {
        int cli_status = h3_cli_run(ctx, model_dir, &params, show, seed_given);
        h3_free(ctx);
        return cli_status;
    }
    if (ctx) h3_free(ctx);
    return 0;
}
