#include "h3_dit.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
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

typedef struct {
    uint64_t queries;
    uint64_t cancel_after;
} exact_cancel_state;

typedef void (*streaming_audit_reset_fn)(void);
typedef int (*streaming_audit_snapshot_fn)(char **result, char **error);
typedef void (*streaming_string_free_fn)(char *value);

static int exact_cancel_query(const void *opaque) {
    exact_cancel_state *state = (exact_cancel_state *)opaque;
    if (!state) return 0;
    return state->queries++ >= state->cancel_after;
}

typedef struct {
    double denoise_seconds;
    int finite;
    uint64_t hash;
    uint64_t bytes_read;
    double read_seconds;
    double wait_seconds;
    double gpu_seconds;
} request_result;

static int write_latents(const char *base, int request, int request_count,
                         const float *video, size_t video_elements,
                         const float *audio, size_t audio_elements) {
    char *path = NULL;
    if (request_count == 1) {
        path = strdup(base);
    } else {
        int needed = snprintf(NULL, 0, "%s.request-%d", base, request + 1);
        if (needed >= 0) {
            path = malloc((size_t)needed + 1);
            if (path)
                snprintf(path, (size_t)needed + 1, "%s.request-%d",
                         base, request + 1);
        }
    }
    if (!path) return 0;
    FILE *output = fopen(path, "wb");
    int ok = output != NULL;
    if (ok &&
        fwrite(video, sizeof(*video), video_elements, output) != video_elements)
        ok = 0;
    if (ok &&
        fwrite(audio, sizeof(*audio), audio_elements, output) != audio_elements)
        ok = 0;
    if (output && fclose(output)) ok = 0;
    free(path);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s TRANSFORMER_DIR MEMORY_BUDGET_BYTES OUTPUT_F32 "
                "[REQUESTS]\n",
                argv[0]);
        return 2;
    }
    uint64_t budget = 0;
    if (!parse_u64(argv[2], &budget)) {
        fprintf(stderr, "invalid memory budget: %s\n", argv[2]);
        return 2;
    }
    uint64_t parsed_requests = 1;
    if (argc == 5 &&
        (!parse_u64(argv[4], &parsed_requests) || parsed_requests < 1 ||
         parsed_requests > 32)) {
        fprintf(stderr, "REQUESTS must be in [1,32]: %s\n", argv[4]);
        return 2;
    }
    int request_count = (int)parsed_requests;
    streaming_audit_reset_fn audit_reset =
        (streaming_audit_reset_fn)dlsym(
            RTLD_DEFAULT, "tc_streaming_audit_reset");
    streaming_audit_snapshot_fn audit_snapshot =
        (streaming_audit_snapshot_fn)dlsym(
            RTLD_DEFAULT, "tc_streaming_audit_snapshot_json");
    streaming_string_free_fn string_free =
        (streaming_string_free_fn)dlsym(RTLD_DEFAULT, "tc_string_free");
    if (audit_reset) audit_reset();

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
        argv[1], "h3_shaders.metal", NULL, NULL, &text, &layout, &sigmas,
        50, 1, 0, 1, 0, budget, NULL, 1.0f,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        progress, NULL, error, sizeof(error));
    double load_seconds = now_seconds() - load_started;
    free(text.values);
    h3_layout_free(&layout);
    if (!dit) {
        fprintf(stderr, "cannot load H3 DiT: %s\n", error);
        return 2;
    }
    h3_dit_stream_fill_result_v1 candidate_fill = {0};
    int candidate_fill_enabled = 0;
    const char *candidate_fill_request = getenv(
        "TURBOCIDER_H3_CANDIDATE_FILL");
    if (candidate_fill_request && *candidate_fill_request &&
        strcmp(candidate_fill_request, "0")) {
        const unsigned first = h3_dit_stream_first_block_id(dit);
        if (first == UINT_MAX ||
            !h3_dit_stream_check_source_snapshot(
                dit, error, sizeof(error)) ||
            !h3_dit_stream_fill_slot_v1(
                dit, first, 1, 8u << 20, NULL, NULL,
                &candidate_fill, error, sizeof(error))) {
            fprintf(stderr, "H3 candidate slot fill failed: %s\n", error);
            h3_dit_free(dit);
            return 2;
        }
        candidate_fill_enabled = 1;
    }
    const char *candidate_fill_only = getenv(
        "TURBOCIDER_H3_CANDIDATE_FILL_ONLY");
    if (candidate_fill_only && *candidate_fill_only &&
        strcmp(candidate_fill_only, "0")) {
        if (!candidate_fill_enabled) {
            fprintf(stderr,
                    "candidate fill-only requires "
                    "TURBOCIDER_H3_CANDIDATE_FILL=1\n");
            h3_dit_free(dit);
            return 2;
        }
        printf("{\"schema\":\"turbocider-h3-candidate-fill-v1\","
               "\"block\":%u,\"slot\":%u,"
               "\"source_bytes\":%llu,\"content_bytes\":%llu,"
               "\"seconds\":%.9f}\n",
               candidate_fill.block, candidate_fill.slot,
               (unsigned long long)candidate_fill.source_bytes,
               (unsigned long long)candidate_fill.content_bytes,
               candidate_fill.seconds);
        h3_dit_free(dit);
        return 0;
    }
    int exact_enabled = 0;
    int exact_cancel_enabled = 0;
    exact_cancel_state exact_cancel = {0, UINT64_MAX};
    const char *exact_request = getenv("TURBOCIDER_H3_EXACT_STREAM");
    if (exact_request && *exact_request && strcmp(exact_request, "0")) {
        if (request_count != 1) {
            fprintf(stderr,
                    "H3 exact candidate is one-shot; REQUESTS must be 1\n");
            h3_dit_free(dit);
            return 2;
        }
        h3_dit_exact_stream_options_v1 options = {
            sizeof(options), H3_DIT_EXACT_STREAM_ABI_V1,
            1u, 1u, 1u, 1, NULL, NULL
        };
        const char *cancel_after = getenv(
            "TURBOCIDER_H3_EXACT_CANCEL_AFTER_QUERIES");
        if (cancel_after && *cancel_after) {
            if (!parse_u64(cancel_after, &exact_cancel.cancel_after)) {
                fprintf(stderr, "invalid exact cancel query count: %s\n",
                        cancel_after);
                h3_dit_free(dit);
                return 2;
            }
            options.cancel = exact_cancel_query;
            options.cancel_user = &exact_cancel;
            exact_cancel_enabled = 1;
        }
        h3_dit_exact_stream_options_v1 invalid = options;
        invalid.prefetch_distance = 0;
        if (h3_dit_enable_exact_streaming_v1(
                dit, &invalid, error, sizeof(error))) {
            fprintf(stderr, "invalid H3 exact options were accepted\n");
            h3_dit_free(dit);
            return 2;
        }
        if (!h3_dit_enable_exact_streaming_v1(
                dit, &options, error, sizeof(error))) {
            fprintf(stderr, "cannot enable H3 exact streaming: %s\n", error);
            h3_dit_free(dit);
            return 2;
        }
        if (h3_dit_enable_exact_streaming_v1(
                dit, &options, error, sizeof(error))) {
            fprintf(stderr, "duplicate H3 exact enable was accepted\n");
            h3_dit_free(dit);
            return 2;
        }
        exact_enabled = 1;
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
    request_result *requests = calloc((size_t)request_count, sizeof(*requests));
    if (!requests) {
        fprintf(stderr, "cannot allocate retained-request results\n");
        free(video);
        free(audio);
        h3_dit_free(dit);
        return 2;
    }
    int outputs_equal = 1;
    double denoise_seconds = 0.0;
    for (int request = 0; request < request_count; request++) {
        if (request &&
            !h3_dit_reset_run(dit, NULL, 0, NULL, 0,
                              error, sizeof(error))) {
            fprintf(stderr, "cannot reset retained H3 DiT request: %s\n", error);
            free(requests);
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

        h3_dit_streaming_info stream_before = {0}, stream_after = {0};
        h3_gpu_stats gpu_before = {0}, gpu_after = {0};
        (void)h3_dit_get_streaming_info(dit, &stream_before);
        (void)h3_dit_get_gpu_stats(dit, &gpu_before);
        double denoise_started = now_seconds();
        int ok = h3_dit_denoise_euler(
            dit, video, audio, 1, progress, NULL, error, sizeof(error));
        requests[request].denoise_seconds = now_seconds() - denoise_started;
        denoise_seconds += requests[request].denoise_seconds;
        (void)h3_dit_get_streaming_info(dit, &stream_after);
        (void)h3_dit_get_gpu_stats(dit, &gpu_after);
        if (!ok) {
            if (exact_cancel_enabled) {
                h3_dit_exact_streaming_info cancelled = {0};
                const int info_ok = h3_dit_get_exact_streaming_info(
                    dit, &cancelled);
                char destroy_error[1024] = {0};
                h3_dit *owner = dit;
                const int destroyed = h3_dit_destroy(
                    &owner, destroy_error, sizeof(destroy_error));
                const int expected = info_ok && cancelled.enabled &&
                    cancelled.poisoned && strstr(error, "cancelled") &&
                    destroyed && owner == NULL;
                printf("{\"schema\":\"turbocider-h3-exact-cancel-v1\","
                       "\"cancel_queries\":%llu,\"poisoned\":%s,"
                       "\"destroyed\":%s,\"error\":\"%s\"}\n",
                       (unsigned long long)exact_cancel.queries,
                       cancelled.poisoned ? "true" : "false",
                       destroyed && owner == NULL ? "true" : "false",
                       error);
                free(requests);
                free(video);
                free(audio);
                return expected ? 0 : 2;
            }
            fprintf(stderr, "H3 DiT retained probe request %d failed: %s\n",
                    request + 1, error);
            free(requests);
            free(video);
            free(audio);
            h3_dit_free(dit);
            return 2;
        }

        requests[request].finite = 1;
        for (size_t index = 0; index < video_elements; index++)
            if (!isfinite(video[index])) requests[request].finite = 0;
        for (size_t index = 0; index < audio_elements; index++)
            if (!isfinite(audio[index])) requests[request].finite = 0;
        requests[request].hash = UINT64_C(1469598103934665603);
        requests[request].hash = fnv1a(
            video, video_elements * sizeof(*video), requests[request].hash);
        requests[request].hash = fnv1a(
            audio, audio_elements * sizeof(*audio), requests[request].hash);
        if (request && requests[request].hash != requests[0].hash)
            outputs_equal = 0;
        requests[request].bytes_read =
            stream_after.bytes_read - stream_before.bytes_read;
        requests[request].read_seconds =
            stream_after.read_seconds - stream_before.read_seconds;
        requests[request].wait_seconds =
            stream_after.wait_seconds - stream_before.wait_seconds;
        requests[request].gpu_seconds =
            gpu_after.gpu_seconds - gpu_before.gpu_seconds;
        if (!write_latents(argv[3], request, request_count,
                           video, video_elements, audio, audio_elements)) {
            fprintf(stderr, "cannot write probe output: %s\n", strerror(errno));
            free(requests);
            free(video);
            free(audio);
            h3_dit_free(dit);
            return 2;
        }
    }
    h3_dit_streaming_info streaming = {0};
    h3_dit_exact_streaming_info exact = {0};
    h3_gpu_stats gpu = {0};
    (void)h3_dit_get_streaming_info(dit, &streaming);
    int exact_info_ok = h3_dit_get_exact_streaming_info(dit, &exact);
    (void)h3_dit_get_gpu_stats(dit, &gpu);
    if (exact_enabled) {
        const uint64_t expected_groups =
            (uint64_t)streaming.streamed_blocks * (uint64_t)sigmas.steps;
        char reset_error[256] = {0};
        if (!exact_info_ok || !exact.enabled || !exact.finished ||
            exact.poisoned || exact.completed_passes != (uint32_t)sigmas.steps ||
            exact.pool_creates != 1 || exact.slot_bundles != 2 ||
            exact.fills != expected_groups ||
            exact.groups_submitted != expected_groups ||
            h3_dit_reset_run(dit, NULL, 0, NULL, 0,
                             reset_error, sizeof(reset_error)) ||
            !strstr(reset_error, "one-shot")) {
            fprintf(stderr,
                    "H3 exact lifecycle/counter verification failed: %s\n",
                    reset_error);
            free(requests);
            free(video);
            free(audio);
            h3_dit_free(dit);
            return 2;
        }
    }
    double total_seconds = now_seconds() - total_started;
    const uint64_t logical_fills =
        (uint64_t)streaming.streamed_blocks * (uint64_t)sigmas.steps;
    const request_result *last_request = &requests[request_count - 1];
    const uint64_t request_bytes_loaded = exact_enabled ?
        exact.content_bytes_loaded : last_request->bytes_read;
    const double request_load_seconds = exact_enabled ?
        exact.refill_load_seconds : last_request->read_seconds;
    const double request_wait_seconds = exact_enabled ?
        exact.wait_seconds : last_request->wait_seconds;
    char *audit_json = NULL;
    char *audit_error = NULL;
    if (audit_snapshot && audit_snapshot(&audit_json, &audit_error)) {
        fprintf(stderr, "cannot capture streaming audit snapshot: %s\n",
                audit_error ? audit_error : "unknown error");
        if (audit_json) {
            if (string_free) string_free(audit_json); else free(audit_json);
            audit_json = NULL;
        }
    }
    if (audit_error) {
        if (string_free) string_free(audit_error); else free(audit_error);
        audit_error = NULL;
    }

    int finite = 1;
    for (int request = 0; request < request_count; request++)
        if (!requests[request].finite) finite = 0;
    printf("{\"schema\":\"turbocider-h3-dit-streaming-probe-v3\","
           "\"memory_budget_bytes\":%llu,\"request_count\":%d,"
           "\"candidate_fill_enabled\":%s,"
           "\"candidate_fill_block\":%u,"
           "\"candidate_fill_slot\":%u,"
           "\"candidate_fill_source_bytes\":%llu,"
           "\"candidate_fill_content_bytes\":%llu,"
           "\"candidate_fill_seconds\":%.6f,"
           "\"exact_enabled\":%s,\"exact_finished\":%s,"
           "\"exact_poisoned\":%s,\"exact_completed_passes\":%u,"
           "\"exact_pool_creates\":%llu,\"exact_slot_bundles\":%llu,"
           "\"exact_fills\":%llu,\"exact_content_bytes_loaded\":%llu,"
           "\"exact_groups_submitted\":%llu,"
           "\"exact_refill_load_seconds\":%.9f,"
           "\"exact_max_refill_seconds\":%.9f,"
           "\"exact_max_refill_block\":%d,"
           "\"exact_wait_seconds\":%.9f,"
           "\"finite\":%s,\"outputs_equal\":%s,"
           "\"video_elements\":%zu,\"audio_elements\":%zu,"
           "\"output_fnv1a64\":\"%016llx\","
           "\"load_seconds\":%.9f,\"denoise_seconds\":%.9f,"
           "\"total_seconds\":%.9f,\"pinned_blocks\":%u,"
           "\"streamed_blocks\":%u,\"block_bytes\":%llu,"
           "\"activation_reserve_bytes\":%llu,\"bytes_read\":%llu,"
           "\"read_seconds\":%.9f,\"wait_seconds\":%.9f,"
           "\"gpu_seconds\":%.9f,\"peak_live_bytes\":%llu,"
           "\"timings_seconds\":{\"request_wall\":%.9f,"
           "\"denoise\":%.9f},"
           "\"block_streaming\":{"
           "\"enabled\":true,\"implementation\":\"%s\","
           "\"request_bytes_loaded\":%llu,"
           "\"request_slot_allocations\":2,"
           "\"request_slot_refills\":%llu,"
           "\"request_slot_fills\":%llu,"
           "\"request_load_seconds\":%.9f,"
           "\"request_wait_seconds\":%.9f,"
           "\"actual_layout\":{"
           "\"stage\":\"denoiser\","
           "\"resident_prefix_blocks\":%u,"
           "\"block_group_size\":1,\"slot_count\":2,"
           "\"prefetch_distance\":1,\"io_workers\":1,"
           "\"group_count\":%u,\"pass_count\":%d,"
           "\"startup_policy\":\"prefetch_window_before_prefix\","
           "\"pass_transition\":\"carry_first_group\","
           "\"retention\":\"request\",\"reader_revision\":1,"
           "\"weight_format\":\"bf16-safetensors\","
           "\"kernel_revision\":\"h3-metal-dense-block-v1\","
           "\"conditioning_recipe\":\"h3-synthetic-text-adaln-v1\","
           "\"upsample_boundary\":\"dit-latent-output\"}},"
           "\"audit_snapshot\":%s,"
           "\"requests\":[",
           (unsigned long long)budget, request_count,
           candidate_fill_enabled ? "true" : "false",
           candidate_fill.block, candidate_fill.slot,
           (unsigned long long)candidate_fill.source_bytes,
           (unsigned long long)candidate_fill.content_bytes,
           candidate_fill.seconds,
           exact_enabled ? "true" : "false",
           exact.finished ? "true" : "false",
           exact.poisoned ? "true" : "false",
           exact.completed_passes,
           (unsigned long long)exact.pool_creates,
           (unsigned long long)exact.slot_bundles,
           (unsigned long long)exact.fills,
           (unsigned long long)exact.content_bytes_loaded,
           (unsigned long long)exact.groups_submitted,
           exact.refill_load_seconds, exact.max_refill_seconds,
           exact.max_refill_block, exact.wait_seconds,
           finite ? "true" : "false", outputs_equal ? "true" : "false",
           video_elements, audio_elements,
           (unsigned long long)requests[request_count - 1].hash,
           load_seconds, denoise_seconds, total_seconds,
           streaming.pinned_blocks, streaming.streamed_blocks,
           (unsigned long long)streaming.block_bytes,
           (unsigned long long)streaming.activation_reserve_bytes,
           (unsigned long long)streaming.bytes_read,
           streaming.read_seconds, streaming.wait_seconds, gpu.gpu_seconds,
           (unsigned long long)gpu.peak_live_bytes,
           total_seconds, denoise_seconds,
           exact_enabled ? "generic-stage-executor-v3" :
                           "h3-legacy-pthread-stream-v1",
           (unsigned long long)request_bytes_loaded,
           (unsigned long long)logical_fills,
           (unsigned long long)logical_fills,
           request_load_seconds, request_wait_seconds,
           streaming.pinned_blocks, streaming.streamed_blocks,
           sigmas.steps, audit_json ? audit_json : "null");

    /* Per-request values are deltas from the retained DiT's cumulative
     * counters.  The top-level values remain cumulative for compatibility
     * with the original one-request probe. */
    for (int request = 0; request < request_count; request++) {
        if (request) putchar(',');
        printf("{\"index\":%d,\"finite\":%s,"
               "\"output_fnv1a64\":\"%016llx\","
               "\"denoise_seconds\":%.9f,\"bytes_read_delta\":%llu,"
               "\"read_seconds_delta\":%.9f,"
               "\"wait_seconds_delta\":%.9f,"
               "\"gpu_seconds_delta\":%.9f}",
               request + 1, requests[request].finite ? "true" : "false",
               (unsigned long long)requests[request].hash,
               requests[request].denoise_seconds,
               (unsigned long long)requests[request].bytes_read,
               requests[request].read_seconds,
               requests[request].wait_seconds,
               requests[request].gpu_seconds);
    }
    puts("]}");

    if (audit_json) {
        if (string_free) string_free(audit_json); else free(audit_json);
    }

    free(requests);
    free(video);
    free(audio);
    h3_dit_free(dit);
    return finite ? 0 : 1;
}
