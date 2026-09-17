#include "h3_gpu.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <string>
#include <unistd.h>

extern "C" const char *h3_runtime_getenv(const char *) { return nullptr; }

struct HookState {
    uint64_t budget = 0;
    uint64_t reserved = 0;
    uint64_t committed = 0;
    uint64_t peak = 0;
    uint64_t reserve_calls = 0;
    uint64_t commit_calls = 0;
    uint64_t cancel_calls = 0;
    uint64_t release_calls = 0;
    uint64_t retire_calls = 0;
    uint64_t complete_calls = 0;
    uint32_t last_memory_class = 0;
    std::string last_tag;
    bool reject_next_commit = false;
};

struct HookToken {
    HookState *state = nullptr;
    uint64_t upper = 0;
    uint64_t actual = 0;
    bool committed = false;
    bool retired = false;
    bool completed = false;
};

struct CompletionState {
    std::atomic<uint32_t> next{0};
    std::atomic<uint32_t> completed{0};
    uint32_t queues[2] = {};
    uint64_t sequences[2] = {};
    int statuses[2] = {};
};

static void command_complete(void *opaque, uint32_t queue,
                             uint64_t sequence, int status) {
    auto *state = static_cast<CompletionState *>(opaque);
    assert(state != nullptr);
    const uint32_t index = state->next.fetch_add(
        1, std::memory_order_relaxed);
    assert(index < 2);
    state->queues[index] = queue;
    state->sequences[index] = sequence;
    state->statuses[index] = status;
    state->completed.fetch_add(1, std::memory_order_release);
}

struct CancelState {
    uint32_t queries = 0;
    uint32_t allow_queries = UINT32_MAX;
};

static int cancel_query(const void *opaque) {
    auto *state = const_cast<CancelState *>(
        static_cast<const CancelState *>(opaque));
    return state->queries++ >= state->allow_queries ? 1 : 0;
}

static int reserve_hook(void *opaque, uint32_t memory_class, uint64_t upper,
                        const char *tag, void **out, char *error,
                        size_t error_size) {
    auto *state = static_cast<HookState *>(opaque);
    if (out) *out = nullptr;
    if (!state || !out || !upper || state->reserved > state->budget ||
        state->committed > state->budget - state->reserved ||
        upper > state->budget - state->reserved - state->committed) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_budget_too_small: fake H3 hook");
        return 0;
    }
    auto *token = new (std::nothrow) HookToken{state, upper, 0, false};
    if (!token) return 0;
    state->reserve_calls++;
    state->reserved += upper;
    state->last_memory_class = memory_class;
    state->last_tag = tag ? tag : "";
    *out = token;
    return 1;
}

static int commit_hook(void *, void *opaque_token, uint64_t domain,
                       uint64_t handle, uint64_t actual, uint64_t generation,
                       char *error, size_t error_size) {
    auto *token = static_cast<HookToken *>(opaque_token);
    if (!token || token->committed || !domain || !handle || !generation ||
        !actual || actual > token->upper || token->state->reject_next_commit) {
        if (token && token->state->reject_next_commit)
            token->state->reject_next_commit = false;
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_lifetime_violation: fake H3 commit");
        return 0;
    }
    HookState *state = token->state;
    assert(state->reserved >= token->upper);
    state->reserved -= token->upper;
    state->committed += actual;
    if (state->committed > state->peak) state->peak = state->committed;
    state->commit_calls++;
    token->actual = actual;
    token->committed = true;
    return 1;
}

static void cancel_hook(void *, void *opaque_token) {
    auto *token = static_cast<HookToken *>(opaque_token);
    if (!token) return;
    if (!token->committed) {
        assert(token->state->reserved >= token->upper);
        token->state->reserved -= token->upper;
        token->state->cancel_calls++;
    }
    delete token;
}

static void release_hook(void *, void *opaque_token) {
    auto *token = static_cast<HookToken *>(opaque_token);
    assert(token && token->committed);
    if (!token->retired) {
        assert(token->state->committed >= token->actual);
        token->state->committed -= token->actual;
    } else {
        assert(token->completed);
    }
    token->state->release_calls++;
    delete token;
}

static int retire_hook(void *, void *opaque_token, uint32_t stage_id,
                       uint32_t slot_id, char *error, size_t error_size) {
    auto *token = static_cast<HookToken *>(opaque_token);
    if (!token || !token->committed || token->retired || !stage_id ||
        !slot_id) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_lifetime_violation: fake H3 retire");
        return 0;
    }
    token->retired = true;
    token->state->retire_calls++;
    return 1;
}

static void complete_hook(void *, void *opaque_token, int status) {
    auto *token = static_cast<HookToken *>(opaque_token);
    assert(token && token->retired && !token->completed && status == 0);
    assert(token->state->committed >= token->actual);
    token->state->committed -= token->actual;
    token->state->complete_calls++;
    token->completed = true;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    HookState state{2048};
    h3_gpu_memory_hooks hooks{};
    hooks.struct_size = sizeof(hooks);
    hooks.version = 1;
    hooks.user = &state;
    hooks.reserve = reserve_hook;
    hooks.commit = commit_hook;
    hooks.cancel = cancel_hook;
    hooks.release = release_hook;
    hooks.retire = retire_hook;
    hooks.complete = complete_hook;
    h3_gpu_options options{};
    options.struct_size = sizeof(options);
    options.version = 1;
    options.memory_hooks = &hooks;
    options.memory_allocator_domain = 7;
    options.memory_generation = 11;

    char error[1024] = {};
    h3_gpu *gpu = h3_gpu_create_with_options(
        argv[1], &options, error, sizeof(error));
    if (!gpu && std::strstr(error, "cannot initialize Metal")) {
        std::puts("SKIP: no Metal device is available");
        return 0;
    }
    assert(gpu != nullptr);

    h3_gpu_tensor *activation = h3_gpu_tensor_new_classified(
        gpu, 256, H3_GPU_F32, H3_GPU_MEMORY_ACTIVATION,
        "h3.test.activation");
    assert(activation != nullptr);
    assert(state.committed == 1024 && state.reserved == 0);
    assert(state.reserve_calls == 1 && state.commit_calls == 1);
    assert(state.last_memory_class == H3_GPU_MEMORY_ACTIVATION);
    assert(state.last_tag == "h3.test.activation");

    h3_gpu_tensor *denied = h3_gpu_tensor_new_classified(
        gpu, 300, H3_GPU_F32, H3_GPU_MEMORY_OUTPUT, "h3.test.denied");
    assert(denied == nullptr);
    assert(std::strstr(h3_gpu_error(gpu), "memory_budget_too_small") != nullptr);
    assert(state.committed == 1024 && state.reserved == 0);

    state.reject_next_commit = true;
    h3_gpu_tensor *commit_denied = h3_gpu_tensor_new_classified(
        gpu, 64, H3_GPU_F32, H3_GPU_MEMORY_CONVERSION_SCRATCH,
        "h3.test.commit_denied");
    assert(commit_denied == nullptr);
    assert(state.cancel_calls == 1 && state.reserved == 0);
    assert(state.committed == 1024);

    h3_gpu_tensor_free(activation);
    assert(state.committed == 0 && state.release_calls == 1);

    const float uploaded_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    h3_gpu_tensor *uploaded = h3_gpu_tensor_from_f32_classified(
        gpu, uploaded_values, 4, H3_GPU_MEMORY_CONDITIONING,
        "h3.test.uploaded_conditioning");
    assert(uploaded != nullptr && state.committed == sizeof(uploaded_values));
    assert(state.last_memory_class == H3_GPU_MEMORY_CONDITIONING);
    assert(state.last_tag == "h3.test.uploaded_conditioning");
    h3_gpu_tensor_free(uploaded);
    assert(state.committed == 0 && state.release_calls == 2);

    const uint16_t reader_values[] = {
        0x3f80, 0x4000, 0x4040, 0x4080,
        0x40a0, 0x40c0, 0x40e0, 0x4100,
    };
    h3_gpu_tensor *reader_source = h3_gpu_tensor_from_bf16_classified(
        gpu, reader_values, 8, H3_GPU_MEMORY_REFILL_SLOT,
        "h3.test.reader_source");
    h3_gpu_tensor *reader_output = h3_gpu_tensor_new_classified(
        gpu, 8, H3_GPU_BF16, H3_GPU_MEMORY_OUTPUT,
        "h3.test.reader_output");
    assert(reader_source != nullptr && reader_output != nullptr);
    CompletionState completion_state;
    assert(h3_gpu_begin(gpu));
    assert(h3_gpu_copy_bf16(
        gpu, reader_output, 0, reader_source, 0, 4));
    h3_gpu_completion_v1 first_completion{
        sizeof(h3_gpu_completion_v1), H3_GPU_COMPLETION_ABI_V1,
        &completion_state, 7, 41, command_complete};
    assert(h3_gpu_continue_with_completion(gpu, &first_completion));
    assert(h3_gpu_copy_bf16(
        gpu, reader_output, 4, reader_source, 4, 4));
    h3_gpu_completion_v1 second_completion{
        sizeof(h3_gpu_completion_v1), H3_GPU_COMPLETION_ABI_V1,
        &completion_state, 7, 42, command_complete};
    assert(h3_gpu_continue_with_completion(gpu, &second_completion));
    assert(h3_gpu_flush_and_drain(gpu));
    assert(h3_gpu_flush_and_drain(gpu));
    assert(completion_state.completed.load(std::memory_order_acquire) == 2);
    assert(completion_state.queues[0] == 7 &&
           completion_state.queues[1] == 7);
    assert(completion_state.sequences[0] == 41 &&
           completion_state.sequences[1] == 42);
    assert(completion_state.statuses[0] == 0 &&
           completion_state.statuses[1] == 0);
    uint16_t copied_values[8] = {};
    assert(h3_gpu_tensor_read_bf16(reader_output, copied_values, 8));
    assert(std::memcmp(copied_values, reader_values,
                       sizeof(reader_values)) == 0);
    h3_gpu_tensor_free(reader_output);
    h3_gpu_tensor_free(reader_source);
    assert(state.committed == 0 && state.release_calls == 4);

    char stream_path[] = "/private/tmp/tc-h3-stream-XXXXXX";
    const int stream_fd = mkstemp(stream_path);
    assert(stream_fd >= 0);
    uint16_t stream_values[32] = {};
    for (uint16_t index = 0; index < 32; ++index)
        stream_values[index] = (uint16_t)(0x3f80u + index);
    assert(write(stream_fd, stream_values, sizeof(stream_values)) ==
           (ssize_t)sizeof(stream_values));
    assert(close(stream_fd) == 0);
    h3_gpu_tensor *stream_target = h3_gpu_tensor_new_classified(
        gpu, 32, H3_GPU_BF16, H3_GPU_MEMORY_REFILL_SLOT,
        "h3.test.cancellable_stream");
    assert(stream_target != nullptr);
    uint64_t bytes_read = 0;
    char stream_error[256] = {};
    CancelState no_cancel;
    assert(h3_gpu_tensor_stream_file_bf16_cancellable(
        stream_target, stream_path, 0, 32, 8, cancel_query, &no_cancel,
        &bytes_read, stream_error, sizeof(stream_error)));
    assert(bytes_read == sizeof(stream_values));
    uint16_t streamed_values[32] = {};
    assert(h3_gpu_tensor_read_bf16(stream_target, streamed_values, 32));
    assert(std::memcmp(streamed_values, stream_values,
                       sizeof(stream_values)) == 0);
    CancelState cancel_after_one{0, 1};
    bytes_read = 0;
    assert(!h3_gpu_tensor_stream_file_bf16_cancellable(
        stream_target, stream_path, 0, 32, 8, cancel_query,
        &cancel_after_one, &bytes_read, stream_error,
        sizeof(stream_error)));
    assert(bytes_read == 8);
    assert(std::strstr(stream_error, "cancelled") != nullptr);
    h3_gpu_tensor_free(stream_target);
    assert(unlink(stream_path) == 0);
    assert(state.committed == 0 && state.release_calls == 5);

    h3_gpu_tensor *deferred = h3_gpu_tensor_new_classified(
        gpu, 128, H3_GPU_F32, H3_GPU_MEMORY_OUTPUT,
        "h3.test.deferred");
    assert(deferred != nullptr && state.committed == 512);
    assert(h3_gpu_begin(gpu));
    h3_gpu_tensor *release_list[] = {deferred};
    assert(h3_gpu_continue_releasing(gpu, release_list, 1));
    assert(state.committed == 512 && state.release_calls == 5);
    assert(state.retire_calls == 1 && state.complete_calls == 0);
    assert(h3_gpu_submit(gpu));
    assert(state.committed == 0 && state.release_calls == 6);
    assert(state.retire_calls == 1 && state.complete_calls == 1);

    assert(state.peak == 1024);
    h3_gpu_free(gpu);
    std::puts("H3 GPU memory hooks passed");
    return 0;
}
