#include "h3_gpu.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

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

    h3_gpu_tensor *deferred = h3_gpu_tensor_new_classified(
        gpu, 128, H3_GPU_F32, H3_GPU_MEMORY_OUTPUT,
        "h3.test.deferred");
    assert(deferred != nullptr && state.committed == 512);
    assert(h3_gpu_begin(gpu));
    h3_gpu_tensor *release_list[] = {deferred};
    assert(h3_gpu_continue_releasing(gpu, release_list, 1));
    assert(state.committed == 512 && state.release_calls == 2);
    assert(state.retire_calls == 1 && state.complete_calls == 0);
    assert(h3_gpu_submit(gpu));
    assert(state.committed == 0 && state.release_calls == 3);
    assert(state.retire_calls == 1 && state.complete_calls == 1);

    assert(state.peak == 1024);
    h3_gpu_free(gpu);
    std::puts("H3 GPU memory hooks passed");
    return 0;
}
