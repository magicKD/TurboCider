#include "ltx_gpu.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

struct HookState {
    uint64_t budget = 0;
    uint64_t reserved = 0;
    uint64_t committed = 0;
    uint64_t pending = 0;
    uint64_t peak = 0;
    uint64_t reserve_calls = 0;
    uint64_t commit_calls = 0;
    uint64_t cancel_calls = 0;
    uint64_t release_calls = 0;
    uint64_t retire_calls = 0;
    uint64_t complete_calls = 0;
    uint32_t last_queue_id = 0;
    uint32_t last_stage_id = 0;
    uint32_t last_slot_id = 0;
    int last_completion_status = -1;
};

struct HookToken {
    HookState *state = nullptr;
    uint64_t upper = 0;
    uint64_t actual = 0;
    bool committed = false;
    bool retired = false;
    bool completed = false;
    uint32_t queue_id = 0;
};

static int reserve_hook(void *opaque, uint32_t, uint64_t upper,
                        const char *, void **out, char *error,
                        size_t error_size) {
    auto *state = static_cast<HookState *>(opaque);
    if (out) *out = nullptr;
    if (!state || !out || !upper ||
        state->committed > state->budget - state->reserved ||
        upper > state->budget - state->reserved - state->committed) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_budget_too_small: fake LTX hook");
        return 0;
    }
    auto *token = new (std::nothrow) HookToken{state, upper, 0, false};
    if (!token) return 0;
    state->reserve_calls++;
    state->reserved += upper;
    *out = token;
    return 1;
}

static int commit_hook(void *, void *opaque_token, uint64_t domain,
                       uint64_t handle, uint64_t actual, uint64_t generation,
                       char *error, size_t error_size) {
    auto *token = static_cast<HookToken *>(opaque_token);
    if (!token || token->committed || !domain || !handle || !generation ||
        !actual || actual > token->upper) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_lifetime_violation: fake LTX commit");
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
    if (token->retired) {
        assert(token->completed);
    } else {
        assert(token->state->committed >= token->actual);
        token->state->committed -= token->actual;
    }
    token->state->release_calls++;
    delete token;
}

static int retire_hook(void *, void *opaque_token, uint32_t queue_id,
                       uint32_t stage_id, uint32_t slot_id,
                       char *error, size_t error_size) {
    auto *token = static_cast<HookToken *>(opaque_token);
    if (!token || !token->committed || token->retired || !queue_id ||
        !stage_id || !slot_id) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "memory_lifetime_violation: fake LTX retire");
        return 0;
    }
    HookState *state = token->state;
    assert(state->committed >= token->actual);
    state->committed -= token->actual;
    state->pending += token->actual;
    state->retire_calls++;
    state->last_queue_id = queue_id;
    state->last_stage_id = stage_id;
    state->last_slot_id = slot_id;
    token->retired = true;
    token->queue_id = queue_id;
    return 1;
}

static void complete_hook(void *, void *opaque_token, uint32_t queue_id,
                          int status) {
    auto *token = static_cast<HookToken *>(opaque_token);
    assert(token && token->retired && !token->completed);
    assert(queue_id == token->queue_id);
    assert(token->state->pending >= token->actual);
    token->state->pending -= token->actual;
    token->state->complete_calls++;
    token->state->last_completion_status = status;
    token->completed = true;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    char error[1024] = {};
    ltx_gpu *gpu = ltx_gpu_create(argv[1], error, sizeof(error));
    if (!gpu && std::strstr(error, "no Metal device")) {
        std::puts("SKIP: no Metal device is available");
        return 0;
    }
    assert(gpu != nullptr);

    HookState state{4096};
    ltx_gpu_memory_hooks hooks{};
    hooks.struct_size = sizeof(hooks);
    hooks.version = 2;
    hooks.user = &state;
    hooks.reserve = reserve_hook;
    hooks.commit = commit_hook;
    hooks.cancel = cancel_hook;
    hooks.release = release_hook;
    hooks.retire = retire_hook;
    hooks.complete = complete_hook;

    ltx_gpu_memory_hooks legacy = hooks;
    legacy.version = 1;
    legacy.retire = nullptr;
    legacy.complete = nullptr;
    assert(ltx_gpu_set_memory_hooks(
        gpu, &legacy, 7, 11, error, sizeof(error)));

    ltx_gpu_memory_hooks incomplete = hooks;
    incomplete.complete = nullptr;
    assert(!ltx_gpu_set_memory_hooks(
        gpu, &incomplete, 7, 11, error, sizeof(error)));
    assert(std::strstr(error, "invalid constrained GPU memory hooks"));

    assert(ltx_gpu_set_memory_hooks_for_queue(
        gpu, &hooks, 7, 11, LTX_GPU_MEMORY_QUEUE_VIDEO,
        error, sizeof(error)));

    ltx_gpu_buffer *buffer = ltx_gpu_buffer_new_classified(
        gpu, 1024, LTX_GPU_MEMORY_ACTIVATION, "unit",
        error, sizeof(error));
    assert(buffer != nullptr);
    assert(state.committed == 1024);
    assert(state.reserve_calls == 1 && state.commit_calls == 1);

    ltx_gpu_buffer *alias = ltx_gpu_buffer_retain(buffer);
    assert(alias == buffer);
    ltx_gpu_buffer_free(alias);
    assert(state.committed == 1024 && state.release_calls == 0);

    ltx_gpu_buffer *denied = ltx_gpu_buffer_new(
        gpu, 4096, error, sizeof(error));
    assert(denied == nullptr);
    assert(std::strstr(error, "memory_budget_too_small") != nullptr);
    assert(state.committed == 1024 && state.reserved == 0);

    ltx_gpu_buffer_free(buffer);
    assert(state.committed == 0 && state.release_calls == 1);
    assert(state.peak == 1024);

    constexpr uint32_t elements = 128;
    ltx_gpu_buffer *input = ltx_gpu_buffer_new_classified(
        gpu, elements * sizeof(float), LTX_GPU_MEMORY_ACTIVATION,
        "unit.input", error, sizeof(error));
    ltx_gpu_buffer *output = ltx_gpu_buffer_new_classified(
        gpu, elements * sizeof(float), LTX_GPU_MEMORY_ACTIVATION,
        "unit.output", error, sizeof(error));
    assert(input && output);
    assert(state.committed == 2 * elements * sizeof(float));
    assert(ltx_gpu_batch_begin(gpu, error, sizeof(error)));
    assert(ltx_gpu_scale_f32(
        gpu, output, input, 1.0f, elements, error, sizeof(error)));
    ltx_gpu_buffer_free(input);
    ltx_gpu_buffer_free(output);
    assert(state.committed == 0);
    assert(state.pending == 2 * elements * sizeof(float));
    assert(state.retire_calls == 2);
    assert(state.complete_calls == 0);
    assert(ltx_gpu_batch_end(gpu, error, sizeof(error)));
    assert(state.pending == 0);
    assert(state.complete_calls == 2);
    assert(state.last_queue_id == LTX_GPU_MEMORY_QUEUE_VIDEO);
    assert(state.last_stage_id != 0 && state.last_slot_id != 0);
    assert(state.last_completion_status == 0);
    assert(state.release_calls == 3);

    assert(ltx_gpu_drain(gpu, error, sizeof(error)));
    ltx_gpu_free(gpu);
    std::puts("LTX GPU memory hooks passed");
    return 0;
}
