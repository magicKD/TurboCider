#include "memory_schedule_adapter.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    unsigned calls;
    tc_memory_schedule_event_v1 last;
    int accept;
    int write_error;
} capture;

static int emit(void *opaque, const tc_memory_schedule_event_v1 *event,
                char *error, size_t error_size) {
    capture *state = opaque;
    state->calls++;
    state->last = *event;
    if (state->accept) return 1;
    if (state->write_error && error && error_size)
        snprintf(error, error_size, "%s", "synthetic callback rejection");
    return 0;
}

static tc_memory_schedule_hooks_v1 hooks(capture *state) {
    tc_memory_schedule_hooks_v1 result = {0};
    result.struct_size = sizeof(result);
    result.version = TC_MEMORY_SCHEDULE_HOOKS_VERSION_1;
    result.user = state;
    result.emit = emit;
    return result;
}

static void test_disabled_is_inert(void) {
    char error[64] = "unchanged";
    assert(tc_memory_schedule_emit_fields_v1(
        NULL, TC_MEMORY_STAGE_TEXT, TC_MEMORY_ACTION_BEGIN,
        TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE,
        TC_MEMORY_BRANCH_TEXT, TC_MEMORY_INDEX_NONE, 0,
        error, sizeof(error)));
    assert(strcmp(error, "unchanged") == 0);
}

static void test_event_is_forwarded_exactly(void) {
    capture state = {.accept = 1};
    tc_memory_schedule_hooks_v1 value = hooks(&state);
    char error[64] = {0};
    assert(tc_memory_schedule_emit_fields_v1(
        &value, TC_MEMORY_STAGE_DENOISER, TC_MEMORY_ACTION_COMPUTE,
        7, 11, 3, TC_MEMORY_BRANCH_VIDEO, 1,
        TC_MEMORY_EVENT_PREFETCH_ALLOWED, error, sizeof(error)));
    assert(state.calls == 1);
    assert(state.last.struct_size == sizeof(state.last));
    assert(state.last.version == TC_MEMORY_SCHEDULE_EVENT_VERSION_1);
    assert(state.last.stage == TC_MEMORY_STAGE_DENOISER);
    assert(state.last.action == TC_MEMORY_ACTION_COMPUTE);
    assert(state.last.step == 7 && state.last.block == 11);
    assert(state.last.tile == 3 && state.last.slot == 1);
    assert(state.last.branch == TC_MEMORY_BRANCH_VIDEO);
    assert(state.last.flags == TC_MEMORY_EVENT_PREFETCH_ALLOWED);
    assert(error[0] == '\0');
}

static void test_invalid_hook_and_callback_errors(void) {
    capture state = {0};
    tc_memory_schedule_hooks_v1 value = hooks(&state);
    char error[128] = {0};
    value.version = 99;
    assert(!tc_memory_schedule_emit_fields_v1(
        &value, TC_MEMORY_STAGE_TEXT, TC_MEMORY_ACTION_BEGIN,
        TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE,
        TC_MEMORY_BRANCH_TEXT, TC_MEMORY_INDEX_NONE, 0,
        error, sizeof(error)));
    assert(strstr(error, "unsupported schedule hook ABI"));
    assert(state.calls == 0);

    value = hooks(&state);
    memset(error, 0, sizeof(error));
    assert(!tc_memory_schedule_emit_fields_v1(
        &value, TC_MEMORY_STAGE_TEXT, TC_MEMORY_ACTION_BEGIN,
        TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE,
        TC_MEMORY_BRANCH_TEXT, TC_MEMORY_INDEX_NONE, 0,
        error, sizeof(error)));
    assert(strstr(error, "schedule callback rejected event"));
    assert(state.calls == 1);

    state.write_error = 1;
    memset(error, 0, sizeof(error));
    assert(!tc_memory_schedule_emit_fields_v1(
        &value, TC_MEMORY_STAGE_TEXT, TC_MEMORY_ACTION_END,
        TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE, TC_MEMORY_INDEX_NONE,
        TC_MEMORY_BRANCH_TEXT, TC_MEMORY_INDEX_NONE, 0,
        error, sizeof(error)));
    assert(strcmp(error, "synthetic callback rejection") == 0);
    assert(state.calls == 2);
}

int main(void) {
    test_disabled_is_inert();
    test_event_is_forwarded_exactly();
    test_invalid_hook_and_callback_errors();
    puts("memory schedule adapter tests passed");
    return 0;
}
