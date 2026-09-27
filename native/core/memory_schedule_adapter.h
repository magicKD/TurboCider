#ifndef TC_MEMORY_SCHEDULE_ADAPTER_H
#define TC_MEMORY_SCHEDULE_ADAPTER_H

#include "memory_schedule_c.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int tc_memory_schedule_hooks_valid_v1(
        const tc_memory_schedule_hooks_v1 *hooks) {
    return hooks &&
        hooks->struct_size >= sizeof(tc_memory_schedule_hooks_v1) &&
        hooks->version == TC_MEMORY_SCHEDULE_HOOKS_VERSION_1 &&
        hooks->user && hooks->emit;
}

static inline int tc_memory_schedule_emit_fields_v1(
        const tc_memory_schedule_hooks_v1 *hooks,
        uint32_t stage, uint32_t action,
        uint32_t step, uint32_t block, uint32_t tile,
        uint32_t branch, uint32_t slot, uint32_t flags,
        char *error, size_t error_size) {
    if (!hooks) return 1;
    if (!tc_memory_schedule_hooks_valid_v1(hooks)) {
        if (error && error_size)
            snprintf(error, error_size, "%s",
                     "memory_schedule_invalid: unsupported schedule hook ABI");
        return 0;
    }
    tc_memory_schedule_event_v1 event;
    event.struct_size = sizeof(event);
    event.version = TC_MEMORY_SCHEDULE_EVENT_VERSION_1;
    event.stage = stage;
    event.action = action;
    event.step = step;
    event.block = block;
    event.tile = tile;
    event.branch = branch;
    event.slot = slot;
    event.flags = flags;
    if (hooks->emit(hooks->user, &event, error, error_size)) return 1;
    if (error && error_size && !error[0])
        snprintf(error, error_size, "%s",
                 "memory_lifetime_violation: schedule callback rejected event");
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
