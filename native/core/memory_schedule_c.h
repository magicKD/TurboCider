#ifndef TC_MEMORY_SCHEDULE_C_H
#define TC_MEMORY_SCHEDULE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TC_MEMORY_INDEX_NONE UINT32_MAX
#define TC_MEMORY_SCHEDULE_EVENT_VERSION_1 1u
#define TC_MEMORY_SCHEDULE_HOOKS_VERSION_1 1u

typedef enum {
    TC_MEMORY_STAGE_ADMISSION = 1,
    TC_MEMORY_STAGE_TEXT = 2,
    TC_MEMORY_STAGE_CONDITIONING_HANDOFF = 3,
    TC_MEMORY_STAGE_DENOISER_LOAD = 4,
    TC_MEMORY_STAGE_DENOISER = 5,
    TC_MEMORY_STAGE_LATENT_HANDOFF = 6,
    TC_MEMORY_STAGE_VIDEO_VAE = 7,
    TC_MEMORY_STAGE_AUDIO_VAE = 8,
    TC_MEMORY_STAGE_EXPORT = 9,
    TC_MEMORY_STAGE_TERMINAL_DRAIN = 10,
} tc_memory_schedule_stage_v1;

typedef enum {
    TC_MEMORY_ACTION_BEGIN = 1,
    TC_MEMORY_ACTION_PREFETCH = 2,
    TC_MEMORY_ACTION_UPLOAD = 3,
    TC_MEMORY_ACTION_COMPUTE = 4,
    TC_MEMORY_ACTION_LAST_USE = 5,
    TC_MEMORY_ACTION_RETIRE = 6,
    TC_MEMORY_ACTION_DRAIN = 7,
    TC_MEMORY_ACTION_END = 8,
} tc_memory_schedule_action_v1;

typedef enum {
    TC_MEMORY_BRANCH_COMMON = 1,
    TC_MEMORY_BRANCH_TEXT = 2,
    TC_MEMORY_BRANCH_VIDEO = 3,
    TC_MEMORY_BRANCH_AUDIO = 4,
    TC_MEMORY_BRANCH_VAE = 5,
    TC_MEMORY_BRANCH_EXPORT = 6,
} tc_memory_schedule_branch_v1;

enum {
    TC_MEMORY_EVENT_SAFE_POINT = 1u << 0,
    TC_MEMORY_EVENT_GPU_DRAINED = 1u << 1,
    TC_MEMORY_EVENT_PREFETCH_ALLOWED = 1u << 2,
    TC_MEMORY_EVENT_TERMINAL = 1u << 3,
};

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint32_t stage;
    uint32_t action;
    uint32_t step;
    uint32_t block;
    uint32_t tile;
    uint32_t branch;
    uint32_t slot;
    uint32_t flags;
} tc_memory_schedule_event_v1;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*emit)(void *user, const tc_memory_schedule_event_v1 *event,
                char *error, size_t error_size);
} tc_memory_schedule_hooks_v1;

#ifdef __cplusplus
}
#endif

#endif
