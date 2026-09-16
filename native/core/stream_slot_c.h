#ifndef TC_STREAM_SLOT_C_H
#define TC_STREAM_SLOT_C_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TC_STREAM_SLOT_ABI_V1 1u
#define TC_STREAM_MAX_READER_QUEUES 8u

typedef struct {
    uint32_t stage, pass, step, group;
} tc_stream_work_item_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t pool, slot;
    uint64_t request_generation, content_generation;
    tc_stream_work_item_v1 item;
} tc_stream_slot_ticket_v1;

typedef struct {
    uint32_t queue;
    uint64_t sequence;
} tc_stream_reader_fence_v1;

// Completion callbacks only publish POD records. They never mutate slot state.
typedef enum {
    TC_STREAM_FILL_COMPLETE = 1,
    TC_STREAM_READER_COMPLETE = 2,
} tc_stream_completion_kind_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t kind;
    int32_t status;
    tc_stream_slot_ticket_v1 ticket;
    tc_stream_reader_fence_v1 fence;
    uint64_t bytes;
} tc_stream_completion_v1;

typedef struct {
    uint32_t group, slot, block_count;
    const uint32_t *blocks;
    uint64_t content_bytes;
} tc_stream_group_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage, pool, slot_count, prefetch_distance, io_workers, pass_count;
    uint64_t request_generation;
    const uint64_t *slot_capacity_bytes;
    uint32_t group_count;
    const tc_stream_group_v1 *groups;
} tc_stream_stage_plan_v1;

typedef struct {
    void *user;
    /* Copy this sink if used asynchronously; never retain the stack address. */
    int (*post)(void *user, const tc_stream_slot_ticket_v1 *,
                tc_stream_reader_fence_v1, int32_t status);
} tc_stream_completion_sink_v1;

typedef struct {
    uint32_t count;
    tc_stream_reader_fence_v1 fences[TC_STREAM_MAX_READER_QUEUES];
} tc_stream_reader_set_v1;

typedef int (*tc_stream_cancel_query_v1)(const void *);
/* Callbacks report errors through their return value and error buffer.
 * destroy_pool is called only after drain succeeds and must not throw. */
typedef struct {
    uint32_t struct_size, version;
    void *user;
    int (*allocate_slot)(void *, uint32_t slot, uint64_t capacity, char *, size_t);
    void (*destroy_pool)(void *);
    int (*fill)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
                tc_stream_cancel_query_v1, const void *cancel_user,
                uint64_t *content_bytes, char *, size_t);
    int (*prefix)(void *, uint32_t pass, char *, size_t);
    int (*prepare)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *, char *, size_t);
    int (*encode)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
                  const tc_stream_completion_sink_v1 *, tc_stream_reader_set_v1 *, char *, size_t);
    int (*drain)(void *, char *, size_t);
} tc_stream_adapter_v1;

typedef struct {
    uint64_t pool_creates, slot_bundles, fills, content_bytes_loaded, groups_submitted;
} tc_stream_counters_v1;
typedef struct tc_stream_executor tc_stream_executor;

/* Metadata arrays and callback table are copied; adapter.user must outlive the
 * handle. On failure a non-null output is quarantined and must NOT be freed
 * until tc_stream_executor_destroy succeeds. All calls except cancel are owner-only. */
int tc_stream_executor_create_v1(const tc_stream_stage_plan_v1 *, const tc_stream_adapter_v1 *,
                                 tc_stream_executor **out, char *, size_t);
int tc_stream_executor_run_pass(tc_stream_executor *, uint32_t pass, uint32_t step, char *, size_t);
int tc_stream_executor_finish(tc_stream_executor *, char *, size_t);
int tc_stream_executor_counters(tc_stream_executor *, tc_stream_counters_v1 *, char *, size_t);
void tc_stream_executor_cancel(tc_stream_executor *);
/* Stops workers and drains before freeing. Returns 0 and retains *handle if
 * safety cannot be proved; caller must quarantine the owning model session. */
int tc_stream_executor_destroy(tc_stream_executor **handle, char *, size_t);

#ifdef __cplusplus
}
#endif
#endif
