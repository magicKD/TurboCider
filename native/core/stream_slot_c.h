#ifndef TC_STREAM_SLOT_C_H
#define TC_STREAM_SLOT_C_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TC_STREAM_SLOT_ABI_V1 1u
#define TC_STREAM_SLOT_ABI_V2 2u
#define TC_STREAM_SLOT_ABI_V3 3u
#define TC_STREAM_RECEIPT_ABI_V1 1u
#define TC_STREAM_RECEIPT_ABI_V2 2u
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
    uint32_t group, pool, slot, block_count;
    const uint32_t *blocks;
    uint64_t content_bytes;
} tc_stream_group_v2;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage, pool, slot_count, prefetch_distance, io_workers, pass_count;
    uint64_t request_generation;
    const uint64_t *slot_capacity_bytes;
    uint32_t group_count;
    const tc_stream_group_v1 *groups;
} tc_stream_stage_plan_v1;

/* V2 keeps the v1 callback/slot protocol but describes ordered compatible
 * pools explicitly.  The executor visits one pool at a time and inserts a
 * drain barrier before switching classes. */
typedef struct {
    uint32_t struct_size, version;
    uint32_t pool, slot_count;
    const uint64_t *slot_capacity_bytes;
} tc_stream_pool_plan_v2;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage, pool_count, slot_count, prefetch_distance, io_workers, pass_count;
    uint64_t request_generation;
    const tc_stream_pool_plan_v2 *pools;
    uint32_t group_count;
    const tc_stream_group_v2 *groups;
} tc_stream_stage_plan_v2;

typedef enum {
    TC_STREAM_PASS_RELOAD_V3 = 0,
    TC_STREAM_PASS_CARRY_FIRST_GROUP_V3 = 1,
} tc_stream_pass_transition_v3;

/* V3 retains the V1 single-pool group/callback ABI and adds an explicit pass
 * transition.  CARRY_FIRST_GROUP means the next pass's first fill is issued
 * before the current pass's final encode and remains Ready at the boundary. */
typedef struct {
    uint32_t struct_size, version;
    uint32_t stage, pool, slot_count, prefetch_distance, io_workers, pass_count;
    uint32_t pass_transition;
    uint64_t request_generation;
    const uint64_t *slot_capacity_bytes;
    uint32_t group_count;
    const tc_stream_group_v1 *groups;
} tc_stream_stage_plan_v3;

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
    uint32_t struct_size, version;
    void *user;
    int (*allocate_slot)(void *, uint32_t pool, uint32_t slot,
                         uint64_t capacity, char *, size_t);
    void (*destroy_pool)(void *, uint32_t pool);
    int (*fill)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
                tc_stream_cancel_query_v1, const void *cancel_user,
                uint64_t *content_bytes, char *, size_t);
    int (*prefix)(void *, uint32_t pass, char *, size_t);
    int (*prepare)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *, char *, size_t);
    int (*encode)(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
                  const tc_stream_completion_sink_v1 *, tc_stream_reader_set_v1 *, char *, size_t);
    int (*drain)(void *, char *, size_t);
} tc_stream_adapter_v2;

typedef struct {
    uint64_t pool_creates, slot_bundles, fills, content_bytes_loaded, groups_submitted;
    double wait_seconds;
} tc_stream_counters_v1;

/* Additive receipt control. Existing V1/V2/V3 plan and callback layouts are
 * unchanged. Enable is owner-only and valid after create but before run_pass;
 * receipt is available only after a successful finish. */
typedef struct {
    uint32_t struct_size, version;
    uint64_t source_generation;
    char layout_digest[65];
    char implementation[64];
} tc_stream_receipt_config_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage_index, completed_passes, completed_groups;
    uint64_t fills, groups_submitted, logical_read_bytes;
    uint64_t reader_fences_issued, reader_fences_completed;
    uint64_t source_generation;
    uint8_t drained, verified;
    uint8_t reserved[6];
    char event_digest[65];
    char canonical_digest[65];
} tc_stream_receipt_v1;

/* V2 is an additive, owner-thread-only clone of the recorder's real event
 * matrix.  It is deliberately caller-allocated so C model runtimes can copy
 * a sealed receipt without taking ownership of C++ vectors.  Calling the V2
 * getter with all array pointers NULL and capacities zero is a size query; a
 * second call supplies arrays of at least the returned counts. */
typedef struct {
    tc_stream_reader_fence_v1 fence;
    uint8_t completed;
    uint8_t reserved[7];
} tc_stream_reader_receipt_v2;

enum {
    TC_STREAM_GROUP_FILL_COMPLETED_V2 = 1u << 0,
    TC_STREAM_GROUP_SUBMITTED_V2 = 1u << 1,
};

typedef struct {
    uint32_t pass, group, pool, slot;
    uint32_t step, fill_count, reader_count, flags;
    uint64_t request_generation, content_generation;
    uint64_t expected_bytes, actual_bytes, source_generation;
    tc_stream_reader_receipt_v2 readers[TC_STREAM_MAX_READER_QUEUES];
} tc_stream_group_receipt_v2;

typedef struct {
    uint32_t pass, ordinal, pool;
} tc_stream_pool_selection_receipt_v2;

typedef struct {
    uint32_t from_pass, to_pass, group, pool, slot;
    uint64_t content_generation;
} tc_stream_carry_receipt_v2;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage_index, completed_passes, completed_groups;
    uint64_t fills, groups_submitted, logical_read_bytes;
    uint64_t reader_fences_issued, reader_fences_completed;
    uint64_t source_generation;
    uint8_t drained, verified;
    uint8_t reserved[6];
    char stage_id[64];
    char layout_digest[65];
    char implementation[64];
    char event_digest[65];
    char canonical_digest[65];
    uint32_t group_capacity, group_count;
    tc_stream_group_receipt_v2 *groups;
    uint32_t pool_selection_capacity, pool_selection_count;
    tc_stream_pool_selection_receipt_v2 *pool_selections;
    uint32_t carry_capacity, carry_count;
    tc_stream_carry_receipt_v2 *carries;
} tc_stream_receipt_v2;

typedef struct tc_stream_executor tc_stream_executor;

/* Metadata arrays and callback table are copied; adapter.user must outlive the
 * handle. On failure a non-null output is quarantined and must NOT be freed
 * until tc_stream_executor_destroy succeeds. All calls except cancel are owner-only. */
int tc_stream_executor_create_v1(const tc_stream_stage_plan_v1 *, const tc_stream_adapter_v1 *,
                                 tc_stream_executor **out, char *, size_t);
int tc_stream_executor_create_v2(const tc_stream_stage_plan_v2 *, const tc_stream_adapter_v2 *,
                                 tc_stream_executor **out, char *, size_t);
/* The callback protocol is unchanged from V1; only the immutable plan is V3. */
int tc_stream_executor_create_v3(const tc_stream_stage_plan_v3 *, const tc_stream_adapter_v1 *,
                                 tc_stream_executor **out, char *, size_t);
int tc_stream_executor_run_pass(tc_stream_executor *, uint32_t pass, uint32_t step, char *, size_t);
int tc_stream_executor_finish(tc_stream_executor *, char *, size_t);
int tc_stream_executor_counters(tc_stream_executor *, tc_stream_counters_v1 *, char *, size_t);
int tc_stream_executor_enable_receipt_v1(
    tc_stream_executor *, const tc_stream_receipt_config_v1 *, char *, size_t);
int tc_stream_executor_receipt_v1(
    tc_stream_executor *, tc_stream_receipt_v1 *, char *, size_t);
int tc_stream_executor_receipt_v2(
    tc_stream_executor *, tc_stream_receipt_v2 *, char *, size_t);
void tc_stream_executor_cancel(tc_stream_executor *);
/* Stops workers and drains before freeing. Returns 0 and retains *handle if
 * safety cannot be proved; caller must quarantine the owning model session. */
int tc_stream_executor_destroy(tc_stream_executor **handle, char *, size_t);

#ifdef __cplusplus
}
#endif
#endif
