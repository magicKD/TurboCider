#ifndef H3_STREAMING_POLICY_H
#define H3_STREAMING_POLICY_H

#include <stdint.h>

typedef enum {
    H3_STREAM_PLAN_OK = 0,
    H3_STREAM_PLAN_INVALID_ARGUMENT,
    H3_STREAM_PLAN_OVERFLOW,
    H3_STREAM_PLAN_BUDGET_TOO_SMALL,
    H3_STREAM_PLAN_PREFIX_EXHAUSTS_STREAM,
    H3_STREAM_PLAN_PREFIX_EXCEEDS_BUDGET
} h3_stream_plan_status;

typedef struct {
    uint64_t memory_budget_bytes;
    uint64_t activation_reserve_bytes;
    uint64_t block_bytes;
    uint64_t slot_bytes;
    uint64_t minimum_bytes;
    unsigned active_blocks;
    unsigned requested_pinned_blocks;
    unsigned maximum_pinned_blocks;
    unsigned pinned_blocks;
    unsigned streamed_blocks;
} h3_stream_plan;

/* Select a BF16 pinned prefix without silently exceeding the requested
 * working-set target. A zero requested prefix means "choose the largest that
 * fits" when a budget is present, and "preserve two-slot streaming" when it
 * is absent. At least one active block always remains stream-backed. */
h3_stream_plan_status h3_stream_plan_build(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, h3_stream_plan *plan);

const char *h3_stream_plan_status_string(h3_stream_plan_status status);

#endif
