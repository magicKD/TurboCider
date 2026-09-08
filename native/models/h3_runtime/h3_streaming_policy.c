#include "h3_streaming_policy.h"

#include <limits.h>
#include <string.h>

h3_stream_plan_status h3_stream_plan_build(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, h3_stream_plan *plan) {
    if (!plan || !block_bytes || active_blocks < 1) {
        return H3_STREAM_PLAN_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    plan->memory_budget_bytes = memory_budget_bytes;
    plan->activation_reserve_bytes = activation_reserve_bytes;
    plan->block_bytes = block_bytes;
    plan->active_blocks = active_blocks;
    plan->requested_pinned_blocks = requested_pinned_blocks;
    if (requested_pinned_blocks >= active_blocks) {
        return H3_STREAM_PLAN_PREFIX_EXHAUSTS_STREAM;
    }
    if (block_bytes > UINT64_MAX / 2u) {
        return H3_STREAM_PLAN_OVERFLOW;
    }
    plan->slot_bytes = block_bytes * 2u;
    if (activation_reserve_bytes > UINT64_MAX - plan->slot_bytes) {
        return H3_STREAM_PLAN_OVERFLOW;
    }
    plan->minimum_bytes = activation_reserve_bytes + plan->slot_bytes;
    plan->maximum_pinned_blocks = active_blocks - 1u;
    if (!memory_budget_bytes) {
        plan->pinned_blocks = requested_pinned_blocks;
        plan->streamed_blocks = active_blocks - requested_pinned_blocks;
        return H3_STREAM_PLAN_OK;
    }
    if (memory_budget_bytes < plan->minimum_bytes) {
        return H3_STREAM_PLAN_BUDGET_TOO_SMALL;
    }
    uint64_t available = memory_budget_bytes - plan->minimum_bytes;
    uint64_t by_budget = available / block_bytes;
    if (by_budget < plan->maximum_pinned_blocks) {
        plan->maximum_pinned_blocks = (unsigned)by_budget;
    }
    if (requested_pinned_blocks > plan->maximum_pinned_blocks) {
        return H3_STREAM_PLAN_PREFIX_EXCEEDS_BUDGET;
    }
    plan->pinned_blocks = requested_pinned_blocks ?
        requested_pinned_blocks : plan->maximum_pinned_blocks;
    plan->streamed_blocks = active_blocks - plan->pinned_blocks;
    return H3_STREAM_PLAN_OK;
}

const char *h3_stream_plan_status_string(h3_stream_plan_status status) {
    switch (status) {
    case H3_STREAM_PLAN_OK: return "ok";
    case H3_STREAM_PLAN_INVALID_ARGUMENT: return "invalid argument";
    case H3_STREAM_PLAN_OVERFLOW: return "size overflow";
    case H3_STREAM_PLAN_BUDGET_TOO_SMALL: return "budget below minimum";
    case H3_STREAM_PLAN_PREFIX_EXHAUSTS_STREAM:
        return "pinned prefix leaves no streamed block";
    case H3_STREAM_PLAN_PREFIX_EXCEEDS_BUDGET:
        return "pinned prefix exceeds budget";
    }
    return "unknown streaming-plan failure";
}
