#include "h3_streaming_policy.h"

#include "../../runtime/block_residency.h"

#include <limits.h>
#include <string.h>

h3_stream_plan_status h3_stream_plan_build(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, h3_stream_plan *plan) {
    tc_block_residency_plan common;
    tc_block_residency_status status = tc_block_residency_plan_build(
        memory_budget_bytes, activation_reserve_bytes, block_bytes,
        active_blocks, requested_pinned_blocks, 2u, 0, 0, &common);
    switch (status) {
    case TC_BLOCK_RESIDENCY_OK:
        memset(plan, 0, sizeof(*plan));
        plan->memory_budget_bytes = common.memory_budget_bytes;
        plan->activation_reserve_bytes = common.activation_reserve_bytes;
        plan->block_bytes = common.block_bytes;
        plan->slot_bytes = common.slot_bytes;
        plan->minimum_bytes = common.minimum_bytes;
        plan->active_blocks = common.active_blocks;
        plan->requested_pinned_blocks = common.requested_pinned_blocks;
        plan->maximum_pinned_blocks = common.maximum_pinned_blocks;
        plan->pinned_blocks = common.pinned_blocks;
        plan->streamed_blocks = common.streamed_blocks;
        return H3_STREAM_PLAN_OK;
    case TC_BLOCK_RESIDENCY_INVALID_ARGUMENT:
        return H3_STREAM_PLAN_INVALID_ARGUMENT;
    case TC_BLOCK_RESIDENCY_OVERFLOW:
        return H3_STREAM_PLAN_OVERFLOW;
    case TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL:
        return H3_STREAM_PLAN_BUDGET_TOO_SMALL;
    case TC_BLOCK_RESIDENCY_PINNED_EXHAUSTS_STREAM:
        return H3_STREAM_PLAN_PREFIX_EXHAUSTS_STREAM;
    case TC_BLOCK_RESIDENCY_PINNED_EXCEEDS_BUDGET:
        return H3_STREAM_PLAN_PREFIX_EXCEEDS_BUDGET;
    }
    return H3_STREAM_PLAN_INVALID_ARGUMENT;
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
