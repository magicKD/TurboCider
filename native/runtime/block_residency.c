#include "block_residency.h"

#include <limits.h>
#include <string.h>

static int add_overflow_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left > UINT64_MAX - right) return 1;
    *result = left + right;
    return 0;
}

static int mul_overflow_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left && right > UINT64_MAX / left) return 1;
    *result = left * right;
    return 0;
}

static tc_block_residency_status finish_plan(
        tc_block_residency_plan *plan, unsigned pinned, unsigned slots,
        int fully_resident) {
    uint64_t block_total = 0;
    if (mul_overflow_u64((uint64_t)pinned + (uint64_t)slots,
                         plan->block_bytes, &block_total) ||
        add_overflow_u64(plan->activation_reserve_bytes, block_total,
                         &plan->estimated_working_set_bytes)) {
        return TC_BLOCK_RESIDENCY_OVERFLOW;
    }
    plan->pinned_blocks = pinned;
    plan->streamed_blocks = plan->active_blocks - pinned;
    plan->refill_slots = slots;
    plan->fully_resident = fully_resident;
    return TC_BLOCK_RESIDENCY_OK;
}

tc_block_residency_status tc_block_residency_plan_build(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, unsigned max_refill_slots,
    int adaptive_refill_slots, int allow_full_resident,
    tc_block_residency_plan *plan) {
    if (!plan || !block_bytes || !active_blocks || !max_refill_slots ||
        (adaptive_refill_slots != 0 && adaptive_refill_slots != 1) ||
        (allow_full_resident != 0 && allow_full_resident != 1)) {
        return TC_BLOCK_RESIDENCY_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    plan->memory_budget_bytes = memory_budget_bytes;
    plan->activation_reserve_bytes = activation_reserve_bytes;
    plan->block_bytes = block_bytes;
    plan->active_blocks = active_blocks;
    plan->requested_pinned_blocks = requested_pinned_blocks;
    if (block_bytes > UINT64_MAX / 2u) {
        return TC_BLOCK_RESIDENCY_OVERFLOW;
    }
    plan->slot_bytes = block_bytes * 2u;
    if (
        add_overflow_u64(activation_reserve_bytes, plan->slot_bytes,
                         &plan->minimum_bytes)) {
        return TC_BLOCK_RESIDENCY_OVERFLOW;
    }
    if (requested_pinned_blocks > active_blocks ||
        (!allow_full_resident && requested_pinned_blocks == active_blocks)) {
        return TC_BLOCK_RESIDENCY_PINNED_EXHAUSTS_STREAM;
    }

    if (!memory_budget_bytes) {
        if (allow_full_resident && requested_pinned_blocks == active_blocks) {
            plan->maximum_pinned_blocks = active_blocks;
            return finish_plan(plan, active_blocks, 0u, 1);
        }
        unsigned pinned = requested_pinned_blocks;
        unsigned streamed = active_blocks - pinned;
        unsigned slots = adaptive_refill_slots && max_refill_slots > streamed ?
            streamed : max_refill_slots;
        if (!streamed || !slots)
            return TC_BLOCK_RESIDENCY_PINNED_EXHAUSTS_STREAM;
        plan->maximum_pinned_blocks = active_blocks - 1u;
        return finish_plan(plan, pinned, slots, 0);
    }

    if (memory_budget_bytes < activation_reserve_bytes)
        return TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL;
    uint64_t available = memory_budget_bytes - activation_reserve_bytes;
    uint64_t capacity = available / block_bytes;
    if (allow_full_resident && capacity >= active_blocks &&
        (!requested_pinned_blocks || requested_pinned_blocks == active_blocks)) {
        plan->maximum_pinned_blocks = active_blocks;
        return finish_plan(plan, active_blocks, 0u, 1);
    }
    if (capacity < 2u)
        return TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL;

    unsigned slots = max_refill_slots;
    uint64_t pinned_capacity;
    if (adaptive_refill_slots) {
        uint64_t candidate = capacity - 1u;
        slots = candidate < slots ? (unsigned)candidate : slots;
        if (!slots) return TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL;
        pinned_capacity = capacity - slots;
    } else {
        uint64_t reserved = (uint64_t)slots;
        if (capacity < reserved)
            return TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL;
        pinned_capacity = capacity - reserved;
    }
    if (pinned_capacity >= active_blocks)
        pinned_capacity = active_blocks - 1u;
    plan->maximum_pinned_blocks = (unsigned)pinned_capacity;
    if ((uint64_t)requested_pinned_blocks > pinned_capacity)
        return TC_BLOCK_RESIDENCY_PINNED_EXCEEDS_BUDGET;
    unsigned pinned = requested_pinned_blocks ? requested_pinned_blocks :
        (unsigned)pinned_capacity;
    return finish_plan(plan, pinned, slots, 0);
}

const char *tc_block_residency_status_string(tc_block_residency_status status) {
    switch (status) {
    case TC_BLOCK_RESIDENCY_OK: return "ok";
    case TC_BLOCK_RESIDENCY_INVALID_ARGUMENT: return "invalid argument";
    case TC_BLOCK_RESIDENCY_OVERFLOW: return "size overflow";
    case TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL: return "budget below minimum";
    case TC_BLOCK_RESIDENCY_PINNED_EXHAUSTS_STREAM:
        return "pinned prefix leaves no streamed block";
    case TC_BLOCK_RESIDENCY_PINNED_EXCEEDS_BUDGET:
        return "pinned prefix exceeds budget";
    }
    return "unknown residency-plan failure";
}
