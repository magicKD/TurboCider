#include "h3_streaming_policy.h"
#include "../../native/runtime/block_residency.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const uint64_t gib = UINT64_C(1024) * 1024 * 1024;
    const uint64_t block = UINT64_C(770725376);
    const uint64_t reserve = 4 * gib;
    h3_stream_plan plan;

    assert(h3_stream_plan_build(
               0, reserve, block, 50, 0, &plan) == H3_STREAM_PLAN_OK);
    assert(plan.pinned_blocks == 0 && plan.streamed_blocks == 50);

    assert(h3_stream_plan_build(
               0, reserve, block, 50, 5, &plan) == H3_STREAM_PLAN_OK);
    assert(plan.pinned_blocks == 5 && plan.streamed_blocks == 45);

    assert(h3_stream_plan_build(
               16 * gib, reserve, block, 50, 0, &plan) == H3_STREAM_PLAN_OK);
    unsigned expected = (unsigned)((16 * gib - reserve - 2 * block) / block);
    assert(plan.pinned_blocks == expected);
    assert(plan.streamed_blocks == 50 - expected);
    assert(plan.minimum_bytes == reserve + 2 * block);

    assert(h3_stream_plan_build(
               plan.minimum_bytes - 1, reserve, block, 50, 0, &plan) ==
           H3_STREAM_PLAN_BUDGET_TOO_SMALL);
    assert(h3_stream_plan_build(
               8 * gib, reserve, block, 50, 12, &plan) ==
           H3_STREAM_PLAN_PREFIX_EXCEEDS_BUDGET);
    assert(h3_stream_plan_build(
               64 * gib, reserve, block, 50, 50, &plan) ==
           H3_STREAM_PLAN_PREFIX_EXHAUSTS_STREAM);

    assert(h3_stream_plan_build(
               8 * gib, reserve, block, 1, 0, &plan) == H3_STREAM_PLAN_OK);
    assert(plan.pinned_blocks == 0 && plan.streamed_blocks == 1);

    assert(h3_stream_plan_build(
               UINT64_MAX, reserve, UINT64_MAX, 50, 0, &plan) ==
           H3_STREAM_PLAN_OVERFLOW);

    uint8_t mask[50];
    assert(h3_stream_uniform_active_mask(50, 50, mask, 50));
    for (unsigned block_id = 0; block_id < 50; block_id++)
        assert(mask[block_id] == 1);
    for (unsigned requested = 25; requested <= 50; requested++) {
        assert(h3_stream_uniform_active_mask(50, requested, mask, 50));
        assert(mask[0] && mask[49]);
        unsigned active = 0;
        for (unsigned block_id = 0; block_id < 50; block_id++)
            active += mask[block_id] != 0;
        assert(active == requested);

        uint8_t legacy[50];
        memset(legacy, 1, sizeof(legacy));
        unsigned skipped = 50 - requested;
        for (unsigned index = 0; index < skipped; index++) {
            unsigned block_id = ((2 * index + 1) * 50) / (2 * skipped);
            if (block_id == 0) block_id = 1;
            if (block_id >= 49) block_id = 48;
            legacy[block_id] = 0;
        }
        assert(!memcmp(mask, legacy, sizeof(mask)));

        uint8_t repeat[50];
        assert(h3_stream_uniform_active_mask(50, requested, repeat, 50));
        assert(!memcmp(mask, repeat, sizeof(mask)));
    }
    assert(!h3_stream_uniform_active_mask(50, 1, mask, 50));
    assert(!h3_stream_uniform_active_mask(50, 51, mask, 50));
    assert(!h3_stream_uniform_active_mask(50, 45, mask, 49));

    /* The shared runtime policy must preserve the two pre-refactor formulas
     * over every meaningful capacity, not just the benchmarked budgets. */
    for (unsigned capacity = 2; capacity <= 70; capacity++) {
        uint64_t budget = reserve + (uint64_t)capacity * block;
        assert(h3_stream_plan_build(
                   budget, reserve, block, 50, 0, &plan) ==
               H3_STREAM_PLAN_OK);
        unsigned expected = capacity - 2u;
        if (expected > 49u) expected = 49u;
        assert(plan.pinned_blocks == expected);
        assert(plan.streamed_blocks == 50u - expected);
    }

    const uint64_t ltx_block = 768 * UINT64_C(1024) * 1024;
    for (unsigned capacity = 2; capacity <= 60; capacity++) {
        tc_block_residency_plan common;
        uint64_t budget = reserve + (uint64_t)capacity * ltx_block;
        assert(tc_block_residency_plan_build(
                   budget, reserve, ltx_block, 48u, 0u, 3u, 1, 1,
                   &common) == TC_BLOCK_RESIDENCY_OK);
        if (capacity >= 48u) {
            assert(common.fully_resident);
            assert(common.pinned_blocks == 48u);
            assert(common.streamed_blocks == 0u);
            assert(common.refill_slots == 0u);
        } else {
            unsigned expected_slots = capacity >= 4u ? 3u :
                capacity >= 3u ? 2u : 1u;
            assert(!common.fully_resident);
            assert(common.refill_slots == expected_slots);
            assert(common.pinned_blocks == capacity - expected_slots);
            assert(common.streamed_blocks ==
                   48u - common.pinned_blocks);
        }
        assert(common.estimated_working_set_bytes <= budget);
    }
    puts("PASS: shared H3/LTX block residency policy");
    return 0;
}
