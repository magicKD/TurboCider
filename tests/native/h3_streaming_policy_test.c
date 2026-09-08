#include "h3_streaming_policy.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

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
    puts("PASS: H3 streaming budget and pinned-prefix policy");
    return 0;
}
