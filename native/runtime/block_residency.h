#ifndef TC_BLOCK_RESIDENCY_H
#define TC_BLOCK_RESIDENCY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define TC_BLOCK_RESIDENCY_INTERNAL __attribute__((visibility("hidden")))
#else
#define TC_BLOCK_RESIDENCY_INTERNAL
#endif

typedef enum {
    TC_BLOCK_RESIDENCY_OK = 0,
    TC_BLOCK_RESIDENCY_INVALID_ARGUMENT,
    TC_BLOCK_RESIDENCY_OVERFLOW,
    TC_BLOCK_RESIDENCY_BUDGET_TOO_SMALL,
    TC_BLOCK_RESIDENCY_PINNED_EXHAUSTS_STREAM,
    TC_BLOCK_RESIDENCY_PINNED_EXCEEDS_BUDGET
} tc_block_residency_status;

/* A model-independent working-set decision shared by native C runtimes.
 * `pinned_blocks` is a prefix retained for the request; the rest are loaded
 * through `refill_slots` reusable slots.  The policy never treats a request
 * budget as a process RSS cap. */
typedef struct {
    uint64_t memory_budget_bytes;
    uint64_t activation_reserve_bytes;
    uint64_t block_bytes;
    uint64_t slot_bytes;
    uint64_t minimum_bytes;
    uint64_t estimated_working_set_bytes;
    unsigned active_blocks;
    unsigned requested_pinned_blocks;
    unsigned maximum_pinned_blocks;
    unsigned pinned_blocks;
    unsigned streamed_blocks;
    unsigned refill_slots;
    int fully_resident;
} tc_block_residency_plan;

/* Build a conservative block residency plan.
 *
 * A zero budget means "do not impose a budget" and preserves the caller's
 * requested prefix.  Fixed-slot runtimes pass adaptive_refill_slots=0; they
 * reserve exactly max_refill_slots slots before selecting a prefix.  Runtimes
 * with adaptive look-ahead pass adaptive_refill_slots=1 and use as many slots
 * as fit while retaining at least one streamed block.  If
 * allow_full_resident is nonzero, a budget that fits every block selects a
 * normal fully resident session with zero refill slots. */
TC_BLOCK_RESIDENCY_INTERNAL tc_block_residency_status
tc_block_residency_plan_build(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, unsigned max_refill_slots,
    int adaptive_refill_slots, int allow_full_resident,
    tc_block_residency_plan *plan);

TC_BLOCK_RESIDENCY_INTERNAL const char *tc_block_residency_status_string(
    tc_block_residency_status status);

#ifdef __cplusplus
}
#endif

#undef TC_BLOCK_RESIDENCY_INTERNAL

#endif
