#pragma once
#include "block_residency.h"
#include "session.hpp"
namespace tc {
struct BlockResidencyPlan {
    uint64_t memory_budget_bytes = 0;
    uint64_t activation_reserve_bytes = 0;
    uint64_t block_bytes = 0;
    uint64_t slot_bytes = 0;
    uint64_t minimum_bytes = 0;
    uint64_t estimated_working_set_bytes = 0;
    unsigned active_blocks = 0;
    unsigned requested_pinned_blocks = 0;
    unsigned maximum_pinned_blocks = 0;
    unsigned pinned_blocks = 0;
    unsigned streamed_blocks = 0;
    unsigned refill_slots = 0;
    bool fully_resident = false;
};

/* Typed C++ view of the same policy used by H3 and LTX C runtimes. */
BlockResidencyPlan make_block_residency_plan(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, unsigned max_refill_slots,
    bool adaptive_refill_slots, bool allow_full_resident);

struct ResidencyPolicy {
    bool retain_images_during_text = false;
    bool release_before_denoise = false;
    bool release_after_denoise = false;
    bool release_after_decode = false;
    static ResidencyPolicy for_request(const Request &, uint64_t physical_memory);
    static void validate_budget(const ExecutionPlan &, uint64_t physical_memory);
};
} // namespace tc
