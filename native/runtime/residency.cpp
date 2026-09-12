#include "residency.hpp"
namespace tc {
BlockResidencyPlan make_block_residency_plan(
    uint64_t memory_budget_bytes, uint64_t activation_reserve_bytes,
    uint64_t block_bytes, unsigned active_blocks,
    unsigned requested_pinned_blocks, unsigned max_refill_slots,
    bool adaptive_refill_slots, bool allow_full_resident) {
    tc_block_residency_plan raw{};
    const auto status = tc_block_residency_plan_build(
        memory_budget_bytes, activation_reserve_bytes, block_bytes,
        active_blocks, requested_pinned_blocks, max_refill_slots,
        adaptive_refill_slots ? 1 : 0, allow_full_resident ? 1 : 0, &raw);
    require(status == TC_BLOCK_RESIDENCY_OK,
            std::string("cannot build block residency plan: ") +
                tc_block_residency_status_string(status));
    return {
        raw.memory_budget_bytes,
        raw.activation_reserve_bytes,
        raw.block_bytes,
        raw.slot_bytes,
        raw.minimum_bytes,
        raw.estimated_working_set_bytes,
        raw.active_blocks,
        raw.requested_pinned_blocks,
        raw.maximum_pinned_blocks,
        raw.pinned_blocks,
        raw.streamed_blocks,
        raw.refill_slots,
        raw.fully_resident != 0,
    };
}

ResidencyPolicy ResidencyPolicy::for_request(const Request &r, uint64_t physical) {
    bool staged = r.residency == "component_staged";
    return {r.residency == "resident" && physical >= (32ull << 30) &&
                (!r.memory_budget_bytes || r.memory_budget_bytes >= (24ull << 30)),
            staged, staged, staged};
}
void ResidencyPolicy::validate_budget(const ExecutionPlan &plan, uint64_t physical) {
    require(plan.memory_estimate_bytes.has_value(), "model has no validated memory estimate");
    auto estimate = *plan.memory_estimate_bytes;
    require(physical >= estimate + (4ull << 30),
            "insufficient physical memory for the conservative BF16 plan");
    require(!plan.request.memory_budget_bytes || plan.request.memory_budget_bytes >= estimate,
            "profile memory budget is below the BF16 plan estimate");
}
} // namespace tc
