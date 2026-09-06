#include "residency.hpp"
namespace tc {
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
