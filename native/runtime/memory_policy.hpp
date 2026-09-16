#pragma once

#include "../core/contracts.hpp"

#include <cstdint>
#include <string>

namespace tc {

// Capability is deliberately separate from the numerical estimate.  A route
// can be fully describable and still be unsafe to execute until every
// allocation site is guarded and a matching manifest/evidence record exists.
enum class MemoryCapabilityLevel : uint8_t {
    Declared = 0,
    HookBridged,
    AllocationGuarded,
    EnvelopeValidated,
    L3Certified,
};

enum class MemoryCertificationState : uint8_t {
    Unsupported = 0,
    PlanOnly,
    ExperimentalGuarded,
    Certified,
};

const char *memory_capability_level_name(MemoryCapabilityLevel);
const char *memory_certification_state_name(MemoryCertificationState);

struct EffectiveMemoryPolicy {
    bool enabled = false;
    uint64_t user_limit_bytes = 0;
    uint64_t effective_budget_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    unsigned buffer_percent = 0;
    unsigned max_refill_slots = 0;
    bool allow_quality_preserving_tiling = false;
    bool estimate_fits = false;
    bool route_available = false;
    bool execution_supported = false;
    MemoryCapabilityLevel capability_level = MemoryCapabilityLevel::Declared;
    MemoryCertificationState certification_state =
        MemoryCertificationState::Unsupported;
    bool release_stable = false;
    std::string manifest_digest;
    std::string evidence_digest;
    uint64_t planned_upper_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t non_denoiser_reserve_bytes = 0;
    uint64_t denoiser_budget_bytes = 0;
    unsigned refill_slots = 0;
    std::string adapter_candidate;
    std::string candidate_backend;
    std::string candidate_dtype;
    std::string candidate_model_variant;
    std::string candidate_sampler_mode;
    std::string candidate_tiling_mode;
    std::string effective_residency;
    std::string estimate_provenance = "unavailable";
    std::string admission_state = "disabled";
    std::string enforcement_scope = "none";
    std::string reason;
    std::string digest;
};

void overlay_memory_config(MemoryConstrainedConfig &base,
                           const MemoryConstrainedConfig &override_config);
void validate_memory_constrained_request(const Request &, uint64_t physical_memory);
EffectiveMemoryPolicy make_effective_memory_policy(const Request &);
void resolve_memory_constrained_candidate(Request &, EffectiveMemoryPolicy &);
void finalize_memory_policy_estimate(EffectiveMemoryPolicy &, uint64_t estimate_bytes,
                                     bool estimate_available);
void require_memory_constrained_execution_supported(const EffectiveMemoryPolicy &);

} // namespace tc
