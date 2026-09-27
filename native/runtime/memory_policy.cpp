#include "memory_policy.hpp"

#include "../core/common.hpp"

#include <limits>
#include <sstream>

namespace tc {
namespace {
constexpr uint64_t gib = 1ull << 30;

uint64_t effective_budget(uint64_t limit, unsigned buffer_percent) {
    require(buffer_percent <= 100, "memory_policy_invalid: buffer percent overflow");
    const uint64_t multiplier = 100u - buffer_percent;
    require(!limit || multiplier <= std::numeric_limits<uint64_t>::max() / limit,
            "memory_policy_invalid: effective budget overflow");
    return (limit * multiplier) / 100u;
}

uint64_t fnv1a64(const std::string &value) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t checked_subtract(uint64_t budget, uint64_t reserve,
                          const char *adapter) {
    require(budget > reserve,
            std::string("memory_budget_too_small: ") + adapter +
                " non-denoiser reserve exhausts the effective budget");
    return budget - reserve;
}
} // namespace

const char *memory_capability_level_name(MemoryCapabilityLevel level) {
    switch (level) {
    case MemoryCapabilityLevel::Declared: return "declared";
    case MemoryCapabilityLevel::HookBridged: return "hook_bridged";
    case MemoryCapabilityLevel::AllocationGuarded: return "allocation_guarded";
    case MemoryCapabilityLevel::EnvelopeValidated: return "envelope_validated";
    case MemoryCapabilityLevel::L3Certified: return "l3_certified";
    }
    return "declared";
}

const char *memory_certification_state_name(MemoryCertificationState state) {
    switch (state) {
    case MemoryCertificationState::Unsupported: return "unsupported";
    case MemoryCertificationState::PlanOnly: return "plan_only";
    case MemoryCertificationState::ExperimentalGuarded:
        return "experimental_guarded";
    case MemoryCertificationState::Certified: return "certified";
    }
    return "unsupported";
}

void overlay_memory_config(MemoryConstrainedConfig &base,
                           const MemoryConstrainedConfig &override_config) {
    if (override_config.has(MemoryFieldEnabled))
        base.enabled = override_config.enabled;
    if (override_config.has(MemoryFieldLimit))
        base.limit_bytes = override_config.limit_bytes;
    if (override_config.has(MemoryFieldBufferPercent))
        base.buffer_percent = override_config.buffer_percent;
    if (override_config.has(MemoryFieldMinFree))
        base.min_free_bytes = override_config.min_free_bytes;
    if (override_config.has(MemoryFieldMaxRefillSlots))
        base.max_refill_slots = override_config.max_refill_slots;
    if (override_config.has(MemoryFieldAllowTiling))
        base.allow_quality_preserving_tiling =
            override_config.allow_quality_preserving_tiling;
    base.specified_fields |= override_config.specified_fields;
}

void validate_memory_constrained_request(const Request &request,
                                         uint64_t physical_memory) {
    const auto &config = request.memory_constrained;
    if (!config.specified())
        return;
    require(config.buffer_percent >= 5 && config.buffer_percent <= 30,
            "memory_policy_invalid: buffer_percent must be 5...30");
    require(config.max_refill_slots >= 1 && config.max_refill_slots <= 3,
            "memory_policy_invalid: max_refill_slots must be 1...3");
    if (physical_memory) {
        require(config.limit_bytes <= physical_memory,
                "memory_policy_invalid: limit_bytes exceeds physical memory");
        require(config.min_free_bytes < physical_memory,
                "memory_policy_invalid: min_free_bytes must be below physical memory");
    }
    if (!config.enabled)
        return;
    require(config.limit_bytes > 0,
            "memory_policy_invalid: enabled mode requires limit_bytes");
    require(effective_budget(config.limit_bytes, config.buffer_percent) > 0,
            "memory_budget_too_small: effective budget is zero");
    require(request.memory_budget_bytes == 0 || config.normalized,
            "memory_policy_conflict: memory_constrained cannot be combined with "
            "legacy memory_budget_bytes");
    require(request.execution != "gpu_ane" && request.ane_manifest.empty() &&
                request.encoder_ane_manifest.empty(),
            "memory_policy_conflict: the first memory-constrained release is GPU-only");
}

EffectiveMemoryPolicy make_effective_memory_policy(const Request &request) {
    const auto &config = request.memory_constrained;
    EffectiveMemoryPolicy policy;
    policy.enabled = config.enabled;
    if (!policy.enabled)
        return policy;
    policy.user_limit_bytes = config.limit_bytes;
    policy.effective_budget_bytes =
        effective_budget(config.limit_bytes, config.buffer_percent);
    policy.system_reserve_bytes = config.min_free_bytes;
    policy.buffer_percent = config.buffer_percent;
    policy.max_refill_slots = config.max_refill_slots;
    policy.allow_quality_preserving_tiling =
        config.allow_quality_preserving_tiling;
    policy.admission_state = "estimate_only";
    policy.enforcement_scope = "inference_process_tree_v1";
    policy.reason =
        "model checkpoint envelope has not been verified by an execution adapter";
    std::ostringstream identity;
    identity << "memory-v1:" << config.limit_bytes << ':'
             << config.buffer_percent << ':' << config.min_free_bytes << ':'
             << config.max_refill_slots << ':'
             << (config.allow_quality_preserving_tiling ? 1 : 0);
    std::ostringstream digest;
    digest << std::hex << fnv1a64(identity.str());
    policy.digest = digest.str();
    return policy;
}

void resolve_memory_constrained_candidate(Request &request,
                                          EffectiveMemoryPolicy &policy) {
    if (!policy.enabled) return;
    request.memory_constrained.effective_budget_bytes =
        policy.effective_budget_bytes;

    if (request.model == "minimax-h3-turbo" &&
        (request.execution == "gpu" || request.execution == "auto") &&
        request.operation == "video.generate" && !request.audio &&
        request.inputs.empty() && request.loras.empty() &&
        request.quantized_cache.empty()) {
        policy.route_available = true;
        policy.adapter_candidate = "h3_c_metal_streamed_v1";
        policy.candidate_backend = "c_metal";
        policy.candidate_dtype = "h3_native_mixed";
        policy.candidate_model_variant = "fl2va_t2v_video_only";
        policy.candidate_sampler_mode = "h3_distilled_euler_v1";
        policy.candidate_tiling_mode = "h3_video_vae_tiled_v1";
        policy.capability_level = MemoryCapabilityLevel::HookBridged;
        policy.certification_state = MemoryCertificationState::PlanOnly;
        policy.effective_residency = "streamed";
        policy.non_denoiser_reserve_bytes = 4ull * gib;
        policy.denoiser_budget_bytes = checked_subtract(
            policy.effective_budget_bytes,
            policy.non_denoiser_reserve_bytes,
            policy.adapter_candidate.c_str());
        policy.refill_slots = 2;
        require(policy.max_refill_slots >= policy.refill_slots,
                "memory_policy_unsupported: H3 C/Metal currently requires "
                "two certified refill slots");
        request.execution = "gpu";
        request.residency = "streamed";
        request.memory_budget_bytes = policy.denoiser_budget_bytes;
        request.memory_constrained.normalized = true;
        request.memory_constrained.denoiser_budget_bytes =
            policy.denoiser_budget_bytes;
        request.memory_constrained.refill_slots = policy.refill_slots;
        policy.admission_state = "plan_only";
        policy.reason =
            "H3 C/Metal route is plan-only: no verified capability manifest "
            "matches this checkpoint, shape, device, and runtime";
        return;
    }

    if (request.model == "ltx-2.5-distilled" &&
        (request.execution == "gpu" || request.execution == "auto") &&
        (request.ltx_backend == "auto" || request.ltx_backend == "c_metal") &&
        request.operation == "video.generate" && !request.audio &&
        request.inputs.empty() && request.loras.empty()) {
        policy.route_available = true;
        policy.adapter_candidate = "ltx_c_metal_streamed_video_v1";
        policy.candidate_backend = "c_metal";
        policy.candidate_dtype = "int8_convrot_bf16";
        policy.candidate_model_variant =
            "ltx_2_5_distilled_t2v_video_only";
        policy.candidate_sampler_mode = "ltx_distilled_euler_v1";
        policy.candidate_tiling_mode = "ltx_video_vae_helper_v1";
        policy.capability_level = MemoryCapabilityLevel::HookBridged;
        policy.certification_state = MemoryCertificationState::PlanOnly;
        policy.effective_residency = "streamed";
        policy.non_denoiser_reserve_bytes = 4ull * gib;
        policy.denoiser_budget_bytes = checked_subtract(
            policy.effective_budget_bytes,
            policy.non_denoiser_reserve_bytes,
            policy.adapter_candidate.c_str());
        policy.refill_slots = policy.max_refill_slots;
        request.execution = "gpu";
        request.ltx_backend = "c_metal";
        request.residency = "streamed";
        request.memory_budget_bytes = policy.denoiser_budget_bytes;
        request.memory_constrained.normalized = true;
        request.memory_constrained.denoiser_budget_bytes =
            policy.denoiser_budget_bytes;
        request.memory_constrained.refill_slots = policy.refill_slots;
        policy.admission_state = "plan_only";
        policy.reason =
            "LTX C/Metal video route is plan-only: no verified capability "
            "manifest matches this checkpoint, shape, device, and runtime";
        return;
    }

    policy.reason =
        "no certified GPU-only memory-constrained route matches this model, "
        "backend, operation, audio, and input combination";
}

void finalize_memory_policy_estimate(EffectiveMemoryPolicy &policy,
                                     uint64_t estimate_bytes,
                                     bool estimate_available) {
    if (!policy.enabled)
        return;
    policy.estimate_fits =
        estimate_available && estimate_bytes <= policy.effective_budget_bytes;
    policy.planned_upper_bytes = estimate_available ? estimate_bytes : 0;
    policy.estimate_provenance = estimate_available ?
        "conservative_heuristic_not_hard_limit" : "unavailable";
    if (!estimate_available) {
        policy.reason = "memory estimate is unavailable";
        policy.admission_state = policy.route_available ? "plan_only" :
                                 "estimate_unavailable";
    } else if (!policy.estimate_fits) {
        std::ostringstream reason;
        reason << "conservative estimate " << estimate_bytes
               << " exceeds effective budget "
               << policy.effective_budget_bytes;
        policy.reason = reason.str();
        policy.admission_state = "rejected";
    } else if (policy.route_available && !policy.execution_supported) {
        // A fit estimate is necessary but never sufficient.  Keep the
        // candidate executable=false until a manifest-backed capability is
        // loaded by the runtime.  This prevents a heuristic from becoming a
        // hidden resident/unbounded fallback.
        policy.admission_state = "plan_only";
    } else if (!policy.route_available) {
        policy.admission_state = "unsupported";
    }
}

void require_memory_constrained_execution_supported(
    const EffectiveMemoryPolicy &policy) {
    if (!policy.enabled)
        return;
    require(policy.estimate_fits,
            "memory_budget_too_small: " + policy.reason);
    require(policy.execution_supported,
            "memory_policy_unsupported: " + policy.reason);
}

} // namespace tc
