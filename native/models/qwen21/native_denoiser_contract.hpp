#pragma once

#include "native_denoiser.hpp"

namespace tc::qwen21::native_graph::detail {

using DenoiserDeadline = std::chrono::steady_clock::time_point;
inline DenoiserDeadline step_deadline(std::chrono::milliseconds timeout, DenoiserDeadline now) {
    if (timeout.count() < 0) throw std::invalid_argument("native denoiser negative deadline");
    const auto maximum=std::chrono::duration_cast<std::chrono::milliseconds>(DenoiserDeadline::max()-now);
    return timeout>=maximum ? DenoiserDeadline::max() : now+timeout;
}
inline std::chrono::milliseconds step_remaining(DenoiserDeadline end, DenoiserDeadline now) {
    return end<=now ? std::chrono::milliseconds(0) : std::chrono::ceil<std::chrono::milliseconds>(end-now);
}

// The first executable recipe is deliberately narrower than the individual
// operator geometry. Tiny is generated-data coverage, not model qualification.
inline bool production_hybrid_geometry(DenoiserGeometry g) {
    return g.block.hidden==4096 && g.block.heads==32 && g.block.head_dim==128 &&
           g.block.ffn_width==12288 && g.layers==32;
}
inline bool tiny_hybrid_geometry(DenoiserGeometry g) {
    return g.block.hidden==128 && g.block.heads==1 && g.block.head_dim==128 &&
           g.block.ffn_width==1024 && g.layers==2;
}
inline DenoiserScheduleVariant validate_lora_recipe(DenoiserGeometry g,
    const NativeDenoiserExecutionOptions& o,bool has_terms) {
    if (has_terms != !o.adapter_revision.empty())
        throw std::invalid_argument("native adapter revision must exactly describe the resolved ordered terms");
    (void)metal_graph::native_lora::recipe_for_variant(o.lora_kernel);
    if (!has_terms && o.lora_kernel!=metal_graph::native_lora::KernelVariant::Scalar)
        throw std::invalid_argument("native non-scalar LoRA kernel requires resolved adapter terms");
    if (has_terms && o.hybrid_ffn)
        throw std::invalid_argument("native hybrid LoRA has no supported execution ABI");
    auto variant=o.schedule_variant;
    if (variant==DenoiserScheduleVariant::AutomaticBase) {
        if (has_terms) throw std::invalid_argument("native adapters require an explicit schedule variant");
        variant=production_hybrid_geometry(g) ? DenoiserScheduleVariant::Base20 :
            DenoiserScheduleVariant::GeneratedTiny;
    }
    switch (variant) {
    case DenoiserScheduleVariant::Base20:
        if (!production_hybrid_geometry(g))
            throw std::invalid_argument("native Base20 requires the complete production geometry");
        break;
    case DenoiserScheduleVariant::ViggleLoRA6:
        if (!production_hybrid_geometry(g) || !has_terms)
            throw std::invalid_argument("native ViggleLoRA6 requires complete geometry and resolved adapters");
        break;
    case DenoiserScheduleVariant::GeneratedTiny:
        if (production_hybrid_geometry(g))
            throw std::invalid_argument("native generated schedule cannot bypass production qualification");
        break;
    default:
        throw std::invalid_argument("native unknown schedule variant");
    }
    return variant;
}
inline void validate_schedule_variant(DenoiserScheduleVariant variant,size_t steps) {
    if (!steps || steps>256 ||
        (variant==DenoiserScheduleVariant::Base20 && steps!=20) ||
        (variant==DenoiserScheduleVariant::ViggleLoRA6 && steps!=6) ||
        (variant!=DenoiserScheduleVariant::Base20 && variant!=DenoiserScheduleVariant::ViggleLoRA6 &&
         variant!=DenoiserScheduleVariant::GeneratedTiny))
        throw std::invalid_argument("native schedule does not match the prepared model variant");
}
inline void validate_adapter_identity(const std::string& prepared,const std::string& requested) {
    if (prepared!=requested)
        throw std::invalid_argument("native adapter identity differs from the prepared immutable factors");
}
inline void validate_execution_options(DenoiserGeometry g,const NativeDenoiserExecutionOptions& o,
                                       bool private_ane_available) {
    if (!o.hybrid_ffn) {
        if (o.register_weight_staging || o.cache_weight_scales)
            throw std::invalid_argument("native register weight staging and scale cache require hybrid FFN");
        return;
    }
    if (!private_ane_available)
        throw std::invalid_argument("native hybrid FFN requires a private-ANE build");
    if (!production_hybrid_geometry(g) && !tiny_hybrid_geometry(g))
        throw std::invalid_argument("native hybrid denoiser geometry is outside the initial base recipe");
    if (o.ane_program_cache.empty() || !o.ane_program_cache.is_absolute() ||
        bool(o.allocation.reserve)!=bool(o.allocation.allocated))
        throw std::invalid_argument("native hybrid needs an explicit absolute cache and paired memory hooks");
}
inline void validate_hybrid_request(DenoiserGeometry g,uint32_t target,size_t references,size_t steps) {
    const bool production=production_hybrid_geometry(g) && target==1024 && steps==20;
    const bool tiny=tiny_hybrid_geometry(g) && target==32 && steps==2;
    if (references || (!production && !tiny))
        throw std::invalid_argument("native hybrid denoiser supports only base 512/20 or generated 32-row/2-step inputs");
}

// CPU-only completion-domain reducer. Absence of an attempted domain's receipt
// must be represented by physically_complete=false by its owning controller.
// The domain is additive only for physical coverage, not overlapping timings.
inline metal_graph::Completion join_physical_domain(metal_graph::Completion gpu,
    bool active,bool domain_ok,bool domain_physical,bool domain_timeout,
    const std::string& domain_error) {
    if (!active) return gpu;
    gpu.ok=gpu.ok && domain_ok;
    gpu.physically_complete=gpu.physically_complete && domain_physical;
    gpu.timed_out=gpu.timed_out || domain_timeout;
    gpu.gpu_seconds=0; // no whole-step GPU interval is measured for hybrid
    if (!domain_error.empty()) {
        if (!gpu.error.empty()) gpu.error+="; ";
        gpu.error+=domain_error;
    }
    return gpu;
}

} // namespace tc::qwen21::native_graph::detail
