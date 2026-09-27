#include "public_request_validation.hpp"

#include "../../core/common.hpp"
#include "../../core/streaming_contracts.hpp"

namespace tc::streaming {

void validate_public_streaming_request(const Request &request) {
    require(request.streaming_selector &&
                request.streaming_selector->active(),
            "streaming_selector_required");
    validate_streaming_selector(*request.streaming_selector);
    require(!request.streaming.specified(),
            "streaming_config_conflict: public selector conflicts with "
            "manual streaming layout");
    require(!request.residency_specified &&
                !request.memory_budget_specified &&
                !request.streaming_offload_specified &&
                !request.memory_budget_bytes &&
                !request.streaming_offload,
            "streaming_config_conflict: public selector conflicts with "
            "explicit legacy residency/budget/offload");
    require(!request.memory_constrained.enabled,
            "streaming_config_conflict: public presets are not "
            "bounded-memory certified");
    require(request.execution == "gpu" &&
                request.ane_manifest.empty() &&
                request.encoder_ane_manifest.empty(),
            "streaming_route_unsupported: public streaming v1 requires "
            "GPU-only execution without ANE manifests");
    require(!request.allow_approximation,
            "streaming_route_unsupported: public streaming v1 excludes "
            "approximation");
    require(!request.compile_gpu,
            "streaming_route_unsupported: public streaming v1 excludes "
            "compiled GPU graphs");
    require(request.loras.empty(),
            "streaming_route_unsupported: public streaming v1 excludes "
            "LoRA");
    require(request.quantized_cache.empty(),
            "streaming_route_unsupported: public streaming v1 excludes "
            "quantized caches");
}

} // namespace tc::streaming
