#include "public_runtime.hpp"

#include "preset_resolver.hpp"
#include "public_request_validation.hpp"
#include "../session.hpp"
#include "../../core/common.hpp"

namespace tc::streaming {

PublicStreamingCoordinator::PublicStreamingCoordinator(
        ModelSession &session, std::string engine_model_id,
        std::string execution_container,
        const StreamingCatalogProvider &catalog_provider)
    : session_(session), engine_model_id_(std::move(engine_model_id)),
      execution_container_(std::move(execution_container)),
      catalog_provider_(catalog_provider) {
    require(!engine_model_id_.empty(), "streaming_engine_unavailable");
    require(!execution_container_.empty(),
            "streaming_execution_container_mismatch");
}

void PublicStreamingCoordinator::preflight(const Request &request) const {
    require(request.model == engine_model_id_,
            "streaming_engine_model_mismatch");
    validate_public_streaming_request(request);
    // This check deliberately precedes make_plan/model validation at the API
    // layer while still allowing pure host tests to inject a read-only
    // provider directly into this C++ coordinator.
    require(!catalog_provider_.catalog().records.empty(),
            "catalog_has_no_public_records");
}

std::shared_ptr<const ResolvedRequestExecution>
PublicStreamingCoordinator::resolve_normalized(
        Request request, const StreamingDeviceIdentity &device) const {
    preflight(request);
    PublicResolveInput input{request, device, execution_container_};
    auto probe = session_.probe_public_streaming(input);
    require(probe != nullptr, "streaming_public_probe_unavailable");
    require(probe->model_id() == engine_model_id_ &&
                probe->workload_identity().model == engine_model_id_ &&
                probe->workload_identity().execution_container ==
                    execution_container_,
            "streaming_probe_identity_mismatch");

    auto selected = PublicPresetResolver::select(
        *request.streaming_selector, *probe, device,
        catalog_provider_.catalog());
    require(probe->component_policy_revision() ==
                selected.record.plan.component_policy_revision,
            "streaming_probe_identity_mismatch");
    auto snapshot = session_.compile_public_streaming(
        probe, selected.record);
    require(snapshot != nullptr,
            "streaming_public_snapshot_unavailable");
    auto selection = PublicPresetResolver::authorize(
        selected, *probe, *snapshot, device);

    request.streaming_selector.reset();
    request.streaming_selector_requested.reset();
    request.streaming = selection.record.plan.canonical_config;
    request.streaming_requested = request.streaming;
    const auto request_digest = streaming_workload_identity_digest(
        probe->workload_identity());
    return std::make_shared<const ResolvedRequestExecution>(
        ResolvedRequestExecution{
            std::move(request), std::move(selection), std::move(probe),
            std::move(snapshot), request_digest});
}

void PublicStreamingCoordinator::revalidate(
        const ResolvedRequestExecution &execution,
        const StreamingDeviceIdentity &current_device) const {
    require(execution.probe != nullptr &&
                execution.model_snapshot != nullptr,
            "streaming_authority_mismatch");
    execution.model_snapshot->revalidate_source();
    const auto replay = PublicPresetResolver::select(
        execution.selection.exact_selector, *execution.probe,
        current_device, catalog_provider_.catalog());
    require(replay.record.canonical_record_digest ==
                execution.selection.record.canonical_record_digest,
            "streaming_resolution_stale");
    require(execution.selection.authority &&
                execution.selection.authority->matches(
                    execution.selection.record,
                    *execution.model_snapshot, current_device),
            "streaming_authority_mismatch");
    require(execution.selection.exact_selector.expected_resolution_digest &&
                *execution.selection.exact_selector
                     .expected_resolution_digest ==
                    execution.selection.resolution_digest,
            "streaming_resolution_stale");
}

} // namespace tc::streaming
