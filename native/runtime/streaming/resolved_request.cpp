#include "resolved_request.hpp"

namespace tc::streaming {

StreamingAuthority::StreamingAuthority(
        std::string preset_digest, std::string source_digest,
        std::string workload_digest, std::string runtime_digest,
        std::string layout_digest,
        std::string component_policy_revision,
        std::string device_digest, uint64_t source_generation,
        std::string source_binding_digest,
        std::string resolution_digest)
    : preset_digest_(std::move(preset_digest)),
      source_digest_(std::move(source_digest)),
      workload_digest_(std::move(workload_digest)),
      runtime_digest_(std::move(runtime_digest)),
      layout_digest_(std::move(layout_digest)),
      component_policy_revision_(std::move(component_policy_revision)),
      device_digest_(std::move(device_digest)),
      source_binding_digest_(std::move(source_binding_digest)),
      source_generation_(source_generation),
      resolution_digest_(std::move(resolution_digest)) {}

bool StreamingAuthority::matches(
        const StreamingPresetRecord &record,
        const ModelStreamingSnapshot &snapshot,
        const StreamingDeviceIdentity &device) const {
    return preset_digest_ == record.canonical_record_digest &&
        source_digest_ ==
            streaming_source_identity_digest(snapshot.source_identity()) &&
        workload_digest_ ==
            streaming_workload_identity_digest(record.workload) &&
        runtime_digest_ ==
            streaming_runtime_identity_digest(snapshot.runtime_identity()) &&
        layout_digest_ == snapshot.layout().digest &&
        component_policy_revision_ == snapshot.component_policy_revision() &&
        device_digest_ == streaming_device_identity_digest(device) &&
        snapshot.source_lease() != nullptr &&
        snapshot.source_lease()->generation() == source_generation_ &&
        snapshot.source_lease()->digest() ==
            source_binding_digest_ &&
        streaming_source_matches_lease(snapshot.source_identity(), *snapshot.source_lease());
}

} // namespace tc::streaming
