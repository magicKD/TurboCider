#include "resolved_request.hpp"

namespace tc::streaming {

StreamingAuthority::StreamingAuthority(
        std::string preset_digest, std::string source_digest,
        std::string workload_digest, std::string runtime_digest,
        std::string layout_digest,
        std::string component_policy_revision,
        std::string device_digest, std::string resolution_digest)
    : preset_digest_(std::move(preset_digest)),
      source_digest_(std::move(source_digest)),
      workload_digest_(std::move(workload_digest)),
      runtime_digest_(std::move(runtime_digest)),
      layout_digest_(std::move(layout_digest)),
      component_policy_revision_(std::move(component_policy_revision)),
      device_digest_(std::move(device_digest)),
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
        device_digest_ == streaming_device_identity_digest(device);
}

} // namespace tc::streaming
