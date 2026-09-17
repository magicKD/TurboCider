#include "preset_resolver.hpp"

#include "canonical_encoding.hpp"

#include <stdexcept>

namespace tc::streaming {
namespace {

[[noreturn]] void resolution_error(const std::string &code) {
    throw std::invalid_argument(code);
}

void require_resolution(bool value, const std::string &code) {
    if (!value) resolution_error(code);
}

} // namespace

std::string streaming_source_identity_digest(
        const PresetSourceIdentity &source) {
    CanonicalEncoder out("tc-streaming-source-identity-v1");
    out.string_field("model_variant", source.model_variant);
    out.string_field("weight_format", source.weight_format);
    out.string_field(
        "artifact_manifest_digest", source.artifact_manifest_digest);
    out.string_field("source_snapshot_digest", source.source_snapshot_digest);
    return out.sha256();
}

std::string streaming_workload_identity_digest(
        const PresetWorkload &workload) {
    CanonicalEncoder out("tc-streaming-workload-identity-v1");
    out.string_field("model", workload.model);
    out.string_field("operation", workload.operation);
    out.string_field("execution", workload.execution);
    out.string_field("device_class", workload.device_class);
    out.string_field("execution_container", workload.execution_container);
    out.unsigned_field("width", workload.width);
    out.unsigned_field("height", workload.height);
    out.unsigned_field("frames", workload.frames);
    out.unsigned_field("fps", workload.fps);
    out.unsigned_field("steps", workload.steps);
    out.unsigned_field("batch", workload.batch);
    out.boolean_field("audio", workload.audio);
    out.boolean_field("dynamic_text", workload.dynamic_text);
    out.boolean_field("approximation", workload.approximation);
    out.string_field(
        "conditioning_revision", workload.conditioning_revision);
    out.string_field("vae_policy_revision", workload.vae_policy_revision);
    out.string_field("feature_digest", workload.feature_digest);
    out.begin_list("token_shapes", workload.token_shapes.size());
    for (const auto &token : workload.token_shapes) {
        out.string_field("token.encoder", token.encoder);
        out.string_field(
            "token.tokenizer_revision", token.tokenizer_revision);
        out.string_field("token.template_revision", token.template_revision);
        out.unsigned_field("token.valid_rows", token.valid_rows);
        out.unsigned_field("token.padded_rows", token.padded_rows);
        out.unsigned_field("token.compute_rows", token.compute_rows);
    }
    return out.sha256();
}

std::string streaming_runtime_identity_digest(
        const PresetRuntimeIdentity &runtime) {
    CanonicalEncoder out("tc-streaming-runtime-identity-v1");
    out.string_field("turbocider_build_id", runtime.turbocider_build_id);
    out.string_field("runtime_revision", runtime.runtime_revision);
    out.string_field("adapter_revision", runtime.adapter_revision);
    out.string_field("reader_revision", runtime.reader_revision);
    out.string_field("kernel_revision", runtime.kernel_revision);
    out.string_field(
        "allocator_policy_revision", runtime.allocator_policy_revision);
    return out.sha256();
}

std::string streaming_device_identity_digest(
        const StreamingDeviceIdentity &device) {
    CanonicalEncoder out("tc-streaming-device-identity-v1");
    out.string_field("gpu_name", device.gpu_name);
    out.string_field("device_class", device.device_class);
    out.string_field("os_build_family", device.os_build_family);
    out.unsigned_field("physical_memory_bytes", device.physical_memory_bytes);
    return out.sha256();
}

namespace {

std::string resolution_digest(
        const StreamingPresetRecord &record,
        const ModelStreamingProbe &probe,
        const ModelStreamingSnapshot &snapshot,
        const StreamingDeviceIdentity &device) {
    CanonicalEncoder out("tc-streaming-resolution-v1");
    out.string_field("preset_digest", record.canonical_record_digest);
    out.string_field(
        "source_digest",
        streaming_source_identity_digest(probe.source_identity()));
    out.string_field(
        "workload_digest",
        streaming_workload_identity_digest(probe.workload_identity()));
    out.string_field(
        "runtime_digest",
        streaming_runtime_identity_digest(probe.runtime_identity()));
    out.string_field("layout_digest", snapshot.layout().digest);
    out.string_field(
        "component_policy_revision", snapshot.component_policy_revision());
    out.string_field(
        "device_digest", streaming_device_identity_digest(device));
    return out.sha256();
}

} // namespace

SelectedStreamingPreset PublicPresetResolver::select(
        const StreamingSelector &selector,
        const ModelStreamingProbe &probe,
        const StreamingDeviceIdentity &device,
        const StreamingPresetCatalog &catalog) {
    validate_streaming_selector(selector);
    require_resolution(selector.active(), "streaming_selector_required");
    require_resolution(probe.model_id() == probe.workload_identity().model,
                       "streaming_probe_identity_mismatch");
    require_resolution(
        !probe.workload_identity().execution_container.empty(),
        "streaming_probe_identity_mismatch");
    require_resolution(
        probe.workload_identity().device_class == device.device_class,
        "unvalidated_device");

    PresetResolveQuery query;
    query.source = probe.source_identity();
    query.workload = probe.workload_identity();
    query.runtime = probe.runtime_identity();
    query.target_request_memory_bytes =
        *selector.target_request_memory_bytes;
    query.physical_memory_bytes = device.physical_memory_bytes;
    query.require_exact_identity = true;
    if (*selector.selection == "preset") {
        query.preset_id = selector.preset_id;
        query.preset_revision = selector.preset_revision;
        query.catalog_revision = selector.catalog_revision;
    }
    auto resolution = resolve_streaming_preset(query, catalog);
    if (!resolution.selected)
        resolution_error(resolution.rejection_code.empty()
                             ? "streaming_resolution_failed"
                             : resolution.rejection_code);
    return {*resolution.selected, selector};
}

ResolvedStreamingSelection PublicPresetResolver::authorize(
        const SelectedStreamingPreset &selected,
        const ModelStreamingProbe &probe,
        const ModelStreamingSnapshot &snapshot,
        const StreamingDeviceIdentity &device) {
    const auto &record = selected.record;
    require_resolution(record.source == probe.source_identity(),
                       "artifact_changed");
    require_resolution(record.workload == probe.workload_identity(),
                       "streaming_resolution_stale");
    require_resolution(record.runtime == probe.runtime_identity(),
                       "streaming_resolution_stale");
    require_resolution(snapshot.model_id() == probe.model_id(),
                       "streaming_authority_mismatch");
    require_resolution(snapshot.source_identity() == probe.source_identity(),
                       "artifact_changed");
    require_resolution(snapshot.runtime_identity() == probe.runtime_identity(),
                       "streaming_resolution_stale");
    require_resolution(snapshot.layout().digest == record.plan.layout_digest,
                       "streaming_actual_plan_mismatch");
    require_resolution(snapshot.component_policy_revision() ==
                           record.plan.component_policy_revision,
                       "streaming_actual_plan_mismatch");
    require_resolution(snapshot.layout().materializations_complete,
                       "streaming_source_identity_incomplete");

    const auto digest = resolution_digest(record, probe, snapshot, device);
    if (selected.requested_selector.expected_resolution_digest)
        require_resolution(
            *selected.requested_selector.expected_resolution_digest == digest,
            "streaming_resolution_stale");

    StreamingSelector exact;
    exact.schema_version = 2;
    exact.enabled = true;
    exact.selection = "preset";
    exact.retention = "request";
    exact.target_request_memory_bytes =
        selected.requested_selector.target_request_memory_bytes;
    exact.preset_id = record.id;
    exact.preset_revision = record.revision;
    exact.catalog_revision = record.catalog_revision;
    exact.expected_resolution_digest = digest;

    auto authority = std::shared_ptr<const StreamingAuthority>(
        new StreamingAuthority(
            record.canonical_record_digest,
            streaming_source_identity_digest(snapshot.source_identity()),
            streaming_workload_identity_digest(record.workload),
            streaming_runtime_identity_digest(snapshot.runtime_identity()),
            snapshot.layout().digest,
            std::string(snapshot.component_policy_revision()),
            streaming_device_identity_digest(device), digest));
    return {record, selected.requested_selector, exact, device, digest,
            std::move(authority)};
}

} // namespace tc::streaming
