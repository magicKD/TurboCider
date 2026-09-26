#include "resolved_request.hpp"

#include "../../core/common.hpp"

#include <utility>

namespace tc::streaming {
namespace {

void value_require(bool condition, const char *reason) {
    require(condition, std::string("streaming_value: ") + reason);
}

void validate_common(std::string_view model_id,
                     const PresetSourceIdentity &source,
                     const PresetRuntimeIdentity &runtime,
                     std::string_view component,
                     const std::shared_ptr<const SourceLease> &lease) {
    value_require(!model_id.empty(), "model id is empty");
    value_require(!source.model_variant.empty(), "source model variant is empty");
    value_require(!source.weight_format.empty(), "source weight format is empty");
    value_require(!runtime.runtime_revision.empty(),
                  "runtime revision is empty");
    value_require(!runtime.adapter_revision.empty(),
                  "adapter revision is empty");
    value_require(!runtime.reader_revision.empty(),
                  "reader revision is empty");
    value_require(!runtime.kernel_revision.empty(),
                  "kernel revision is empty");
    value_require(!runtime.allocator_policy_revision.empty(),
                  "allocator revision is empty");
    value_require(!component.empty(), "component policy is empty");
    value_require(lease != nullptr, "source lease is missing");
    value_require(!lease->digest().empty(), "source lease digest is empty");
    value_require(lease->generation() != 0, "source lease generation is zero");
    value_require(streaming_source_matches_lease(source, *lease),
                  "source lease digest differs from source identity");
}

} // namespace

ValueModelStreamingProbe::ValueModelStreamingProbe(Values values)
    : values_(std::move(values)) {
    value_require(values_.workload.model == values_.model_id,
                  "workload model differs from model id");
    value_require(!values_.workload.execution_container.empty(),
                  "execution container is empty");
    validate_common(values_.model_id, values_.source, values_.runtime,
                    values_.component_policy_revision, values_.lease);
}

std::string_view ValueModelStreamingProbe::model_id() const noexcept {
    return values_.model_id;
}

const PresetSourceIdentity &ValueModelStreamingProbe::source_identity() const noexcept {
    return values_.source;
}

const PresetWorkload &ValueModelStreamingProbe::workload_identity() const noexcept {
    return values_.workload;
}

const PresetRuntimeIdentity &ValueModelStreamingProbe::runtime_identity() const noexcept {
    return values_.runtime;
}

std::string_view ValueModelStreamingProbe::component_policy_revision() const noexcept {
    return values_.component_policy_revision;
}

const SourceLease *ValueModelStreamingProbe::source_lease() const noexcept {
    return values_.lease.get();
}

const SourceLease &ValueModelStreamingProbe::lease() const {
    value_require(values_.lease != nullptr, "source lease is missing");
    return *values_.lease;
}

std::shared_ptr<const SourceLease>
ValueModelStreamingProbe::lease_ptr() const noexcept {
    return values_.lease;
}

ValueModelStreamingSnapshot::ValueModelStreamingSnapshot(Values values)
    : values_(std::move(values)) {
    value_require(values_.descriptor.model == values_.model_id,
                  "descriptor model differs from model id");
    value_require(values_.layout.materializations_complete,
                  "layout materializations are incomplete");
    value_require(!values_.layout.digest.empty(), "layout digest is empty");
    validate_common(values_.model_id, values_.source, values_.runtime,
                    values_.component_policy_revision, values_.lease);
}

std::string_view ValueModelStreamingSnapshot::model_id() const noexcept {
    return values_.model_id;
}

const PresetSourceIdentity &ValueModelStreamingSnapshot::source_identity() const noexcept {
    return values_.source;
}

const PresetRuntimeIdentity &ValueModelStreamingSnapshot::runtime_identity() const noexcept {
    return values_.runtime;
}

const Descriptor &ValueModelStreamingSnapshot::descriptor() const noexcept {
    return values_.descriptor;
}

const Layout &ValueModelStreamingSnapshot::layout() const noexcept {
    return values_.layout;
}

std::string_view ValueModelStreamingSnapshot::component_policy_revision() const noexcept {
    return values_.component_policy_revision;
}

void ValueModelStreamingSnapshot::revalidate_source() const {
    value_require(values_.lease != nullptr, "source lease is missing");
    values_.lease->revalidate_paths();
    values_.lease->revalidate_open_files();
}

const SourceLease *ValueModelStreamingSnapshot::source_lease() const noexcept {
    return values_.lease.get();
}

const SourceLease &ValueModelStreamingSnapshot::lease() const {
    value_require(values_.lease != nullptr, "source lease is missing");
    return *values_.lease;
}

std::shared_ptr<const SourceLease>
ValueModelStreamingSnapshot::lease_ptr() const noexcept {
    return values_.lease;
}

} // namespace tc::streaming
