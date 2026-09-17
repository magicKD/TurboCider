#pragma once

#include "../../core/contracts.hpp"
#include "layout.hpp"
#include "preset_catalog.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace tc::streaming {

struct StreamingDeviceIdentity {
    std::string gpu_name, device_class, os_build_family;
    uint64_t physical_memory_bytes = 0;
    bool operator==(const StreamingDeviceIdentity &) const = default;
};

std::string streaming_source_identity_digest(
    const PresetSourceIdentity &);
std::string streaming_workload_identity_digest(
    const PresetWorkload &);
std::string streaming_runtime_identity_digest(
    const PresetRuntimeIdentity &);
std::string streaming_device_identity_digest(
    const StreamingDeviceIdentity &);

struct PublicResolveInput {
    Request request;
    StreamingDeviceIdentity device;
    std::string execution_container;
};

class ModelStreamingProbe {
  public:
    virtual ~ModelStreamingProbe() = default;
    virtual std::string_view model_id() const noexcept = 0;
    virtual const PresetSourceIdentity &source_identity() const noexcept = 0;
    virtual const PresetWorkload &workload_identity() const noexcept = 0;
    virtual const PresetRuntimeIdentity &runtime_identity() const noexcept = 0;
    virtual std::string_view component_policy_revision() const noexcept = 0;
};

class ModelStreamingSnapshot {
  public:
    virtual ~ModelStreamingSnapshot() = default;
    virtual std::string_view model_id() const noexcept = 0;
    virtual const PresetSourceIdentity &source_identity() const noexcept = 0;
    virtual const PresetRuntimeIdentity &runtime_identity() const noexcept = 0;
    virtual const Descriptor &descriptor() const noexcept = 0;
    virtual const Layout &layout() const noexcept = 0;
    virtual std::string_view component_policy_revision() const noexcept = 0;
};

struct SelectedStreamingPreset {
    StreamingPresetRecord record;
    StreamingSelector requested_selector;
};

class StreamingAuthority final {
  public:
    StreamingAuthority(const StreamingAuthority &) = delete;
    StreamingAuthority &operator=(const StreamingAuthority &) = delete;

    const std::string &resolution_digest() const noexcept {
        return resolution_digest_;
    }
    bool matches(const StreamingPresetRecord &record,
                 const ModelStreamingSnapshot &snapshot,
                 const StreamingDeviceIdentity &device) const;

  private:
    friend class PublicPresetResolver;
    StreamingAuthority(std::string preset_digest, std::string source_digest,
                       std::string workload_digest, std::string runtime_digest,
                       std::string layout_digest,
                       std::string component_policy_revision,
                       std::string device_digest,
                       std::string resolution_digest);

    std::string preset_digest_, source_digest_, workload_digest_;
    std::string runtime_digest_, layout_digest_;
    std::string component_policy_revision_;
    std::string device_digest_;
    std::string resolution_digest_;
};

struct ResolvedStreamingSelection {
    StreamingPresetRecord record;
    StreamingSelector requested_selector;
    StreamingSelector exact_selector;
    StreamingDeviceIdentity device;
    std::string resolution_digest;
    std::shared_ptr<const StreamingAuthority> authority;
};

// Immutable data shared by one resolved generate call.  Model-specific source
// leases live in model_snapshot; mutable tickets, workers and slots do not.
struct ResolvedRequestExecution {
    Request request;
    ResolvedStreamingSelection selection;
    std::shared_ptr<const ModelStreamingProbe> probe;
    std::shared_ptr<const ModelStreamingSnapshot> model_snapshot;
    std::string request_digest;
};

} // namespace tc::streaming
