#pragma once

#include "../../core/contracts.hpp"
#include "layout.hpp"
#include "preset_catalog.hpp"
#include "source_lease.hpp"

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
    // Optional for legacy/private probes. Public adapters must provide a
    // request-scoped lease before they can report source_lease_verified.
    virtual const SourceLease *source_lease() const noexcept { return nullptr; }
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
    // Metadata/source lease check performed under the engine/global execution
    // locks immediately before the adapter is allowed to open readers or
    // submit GPU work. Implementations must not allocate GPU payloads here.
    virtual void revalidate_source() const {}
    // Optional for legacy/private snapshots. Public snapshots override this
    // with the same immutable lease used by their readers.
    virtual const SourceLease *source_lease() const noexcept { return nullptr; }
};

// Shared value implementation used by model adapters after their
// model-specific metadata has been converted to the common identity/lease
// contract. It deliberately owns no GPU payload or worker.
class ValueModelStreamingProbe : public ModelStreamingProbe {
  public:
    struct Values {
        std::string model_id;
        PresetSourceIdentity source;
        PresetWorkload workload;
        PresetRuntimeIdentity runtime;
        std::string component_policy_revision;
        std::shared_ptr<const SourceLease> lease;
    };

    explicit ValueModelStreamingProbe(Values);

    std::string_view model_id() const noexcept override;
    const PresetSourceIdentity &source_identity() const noexcept override;
    const PresetWorkload &workload_identity() const noexcept override;
    const PresetRuntimeIdentity &runtime_identity() const noexcept override;
    std::string_view component_policy_revision() const noexcept override;
    const SourceLease *source_lease() const noexcept override;
    const SourceLease &lease() const;

  private:
    Values values_;
};

class ValueModelStreamingSnapshot : public ModelStreamingSnapshot {
  public:
    struct Values {
        std::string model_id;
        PresetSourceIdentity source;
        PresetRuntimeIdentity runtime;
        Descriptor descriptor;
        Layout layout;
        std::string component_policy_revision;
        std::shared_ptr<const SourceLease> lease;
    };

    explicit ValueModelStreamingSnapshot(Values);

    std::string_view model_id() const noexcept override;
    const PresetSourceIdentity &source_identity() const noexcept override;
    const PresetRuntimeIdentity &runtime_identity() const noexcept override;
    const Descriptor &descriptor() const noexcept override;
    const Layout &layout() const noexcept override;
    std::string_view component_policy_revision() const noexcept override;
    void revalidate_source() const override;
    const SourceLease *source_lease() const noexcept override;
    const SourceLease &lease() const;

  private:
    Values values_;
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
                       uint64_t source_generation,
                       std::string resolution_digest);

    std::string preset_digest_, source_digest_, workload_digest_;
    std::string runtime_digest_, layout_digest_;
    std::string component_policy_revision_;
    std::string device_digest_;
    uint64_t source_generation_ = 0;
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
