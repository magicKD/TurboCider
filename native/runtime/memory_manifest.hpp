#pragma once

#include <atomic>

#include "memory_accounting.hpp"
#include "memory_schedule.hpp"
#include "memory_policy.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tc {

enum class AllocationLifetime : uint8_t {
    Request,
    Stage,
    Block,
    Tile,
    CommandBuffer,
    Export,
};

enum class UpperProvenance : uint8_t {
    ExactMetadata,
    ExactShapeFormula,
    ValidatedEnvelope,
    HeuristicNotExecutable,
};

const char *allocation_lifetime_name(AllocationLifetime);
const char *upper_provenance_name(UpperProvenance);

struct MemoryCheckpointFileIdentity {
    std::string logical_name;
    uint64_t size_bytes = 0;
    std::string sha256;
};

std::string memory_checkpoint_identity_digest(
    const std::vector<MemoryCheckpointFileIdentity> &files);

struct AllocationSiteSpec {
    std::string site_id;
    std::string component;
    std::string stage;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    AllocationLifetime lifetime = AllocationLifetime::Request;
    UpperProvenance provenance = UpperProvenance::HeuristicNotExecutable;
    bool required = true;
    bool asynchronous = false;
    bool aliasable = false;
    uint64_t guard_threshold_bytes = 0;
};

struct AllocationInstance {
    std::string site_id;
    uint32_t instance_id = 0;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    uint64_t alias_group = 0;
    std::string last_use_event;
};

struct MemoryManifest {
    std::string schema = "turbocider.memory_manifest.v1";
    std::string candidate_id;
    std::string checkpoint_digest;
    std::string backend_revision;
    std::string runtime_revision;
    std::vector<AllocationSiteSpec> sites;
    std::vector<AllocationInstance> instances;

    void validate() const;
    std::string digest() const;
    const AllocationSiteSpec *find_site(std::string_view site_id) const;
};

/* Deterministic construction helper used by metadata-only probes and
 * campaign tooling.  It owns no model tensors and performs no I/O; callers
 * provide already validated metadata/layout values. */
class MemoryManifestBuilder {
  public:
    MemoryManifestBuilder(std::string candidate_id,
                          std::string checkpoint_digest,
                          std::string backend_revision,
                          std::string runtime_revision);

    MemoryManifestBuilder(const MemoryManifestBuilder &) = delete;
    MemoryManifestBuilder &operator=(const MemoryManifestBuilder &) = delete;
    MemoryManifestBuilder(MemoryManifestBuilder &&) noexcept = default;
    MemoryManifestBuilder &operator=(MemoryManifestBuilder &&) noexcept = default;

    void add_site(AllocationSiteSpec site);
    void add_instance(AllocationInstance instance);
    MemoryManifest build();

  private:
    MemoryManifest manifest_;
    bool built_ = false;
};

struct MemoryCandidateKey {
    std::string adapter;
    std::string model_id;
    std::string checkpoint_digest;
    std::string backend;
    std::string dtype;
    std::string model_variant;
    std::string operation;
    std::string shape_bucket;
    std::string sampler_mode;
    unsigned refill_slots = 0;
    std::string tiling_mode;
    std::string runtime_revision;
    std::string device_family;

    bool operator==(const MemoryCandidateKey &other) const;
    bool operator<(const MemoryCandidateKey &other) const;
    std::string canonical() const;
    std::string digest() const;
};

struct MemoryCapabilityRecord {
    MemoryCandidateKey key;
    MemoryCapabilityLevel level = MemoryCapabilityLevel::Declared;
    MemoryCertificationState state = MemoryCertificationState::Unsupported;
    std::string manifest_digest;
    std::string evidence_digest;
    std::vector<unsigned> verified_refill_slots;
    std::vector<std::string> verified_tiling_modes;
    uint64_t framework_upper_bytes = 0;
    std::string framework_provenance;
    uint64_t maximum_validated_upper_bytes = 0;
    bool require_explicit_epoch = true;
    bool require_explicit_schedule = false;
    std::vector<MemoryScheduleBindingSpec> schedule;
    bool release_enabled = false;
};

struct MemoryCapabilityMatch {
    bool matched = false;
    std::string reason;
    std::optional<MemoryCapabilityRecord> record;
};

struct MemoryDeviceIdentity {
    std::string device_family;
    std::string runtime_revision;
};

/* A model session constructs this from its private model root without
 * materializing model tensors or creating GPU buffers.  Catalog lookup is
 * deliberately kept outside the session so model code cannot grant itself
 * execution capability. */
struct MemoryCapabilityProbe {
    MemoryCandidateKey key;
    MemoryManifest manifest;
};

class MemoryCapabilityRegistry {
  public:
    void add(MemoryCapabilityRecord record);
    MemoryCapabilityMatch lookup(const MemoryCandidateKey &key,
                                 std::string_view manifest_digest,
                                 bool allow_experimental = false) const;
    size_t size() const { return records_.size(); }

  private:
    std::vector<MemoryCapabilityRecord> records_;
};

/* Portable SHA-256 used for plan/capability identity.  The runtime does not
 * depend on Foundation or Objective-C JSON; platform bridges may use this
 * helper when converting a canonical manifest to a release record. */
std::string memory_sha256_hex(std::string_view value);

// Hash exactly [0, bytes) from a borrowed descriptor with bounded heap memory.
// Uses pread so callers' offsets are preserved. The caller must revalidate the
// file generation after hashing before treating this as an artifact identity.
std::string memory_sha256_fd(int fd, uint64_t bytes,
                             const std::atomic<bool> *cancelled = nullptr);

} // namespace tc
