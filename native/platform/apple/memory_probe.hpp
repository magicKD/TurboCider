#pragma once

#include "../../runtime/session.hpp"

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>

namespace tc {

enum class MemoryModelRootTrust : uint8_t {
    ExternalMutable,
    OwnedImmutable,
};

/* Content hash cache for metadata-only capability probes. Device/inode,
 * size, ctime and mtime are cache invalidation hints only; a cache miss hashes
 * one opened descriptor and verifies its metadata did not change while the
 * complete file was read. */
class MemoryCheckpointHashCache {
  public:
    std::string sha256(const std::filesystem::path &,
                       MemoryModelRootTrust) const;

  private:
    struct Entry {
        uint64_t device = 0;
        uint64_t inode = 0;
        uintmax_t size_bytes = 0;
        int64_t change_time_seconds = 0;
        int64_t change_time_nanoseconds = 0;
        int64_t modify_time_seconds = 0;
        int64_t modify_time_nanoseconds = 0;
        std::string sha256;
    };
    mutable std::mutex mutex_;
    mutable std::map<std::filesystem::path, Entry> entries_;
};

std::filesystem::path memory_capability_probe_path(
    const std::filesystem::path &model_root, std::string_view adapter,
    const Request &, unsigned refill_slots);

MemoryCandidateKey expected_memory_candidate_key(
    const ExecutionPlan &, const MemoryDeviceIdentity &,
    std::string checkpoint_digest = {});

/* Load and validate an exact, model-preparation-generated probe sidecar.
 * Missing sidecars return nullopt so production preflight remains fail-
 * closed. Existing but malformed or stale sidecars throw a typed memory
 * policy error. This function performs file/hash/JSON I/O only: no Metal,
 * MLX, graph compilation, tensor materialization, cache clear, or unload. */
std::optional<MemoryCapabilityProbe> load_memory_capability_probe(
    const std::filesystem::path &model_root,
    const std::filesystem::path &sidecar,
    const ExecutionPlan &, const MemoryDeviceIdentity &,
    MemoryModelRootTrust, const MemoryCheckpointHashCache &);

} // namespace tc
