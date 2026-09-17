#pragma once

#include <cstdint>
#include <array>
#include <map>
#include <optional>
#include <string>

namespace tc {

inline constexpr uint64_t streaming_gib = 1ull << 30;
inline constexpr std::array<uint64_t, 5> public_streaming_targets{
    8 * streaming_gib, 10 * streaming_gib, 12 * streaming_gib,
    16 * streaming_gib, 20 * streaming_gib};

// Presence is part of the contract: zero, false and absent are distinct.
struct StreamingStageConfig {
    std::optional<std::string> residency;
    std::optional<uint32_t> block_group_size, slot_count, resident_prefix_blocks;
    std::optional<uint32_t> prefetch_distance, io_workers;
};

struct StreamingConfig {
    std::optional<bool> enabled;
    std::optional<uint32_t> schema_version;
    std::optional<std::string> selection, retention;
    std::map<std::string, StreamingStageConfig> stages;
    // Dotted field path -> request/profile/descriptor. Not part of layout identity.
    std::map<std::string, std::string> provenance;

    bool active() const noexcept { return enabled.value_or(false); }
    bool specified() const noexcept {
        return enabled || schema_version || selection || retention || !stages.empty();
    }
};

// Public selectors describe user intent only. They never carry an execution
// authority and must be resolved against the native, read-only preset catalog
// before a model adapter may consume the resulting manual layout.
struct StreamingSelector {
    std::optional<bool> enabled;
    std::optional<uint32_t> schema_version;
    std::optional<std::string> selection, retention;
    std::optional<uint64_t> target_request_memory_bytes;
    std::optional<std::string> preset_id, catalog_revision;
    std::optional<uint32_t> preset_revision;
    std::optional<std::string> expected_resolution_digest;
    // Dotted field path -> request/profile. Not part of selector identity.
    std::map<std::string, std::string> provenance;

    bool active() const noexcept { return enabled.value_or(false); }
    bool specified() const noexcept {
        return enabled || schema_version || selection || retention ||
               target_request_memory_bytes || preset_id || preset_revision ||
               catalog_revision || expected_resolution_digest;
    }
};

// Merge before validation. A residency change replaces the entire stage.
void overlay_streaming_config(StreamingConfig &, const StreamingConfig &);
void validate_streaming_config(const StreamingConfig &);
void validate_streaming_selector(const StreamingSelector &);

} // namespace tc
