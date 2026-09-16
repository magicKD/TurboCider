#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace tc {

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

// Merge before validation. A residency change replaces the entire stage.
void overlay_streaming_config(StreamingConfig &, const StreamingConfig &);
void validate_streaming_config(const StreamingConfig &);

} // namespace tc
