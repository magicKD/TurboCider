#pragma once

#include "../../core/streaming_contracts.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tc::streaming {

inline constexpr uint64_t gib = 1ull << 30;

struct PresetWorkload {
    std::string model, operation, execution;
    std::string device_class, execution_container;
    uint32_t width = 0, height = 0, frames = 1, steps = 0;
    bool audio = false;
};

struct PresetCalibration {
    bool complete = false;
    uint64_t calibrated_request_bytes = 0;
    std::string scope, estimator_revision;
};

struct StreamingPresetRecord {
    std::string id, catalog_revision, release_channel;
    uint32_t revision = 0;
    bool revoked = false;
    PresetWorkload workload;
    StreamingConfig canonical_config;
    PresetCalibration calibration;
    uint64_t minimum_physical_memory_bytes = 0;
    uint64_t maximum_physical_memory_bytes = 0;
    uint64_t logical_read_bytes = 0;
    uint32_t performance_rank = 0;
    std::string evidence_digest;
};

struct StreamingPresetCatalog {
    std::string revision;
    std::vector<StreamingPresetRecord> records;
};

struct PresetResolveQuery {
    PresetWorkload workload;
    uint64_t target_request_memory_bytes = 0;
    uint64_t physical_memory_bytes = 0;
    std::optional<std::string> preset_id, catalog_revision;
    std::optional<uint32_t> preset_revision;
    bool allow_experimental = true;
};

struct PresetResolution {
    std::optional<StreamingPresetRecord> selected;
    std::string rejection_code;
    std::vector<std::string> rejected_preset_ids;
};

uint64_t streaming_target_margin_bytes(uint64_t target);
bool supported_streaming_target(uint64_t target) noexcept;
PresetResolution resolve_streaming_preset(
    const PresetResolveQuery &, const StreamingPresetCatalog &);
const StreamingPresetCatalog &production_streaming_preset_catalog();

} // namespace tc::streaming
