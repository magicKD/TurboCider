#pragma once

#include "../../core/streaming_contracts.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tc { struct Request; }
namespace tc::streaming {

inline constexpr uint64_t gib = 1ull << 30;

struct PresetTokenShape {
    std::string encoder, tokenizer_revision, template_revision;
    uint32_t valid_rows = 0, padded_rows = 0, compute_rows = 0;
    bool operator==(const PresetTokenShape &) const = default;
};

struct PresetWorkload {
    std::string model, operation, execution;
    std::string device_class, execution_container;
    uint32_t width = 0, height = 0, frames = 1, fps = 0, steps = 0;
    uint32_t batch = 1;
    bool audio = false, dynamic_text = false, approximation = false;
    std::string conditioning_revision, vae_policy_revision, feature_digest;
    std::vector<PresetTokenShape> token_shapes;
    bool operator==(const PresetWorkload &) const = default;
};

struct PresetSourceIdentity {
    std::string model_variant, weight_format;
    std::string artifact_manifest_digest, source_snapshot_digest;
    bool operator==(const PresetSourceIdentity &) const = default;
};

struct PresetRuntimeIdentity {
    std::string turbocider_build_id, runtime_revision, adapter_revision;
    std::string reader_revision, kernel_revision, allocator_policy_revision;
    bool operator==(const PresetRuntimeIdentity &) const = default;
};

struct PresetDeviceQualification {
    uint64_t minimum_physical_memory_bytes = 0;
    uint64_t maximum_physical_memory_bytes = 0;
};

struct PresetPlan {
    StreamingConfig canonical_config;
    std::string layout_digest, component_policy_revision;
    std::string pass_transition, multi_pool_policy;
};

struct PresetCalibration {
    bool complete = false;
    uint64_t calibrated_request_bytes = 0;
    std::string scope, estimator_revision, calibration_id;
    std::string execution_container, evidence_digest;
    uint64_t confirmation_sample_count = 0;
    uint64_t maximum_sample_gap_ns = 0;
};

struct PresetPerformance {
    uint32_t rank = 0;
    uint64_t logical_read_bytes = 0;
    std::string profile_id, comparison_kind, confidence_status;
    std::string evidence_digest;
};

struct PresetRelease {
    std::string channel;
    bool revoked = false;
    std::string reviewed_commit, review_digest;
};

struct StreamingPresetRecord {
    std::string id, catalog_revision;
    uint32_t revision = 0;
    PresetSourceIdentity source;
    PresetWorkload workload;
    PresetRuntimeIdentity runtime;
    PresetDeviceQualification device;
    PresetPlan plan;
    PresetCalibration calibration;
    PresetPerformance performance;
    PresetRelease release;
    std::string canonical_record_digest;
};

struct StreamingPresetCatalog {
    std::string revision;
    std::vector<StreamingPresetRecord> records;
};

struct PresetResolveQuery {
    PresetSourceIdentity source;
    PresetWorkload workload;
    PresetRuntimeIdentity runtime;
    uint64_t target_request_memory_bytes = 0;
    uint64_t physical_memory_bytes = 0;
    std::optional<std::string> preset_id, catalog_revision;
    std::optional<uint32_t> preset_revision;
    bool allow_experimental = true;
    bool require_exact_identity = false;
};

struct PresetResolution {
    std::optional<StreamingPresetRecord> selected;
    std::string rejection_code;
    std::vector<std::string> rejected_preset_ids;
};

// Discovery only. A candidate is not execution authority and must pass the
// installed-model resolver before a caller may present it as available.
struct PresetCandidateResolution {
    std::optional<StreamingPresetRecord> candidate;
    std::string rejection_code;
};

PresetWorkload basic_streaming_workload(const Request &,
    std::string device_class, std::string execution_container);

uint64_t streaming_target_margin_bytes(uint64_t target);
bool supported_streaming_target(uint64_t target) noexcept;

std::string canonical_streaming_preset_record(
    const StreamingPresetRecord &);
std::string streaming_preset_record_digest(
    const StreamingPresetRecord &);
StreamingPresetRecord finalize_streaming_preset_record(
    StreamingPresetRecord);
void validate_streaming_preset_record(
    const StreamingPresetRecord &, std::string_view catalog_revision);

PresetResolution resolve_streaming_preset(
    const PresetResolveQuery &, const StreamingPresetCatalog &);
PresetCandidateResolution find_streaming_preset_candidate(
    const PresetResolveQuery &, const StreamingPresetCatalog &);
const StreamingPresetCatalog &production_streaming_preset_catalog();

} // namespace tc::streaming
