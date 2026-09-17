#include "preset_catalog.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace tc::streaming {
namespace {

bool released(const StreamingPresetRecord &record, bool experimental) {
    return record.release_channel == "public-stable" ||
        (experimental && record.release_channel == "public-experimental");
}

bool same_workload(const PresetWorkload &left, const PresetWorkload &right) {
    return left.model == right.model &&
        left.operation == right.operation &&
        left.execution == right.execution &&
        left.device_class == right.device_class &&
        left.execution_container == right.execution_container &&
        left.width == right.width && left.height == right.height &&
        left.frames == right.frames && left.steps == right.steps &&
        left.audio == right.audio;
}

bool device_memory_matches(const StreamingPresetRecord &record,
                           uint64_t physical) {
    if (!physical || physical < record.minimum_physical_memory_bytes)
        return false;
    return !record.maximum_physical_memory_bytes ||
        physical <= record.maximum_physical_memory_bytes;
}

bool fits(const StreamingPresetRecord &record, uint64_t target) {
    if (!record.calibration.complete ||
        record.calibration.scope != "execution_process_tree_v1" ||
        record.calibration.calibrated_request_bytes > target)
        return false;
    const uint64_t margin = streaming_target_margin_bytes(target);
    return margin <= target - record.calibration.calibrated_request_bytes;
}

} // namespace

uint64_t streaming_target_margin_bytes(uint64_t target) {
    if (!supported_streaming_target(target))
        throw std::invalid_argument("streaming_target_unsupported");
    const uint64_t ten_percent = target / 10 + (target % 10 != 0);
    return std::max<uint64_t>(512ull << 20, ten_percent);
}

bool supported_streaming_target(uint64_t target) noexcept {
    return std::find(public_streaming_targets.begin(),
                     public_streaming_targets.end(), target) !=
        public_streaming_targets.end();
}

PresetResolution resolve_streaming_preset(
        const PresetResolveQuery &query,
        const StreamingPresetCatalog &catalog) {
    PresetResolution result;
    if (!supported_streaming_target(query.target_request_memory_bytes)) {
        result.rejection_code = "unsupported_memory_target";
        return result;
    }
    if (query.catalog_revision && *query.catalog_revision != catalog.revision) {
        result.rejection_code = "streaming_resolution_stale";
        return result;
    }

    bool model_seen = false, workload_seen = false, release_seen = false;
    bool calibration_seen = false;
    std::vector<const StreamingPresetRecord *> candidates;
    for (const auto &record : catalog.records) {
        if (record.workload.model != query.workload.model)
            continue;
        model_seen = true;
        if (!same_workload(record.workload, query.workload) ||
            !device_memory_matches(record, query.physical_memory_bytes))
            continue;
        workload_seen = true;
        if (record.revoked || !released(record, query.allow_experimental))
            continue;
        release_seen = true;
        if (query.preset_id &&
            (record.id != *query.preset_id ||
             !query.preset_revision || record.revision != *query.preset_revision))
            continue;
        if (!record.calibration.complete) {
            result.rejected_preset_ids.push_back(record.id);
            continue;
        }
        calibration_seen = true;
        if (!fits(record, query.target_request_memory_bytes)) {
            result.rejected_preset_ids.push_back(record.id);
            continue;
        }
        candidates.push_back(&record);
    }

    if (candidates.empty()) {
        result.rejection_code = !model_seen ? "unsupported_model_or_format" :
            !workload_seen ? "unvalidated_workload" :
            !release_seen ? "preset_not_public" :
            !calibration_seen ? "memory_calibration_incomplete" :
            "no_preset_fits_target";
        return result;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto *left, const auto *right) {
        return std::tie(left->performance_rank,
                        left->calibration.calibrated_request_bytes,
                        left->logical_read_bytes, left->id, left->revision) <
               std::tie(right->performance_rank,
                        right->calibration.calibrated_request_bytes,
                        right->logical_read_bytes, right->id, right->revision);
    });
    result.selected = *candidates.front();
    return result;
}

const StreamingPresetCatalog &production_streaming_preset_catalog() {
    // Deliberately empty until a model/workload/device record has passed the
    // full public acceptance protocol. Tests inject their own fixture catalog.
    static const StreamingPresetCatalog catalog{
        "tc-streaming-catalog-empty-v1", {}};
    return catalog;
}

} // namespace tc::streaming
