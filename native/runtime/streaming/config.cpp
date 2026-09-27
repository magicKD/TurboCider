#include "../../core/streaming_contracts.hpp"

#include <stdexcept>
#include <algorithm>

namespace tc {
namespace {
void check(bool value, const std::string &message) {
    if (!value) throw std::invalid_argument("streaming_config_invalid: " + message);
}
template<class T> void overlay(std::optional<T> &to, const std::optional<T> &from) {
    if (from) to = from;
}
}

void overlay_streaming_config(StreamingConfig &to, const StreamingConfig &from) {
    overlay(to.enabled, from.enabled);
    overlay(to.schema_version, from.schema_version);
    overlay(to.selection, from.selection);
    overlay(to.retention, from.retention);
    for (const auto &[id, src] : from.stages) {
        auto &dst = to.stages[id];
        if (src.residency && src.residency != dst.residency) {
            dst = src;
            const auto prefix = "stages." + id + ".";
            for (auto it = to.provenance.begin(); it != to.provenance.end();) {
                if (it->first.starts_with(prefix)) it = to.provenance.erase(it);
                else ++it;
            }
        } else {
            overlay(dst.residency, src.residency);
            overlay(dst.block_group_size, src.block_group_size);
            overlay(dst.slot_count, src.slot_count);
            overlay(dst.resident_prefix_blocks, src.resident_prefix_blocks);
            overlay(dst.prefetch_distance, src.prefetch_distance);
            overlay(dst.io_workers, src.io_workers);
        }
    }
    for (const auto &[key, origin] : from.provenance) to.provenance[key] = origin;
}

void validate_streaming_config(const StreamingConfig &c) {
    if (c.schema_version) check(*c.schema_version == 1, "schema_version must be 1");
    if (c.selection) check(*c.selection == "manual", "selection must be manual");
    if (c.retention) check(*c.retention == "request", "retention must be request");
    check(c.stages.size() <= 64, "too many stages");
    if (c.active()) {
        check(c.schema_version.has_value(), "schema_version is required");
        check(c.selection.has_value(), "selection is required");
        check(c.retention.has_value(), "retention is required");
    }
    for (const auto &[id, s] : c.stages) {
        check(!id.empty() && id.size() <= 128 && id.find('\0') == std::string::npos,
              "invalid stage id");
        // Partial profile/request overlays are validated only after merge.
        check(s.residency.has_value(), id + ".residency is required");
        if (*s.residency == "resident") {
            check(!s.block_group_size && !s.slot_count && !s.resident_prefix_blocks &&
                  !s.prefetch_distance && !s.io_workers,
                  id + ": resident does not accept streamed fields");
        } else {
            check(*s.residency == "streamed", id + ": unknown residency");
            check(s.block_group_size && s.slot_count && s.resident_prefix_blocks &&
                  s.prefetch_distance && s.io_workers, id + ": incomplete streamed layout");
            check(*s.block_group_size > 0, id + ".block_group_size must be positive");
            check(*s.slot_count > 0, id + ".slot_count must be positive");
            check(*s.prefetch_distance < *s.slot_count, id + ": prefetch_distance >= slot_count");
            check(*s.io_workers > 0 && *s.io_workers <= *s.slot_count,
                  id + ": io_workers must be 1...slot_count");
        }
    }
}

void validate_streaming_selector(const StreamingSelector &s) {
    check(s.schema_version.has_value() && *s.schema_version == 2,
          "selector schema_version must be 2");
    check(s.enabled.has_value(), "selector enabled is required");
    if (!*s.enabled) {
        check(!s.selection && !s.retention &&
                  !s.target_request_memory_bytes && !s.preset_id &&
                  !s.preset_revision && !s.catalog_revision &&
                  !s.expected_resolution_digest,
              "disabled selector accepts only schema_version and enabled");
        return;
    }
    check(s.selection.has_value(), "selector selection is required");
    check(s.retention.has_value() && *s.retention == "request",
          "selector retention must be request");
    check(s.target_request_memory_bytes.has_value() &&
              *s.target_request_memory_bytes > 0 &&
              *s.target_request_memory_bytes <= ((1ull << 53) - 1),
          "selector target_request_memory_bytes is outside the supported range");
    check(std::find(public_streaming_targets.begin(),
                    public_streaming_targets.end(),
                    *s.target_request_memory_bytes) !=
              public_streaming_targets.end(),
          "selector target_request_memory_bytes is not a published target");
    if (*s.selection == "memory_tier") {
        check(!s.preset_id && !s.preset_revision && !s.catalog_revision &&
                  !s.expected_resolution_digest,
              "memory_tier selector does not accept preset fields");
    } else {
        check(*s.selection == "preset", "selector selection is unknown");
        check(s.preset_id.has_value() && !s.preset_id->empty() &&
                  s.preset_id->size() <= 128,
              "preset_id is required and must be at most 128 bytes");
        check(s.preset_revision.has_value() && *s.preset_revision > 0,
              "preset_revision must be positive");
        check(s.catalog_revision.has_value() &&
                  !s.catalog_revision->empty() &&
                  s.catalog_revision->size() <= 128,
              "catalog_revision is required and must be at most 128 bytes");
        if (s.expected_resolution_digest)
            check(!s.expected_resolution_digest->empty() &&
                      s.expected_resolution_digest->size() <= 128,
                  "expected_resolution_digest must be 1...128 bytes");
    }
}
} // namespace tc
