#include "../../core/streaming_contracts.hpp"

#include <stdexcept>

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
} // namespace tc
