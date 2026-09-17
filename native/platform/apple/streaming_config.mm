#include "bridge.hpp"
#include <cmath>

namespace tc {
namespace {
void object(id value, const char *path) {
    require([value isKindOfClass:NSDictionary.class], std::string(path) + " must be an object");
}
void known_keys(NSDictionary *d, NSArray *allowed) {
    for (NSString *key in d)
        require([allowed containsObject:key], "unknown streaming field: " + std::string(key.UTF8String));
}
uint32_t integer(id value, const char *path) {
    require([value isKindOfClass:NSNumber.class] &&
            CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            std::string(path) + " must be an integer");
    const double n = [value doubleValue];
    require(std::isfinite(n) && n >= 0 && n <= UINT32_MAX && n == std::floor(n),
            std::string(path) + " is outside uint32 range");
    return uint32_t(n);
}
uint64_t exact_byte_count(id value, const char *path) {
    require([value isKindOfClass:NSNumber.class] &&
            CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            std::string(path) + " must be an integer");
    const double n = [value doubleValue];
    constexpr double maximum = double((1ull << 53) - 1);
    require(std::isfinite(n) && n > 0 && n <= maximum && n == std::floor(n),
            std::string(path) + " is outside the supported integer range");
    const uint64_t result = [value unsignedLongLongValue];
    require(double(result) == n,
            std::string(path) + " cannot be represented exactly");
    return result;
}
}

void parse_streaming_config(NSDictionary *d, StreamingConfig &c, const char *origin) {
    object(d, "streaming");
    known_keys(d, @[@"enabled", @"schema_version", @"selection", @"retention", @"stages"]);
    if (d[@"enabled"]) {
        require(CFGetTypeID((__bridge CFTypeRef)d[@"enabled"]) == CFBooleanGetTypeID(),
                "streaming.enabled must be boolean");
        c.enabled = [d[@"enabled"] boolValue];
        c.provenance["enabled"] = origin;
    }
    if (d[@"schema_version"]) {
        c.schema_version = integer(d[@"schema_version"], "streaming.schema_version");
        c.provenance["schema_version"] = origin;
    }
    if (d[@"selection"]) {
        c.selection = string_value(d, @"selection");
        c.provenance["selection"] = origin;
    }
    if (d[@"retention"]) {
        c.retention = string_value(d, @"retention");
        c.provenance["retention"] = origin;
    }
    if (d[@"stages"]) {
        NSDictionary *stages = d[@"stages"];
        object(stages, "streaming.stages");
        require(stages.count <= 64, "streaming stage count exceeds 64");
        for (NSString *key in stages) {
            const std::string id = key.UTF8String;
            require(!id.empty() && id.size() <= 128 &&
                        id.size() == [key lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                    "invalid streaming stage id");
            NSDictionary *value = stages[key];
            object(value, "streaming stage");
            known_keys(value, @[@"residency", @"block_group_size", @"slot_count",
                               @"resident_prefix_blocks", @"prefetch_distance", @"io_workers"]);
            auto &s = c.stages[id];
            const std::string prefix = "stages." + id + ".";
            if (value[@"residency"]) {
                s.residency = string_value(value, @"residency");
                c.provenance[prefix + "residency"] = origin;
            }
            auto number = [&](NSString *name, std::optional<uint32_t> &field) {
                if (value[name]) {
                    field = integer(value[name], name.UTF8String);
                    c.provenance[prefix + name.UTF8String] = origin;
                }
            };
            number(@"block_group_size", s.block_group_size);
            number(@"slot_count", s.slot_count);
            number(@"resident_prefix_blocks", s.resident_prefix_blocks);
            number(@"prefetch_distance", s.prefetch_distance);
            number(@"io_workers", s.io_workers);
        }
    }
    // Cross-field validation is deliberately after the profile/request merge.
}

void parse_streaming_input(NSDictionary *d, StreamingConfig &manual,
                           std::optional<StreamingSelector> &selector,
                           const char *origin) {
    object(d, "streaming");
    const uint32_t schema = d[@"schema_version"]
        ? integer(d[@"schema_version"], "streaming.schema_version") : 1;
    if (schema == 1) {
        require(!selector.has_value(),
                "streaming_config_conflict: manual and selector are mutually exclusive");
        parse_streaming_config(d, manual, origin);
        return;
    }
    require(schema == 2, "streaming selector schema_version must be 2");
    require(!manual.specified(),
            "streaming_config_conflict: selector and manual layout are mutually exclusive");
    known_keys(d, @[
        @"enabled", @"schema_version", @"selection", @"retention",
        @"target_request_memory_bytes", @"preset_id", @"preset_revision",
        @"catalog_revision", @"expected_resolution_digest"
    ]);
    StreamingSelector parsed;
    parsed.schema_version = schema;
    parsed.provenance["schema_version"] = origin;
    if (d[@"enabled"]) {
        require(CFGetTypeID((__bridge CFTypeRef)d[@"enabled"]) == CFBooleanGetTypeID(),
                "streaming.enabled must be boolean");
        parsed.enabled = [d[@"enabled"] boolValue];
        parsed.provenance["enabled"] = origin;
    }
    if (d[@"selection"]) {
        parsed.selection = string_value(d, @"selection");
        parsed.provenance["selection"] = origin;
    }
    if (d[@"retention"]) {
        parsed.retention = string_value(d, @"retention");
        parsed.provenance["retention"] = origin;
    }
    if (d[@"target_request_memory_bytes"]) {
        parsed.target_request_memory_bytes = exact_byte_count(
            d[@"target_request_memory_bytes"],
            "streaming.target_request_memory_bytes");
        parsed.provenance["target_request_memory_bytes"] = origin;
    }
    if (d[@"preset_id"]) {
        parsed.preset_id = string_value(d, @"preset_id");
        parsed.provenance["preset_id"] = origin;
    }
    if (d[@"preset_revision"]) {
        parsed.preset_revision = integer(
            d[@"preset_revision"], "streaming.preset_revision");
        parsed.provenance["preset_revision"] = origin;
    }
    if (d[@"catalog_revision"]) {
        parsed.catalog_revision = string_value(d, @"catalog_revision");
        parsed.provenance["catalog_revision"] = origin;
    }
    if (d[@"expected_resolution_digest"]) {
        parsed.expected_resolution_digest = string_value(
            d, @"expected_resolution_digest");
        parsed.provenance["expected_resolution_digest"] = origin;
    }
    validate_streaming_selector(parsed);
    selector = std::move(parsed);
}

NSDictionary *streaming_config_dictionary(const StreamingConfig &c) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (c.enabled) result[@"enabled"] = @(*c.enabled);
    if (c.schema_version) result[@"schema_version"] = @(*c.schema_version);
    if (c.selection) result[@"selection"] = @(c.selection->c_str());
    if (c.retention) result[@"retention"] = @(c.retention->c_str());
    NSMutableDictionary *stages = [NSMutableDictionary dictionary];
    for (const auto &[id, s] : c.stages) {
        NSMutableDictionary *value = [NSMutableDictionary dictionary];
        if (s.residency) value[@"residency"] = @(s.residency->c_str());
        if (s.block_group_size) value[@"block_group_size"] = @(*s.block_group_size);
        if (s.slot_count) value[@"slot_count"] = @(*s.slot_count);
        if (s.resident_prefix_blocks) value[@"resident_prefix_blocks"] = @(*s.resident_prefix_blocks);
        if (s.prefetch_distance) value[@"prefetch_distance"] = @(*s.prefetch_distance);
        if (s.io_workers) value[@"io_workers"] = @(*s.io_workers);
        stages[@(id.c_str())] = value;
    }
    result[@"stages"] = stages;
    return result;
}

NSDictionary *streaming_selector_dictionary(const StreamingSelector &s) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (s.enabled) result[@"enabled"] = @(*s.enabled);
    if (s.schema_version) result[@"schema_version"] = @(*s.schema_version);
    if (s.selection) result[@"selection"] = @(s.selection->c_str());
    if (s.retention) result[@"retention"] = @(s.retention->c_str());
    if (s.target_request_memory_bytes)
        result[@"target_request_memory_bytes"] = @(*s.target_request_memory_bytes);
    if (s.preset_id) result[@"preset_id"] = @(s.preset_id->c_str());
    if (s.preset_revision) result[@"preset_revision"] = @(*s.preset_revision);
    if (s.catalog_revision)
        result[@"catalog_revision"] = @(s.catalog_revision->c_str());
    if (s.expected_resolution_digest)
        result[@"expected_resolution_digest"] =
            @(s.expected_resolution_digest->c_str());
    return result;
}
} // namespace tc
