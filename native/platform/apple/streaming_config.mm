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
} // namespace tc
