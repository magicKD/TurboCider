#include "bridge.hpp"
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <sys/sysctl.h>
namespace tc {
static void profile_keys(NSDictionary *d, NSArray *allowed) {
    auto keys = [NSSet setWithArray:allowed];
    for (NSString *key in d)
        require([keys containsObject:key], "unknown profile field: " + std::string(key.UTF8String));
}
static bool profile_number(id value) {
    return [value isKindOfClass:NSNumber.class] &&
           CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID();
}
void resolve_profile(Request &r) {
    if (r.profile.empty())
        return;
    auto path = std::filesystem::absolute(r.profile);
    auto d = read_json(path);
    profile_keys(d, @[ @"schema_version", @"enabled", @"match", @"models" ]);
    require(profile_number(d[@"schema_version"]) && [d[@"schema_version"] doubleValue] == 1,
            "profile schema must be 1");
    require(d[@"enabled"] && CFGetTypeID((__bridge CFTypeRef)d[@"enabled"]) == CFBooleanGetTypeID(),
            "profile enabled must be boolean");
    if (![d[@"enabled"] boolValue]) {
        require(r.execution != "gpu_ane", "hybrid profile is disabled");
        return;
    }
    auto match = d[@"match"];
    require([match isKindOfClass:NSDictionary.class], "profile requires hardware match");
    profile_keys(match, @[ @"gpu_name", @"memory_bytes" ]);
    require([match[@"gpu_name"] isKindOfClass:NSString.class] &&
                profile_number(match[@"memory_bytes"]),
            "profile hardware identity has invalid types");
    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    require(gpu && [match[@"gpu_name"] isEqual:gpu.name],
            "device profile GPU does not match this machine");
    uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
    require([match[@"memory_bytes"] unsignedLongLongValue] == physical,
            "device profile memory does not match this machine");
    auto models = d[@"models"];
    require([models isKindOfClass:NSDictionary.class], "profile models must be object");
    auto model = models[@(r.model.c_str())];
    require([model isKindOfClass:NSDictionary.class], "profile does not contain requested model");
    profile_keys(model, @[
        @"policy", @"residency", @"allow_approximation", @"ane_manifest", @"memory_budget_bytes",
        @"allocator_cache_bytes", @"warmup_iterations", @"coreml_export"
    ]);
    r.execution = string_value(model, @"policy", "gpu");
    r.residency = string_value(model, @"residency", r.residency);
    if (model[@"allow_approximation"]) {
        require(CFGetTypeID((__bridge CFTypeRef)model[@"allow_approximation"]) ==
                    CFBooleanGetTypeID(),
                "profile approximation must be boolean");
        r.allow_approximation = [model[@"allow_approximation"] boolValue];
    }
    auto artifact = string_value(model, @"ane_manifest");
    if (!artifact.empty())
        r.ane_manifest = (path.parent_path() / artifact).lexically_normal().string();
    auto bounded = [&](NSString *key, uint64_t fallback, uint64_t upper) {
        id value = model[key];
        if (!value)
            return fallback;
        require([value isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
                "profile budget must be numeric");
        double x = [value doubleValue];
        uint64_t n = [value unsignedLongLongValue];
        require(x >= 0 && x <= double(upper) && double(n) == x, "invalid profile budget");
        return n;
    };
    r.memory_budget_bytes = bounded(@"memory_budget_bytes", 0, physical);
    r.allocator_cache_bytes =
        bounded(@"allocator_cache_bytes", r.allocator_cache_bytes, physical / 4);
    r.warmup_iterations = int(bounded(@"warmup_iterations", 0, 8));
    auto content = json(d);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(content.data(), CC_LONG(content.size()), digest);
    for (auto byte : digest) {
        char pair[3];
        snprintf(pair, sizeof(pair), "%02x", byte);
        r.profile_identity += pair;
    }
}
} // namespace tc
