#include "bridge.hpp"
#include "platform.hpp"
#include "../../models/seedvr2/hybrid.hpp"
#include "../../models/seedvr2/qualification.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <set>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace tc::seedvr2 {
namespace {
constexpr const char *source_sha = "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660";
std::string text(NSDictionary *d, NSString *key) {
    id value = d[key];
    require([value isKindOfClass:NSString.class], "SeedVR2 hybrid manifest requires string " + std::string(key.UTF8String));
    const char *raw = [value UTF8String];
    require(raw && std::strlen(raw) == [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding] &&
            [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding] <= 4096, "invalid SeedVR2 hybrid string");
    return raw;
}
uint64_t integer(NSDictionary *d, NSString *key) {
    id value = d[key];
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
            !CFNumberIsFloatType((__bridge CFNumberRef)value) && [value doubleValue] >= 0 &&
            [value doubleValue] <= double(1ull << 53), "invalid SeedVR2 hybrid integer: " + std::string(key.UTF8String));
    return [value unsignedLongLongValue];
}
NSDictionary *object(id value) {
    require([value isKindOfClass:NSDictionary.class], "SeedVR2 hybrid manifest object required");
    return value;
}
void digest_string(const std::string &value) {
    require(value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "invalid SeedVR2 artifact digest");
}
std::filesystem::path relative(const std::string &value) {
    const std::filesystem::path path(value);
    require(!path.empty() && !path.is_absolute() && path.generic_string() == value,
            "SeedVR2 artifact paths must be canonical relative paths");
    for (const auto &part : path)
        require(part != "." && part != ".." && !part.empty(), "SeedVR2 artifact path escapes its root");
    return path;
}
std::string data_sha(NSData *data) {
    require(data.length <= UINT32_MAX, "SeedVR2 artifact index is too large");
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, CC_LONG(data.length), bytes);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (auto byte : bytes) result << std::setw(2) << unsigned(byte);
    return result.str();
}
void verify_tree(const std::filesystem::path &root, NSDictionary *identity) {
    id files = identity[@"files"];
    require([files isKindOfClass:NSArray.class] && [files count] > 0 && [files count] <= 4096,
            "SeedVR2 compiled artifact file index required");
    std::set<std::string> actual;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
        const auto status = entry.symlink_status();
        require(!std::filesystem::is_symlink(status) &&
                (std::filesystem::is_directory(status) || std::filesystem::is_regular_file(status)),
                "SeedVR2 compiled artifact contains unsafe filesystem entries");
        if (std::filesystem::is_regular_file(status)) actual.insert(entry.path().lexically_relative(root).generic_string());
    }
    NSMutableArray *canonical = [NSMutableArray array];
    std::string previous;
    for (id item in (NSArray *)files) {
        auto d = object(item);
        const auto name = text(d, @"path"), hash = text(d, @"sha256");
        const auto path = root / relative(name);
        const auto bytes = integer(d, @"bytes");
        digest_string(hash);
        require(name > previous && actual.erase(name) == 1 && std::filesystem::file_size(path) == bytes &&
                sha256_file(path) == hash, "SeedVR2 compiled artifact identity mismatch: " + name);
        previous = name;
        [canonical addObject:@{@"path": @(name.c_str()), @"bytes": @(bytes), @"sha256": @(hash.c_str())}];
    }
    require(actual.empty(), "SeedVR2 compiled artifact has unregistered files");
    NSError *error = nil;
    auto encoded = [NSJSONSerialization dataWithJSONObject:canonical
        options:(NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes) error:&error];
    require(encoded != nil, "cannot canonicalize SeedVR2 artifact identity");
    const auto expected = text(identity, @"sha256");
    digest_string(expected);
    require(data_sha(encoded) == expected, "SeedVR2 compiled artifact tree digest mismatch");
}
} // namespace

void validate_public_hybrid_manifest(const std::filesystem::path &file) {
    require(std::filesystem::is_regular_file(file) && std::filesystem::file_size(file) <= (16ull << 20) &&
            sha256_file(file) == kQualifiedPrefixManifestSha,
            "SeedVR2 GPU+ANE requires the qualified original-FP16 all-block prefix manifest");
    const auto root = std::filesystem::canonical(file).parent_path();
    for (int block = 0; block < 32; ++block) {
        char name[32]; std::snprintf(name,sizeof(name),"block-%02d.mlmodelc",block);
        require(std::filesystem::is_directory(std::filesystem::symlink_status(root/name)),
                "SeedVR2 qualified compiled prefix bank is incomplete");
    }
}

HybridConfig read_hybrid_config(const std::filesystem::path &file) {
    @autoreleasepool {
        require(std::filesystem::is_regular_file(file) && std::filesystem::file_size(file) <= (16ull << 20),
                "SeedVR2 hybrid manifest is missing or too large");
        const auto canonical_file = std::filesystem::canonical(file), root = canonical_file.parent_path();
        const auto hash = sha256_file(canonical_file);
        const auto d = read_json(canonical_file);
        require(text(d, @"schema") == "turbocider-seedvr2-ffn-prefix-v1" &&
                text(d, @"checkpoint_sha256") == source_sha && integer(d, @"checkpoint_bytes") == 6784268336ull &&
                integer(d, @"hidden") == 2560 && integer(d, @"intermediate") == 6912 &&
                text(d, @"activation") == "silu-gated" && text(d, @"compute_dtype") == "fp16" &&
                text(d, @"variant") == "fp16" && text(d, @"compute_units") == "cpuAndNeuralEngine" &&
                text(d, @"source_operand_recipe") == "original-fp16",
                "SeedVR2 hybrid manifest requires the original-FP16 SwiGLU operand recipe");
        const auto rows = integer(d, @"rows"), channels = integer(d, @"ane_channels");
        require((rows == 128 || rows == 256 || rows == 512 || rows == 1024) &&
                channels > 0 && channels < 6912 && channels % 256 == 0 &&
                integer(d, @"gpu_channels") == 6912 - channels, "invalid SeedVR2 hybrid partition geometry");
        auto artifacts = object(d[@"artifacts"]);
        require(artifacts.count > 0 && artifacts.count <= 32, "SeedVR2 hybrid block coverage required");
        HybridConfig result;
        result.source_operand_recipe = text(d, @"source_operand_recipe");
        result.rows = int(rows); result.channels = int(channels); result.manifest_sha256 = hash;
        for (int block = 0; block < 32; ++block) {
            auto key = [NSString stringWithFormat:@"%d", block];
            if (!artifacts[key]) continue;
            auto artifact = object(artifacts[key]);
            auto name = text(artifact, @"path");
            char expected[32]; std::snprintf(expected, sizeof(expected), "block-%02d.mlmodelc", block);
            require(name == expected, "SeedVR2 compiled artifact block name mismatch");
            const auto path = root / relative(name);
            require(std::filesystem::is_directory(std::filesystem::symlink_status(path)),
                    "SeedVR2 compiled artifact must be a non-symlink directory");
            verify_tree(path, object(artifact[@"identity"]));
            result.blocks.push_back(block); result.artifacts.push_back(path);
        }
        require(result.blocks.size() == artifacts.count && sha256_file(canonical_file) == hash,
                "SeedVR2 hybrid manifest changed or contains unknown blocks");
        return result;
    }
}
} // namespace tc::seedvr2
