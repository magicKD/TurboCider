#include "coreml_bundle.hpp"
#include "../../core/common.hpp"
#include "../../core/json_keys.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#import <Foundation/Foundation.h>
#include <bit>
#include <cmath>
#include <fstream>
#include <set>

namespace tc::z_image {
namespace {
namespace fs = std::filesystem;
NSDictionary *dictionary(id value) {
    require([value isKindOfClass:NSDictionary.class], "Core ML bundle: expected object");
    return value;
}
std::string string(id value) {
    require([value isKindOfClass:NSString.class], "Core ML bundle: expected string");
    auto data = [(NSString *)value dataUsingEncoding:NSUTF8StringEncoding];
    std::string result(static_cast<const char *>(data.bytes), data.length);
    require(result.find('\0') == std::string::npos, "Core ML bundle: embedded NUL");
    return result;
}
double number(id value) {
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "Core ML bundle: expected number");
    double result = [value doubleValue];
    require(std::isfinite(result), "Core ML bundle: nonfinite number");
    return result;
}
uint64_t integer(id value) {
    double result = number(value);
    require(result >= 0 && result <= 9007199254740991.0 && std::floor(result) == result,
            "Core ML bundle: expected exact nonnegative integer");
    return uint64_t(result);
}
fs::path relative(id value) {
    fs::path p(string(value));
    require(!p.empty() && !p.is_absolute() && p == p.lexically_normal(), "Core ML bundle: invalid relative path");
    for (const auto &part : p) require(part != "." && part != "..", "Core ML bundle: path escapes generation");
    return p;
}
void branches(id value) {
    require([value isKindOfClass:NSArray.class] && [value count] == 32, "Core ML bundle: requires 32 branches");
    for (unsigned i = 0; i < 32; ++i) require(integer(value[i]) == i, "Core ML bundle: wrong branch map");
}
void no_lora(NSDictionary *value) {
    id loras = value[@"loras"];
    require(loras == nil || ([loras isKindOfClass:NSArray.class] && [loras count] == 0), "Core ML bundle: LoRA unsupported");
}
}
struct VerifiedCoreMLBundleLease::State {
    std::shared_ptr<const CoreMLGeneration> generation;
    std::shared_ptr<const streaming::SourceLease> parent;
    HybridPartitionSpec partition;
    std::vector<fs::path> models;
};
VerifiedCoreMLBundleLease::VerifiedCoreMLBundleLease(std::unique_ptr<State> s) : state_(std::move(s)) {}
VerifiedCoreMLBundleLease::~VerifiedCoreMLBundleLease() = default;
const HybridPartitionSpec &VerifiedCoreMLBundleLease::partition() const noexcept { return state_->partition; }
const std::vector<fs::path> &VerifiedCoreMLBundleLease::models() const noexcept { return state_->models; }
void VerifiedCoreMLBundleLease::revalidate() const {
    state_->parent->revalidate_after_drain(); state_->generation->revalidate();
}
std::shared_ptr<const VerifiedCoreMLBundleLease> VerifiedCoreMLBundleLease::bind(
        std::shared_ptr<const CoreMLGeneration> generation, const fs::path &manifest,
        std::shared_ptr<const streaming::SourceLease> parent, std::string_view logical_id) {
    require(generation && parent && parent->has_verified_content(), "Core ML bundle: verified sources required");
    generation->revalidate(); parent->revalidate_after_drain();
    auto state = std::make_unique<State>();
    state->generation = std::move(generation); state->parent = std::move(parent);
    const auto &checkpoint = state->parent->file(logical_id);
    @autoreleasepool {
        auto manifest_text = manifest.generic_string();
        require(manifest_text.find('\0') == std::string::npos, "Core ML bundle: embedded NUL manifest path");
        auto name = relative(@(manifest_text.c_str()));
        auto path = state->generation->root() / name;
        require(fs::is_regular_file(path) && fs::file_size(path) <= (4 << 20), "Core ML bundle: invalid manifest size/type");
        std::ifstream stream(path, std::ios::binary);
        require(bool(stream), "Core ML bundle: cannot read manifest");
        std::string bytes{std::istreambuf_iterator<char>(stream), {}};
        require(bytes.size() == fs::file_size(path), "Core ML bundle: incomplete manifest read");
        reject_duplicate_json_keys(bytes);
        NSError *error = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:bytes.data() length:bytes.size()] options:0 error:&error];
        require(parsed != nil && error == nil, "Core ML bundle: invalid JSON");
        auto d = dictionary(parsed), shape = dictionary(d[@"shape"]), source = dictionary(d[@"source"]);
        auto exported = dictionary(d[@"export_identity"]), artifacts = dictionary(d[@"artifacts"]);
        require(integer(d[@"schema_version"]) == 2 && integer(exported[@"recipe"]) == 1 &&
                string(exported[@"owner"]) == "turbocider.z_image.coreml.v1" &&
                string(exported[@"checkpoint_format"]) == "safetensors" && string(exported[@"convrot_mode"]) == "none",
                "Core ML bundle: unsupported export ABI");
        no_lora(source); no_lora(exported);
        branches(source[@"blocks"]); branches(exported[@"blocks"]);
        for (NSDictionary *record in @[source, exported]) {
            require(string(record[@"checkpoint_sha256"]) == checkpoint.content_digest &&
                    integer(record[@"checkpoint_bytes"]) == checkpoint.bytes, "Core ML bundle: parent content mismatch");
        }
        auto &spec = state->partition;
        require(integer(shape[@"K"]) == spec.hidden && integer(shape[@"N"]) == spec.hidden &&
                integer(shape[@"mlp_width"]) == spec.mlp_width && integer(exported[@"hidden"]) == spec.hidden &&
                integer(exported[@"mlp_width"]) == spec.mlp_width, "Core ML bundle: FFN geometry mismatch");
        auto end = integer(shape[@"ane_mlp_end"]);
        require(integer(shape[@"ane_mlp_start"]) == 0 && end > 0 && end < spec.mlp_width && end % 128 == 0 &&
                integer(exported[@"ane_mlp_start"]) == 0 && integer(exported[@"ane_mlp_end"]) == end,
                "Core ML bundle: invalid channel partition");
        spec.ane_end = uint32_t(end);
        id buckets = shape[@"buckets"];
        require([buckets isKindOfClass:NSArray.class] && [buckets count] == 1 && integer(buckets[0]) == 1088 &&
                integer(exported[@"bucket"]) == 1088, "Core ML bundle: requires HY-M0 fixed bucket 1088");
        spec.bucket_rows = 1088;
        auto functions = dictionary(d[@"functions"]);
        require(functions.count == 1 && string(functions[@"1088"]) == "main", "Core ML bundle: unsupported function map");
        spec.precision_revision = string(exported[@"variant"]);
        require(spec.precision_revision == "fp16" || spec.precision_revision == "int8_pc", "Core ML bundle: unsupported precision");
        // Current exporter recipe uses these exact binary powers of two.
        require(number(shape[@"activation_scale"]) == 8 && number(exported[@"activation_scale"]) == 8 &&
                number(shape[@"output_scale"]) == 32 && number(exported[@"output_scale"]) == 32,
                "Core ML bundle: unsupported scale encoding");
        spec.activation_scale = 8; spec.output_scale = 32;
        require(artifacts.count == 32, "Core ML bundle: incomplete artifact map");
        std::set<fs::path> unique;
        for (unsigned i = 0; i < 32; ++i) {
            auto entry = dictionary(artifacts[@(std::to_string(i).c_str())]);
            require(entry.count == 1, "Core ML bundle: ambiguous artifact variant");
            // Legacy exporter uses this key even for uncompressed FP16.
            auto part = relative(entry[@"int8_pc"]);
            auto model = path.parent_path() / part;
            require(part.extension() == ".mlmodelc" && fs::is_directory(model) && unique.insert(model).second,
                    "Core ML bundle: missing or repeated compiled partition");
            bool has_file = false;
            for (const auto &file : fs::recursive_directory_iterator(model)) has_file |= file.is_regular_file();
            require(has_file, "Core ML bundle: empty model");
            state->models.push_back(model);
        }
        spec.parent_checkpoint_digest = checkpoint.content_digest;
        spec.artifact_content_digest = state->generation->content_digest();
        streaming::CanonicalEncoder identity("tc-z-image-hybrid-partition-v1");
        identity.string_field("bundle", spec.artifact_content_digest);
        identity.string_field("parent", spec.parent_checkpoint_digest);
        identity.unsigned_field("hidden", spec.hidden); identity.unsigned_field("mlp_width", spec.mlp_width);
        identity.unsigned_field("ane_begin", 0); identity.unsigned_field("ane_end", spec.ane_end);
        identity.unsigned_field("bucket", spec.bucket_rows);
        identity.string_field("precision", spec.precision_revision);
        identity.unsigned_field("activation_scale_f32_bits", std::bit_cast<uint32_t>(spec.activation_scale));
        identity.unsigned_field("output_scale_f32_bits", std::bit_cast<uint32_t>(spec.output_scale));
        identity.begin_list("branches", 32);
        for (unsigned i = 0; i < 32; ++i) {
            identity.string_field("branch", i < 2 ? "noise_refiner." + std::to_string(i) : "layers." + std::to_string(i - 2));
            identity.string_field("path", state->models[i].lexically_relative(state->generation->root()).generic_string());
        }
        spec.identity = identity.sha256();
    }
    auto result = std::shared_ptr<const VerifiedCoreMLBundleLease>(new VerifiedCoreMLBundleLease(std::move(state)));
    result->revalidate();
    return result;
}
} // namespace tc::z_image
