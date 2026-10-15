#include "ane_smoothquant.hpp"
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <cerrno>
#include <cstring>
#include <set>
#include <sys/stat.h>

namespace tc::ane::smoothquant {
namespace {
void check(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
std::string sha256(const std::string &bytes) {
    check(bytes.size() <= UINT32_MAX, "S1 digest input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), digest);
    constexpr char hex[] = "0123456789abcdef";
    std::string value; value.reserve(64);
    for (const auto byte : digest) { value += hex[byte >> 4]; value += hex[byte & 15]; }
    return value;
}
void append_string(std::string &bytes, const std::string &value) {
    const uint64_t count = value.size();
    for (int shift = 56; shift >= 0; shift -= 8) bytes += char(count >> shift);
    bytes += value;
}
std::string fingerprint(const std::string &identity) {
    // Exact CanonicalEncoder("tc-ane-calibration-source-v1") string_field
    // encoding, independent of MLX and memory-manifest object dependencies.
    std::string bytes = "H";
    append_string(bytes, "tc-ane-calibration-source-v1");
    bytes += 'S'; append_string(bytes, "identity"); append_string(bytes, identity);
    return sha256(bytes);
}
bool same_file(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
        a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
        a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
}
std::string read_profile(const std::filesystem::path &path) {
    check(path.is_absolute() && path == path.lexically_normal(), "S1 path must be absolute and normalized");
    const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    check(fd >= 0, "cannot open S1 profile without following symlinks");
    try {
        struct stat before{}, after{};
        check(::fstat(fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_size > 0 &&
              uint64_t(before.st_size) <= profile_limit_bytes, "S1 profile is not a small regular file");
        std::string bytes(size_t(before.st_size), '\0');
        size_t done = 0;
        while (done < bytes.size()) {
            const auto got = ::read(fd, bytes.data() + done, bytes.size() - done);
            if (got < 0 && errno == EINTR) continue;
            check(got > 0, "S1 profile changed or could not be read"); done += size_t(got);
        }
        char extra;
        check(::read(fd, &extra, 1) == 0 && ::fstat(fd, &after) == 0 && same_file(before,after),
              "S1 profile changed while reading");
        ::close(fd); return bytes;
    } catch (...) { ::close(fd); throw; }
}

// Foundation overwrites duplicate JSON keys. Scan every object first, decoding
// keys through Foundation too, so "hidden" and "\u0068idden" are duplicates.
class UniqueJsonKeys {
    const std::string &source_;
    size_t cursor_ = 0, tokens_ = 0;
    void space() { while (cursor_ < source_.size() && std::strchr(" \t\r\n",source_[cursor_])) ++cursor_; }
    bool take(char value) { space(); if (cursor_ < source_.size() && source_[cursor_] == value) { ++cursor_; return true; } return false; }
    std::string string_token() {
        space(); check(cursor_ < source_.size() && source_[cursor_] == '"', "invalid JSON key/string");
        const size_t first = cursor_++;
        while (cursor_ < source_.size()) {
            const char c = source_[cursor_++];
            if (c == '"') return source_.substr(first,cursor_ - first);
            if (c == '\\') { check(cursor_ < source_.size(), "invalid JSON escape"); ++cursor_; }
        }
        throw std::invalid_argument("unterminated JSON string");
    }
    std::string decoded_key() {
        const auto token = "[" + string_token() + "]";
        NSData *data = [NSData dataWithBytes:token.data() length:token.size()];
        NSError *error = nil;
        id decoded = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        check(!error && [decoded isKindOfClass:NSArray.class] && [(NSArray *)decoded count] == 1 &&
              [decoded[0] isKindOfClass:NSString.class], "invalid JSON object key");
        NSString *name = decoded[0];
        const char *key = [name UTF8String];
        check(key != nullptr, "invalid UTF8 JSON key");
        return std::string(key, [name lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
    }
    void value(int depth) {
        check(depth <= 32 && ++tokens_ <= 524288, "S1 JSON nesting/token budget exceeded");
        space(); check(cursor_ < source_.size(), "incomplete S1 JSON");
        if (take('{')) {
            std::set<std::string> keys;
            if (take('}')) return;
            do {
                check(keys.insert(decoded_key()).second, "duplicate S1 JSON key");
                check(take(':'), "invalid S1 JSON object"); value(depth + 1);
                if (take('}')) return;
            } while (take(','));
            throw std::invalid_argument("invalid S1 JSON object separator");
        }
        if (take('[')) {
            if (take(']')) return;
            do { value(depth + 1); if (take(']')) return; } while (take(','));
            throw std::invalid_argument("invalid S1 JSON array separator");
        }
        if (source_[cursor_] == '"') { string_token(); return; }
        const size_t first = cursor_;
        while (cursor_ < source_.size() && !std::strchr(" \t\r\n,]}",source_[cursor_])) ++cursor_;
        check(cursor_ > first, "invalid S1 JSON token");
    }
public:
    explicit UniqueJsonKeys(const std::string &source) : source_(source) {}
    void validate() { value(0); space(); check(cursor_ == source_.size(), "trailing S1 JSON"); }
};
NSDictionary *object(id value) { check([value isKindOfClass:NSDictionary.class], "missing S1 JSON object"); return value; }
NSArray *array(id value) { check([value isKindOfClass:NSArray.class], "missing S1 JSON array"); return value; }
void keys(NSDictionary *value, std::initializer_list<const char *> expected) {
    check(value.count == expected.size(), "missing or unexpected S1 JSON fields");
    for (const auto key : expected) check(value[@(key)] != nil, "missing S1 JSON field");
}
std::string text(id value) {
    check([value isKindOfClass:NSString.class], "missing S1 string identity");
    const char *raw = [(NSString *)value UTF8String]; check(raw != nullptr, "invalid S1 UTF8 identity");
    std::string result(raw, [(NSString *)value lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
    check(calibration::valid_text(result), "invalid/empty S1 string identity"); return result;
}
bool is_boolean(id value) {
    return [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}
void boolean(id value, bool expected) {
    check(is_boolean(value) && [(NSNumber *)value boolValue] == expected, "invalid S1 qualification/experiment flag");
}
uint64_t integer(id value, uint64_t low, uint64_t high) {
    check([value isKindOfClass:NSNumber.class] && !is_boolean(value), "invalid S1 integer type");
    const char *type = [(NSNumber *)value objCType];
    check(type && std::strchr("cCsSiIlLqQ",type[0]), "S1 integer cannot be a floating value");
    if (!std::strchr("CSILQ",type[0])) check([(NSNumber *)value longLongValue] >= 0, "negative S1 integer");
    const uint64_t result = [(NSNumber *)value unsignedLongLongValue];
    check(result >= low && result <= high, "S1 integer out of range"); return result;
}
double number(id value, double low, double high) {
    check([value isKindOfClass:NSNumber.class] && !is_boolean(value), "invalid S1 numeric type");
    const double result = [(NSNumber *)value doubleValue];
    check(std::isfinite(result) && result >= low && result <= high, "nonfinite/out-of-range S1 number"); return result;
}
bool digest(const std::string &value) {
    return value.size() == 64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
std::vector<int> integers(id value, size_t low_count, size_t high_count, int maximum) {
    const auto values = array(value);
    check(values.count >= low_count && values.count <= high_count, "invalid S1 selection count");
    std::vector<int> result;
    for (id item in values) {
        const int current = int(integer(item,0,uint64_t(maximum)));
        check(std::find(result.begin(),result.end(),current) == result.end(), "duplicate S1 selection"); result.push_back(current);
    }
    return result;
}
std::vector<int> sorted(std::vector<int> value) { std::sort(value.begin(),value.end()); return value; }
std::vector<int> full_layers() { std::vector<int> value; for (int i = 0; i < 32; ++i) value.push_back(i); return value; }
std::vector<int> expected_steps(const Binding &binding) {
    if (!binding.steps.empty()) return sorted(binding.steps);
    return sorted({0,binding.total_steps / 2,binding.total_steps - 1});
}
} // namespace

std::string checkpoint_fingerprint(const std::filesystem::path &checkpoint) {
    const auto path = std::filesystem::canonical(checkpoint);
    struct stat info{};
    check(::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode), "S1 checkpoint identity requires a regular source");
    const auto name = path.string();
    const auto identity = ":" + std::to_string(name.size()) + ":" + name + ":" + std::to_string(info.st_dev) + ":" +
        std::to_string(info.st_ino) + ":" + std::to_string(info.st_size) + ":" +
        std::to_string(info.st_mtimespec.tv_sec) + ":" + std::to_string(info.st_mtimespec.tv_nsec) + ":" +
        std::to_string(info.st_ctimespec.tv_sec) + ":" + std::to_string(info.st_ctimespec.tv_nsec);
    return fingerprint(identity);
}
std::string fingerprint_adapter(const std::string &adapter_identity, size_t index) {
    check(adapter_identity.size() <= 65536, "S1 adapter identity exceeds bound");
    return fingerprint(adapter_identity + ":" + std::to_string(index));
}

ExperimentalProfile ExperimentalProfile::load(const std::filesystem::path &path, const Binding &expected) {
    check(expected.experimental, "S1 requires explicit experiment opt-in");
    check((expected.model_id == "z-image-turbo" || expected.model_id == "qwen-image-2.1") && expected.hidden > 0 &&
          expected.hidden <= 8192 && expected.width == 512 && expected.height == 512 && expected.total_steps >= 3 &&
          expected.total_steps <= 4096 && expected.reference_count >= 0 && expected.reference_count <= 8 &&
          expected.reference_size >= 0 && expected.reference_size <= 4096 &&
          (!expected.reference_count || expected.reference_size > 0), "invalid S1 request/model geometry");
    check(sorted(expected.layers) == full_layers() && expected.rows_per_point == 8,
          "Runtime S1 requires all 32 layers and 8-row samples");
    const auto steps = expected_steps(expected);
    check(steps.size() == 3 && std::adjacent_find(steps.begin(),steps.end()) == steps.end() &&
          steps.front() >= 0 && steps.back() < expected.total_steps, "Runtime S1 requires three valid sample steps");
    check(calibration::valid_text(expected.runtime_recipe) && calibration::valid_text(expected.configured_backend) &&
          expected.loras.size() <= 16, "incomplete S1 runtime binding");
    for (const auto &lora : expected.loras)
        check(calibration::valid_text(lora.id) && digest(lora.fingerprint) && std::isfinite(lora.strength), "invalid S1 adapter binding");
    const auto source_digest = checkpoint_fingerprint(expected.checkpoint);
    const auto bytes = read_profile(path);
    @autoreleasepool {
        UniqueJsonKeys(bytes).validate();
        NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
        NSError *error = nil;
        NSDictionary *root = object([NSJSONSerialization JSONObjectWithData:data options:0 error:&error]);
        check(error == nil, "invalid S1 JSON");
        keys(root,{"schema_version","artifact_type","experimental_only","complete","scope","capture_recipe","selection",
                   "binding","calibration_sha256","recipe","alpha","scale_bounds","runtime_applied","quantization_qualified",
                   "performance_qualified","pending","cpu_replay","layers"});
        check(integer(root[@"schema_version"],2,2) == 2 && text(root[@"artifact_type"]) == "smoothquant_s1_candidate" &&
              text(root[@"scope"]) == "full-model-s1" && text(root[@"capture_recipe"]) == calibration::capture_recipe &&
              text(root[@"recipe"]) == s1_recipe, "unsupported/partial S1 candidate schema");
        boolean(root[@"experimental_only"],true); boolean(root[@"complete"],true);
        boolean(root[@"runtime_applied"],false); boolean(root[@"quantization_qualified"],false); boolean(root[@"performance_qualified"],false);
        ExperimentalProfile result;
        result.alpha_ = number(root[@"alpha"],0,1);
        if (expected.alpha) check(std::isfinite(*expected.alpha) && result.alpha_ == *expected.alpha, "S1 alpha binding mismatch");
        const auto bounds = array(root[@"scale_bounds"]);
        check(bounds.count == 2 && number(bounds[0],1./16,1./16) == 1./16 && number(bounds[1],16,16) == 16,
              "invalid S1 scale bounds");
        result.calibration_digest_ = text(root[@"calibration_sha256"]);
        check(digest(result.calibration_digest_), "invalid S1 calibration digest");
        const auto pending = array(root[@"pending"]); check(pending.count > 0 && pending.count <= 8, "missing S1 pending qualification");
        for (id item in pending) text(item);
        const auto replay = object(root[@"cpu_replay"]);
        keys(replay,{"scope","passed","dot_products","max_normalized_error"});
        check(text(replay[@"scope"]) == "synthetic weight rows bounded by supplied maxima; unquantized S1 algebra only",
              "S1 CPU replay cannot claim real-weight/quantized qualification");
        boolean(replay[@"passed"],true); integer(replay[@"dot_products"],1,4096); number(replay[@"max_normalized_error"],0,1e-12);

        const auto binding = object(root[@"binding"]);
        keys(binding,{"request_id","model_id","model_fingerprint","identity_kind","recipe","loras","seed","width","height",
                      "total_steps","reference_size","reference_count","execution_route","runtime_recipe","configured_backend","actual_execution"});
        auto &metadata = result.provenance_;
        metadata.request_id = text(binding[@"request_id"]); metadata.model_id = text(binding[@"model_id"]);
        metadata.model_fingerprint = text(binding[@"model_fingerprint"]); metadata.identity_kind = text(binding[@"identity_kind"]);
        metadata.recipe = text(binding[@"recipe"]); metadata.seed = integer(binding[@"seed"],0,UINT64_MAX);
        metadata.width = int(integer(binding[@"width"],512,512)); metadata.height = int(integer(binding[@"height"],512,512));
        metadata.total_steps = int(integer(binding[@"total_steps"],3,4096));
        metadata.reference_size = int(integer(binding[@"reference_size"],0,4096));
        metadata.reference_count = int(integer(binding[@"reference_count"],0,8));
        metadata.execution_route = text(binding[@"execution_route"]); metadata.runtime_recipe = text(binding[@"runtime_recipe"]);
        metadata.configured_backend = text(binding[@"configured_backend"]); metadata.actual_execution = text(binding[@"actual_execution"]);
        check(metadata.model_id == expected.model_id && metadata.model_fingerprint == source_digest &&
              metadata.identity_kind == "canonical-stat-identity" && metadata.recipe == calibration::capture_recipe &&
              metadata.total_steps == expected.total_steps && metadata.reference_size == expected.reference_size &&
              metadata.reference_count == expected.reference_count && metadata.runtime_recipe == expected.runtime_recipe &&
              metadata.configured_backend == expected.configured_backend, "S1 model/request/recipe binding mismatch");
        const auto loras = array(binding[@"loras"]);
        check(loras.count == expected.loras.size(), "S1 adapter count mismatch");
        for (size_t i = 0; i < expected.loras.size(); ++i) {
            const auto item = object(loras[i]); keys(item,{"id","fingerprint","strength"});
            calibration::LoRAIdentity lora{text(item[@"id"]),text(item[@"fingerprint"]),number(item[@"strength"],-1e100,1e100)};
            check(lora.id == expected.loras[i].id && lora.fingerprint == expected.loras[i].fingerprint &&
                  lora.strength == expected.loras[i].strength, "S1 adapter identity/order/strength mismatch");
            metadata.loras.push_back(std::move(lora));
        }
        const auto selection = object(root[@"selection"]); keys(selection,{"layers","steps","rows_per_point"});
        check(sorted(integers(selection[@"layers"],32,32,31)) == full_layers() &&
              sorted(integers(selection[@"steps"],3,3,expected.total_steps - 1)) == steps &&
              integer(selection[@"rows_per_point"],8,8) == expected.rows_per_point, "S1 sample selection mismatch");
        const auto layers = array(root[@"layers"]); check(layers.count == 32, "S1 has partial layer coverage");
        std::set<int> covered;
        for (id raw in layers) {
            const auto layer = object(raw); keys(layer,{"layer","hidden","s1","weight_source_fingerprint","contexts"});
            LayerScale scale; scale.layer = int(integer(layer[@"layer"],0,31));
            check(covered.insert(scale.layer).second && integer(layer[@"hidden"],1,8192) == uint64_t(expected.hidden),
                  "S1 duplicate layer or hidden mismatch");
            check(text(layer[@"weight_source_fingerprint"]) == source_digest, "S1 weight statistics source mismatch");
            const auto values = array(layer[@"s1"]); check(values.count == size_t(expected.hidden), "S1 scale channel count mismatch");
            scale.s1.reserve(values.count);
            for (id value in values) scale.s1.push_back(float(number(value,1./16,16)));
            const auto contexts = array(layer[@"contexts"]); check(contexts.count == 3, "S1 partial temporal coverage");
            std::vector<int> context_steps;
            for (id raw_context in contexts) {
                const auto context = object(raw_context); keys(context,{"layer","step","phase","rows","regions"});
                check(integer(context[@"layer"],0,31) == uint64_t(scale.layer), "S1 context layer mismatch");
                context_steps.push_back(int(integer(context[@"step"],0,uint64_t(expected.total_steps - 1))));
                const auto phase = text(context[@"phase"]); check(phase == "prefill" || phase == "denoise", "invalid S1 capture phase");
                const auto rows = integer(context[@"rows"],1,32768);
                const auto regions = array(context[@"regions"]); check(regions.count > 0 && regions.count <= 8, "missing S1 row regions");
                uint64_t end = 0; std::set<std::string> names;
                for (id raw_region in regions) {
                    const auto region = object(raw_region); keys(region,{"name","begin","end"});
                    check(names.insert(text(region[@"name"])).second && integer(region[@"begin"],0,rows) == end,
                          "S1 duplicate/overlapping row region");
                    end = integer(region[@"end"],end + 1,rows);
                }
                check(end == rows, "S1 regions omit source rows");
            }
            check(sorted(context_steps) == steps, "S1 duplicate/missing sampled step");
            result.layers_.push_back(std::move(scale));
        }
        check(covered.size() == 32, "S1 incomplete full model");
        std::sort(result.layers_.begin(),result.layers_.end(),[](const LayerScale &a,const LayerScale &b) { return a.layer < b.layer; });
        result.content_digest_ = sha256(bytes);
        return result;
    }
}
const LayerScale *ExperimentalProfile::find(int layer) const noexcept {
    if (layer < 0 || layer >= int(layers_.size()) || layers_[size_t(layer)].layer != layer) return nullptr;
    return &layers_[size_t(layer)];
}
std::span<const float> ExperimentalProfile::scales_for(int layer) const {
    const auto *value = find(layer); check(value != nullptr, "S1 layer is not covered"); return value->s1;
}
} // namespace tc::ane::smoothquant
