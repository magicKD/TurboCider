// CPU-only tests of the cache component used by the real native session.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include "source_lease.hpp"

namespace tc {
static std::string runtime_identity = "runtime-A";
static std::string catalog_runtime_identity() { return runtime_identity; }
}
namespace {
namespace streaming = tc::streaming;
static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
static std::string recipe = "original-i8-f32-scale-w8a8-rne127-bf16-io-v1";
static std::string ltx_convrot_compute_recipe() { return recipe; }
constexpr uint32_t kLtxVideoDim = 4096, kLtxAudioDim = 2048;
static std::string sha256_text(const std::string& text) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    require(CC_SHA256(text.data(), static_cast<CC_LONG>(text.size()), digest) != nullptr, "sha256");
    char hex[65];
    for (size_t i = 0; i < sizeof(digest); ++i) std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return hex;
}
static std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return sha256_text(std::string(std::istreambuf_iterator<char>(input), {}));
}
static bool valid_sha256(const std::string& value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
static NSDictionary* read_json(const std::filesystem::path& path) {
    NSData* data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "read JSON");
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    require([value isKindOfClass:NSDictionary.class], "JSON dictionary");
    return value;
}
static std::string string_value(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    return [value isKindOfClass:NSString.class] ? [value UTF8String] : "";
}
static std::string json(id value) {
    NSData* data = [NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    require(data != nil, "encode JSON");
    return {static_cast<const char*>(data.bytes), data.length};
}
static void read_exact(const std::filesystem::path& path, void* output, size_t bytes) {
    require(std::filesystem::file_size(path) == bytes, "exact tensor size");
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(static_cast<char*>(output), bytes)), "read tensor");
}
static void write_exact(const std::filesystem::path& path, const void* data, size_t bytes, const char*) {
    std::ofstream output(path, std::ios::binary);
    require(bool(output.write(static_cast<const char*>(data), bytes)), "write tensor");
}
#include "ltx_conditioning_cache.hpp"

static void check(bool value, const char* message) { require(value, message); }
}

int main() {
    @autoreleasepool {
        char name[] = "/tmp/tc-ltx-cache-test-XXXXXX";
        char* created = mkdtemp(name);
        check(created != nullptr, "temporary directory");
        const std::filesystem::path root(created), model = root / "model", checkpoint = model / "transformer";
        setenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR", (root / "cache").c_str(), 1);
        try {
            Conditioning input;
            input.rows = input.raw_rows = 2;
            input.connected = true;
            input.video.assign(input.rows * kLtxVideoDim, 0x3f80);
            input.audio.assign(input.rows * kLtxAudioDim, 0xbf80);
            input.mask = {0x3f80, 0};
            const std::string prompt = "fox\n音乐 🦊", binding = "request-held-source-generation-A";
            auto hit = [&] { return load_conditioning(model, checkpoint, prompt, binding); };
            check(!hit().connected, "initial cache miss");
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, binding);
            auto first = hit();
            check(first.connected && first.cache_hit && first.video == input.video &&
                  first.audio == input.audio && first.mask == input.mask, "all connector outputs round-trip");
            check(!load_conditioning(model, checkpoint, "different prompt", binding).connected, "prompt invalidation");
            check(!load_conditioning(model, checkpoint, prompt, binding + "-changed").connected, "source mutation invalidation");
            tc::runtime_identity = "runtime-B";
            check(!hit().connected, "runtime recipe invalidation");
            tc::runtime_identity = "runtime-A";
            recipe = "different numerical recipe";
            check(!hit().connected, "compute recipe invalidation");
            recipe = "original-i8-f32-scale-w8a8-rne127-bf16-io-v1";
            check(hit().cache_hit, "restored identity hits");
            // Resolution, frame count, audio enablement and random seed do not
            // enter this text-only key; the same arrays serve both stages.
            check(ltx_conditioning_cache_location(model, checkpoint, prompt, binding).key ==
                  ltx_conditioning_cache_location(root / "alternate-path", root / "other-checkpoint", prompt, binding).key,
                  "held source identity controls public cache, no path reopening");
            const auto location = ltx_conditioning_cache_location(model, checkpoint, prompt, binding);
            const auto video = location.directory / "video_context.bf16";
            std::fstream corrupt(video, std::ios::in | std::ios::out | std::ios::binary);
            corrupt.put('\0'); corrupt.close();
            check(!hit().connected && !std::filesystem::exists(location.directory), "corrupt tensor removed and recomputed");
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, binding);
            std::filesystem::resize_file(location.directory / "audio_context.bf16", 2);
            check(!hit().connected, "truncated tensor is a miss");
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, binding);
            NSMutableDictionary* metadata = [read_json(location.directory / "conditioning.json") mutableCopy];
            metadata[@"cache_identity"] = @"wrong-key";
            const auto encoded = json(metadata);
            write_exact(location.directory / "conditioning.json", encoded.data(), encoded.size(), "metadata");
            check(!hit().connected, "forged identity is a miss");
            // An unbound model-side fixture must never satisfy a public lease.
            write_ltx_conditioning_cache(model, checkpoint, prompt, input);
            auto legacy = ltx_conditioning_cache_location(model, checkpoint, prompt);
            std::filesystem::create_directories(model);
            std::filesystem::copy(legacy.directory, model / "conditioning", std::filesystem::copy_options::recursive);
            check(!hit().connected, "public miss ignores model-side connected fixture");
            check(load_conditioning(model, checkpoint, prompt).connected, "legacy fixture retained");
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, binding);
            input.video[0] = 0x4000;
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, binding);
            check(hit().video[0] == 0x3f80, "atomic published entry is immutable");
            for (const auto& file : std::filesystem::directory_iterator(root / "cache"))
                check(!file.path().filename().string().starts_with(".tmp-"), "staging directory cleanup");
            const auto source_path = root / "source.bin";
            write_exact(source_path, "held-source", 11, "source");
            tc::streaming::SourceFileIdentity source;
            source.logical_id = "fixture"; source.path = source_path;
            auto held = tc::streaming::SourceLease::capture({source});
            write_ltx_conditioning_cache(model, checkpoint, prompt, input, std::string(held->digest()));
            check(load_conditioning_with_sources(model, checkpoint, prompt, held.get()).cache_hit,
                  "actual held SourceLease cache hit");
            const auto old_digest = std::string(held->digest());
            write_exact(source_path, "mutated-now", 11, "source mutation");
            bool rejected = false;
            try { (void)load_conditioning_with_sources(model, checkpoint, prompt, held.get()); }
            catch (const std::exception&) { rejected = true; }
            check(rejected, "mutated held source rejected before cache use");
            auto updated = tc::streaming::SourceLease::capture({source});
            check(old_digest != updated->digest(), "source mutation changes request binding");
            check(!load_conditioning_with_sources(model, checkpoint, prompt, updated.get()).connected,
                  "new held source misses the previous derived entry");
            unsetenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR");
            check(!hit().connected, "disabled cache is a miss");
            std::filesystem::remove_all(root);
            std::puts("PASS connected-conditioning cache: round-trip video/audio/mask, prompt/source/runtime/recipe identity, corrupt/truncated/forged entries, unbound fixture isolation, atomic publication, actual SourceLease validation/mutation and disabled cache; CPU only");
        } catch (const std::exception& error) {
            std::filesystem::remove_all(root);
            std::fprintf(stderr, "FAIL %s\n", error.what());
            return 1;
        }
    }
}
