#include "bridge.hpp"
#include "../../backends/mlx.hpp"
#include "../../media/image.hpp"
#include "../../media/audio.hpp"
#include "../../media/video.hpp"
#include "../../models/ltx_runtime/ltx.h"
#include "../../models/ltx_runtime/ltx_gemma_encoder.h"
#include "../../models/ltx_runtime/ltx_mlx_audio_vae.h"
#include "../../models/ltx_runtime/ltx_mlx_bwe.h"
#include "../../models/ltx_runtime/ltx_mlx_vocoder.h"
#include "../../models/ltx_runtime/ltx_mlx_video_vae.h"
#include "../../models/ltx_runtime/ltx_native.h"
#include "../../models/ltx_runtime/ltx_rng.h"
#include "../../models/ltx_runtime/ltx_video_convert.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace tc {
namespace {

constexpr uint32_t kLtxVideoChannels = 128;
constexpr uint32_t kLtxAudioChannels = 128;
constexpr uint32_t kLtxVideoDim = 4096;
constexpr uint32_t kLtxAudioDim = 2048;
constexpr const char* kLtxRevision =
    "bf86adedf518142442575d1ce2e767b7d01c8c76";

static std::filesystem::path ltx_runtime_resource(
        const std::filesystem::path& model_root, const char* name) {
    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("TURBOCIDER_RESOURCE_DIR"))
        if (*configured) candidates.emplace_back(
            std::filesystem::path(configured) / name);
    Dl_info location{};
    if (dladdr((void*)&ltx_runtime_resource, &location) &&
        location.dli_fname) {
        auto library_directory = std::filesystem::path(location.dli_fname)
            .parent_path();
        candidates.emplace_back(library_directory / name);
        candidates.emplace_back(library_directory.parent_path() / "Resources" /
                                "TurboCider" / name);
    }
    candidates.emplace_back(model_root.parent_path().parent_path() /
                            "build/native" / name);
    candidates.emplace_back(std::filesystem::current_path() /
                            "build/native" / name);
    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate))
            return std::filesystem::absolute(candidate);
    return {};
}

struct HashCache {
    std::filesystem::path path;
    std::filesystem::file_time_type mtime{};
    uintmax_t bytes = 0;
    std::string sha256;
};

static std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) return {};
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) return {};
    std::array<char, 1 << 20> buffer{};
    while (stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize count = stream.gcount();
        if (count > 0 && CC_SHA256_Update(&context, buffer.data(),
                                          static_cast<CC_LONG>(count)) != 1)
            return {};
    }
    if (!stream.eof()) return {};
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (CC_SHA256_Final(digest, &context) != 1) return {};
    char output[CC_SHA256_DIGEST_LENGTH * 2 + 1] = {};
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        snprintf(output + index * 2, 3, "%02x", digest[index]);
    return output;
}

static std::string sha256_text(const std::string& value) {
    CC_SHA256_CTX context;
    require(CC_SHA256_Init(&context) == 1,
            "cannot initialize LTX conditioning cache hash");
    require(CC_SHA256_Update(&context, value.data(),
                             static_cast<CC_LONG>(value.size())) == 1,
            "cannot hash LTX conditioning cache identity");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    require(CC_SHA256_Final(digest, &context) == 1,
            "cannot finalize LTX conditioning cache hash");
    char output[CC_SHA256_DIGEST_LENGTH * 2 + 1] = {};
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        snprintf(output + index * 2, 3, "%02x", digest[index]);
    return output;
}

static std::string ltx_cache_file_identity(
        const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error).lexically_normal();
    if (error) return path.string() + ":unresolved";
    auto bytes = std::filesystem::file_size(absolute, error);
    if (error) return absolute.string() + ":missing";
    auto mtime = std::filesystem::last_write_time(absolute, error);
    if (error) return absolute.string() + ":" + std::to_string(bytes) +
        ":unknown-mtime";
    return absolute.string() + ":" + std::to_string(bytes) + ":" +
        std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

static std::string cached_sha256(const std::filesystem::path& path,
                                 HashCache& cache) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    require(!error, "cannot resolve LTX checkpoint path: " + path.string());
    auto bytes = std::filesystem::file_size(absolute, error);
    require(!error, "cannot inspect LTX checkpoint size: " + absolute.string());
    auto mtime = std::filesystem::last_write_time(absolute, error);
    require(!error, "cannot inspect LTX checkpoint timestamp: " + absolute.string());
    if (absolute != cache.path || bytes != cache.bytes || mtime != cache.mtime) {
        cache.sha256 = sha256_file(absolute);
        require(!cache.sha256.empty(),
                "cannot hash LTX checkpoint: " + absolute.string());
        auto final_bytes = std::filesystem::file_size(absolute, error);
        require(!error && final_bytes == bytes,
                "LTX checkpoint changed while it was being hashed: " +
                    absolute.string());
        auto final_mtime = std::filesystem::last_write_time(absolute, error);
        require(!error && final_mtime == mtime,
                "LTX checkpoint changed while it was being hashed: " +
                    absolute.string());
        cache.path = absolute;
        cache.bytes = bytes;
        cache.mtime = mtime;
    }
    return cache.sha256;
}

static std::string checkpoint_stat_identity(const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    require(!error, "cannot resolve LTX checkpoint path: " + path.string());
    auto bytes = std::filesystem::file_size(absolute, error);
    require(!error, "cannot inspect LTX checkpoint size: " + absolute.string());
    auto mtime = std::filesystem::last_write_time(absolute, error);
    require(!error, "cannot inspect LTX checkpoint timestamp: " + absolute.string());
    return absolute.string() + ":" + std::to_string(bytes) + ":" +
        std::to_string(static_cast<long long>(
            mtime.time_since_epoch().count()));
}

static uintmax_t manifest_bytes(NSDictionary* dictionary, NSString* key,
                                const std::string& context) {
    id value = dictionary[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            context + " must contain numeric " + std::string(key.UTF8String));
    double number = [value doubleValue];
    require(std::isfinite(number) && number >= 0.0 &&
                number == std::floor(number),
            context + " contains an invalid " + std::string(key.UTF8String));
    return static_cast<uintmax_t>([value unsignedLongLongValue]);
}

static double manifest_number(NSDictionary* dictionary, NSString* key,
                               const std::string& context) {
    id value = dictionary[key];
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            context + " must contain numeric " + std::string(key.UTF8String));
    double number = [value doubleValue];
    require(std::isfinite(number),
            context + " contains a non-finite " + std::string(key.UTF8String));
    return number;
}

static bool safe_filename(const std::string& value) {
    if (value.empty() || value == "." || value == "..") return false;
    std::filesystem::path path(value);
    return !path.is_absolute() && path == path.filename();
}

static bool valid_sha256(const std::string& value) {
    if (value.size() != CC_SHA256_DIGEST_LENGTH * 2) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
    });
}

/* Audio finalization remains a separate readiness gate. The native Audio VAE,
 * base vocoder, BWE, WAV, and AAC paths are linked, but only provenance-bound
 * assets may qualify for an eventual public executor. */
constexpr const char* kLtxAudioManifestSchema =
    "turbocider-ltx-audio-assets-v1";
static constexpr std::array<const char*, 5> kLtxAudioArtifactNames = {
    "ltx-2.5-audio-vae-bf16.safetensors",
    "ltx-2.5-audio-vae-vocoder-bf16.safetensors",
    "ltx-2.5-audio-vae-vocoder.safetensors",
    "audio_vae_vocoder.safetensors",
    "audio_vae.safetensors",
};

struct LtxAudioAssetStatus {
    std::filesystem::path artifact;
    std::filesystem::path manifest;
    std::string artifact_sha256;
    std::string status = "missing";
    std::string reason;
    bool assets_verified = false;
    bool native_supported = false;
};

static std::filesystem::path find_ltx_audio_artifact(
        const std::filesystem::path& root) {
    const std::array<std::filesystem::path, 6> directories = {
        root, root / "vae", root / "audio_vae", root / "vocoder",
        root / "audio", root / "audio_vae_vocoder",
    };
    for (const auto& directory : directories) {
        for (const auto* name : kLtxAudioArtifactNames) {
            auto path = directory / name;
            if (std::filesystem::is_regular_file(path)) return path;
        }
    }
    return {};
}

static std::filesystem::path find_ltx_audio_manifest(
        const std::filesystem::path& artifact) {
    const auto sidecar = std::filesystem::path(artifact.string() +
                                                ".manifest.json");
    if (std::filesystem::is_regular_file(sidecar)) return sidecar;
    const auto named = artifact.parent_path() / "ltx-audio.manifest.json";
    if (std::filesystem::is_regular_file(named)) return named;
    return {};
}

static void require_audio_component(NSDictionary* components, NSString* key) {
    id value = components[key];
    require([value isKindOfClass:NSString.class] &&
                [value length] > 0,
            "LTX audio manifest component is missing " +
                std::string(key.UTF8String));
    auto name = std::string([value UTF8String]);
    require(safe_filename(name),
            "LTX audio manifest component must be a plain filename: " + name);
}

static LtxAudioAssetStatus inspect_ltx_audio_assets(
        const std::filesystem::path& root) {
    require(std::filesystem::is_directory(root),
            "LTX model directory missing: " + root.string());
    LtxAudioAssetStatus result;
    result.artifact = find_ltx_audio_artifact(root);
    if (result.artifact.empty()) {
        result.reason =
            "no LTX Audio VAE/base-vocoder/BWE artifact found; expected " +
            std::string(kLtxAudioArtifactNames.front());
        return result;
    }
    result.artifact = std::filesystem::absolute(result.artifact);
    result.manifest = find_ltx_audio_manifest(result.artifact);
    if (result.manifest.empty()) {
        result.status = "unverified";
        result.reason = "audio artifact exists but its provenance manifest is missing";
        return result;
    }
    try {
        auto manifest = read_json(result.manifest);
        require(string_value(manifest, @"schema") == kLtxAudioManifestSchema,
                "unsupported LTX audio asset manifest schema");
        require(string_value(manifest, @"repository") == "Lightricks/LTX-2.5",
                "LTX audio asset manifest repository is not trusted");
        require(string_value(manifest, @"revision") == kLtxRevision,
                "LTX audio asset manifest revision is not trusted");
        NSDictionary* artifact = [manifest[@"artifact"] isKindOfClass:NSDictionary.class] ?
            manifest[@"artifact"] : nil;
        NSDictionary* components = [manifest[@"components"] isKindOfClass:NSDictionary.class] ?
            manifest[@"components"] : nil;
        require(artifact && components,
                "LTX audio asset manifest is missing artifact/components records");
        auto filename = string_value(artifact, @"filename");
        require(safe_filename(filename) &&
                    filename == result.artifact.filename().string(),
                "LTX audio asset manifest names a different artifact");
        auto bytes = manifest_bytes(artifact, @"bytes", "LTX audio artifact");
        require(bytes == std::filesystem::file_size(result.artifact),
                "LTX audio asset manifest size does not match the installed artifact");
        auto expected_sha = string_value(artifact, @"sha256");
        require(valid_sha256(expected_sha),
                "LTX audio asset manifest contains a malformed SHA-256 identity");
        require_audio_component(components, @"audio_vae_decoder");
        require_audio_component(components, @"base_vocoder");
        require_audio_component(components, @"bwe_vocoder");
        require_audio_component(components, @"mel_stft");
        id sample_rates_value = manifest[@"sample_rates"];
        require([sample_rates_value isKindOfClass:NSArray.class],
                "LTX audio asset manifest must declare 16 kHz and 48 kHz rates");
        NSArray* sample_rates = sample_rates_value;
        require(sample_rates.count == 2,
                "LTX audio asset manifest must declare exactly two sample rates");
        bool has_16k = false, has_48k = false;
        for (id value in sample_rates) {
            require([value isKindOfClass:NSNumber.class] &&
                        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
                    "LTX audio asset manifest sample_rates must be numeric");
            has_16k |= [value unsignedIntValue] == 16000u;
            has_48k |= [value unsignedIntValue] == 48000u;
        }
        require(has_16k && has_48k,
                "LTX audio asset manifest must declare 16000 and 48000 Hz");
        id channels = manifest[@"channels"];
        require([channels isKindOfClass:NSNumber.class] &&
                    CFGetTypeID((__bridge CFTypeRef)channels) != CFBooleanGetTypeID() &&
                    [channels unsignedIntValue] == 2u,
                "LTX audio asset manifest must declare stereo output");
        result.artifact_sha256 = sha256_file(result.artifact);
        require(result.artifact_sha256 == expected_sha,
                "LTX audio artifact SHA-256 does not match its manifest");
        result.assets_verified = true;
        result.native_supported = true;
        result.status = "verified_native_candidate_session_parity_pending";
        result.reason =
            "Audio assets and the native decode/mux path are available; end-to-end Session parity and public executor validation remain pending";
    } catch (const std::exception& exception) {
        result.status = "invalid";
        result.reason = exception.what();
    }
    return result;
}

struct LtxCheckpointSelection {
    std::filesystem::path checkpoint;
    std::filesystem::path manifest;
    std::string checkpoint_sha256;
    std::string lora_sha256;
    double strength = 0.0;
};

static void add_manifest_candidate(std::vector<std::filesystem::path>& paths,
                                   const std::filesystem::path& path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    if (error || !std::filesystem::is_regular_file(absolute)) return;
    for (const auto& existing : paths)
        if (existing == absolute) return;
    paths.push_back(std::move(absolute));
}

static void add_manifest_directory(std::vector<std::filesystem::path>& paths,
                                   const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error) || error) return;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error) return;
    for (const auto& entry : iterator) {
        if (entry.is_regular_file(error) && !error &&
            entry.path().filename().string().ends_with(".manifest.json"))
            add_manifest_candidate(paths, entry.path());
        error.clear();
    }
    std::sort(paths.begin(), paths.end());
}

static bool inspect_ltx_manifest(
        const std::filesystem::path& manifest_path,
        const LoRAAsset& requested,
        HashCache& base_hash_cache,
        HashCache& lora_hash_cache,
        HashCache& output_hash_cache,
        LtxCheckpointSelection& selection,
        std::string& failure) {
    try {
        auto manifest = read_json(manifest_path);
        auto schema = string_value(manifest, @"schema");
        const bool hybrid = schema == "h3-super-hybrid-refiner-v1";
        require(schema == "h3-super-merged-refiner-v1" || hybrid,
                "unsupported LTX merge manifest schema");
        auto algorithm = string_value(manifest, @"algorithm");
        require((!hybrid && algorithm ==
                     "comfy-runtime-equivalent-fp16-lora-fp32-mm-convrot-int8-v1") ||
                    (hybrid && algorithm ==
                     "selective-video-self-attention-bf16-comfy-runtime-equivalent-fp16-lora-fp32-mm-v1"),
                "unsupported LTX merge algorithm");
        require(string_value(manifest, @"repository") == "Lightricks/LTX-2.5",
                "LTX merge manifest repository is not trusted");
        require(string_value(manifest, @"revision") == kLtxRevision,
                "LTX merge manifest revision is not trusted");
        NSDictionary* base = [manifest[@"base"] isKindOfClass:NSDictionary.class] ?
            manifest[@"base"] : nil;
        NSDictionary* lora = [manifest[@"lora"] isKindOfClass:NSDictionary.class] ?
            manifest[@"lora"] : nil;
        NSDictionary* output = [manifest[@"output"] isKindOfClass:NSDictionary.class] ?
            manifest[@"output"] : nil;
        require([base isKindOfClass:NSDictionary.class] &&
                    [lora isKindOfClass:NSDictionary.class] &&
                    [output isKindOfClass:NSDictionary.class],
                "LTX merge manifest is missing base/lora/output records");
        auto base_filename = string_value(base, @"filename");
        auto lora_filename = string_value(lora, @"filename");
        auto output_filename = string_value(output, @"filename");
        require(safe_filename(base_filename) && safe_filename(lora_filename) &&
                    safe_filename(output_filename),
                "LTX merge manifest artifact names must be plain filenames");
        require(lora_filename == std::filesystem::path(requested.path).filename().string(),
                "LTX merge manifest does not describe the requested LoRA file");
        auto expected_strength = manifest_number(
            lora, @"strength", "LTX LoRA record");
        require(expected_strength > 0.0 && expected_strength <= 4.0 &&
                    std::abs(requested.strength -
                             static_cast<float>(expected_strength)) <= 1e-6f,
                "LTX requested LoRA strength does not match the merge manifest");
        auto base_bytes = manifest_bytes(base, @"bytes", "LTX base record");
        auto lora_bytes = manifest_bytes(lora, @"bytes", "LTX LoRA record");
        auto output_bytes = manifest_bytes(output, @"bytes", "LTX output record");
        auto base_checkpoint = manifest_path.parent_path() / base_filename;
        require(std::filesystem::is_regular_file(base_checkpoint),
                "LTX refiner base checkpoint named by the manifest is missing");
        require(base_bytes == std::filesystem::file_size(base_checkpoint),
                "LTX merge manifest base size does not match the installed checkpoint");
        auto lora_path = std::filesystem::path(requested.path);
        require(std::filesystem::is_regular_file(lora_path),
                "requested LTX LoRA file is missing");
        require(lora_bytes == std::filesystem::file_size(lora_path),
                "LTX merge manifest LoRA size does not match the requested file");
        auto output_path = manifest_path.parent_path() / output_filename;
        require(std::filesystem::is_regular_file(output_path),
                "LTX premerged checkpoint named by the manifest is missing");
        require(output_bytes == std::filesystem::file_size(output_path),
                "LTX merge manifest output size does not match the premerged checkpoint");
        auto expected_base_sha = string_value(base, @"sha256");
        auto expected_lora_sha = string_value(lora, @"sha256");
        auto expected_output_sha = string_value(output, @"sha256");
        require(valid_sha256(expected_base_sha) &&
                    valid_sha256(expected_lora_sha) &&
                    valid_sha256(expected_output_sha),
                "LTX merge manifest contains a malformed SHA-256 identity");
        NSDictionary* mapping = [manifest[@"mapping"] isKindOfClass:NSDictionary.class] ?
            manifest[@"mapping"] : nil;
        auto mapping_total = mapping ? manifest_bytes(mapping, @"total", "LTX mapping") : 0;
        auto mapping_missing = mapping ? manifest_bytes(mapping, @"missing", "LTX mapping") : 1;
        require(mapping && mapping_total > 0 && mapping_missing == 0,
                "LTX merge manifest mapping is incomplete");
        if (hybrid) {
            auto int8 = manifest_bytes(mapping, @"int8_convrot", "LTX mapping");
            auto bf16 = manifest_bytes(mapping, @"bf16_total", "LTX mapping");
            auto original = manifest_bytes(mapping, @"bf16_original", "LTX mapping");
            auto promoted = manifest_bytes(mapping, @"bf16_promoted", "LTX mapping");
            require(int8 > 0 && bf16 > 0 && int8 + bf16 == mapping_total &&
                        original + promoted == bf16,
                    "LTX hybrid refiner precision mapping is incomplete");
        } else {
            auto int8 = manifest_bytes(mapping, @"int8_convrot", "LTX mapping");
            auto bf16 = manifest_bytes(mapping, @"bf16", "LTX mapping");
            require(int8 > 0 && bf16 > 0 && int8 + bf16 == mapping_total,
                    "LTX merged refiner precision mapping is incomplete");
        }
        require(cached_sha256(base_checkpoint, base_hash_cache) == expected_base_sha,
                "LTX installed base checkpoint SHA-256 does not match the merge manifest");
        require(cached_sha256(lora_path, lora_hash_cache) == expected_lora_sha,
                "LTX requested LoRA SHA-256 does not match the merge manifest");
        require(cached_sha256(output_path, output_hash_cache) == expected_output_sha,
                "LTX premerged checkpoint SHA-256 does not match the merge manifest");
        selection.checkpoint = std::move(output_path);
        selection.manifest = manifest_path;
        selection.checkpoint_sha256 = expected_output_sha;
        selection.lora_sha256 = expected_lora_sha;
        selection.strength = expected_strength;
        return true;
    } catch (const std::exception& exception) {
        failure = exception.what();
        return false;
    }
}

static LtxCheckpointSelection resolve_ltx_checkpoint(
        const std::filesystem::path& root,
        const std::filesystem::path& default_checkpoint,
        const LoRAAsset& requested,
        HashCache& base_hash_cache,
        HashCache& lora_hash_cache,
        HashCache& output_hash_cache) {
    std::filesystem::path lora_path = requested.path;
    require(std::filesystem::is_regular_file(lora_path),
            "requested LTX LoRA file is missing: " + lora_path.string());
    std::vector<std::filesystem::path> candidates;
    add_manifest_candidate(candidates, lora_path.string() + ".manifest.json");
    add_manifest_directory(candidates, lora_path.parent_path());
    if (lora_path.parent_path().has_parent_path())
        add_manifest_directory(candidates,
                               lora_path.parent_path().parent_path() / "diffusion_models");
    add_manifest_directory(candidates, root / "diffusion_models");
    add_manifest_directory(candidates, root);
    std::error_code canonical_error;
    auto canonical_checkpoint = std::filesystem::canonical(
        default_checkpoint, canonical_error);
    if (!canonical_error)
        add_manifest_directory(candidates, canonical_checkpoint.parent_path());
    std::sort(candidates.begin(), candidates.end());
    std::string failure = "no candidate manifest found";
    LtxCheckpointSelection selection;
    for (const auto& candidate : candidates) {
        if (inspect_ltx_manifest(candidate, requested,
                                 base_hash_cache, lora_hash_cache,
                                 output_hash_cache, selection, failure))
            return selection;
    }
    require(false, "LTX LoRA requires an offline-premerged checkpoint and "
                   "matching provenance manifest: " + failure);
    return selection;
}

static void dump_ltx_bf16(const std::filesystem::path& directory,
                          const char* name,
                          const uint16_t* values,
                          size_t elements) {
    if (directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    require(!error, "cannot create LTX tensor dump directory: " +
                directory.string());
    auto path = directory / (std::string(name) + ".bf16");
    std::ofstream stream(path, std::ios::binary);
    require(stream.good(), "cannot create LTX tensor dump: " + path.string());
    stream.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(elements * sizeof(uint16_t)));
    require(stream.good(), "cannot write LTX tensor dump: " + path.string());
}

static void write_ltx_dump_metadata(const std::filesystem::path& directory,
                                    const ltx_workload& workload) {
    if (directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    require(!error, "cannot create LTX tensor dump directory: " +
                directory.string());
    auto path = directory / "metadata.json";
    std::ofstream stream(path);
    require(stream.good(), "cannot create LTX dump metadata: " + path.string());
    stream << "{\n"
           << "  \"stage1_video_tokens\": " << workload.stage1_video_tokens << ",\n"
           << "  \"stage2_video_tokens\": " << workload.stage2_video_tokens << ",\n"
           << "  \"audio_tokens\": " << workload.audio_tokens << ",\n"
           << "  \"video_channels\": " << kLtxVideoChannels << ",\n"
           << "  \"audio_channels\": " << kLtxAudioChannels << "\n"
           << "}\n";
    require(stream.good(), "cannot write LTX dump metadata: " + path.string());
}

static void read_exact(const std::filesystem::path& path,
                       void* destination,
                       size_t bytes) {
    std::error_code size_error;
    auto file_bytes = std::filesystem::file_size(path, size_error);
    require(!size_error, "missing LTX conditioning file: " + path.string());
    require(file_bytes == bytes,
            "LTX conditioning file has an unexpected size: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "missing LTX conditioning file: " + path.string());
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    require(stream.good() && static_cast<size_t>(stream.gcount()) == bytes,
            "LTX conditioning file has an unexpected size: " + path.string());
}

static void write_exact(const std::filesystem::path& path,
                        const void* source, size_t bytes,
                        const char* label) {
    std::ofstream stream(path, std::ios::binary);
    require(stream.good(), "cannot create " + std::string(label) + ": " +
                path.string());
    stream.write(static_cast<const char*>(source),
                 static_cast<std::streamsize>(bytes));
    require(stream.good(), "cannot write " + std::string(label) + ": " +
                path.string());
}

static void read_exact_output(const std::filesystem::path& path,
                              void* destination, size_t bytes,
                              const char* label) {
    std::error_code size_error;
    auto file_bytes = std::filesystem::file_size(path, size_error);
    require(!size_error && file_bytes == bytes,
            std::string(label) + " has an unexpected size: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open " + std::string(label) + ": " +
                path.string());
    stream.read(static_cast<char*>(destination),
                static_cast<std::streamsize>(bytes));
    require(stream.good() && static_cast<size_t>(stream.gcount()) == bytes,
            "cannot read " + std::string(label) + ": " + path.string());
}

struct TemporaryDirectoryCleanup {
    std::filesystem::path path;
    ~TemporaryDirectoryCleanup() noexcept {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

static std::vector<uint16_t> decode_ltx_video_isolated(
        const std::filesystem::path& helper,
        const std::filesystem::path& checkpoint,
        const std::vector<uint16_t>& latent,
        const ltx_workload& workload,
        std::atomic<bool>& cancel) {
    require(std::filesystem::is_regular_file(helper),
            "LTX Video VAE helper is missing: " + helper.string());
    auto template_path = std::filesystem::temp_directory_path() /
        "turbocider-ltx-vae-XXXXXX";
    std::string template_string = template_path.string();
    std::vector<char> directory_text(template_string.begin(),
                                     template_string.end());
    directory_text.push_back('\0');
    char* created = ::mkdtemp(directory_text.data());
    require(created != nullptr,
            "cannot create LTX Video VAE temporary directory: " +
                std::string(std::strerror(errno)));
    std::filesystem::path directory(created);
    TemporaryDirectoryCleanup cleanup{directory};
    auto input = directory / "video_latent.bf16";
    auto output = directory / "video_pixels.bf16";
    write_exact(input, latent.data(), latent.size() * sizeof(uint16_t),
                "LTX Video VAE latent input");
    std::array<std::string, 7> arguments = {
        helper.string(), checkpoint.string(), input.string(),
        std::to_string(workload.latent_frames),
        std::to_string(workload.stage2_latent_height),
        std::to_string(workload.stage2_latent_width), output.string(),
    };
    std::array<char*, 8> argv{};
    for (size_t index = 0; index < arguments.size(); ++index)
        argv[index] = arguments[index].data();
    pid_t child = -1;
    int spawn_status = ::posix_spawn(
        &child, helper.c_str(), nullptr, nullptr, argv.data(), ::environ);
    require(spawn_status == 0 && child > 0,
            "cannot spawn LTX Video VAE helper: " +
                std::string(std::strerror(spawn_status)));
    int status = 0;
    for (;;) {
        pid_t waited = ::waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno == EINTR) continue;
        if (waited < 0)
            throw std::runtime_error(
                "cannot wait for LTX Video VAE helper: " +
                std::string(std::strerror(errno)));
        if (cancel.load()) {
            ::kill(child, SIGTERM);
            while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            throw Cancelled();
        }
        ::usleep(10000);
    }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "LTX Video VAE helper failed with status " +
                std::to_string(status));
    size_t pixel_count = static_cast<size_t>(3) * workload.frames *
        workload.output_height * workload.output_width;
    std::vector<uint16_t> pixels(pixel_count);
    read_exact_output(output, pixels.data(),
                      pixels.size() * sizeof(uint16_t),
                      "LTX Video VAE decoded pixels");
    return pixels;
}

[[noreturn]] static void exec_ltx_video_finalizer(
        const std::filesystem::path& helper,
        const std::filesystem::path& checkpoint,
        const std::vector<uint16_t>& latent,
        const ltx_workload& workload,
        const Request& request,
        const std::string& execution) {
    require(std::filesystem::is_regular_file(helper),
            "LTX Video VAE exec finalizer is missing: " + helper.string());
    auto template_path = std::filesystem::temp_directory_path() /
        "turbocider-ltx-exec-finalizer-XXXXXX";
    std::string directory_text = template_path.string();
    std::vector<char> mutable_directory(directory_text.begin(),
                                         directory_text.end());
    mutable_directory.push_back('\0');
    char* created = ::mkdtemp(mutable_directory.data());
    require(created != nullptr,
            "cannot create LTX exec finalizer directory: " +
                std::string(std::strerror(errno)));
    const auto directory = std::filesystem::path(created);
    const auto input = directory / "video_latent.bf16";
    write_exact(input, latent.data(), latent.size() * sizeof(uint16_t),
                "LTX exec finalizer latent");
    std::array<std::string, 9> arguments = {
        helper.string(), checkpoint.string(), input.string(),
        std::to_string(workload.latent_frames),
        std::to_string(workload.stage2_latent_height),
        std::to_string(workload.stage2_latent_width), request.output,
        request.operation, execution,
    };
    std::array<char*, 10> argv{};
    for (size_t index = 0; index < arguments.size(); ++index)
        argv[index] = arguments[index].data();
    fflush(nullptr);
    ::execve(helper.c_str(), argv.data(), ::environ);
    throw std::runtime_error("exec LTX Video VAE finalizer: " +
                             std::string(std::strerror(errno)));
}

struct Conditioning {
    std::filesystem::path directory;
    uint32_t rows = 0;
    uint32_t raw_rows = 0;
    bool connected = false;
    bool cache_hit = false;
    std::vector<uint16_t> video;
    std::vector<uint16_t> audio;
    std::vector<uint16_t> mask;
};

struct ConditioningCacheLocation {
    std::filesystem::path root;
    std::filesystem::path directory;
    std::string key;
};

struct LtxAneConfig {
    std::filesystem::path mlp_stage1;
    std::filesystem::path mlp_stage2;
    std::filesystem::path v2a_stage1;
    std::filesystem::path v2a_stage2;
    std::filesystem::path kv;
    std::filesystem::path qkv_stage1;
    std::filesystem::path qkv_stage2;
    std::string variant = "int8_pc";
    bool parallel_av = false;
    bool preload_stage2 = false;
    bool release_full_gpu_mlp = false;
    bool detach_stage1 = false;
    bool detach_stage2 = false;
    bool release_blocks_final_step = false;
    bool fused_mlp_residual = false;
    bool fused_mlp_adaln_pack = false;
    uint32_t kv_stage_mask = 1u;
    std::string identity;
};

static void validate_ane_profile_keys(NSDictionary* dictionary) {
    NSSet* allowed = [NSSet setWithArray:@[
        @"schema", @"mlp_stage1", @"mlp_stage2", @"v2a_stage1",
        @"v2a_stage2", @"kv", @"qkv_stage1", @"qkv_stage2",
        @"variant", @"parallel_av", @"preload_stage2",
        @"release_full_gpu_mlp", @"detach_stage1", @"detach_stage2",
        @"release_blocks_final_step", @"fused_mlp_residual",
        @"fused_mlp_adaln_pack", @"kv_stage_mask"]];
    for (NSString* key in dictionary)
        require([allowed containsObject:key],
                "unknown LTX ANE profile field: " +
                    std::string(key.UTF8String));
}

static bool complete_ane_directory(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) return false;
    for (uint32_t block = 0; block < 48u; ++block) {
        if (!std::filesystem::is_regular_file(
                directory / ("block-" + std::to_string(block)) /
                "manifest.json")) return false;
    }
    return true;
}

static uint64_t ane_source_mtime_ns(const std::filesystem::path& path,
                                    const std::string& context) {
    struct stat info {};
    require(::stat(path.c_str(), &info) == 0,
            "cannot stat " + context + ": " + path.string());
#if defined(__APPLE__)
    require(info.st_mtimespec.tv_sec >= 0 && info.st_mtimespec.tv_nsec >= 0,
            context + " has an invalid modification timestamp");
    return static_cast<uint64_t>(info.st_mtimespec.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(info.st_mtimespec.tv_nsec);
#else
    require(info.st_mtim.tv_sec >= 0 && info.st_mtim.tv_nsec >= 0,
            context + " has an invalid modification timestamp");
    return static_cast<uint64_t>(info.st_mtim.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(info.st_mtim.tv_nsec);
#endif
}

static void validate_ane_source_identity(
        NSDictionary* manifest, const std::filesystem::path& checkpoint,
        const std::string& context) {
    if (checkpoint.empty()) return;
    id source_value = manifest[@"source"];
    require([source_value isKindOfClass:NSDictionary.class],
            context + " is missing checkpoint provenance");
    auto source = (NSDictionary*)source_value;
    auto source_path_value = string_value(source, @"path");
    require(!source_path_value.empty(),
            context + " provenance is missing source.path");
    std::error_code source_error, checkpoint_error;
    auto source_path = std::filesystem::canonical(
        source_path_value, source_error);
    auto checkpoint_path = std::filesystem::canonical(
        checkpoint, checkpoint_error);
    require(!source_error && !checkpoint_error,
            context + " checkpoint provenance path cannot be resolved");
    require(source_path == checkpoint_path,
            context + " was exported from a different checkpoint");
    auto expected_bytes = manifest_bytes(
        source, @"bytes", context + " source");
    require(expected_bytes == std::filesystem::file_size(checkpoint_path),
            context + " checkpoint size does not match its provenance");
    auto expected_mtime = manifest_bytes(
        source, @"mtime_ns", context + " source");
    require(expected_mtime == ane_source_mtime_ns(checkpoint_path, context),
            context + " checkpoint timestamp does not match its provenance");
}

static void validate_ane_directory_geometry(
        const std::filesystem::path& directory, const char* name,
        NSString* schema, NSString* shape_key, uint32_t expected_rows,
        const std::filesystem::path& checkpoint = {}) {
    require(complete_ane_directory(directory),
            std::string("LTX ANE ") + name +
                " must contain all 48 block manifests");
    uint32_t observed_rows = 0;
    for (uint32_t block = 0; block < 48u; ++block) {
        auto manifest_path = directory / ("block-" + std::to_string(block)) /
            "manifest.json";
        auto manifest = read_json(manifest_path);
        require(string_value(manifest, @"schema") ==
                    std::string(schema.UTF8String),
                std::string("LTX ANE ") + name +
                    " block manifest has an unexpected schema");
        require(manifest_bytes(
                    manifest, @"block_index",
                    std::string("LTX ANE ") + name + " manifest") == block,
                std::string("LTX ANE ") + name +
                    " block manifest has the wrong block_index");
        validate_ane_source_identity(
            manifest, checkpoint, std::string("LTX ANE ") + name +
                " block " + std::to_string(block));
        id shape_value = manifest[@"shape"];
        require([shape_value isKindOfClass:NSDictionary.class],
                std::string("LTX ANE ") + name +
                    " block manifest is missing shape");
        auto shape = (NSDictionary*)shape_value;
        auto rows = manifest_bytes(
            shape, shape_key, std::string("LTX ANE ") + name + " shape");
        require(rows > 0 && rows <= UINT32_MAX,
                std::string("LTX ANE ") + name +
                    " block manifest has invalid row geometry");
        if (block == 0u) observed_rows = static_cast<uint32_t>(rows);
        require(rows == observed_rows,
                std::string("LTX ANE ") + name +
                    " block manifests disagree on row geometry");
        bool supports_expected = !expected_rows || rows == expected_rows;
        id supported_value = shape[@"supported_rows"];
        if (expected_rows && supported_value) {
            require([supported_value isKindOfClass:NSArray.class],
                    std::string("LTX ANE ") + name +
                        " supported_rows must be an array");
            for (id value in (NSArray*)supported_value) {
                require([value isKindOfClass:NSNumber.class] &&
                            CFGetTypeID((__bridge CFTypeRef)value) !=
                                CFBooleanGetTypeID(),
                        std::string("LTX ANE ") + name +
                            " supported_rows must be numeric");
                supports_expected |= [value unsignedLongLongValue] ==
                    expected_rows;
            }
        }
        require(supports_expected,
                std::string("LTX ANE ") + name + " rows=" +
                    std::to_string(rows) + " do not match request rows=" +
                    std::to_string(expected_rows));
    }
}

static std::filesystem::path ane_profile_path(
        NSDictionary* profile, NSString* key,
        const std::filesystem::path& base, bool required) {
    auto value = string_value(profile, key);
    if (value.empty()) {
        require(!required, "LTX ANE profile is missing " +
                    std::string(key.UTF8String));
        return {};
    }
    std::filesystem::path path(value);
    if (path.is_relative()) path = base / path;
    std::error_code error;
    path = std::filesystem::absolute(path, error).lexically_normal();
    require(!error, "cannot resolve LTX ANE profile path: " + value);
    return path;
}

static LtxAneConfig resolve_ltx_ane_config(
        const std::filesystem::path& manifest_path,
        uint32_t expected_stage1_rows = 0u,
        uint32_t expected_stage2_rows = 0u,
        uint32_t expected_text_rows = 0u,
        const std::filesystem::path& checkpoint = {}) {
    require(!manifest_path.empty(),
            "gpu_ane requires an explicit LTX ANE profile or directory");
    std::error_code error;
    auto absolute = std::filesystem::absolute(manifest_path, error);
    require(!error, "cannot resolve LTX ANE profile path: " +
                manifest_path.string());
    LtxAneConfig result;
    result.parallel_av = true;
    result.preload_stage2 = true;
    result.release_full_gpu_mlp = true;
    result.detach_stage1 = true;
    result.release_blocks_final_step = true;
    result.kv_stage_mask = 1u;
    NSDictionary* profile = nil;
    std::filesystem::path base = absolute;
    if (std::filesystem::is_regular_file(absolute)) {
        profile = read_json(absolute);
        validate_ane_profile_keys(profile);
        require(string_value(profile, @"schema") == "turbocider-ltx-ane-v1",
                "unsupported LTX ANE profile schema");
        base = absolute.parent_path();
        result.mlp_stage1 = ane_profile_path(profile, @"mlp_stage1", base, true);
        result.mlp_stage2 = ane_profile_path(profile, @"mlp_stage2", base, true);
        result.v2a_stage1 = ane_profile_path(profile, @"v2a_stage1", base, false);
        result.v2a_stage2 = ane_profile_path(profile, @"v2a_stage2", base, false);
        result.kv = ane_profile_path(profile, @"kv", base, false);
        result.qkv_stage1 = ane_profile_path(profile, @"qkv_stage1", base, false);
        result.qkv_stage2 = ane_profile_path(profile, @"qkv_stage2", base, false);
        result.variant = string_value(profile, @"variant", result.variant);
        require(result.variant == "int8_pc",
                "unsupported LTX ANE artifact variant");
        id parallel = profile[@"parallel_av"];
        if (parallel) {
            require([parallel isKindOfClass:NSNumber.class] &&
                        CFGetTypeID((__bridge CFTypeRef)parallel) ==
                            CFBooleanGetTypeID(),
                    "LTX ANE profile parallel_av must be bool");
            result.parallel_av = [parallel boolValue];
        }
        auto read_bool = [&](NSString* key, bool current) {
            id value = profile[key];
            if (!value) return current;
            require([value isKindOfClass:NSNumber.class] &&
                        CFGetTypeID((__bridge CFTypeRef)value) ==
                            CFBooleanGetTypeID(),
                    "LTX ANE profile " + std::string(key.UTF8String) +
                        " must be bool");
            return [value boolValue];
        };
        result.preload_stage2 = read_bool(
            @"preload_stage2", result.preload_stage2);
        result.release_full_gpu_mlp = read_bool(
            @"release_full_gpu_mlp", result.release_full_gpu_mlp);
        result.detach_stage1 = read_bool(
            @"detach_stage1", result.detach_stage1);
        result.detach_stage2 = read_bool(
            @"detach_stage2", result.detach_stage2);
        result.release_blocks_final_step = read_bool(
            @"release_blocks_final_step", result.release_blocks_final_step);
        result.fused_mlp_residual = read_bool(
            @"fused_mlp_residual", result.fused_mlp_residual);
        result.fused_mlp_adaln_pack = read_bool(
            @"fused_mlp_adaln_pack", result.fused_mlp_adaln_pack);
        id kv_mask = profile[@"kv_stage_mask"];
        if (kv_mask) {
            require([kv_mask isKindOfClass:NSNumber.class] &&
                        CFGetTypeID((__bridge CFTypeRef)kv_mask) !=
                            CFBooleanGetTypeID(),
                    "LTX ANE profile kv_stage_mask must be numeric");
            double value = [kv_mask doubleValue];
            require(std::isfinite(value) && value == std::floor(value) &&
                        value >= 0.0 && value <= 3.0,
                    "LTX ANE profile kv_stage_mask must be 0..3");
            result.kv_stage_mask = static_cast<uint32_t>(value);
        }
    } else {
        require(std::filesystem::is_directory(absolute),
                "LTX ANE profile directory is missing: " + absolute.string());
        /* A directory profile is intentionally narrow: it must contain both
         * stage directories, so a partial or ambiguous artifact tree cannot
         * silently enable only one denoising stage. */
        result.mlp_stage1 = absolute / "stage1";
        result.mlp_stage2 = absolute / "stage2";
        if (!std::filesystem::is_directory(result.mlp_stage1) ||
            !std::filesystem::is_directory(result.mlp_stage2)) {
            require(false, "LTX ANE directory must contain stage1 and stage2");
        }
        auto kv = absolute / "text_kv";
        if (std::filesystem::is_directory(kv)) result.kv = std::move(kv);
    }
    require(!result.release_full_gpu_mlp || result.preload_stage2,
            "LTX ANE full GPU MLP release requires Stage-2 preload");
    require(!(result.fused_mlp_residual && result.fused_mlp_adaln_pack),
            "LTX ANE MLP residual and AdaLN-pack fusion cannot both be enabled");
    if (result.kv.empty()) result.kv_stage_mask = 0u;
    auto validate = [](const std::filesystem::path& path,
                       const char* name, bool required) {
        if (path.empty()) {
            require(!required, std::string("LTX ANE profile is missing ") + name);
            return;
        }
        require(complete_ane_directory(path),
                std::string("LTX ANE ") + name +
                    " must contain all 48 block manifests");
    };
    validate(result.mlp_stage1, "stage1 MLP", true);
    validate(result.mlp_stage2, "stage2 MLP", true);
    validate(result.v2a_stage1, "stage1 V2A", false);
    validate(result.v2a_stage2, "stage2 V2A", false);
    validate(result.kv, "text K/V", false);
    validate(result.qkv_stage1, "stage1 QKV", false);
    validate(result.qkv_stage2, "stage2 QKV", false);
    if (expected_stage1_rows && expected_stage2_rows) {
        validate_ane_directory_geometry(
            result.mlp_stage1, "stage1 MLP", @"ltx-ane-mlp-v1", @"rows",
            expected_stage1_rows, checkpoint);
        validate_ane_directory_geometry(
            result.mlp_stage2, "stage2 MLP", @"ltx-ane-mlp-v1", @"rows",
            expected_stage2_rows, checkpoint);
        if (!result.v2a_stage1.empty())
            validate_ane_directory_geometry(
                result.v2a_stage1, "stage1 V2A", @"ltx-ane-v2a-v1",
                @"video_rows", expected_stage1_rows, checkpoint);
        if (!result.v2a_stage2.empty())
            validate_ane_directory_geometry(
                result.v2a_stage2, "stage2 V2A", @"ltx-ane-v2a-v1",
                @"video_rows", expected_stage2_rows, checkpoint);
        if (!result.qkv_stage1.empty())
            validate_ane_directory_geometry(
                result.qkv_stage1, "stage1 QKV",
                @"ltx-ane-qkv-sequence-v1", @"rows",
                expected_stage1_rows, checkpoint);
        if (!result.qkv_stage2.empty())
            validate_ane_directory_geometry(
                result.qkv_stage2, "stage2 QKV",
                @"ltx-ane-qkv-sequence-v1", @"rows",
                expected_stage2_rows, checkpoint);
    }
    if (!result.kv.empty() && expected_text_rows)
        validate_ane_directory_geometry(
            result.kv, "text K/V", @"ltx-ane-text-kv-v1", @"text_rows",
            expected_text_rows, checkpoint);
    std::string profile_identity;
    if (std::filesystem::is_regular_file(absolute)) {
        profile_identity = sha256_file(absolute);
        require(!profile_identity.empty(),
                "cannot hash LTX ANE profile: " + absolute.string());
    } else {
        profile_identity = absolute.string();
    }
    result.identity = profile_identity + ":" + result.variant + ":" +
        (result.parallel_av ? "parallel" : "serial") + ":preload=" +
        (result.preload_stage2 ? "1" : "0") + ":release_gpu=" +
        (result.release_full_gpu_mlp ? "1" : "0") + ":detach=" +
        (result.detach_stage1 ? "1" : "0") +
        (result.detach_stage2 ? "1" : "0") + ":kv=" +
        std::to_string(result.kv_stage_mask) + ":fused=" +
        (result.fused_mlp_adaln_pack ? "adaln" :
         result.fused_mlp_residual ? "residual" : "none");
    return result;
}

/* The native denoiser consumes a contiguous BFHWC token prefix for I2V,
 * while the MLX VAE encoder accepts normalized BCFHW pixels.  Keep this
 * conversion here so the session can encode each stage at its own spatial
 * resolution and release the encoder before the denoiser starts. */
static std::vector<uint16_t> encode_ltx_first_frame(
        ltx_mlx_video_vae* encoder,
        const std::filesystem::path& image_path,
        uint32_t width, uint32_t height,
        uint32_t latent_height, uint32_t latent_width,
        char* error, size_t error_size) {
    require(encoder != nullptr, "LTX first-frame encoder is unavailable");
    require(std::filesystem::is_regular_file(image_path),
            "LTX first-frame image is missing: " + image_path.string());
    require(width >= 32 && height >= 32 && width % 32u == 0 &&
                height % 32u == 0,
            "LTX first-frame dimensions must be divisible by 32");
    auto image = load_image_tensor(image_path, static_cast<int>(width),
                                   static_cast<int>(height), false);
    auto pixels = mx::contiguous(mx::astype(
        mx::expand_dims(mx::transpose(image, {0, 3, 1, 2}), 2),
        mx::bfloat16));
    mx::eval(pixels);
    const size_t pixel_elements = static_cast<size_t>(1) * 3u * 1u *
        height * width;
    require(pixels.size() == pixel_elements,
            "LTX first-frame tensor shape mismatch");
    std::vector<uint16_t> encoded_pixels(pixel_elements);
    std::memcpy(encoded_pixels.data(), pixels.data<uint16_t>(),
                encoded_pixels.size() * sizeof(uint16_t));
    const size_t token_elements = static_cast<size_t>(latent_height) *
        latent_width * kLtxVideoChannels;
    std::vector<uint16_t> tokens(token_elements);
    require(ltx_mlx_video_vae_encode_pixels_bf16(
                encoder, tokens.data(), tokens.size(), encoded_pixels.data(),
                encoded_pixels.size(), 1, 1, height, width,
                error, error_size),
            error);
    return tokens;
}

static std::filesystem::path conditioning_directory(
        const std::filesystem::path& root) {
    if (std::filesystem::is_regular_file(root / "conditioning" /
                                         "conditioning.json"))
        return root / "conditioning";
    return {};
}

static std::filesystem::path raw_conditioning_directory(
        const std::filesystem::path& root) {
    if (std::filesystem::is_regular_file(root / "conditioning-raw" /
                                         "raw_conditioning.json"))
        return root / "conditioning-raw";
    return {};
}

static ConditioningCacheLocation ltx_conditioning_cache_location(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& prompt) {
    ConditioningCacheLocation result;
    const char* configured = std::getenv(
        "TURBOCIDER_LTX_CONDITIONING_CACHE_DIR");
    if (!configured || !configured[0]) return result;
    std::error_code error;
    result.root = std::filesystem::absolute(configured, error).lexically_normal();
    if (error) return {};
    const auto absolute_root = std::filesystem::absolute(
        root, error).lexically_normal();
    if (error) return {};
    const auto gemma = root / "text_encoders" /
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
    const auto tokenizer = root / "gemma4-12b-ltx-v1" / "tokenizer.json";
    const std::string material = absolute_root.string() + "\n" +
        ltx_cache_file_identity(checkpoint) + "\n" +
        ltx_cache_file_identity(gemma) + "\n" +
        ltx_cache_file_identity(tokenizer) + "\n" + prompt;
    result.key = sha256_text(material);
    result.directory = result.root / result.key / "conditioning";
    return result;
}

static std::optional<std::string> conditioning_prompt(
        const std::filesystem::path& root,
        const std::filesystem::path& directory,
        NSDictionary* metadata, bool connected) {
    id direct = metadata[@"prompt"];
    if (direct) {
        require([direct isKindOfClass:NSString.class],
                "LTX conditioning prompt must be a string");
        return string_value(metadata, @"prompt");
    }
    if (!connected) return std::nullopt;
    const std::filesystem::path candidates[] = {
        directory.parent_path() / "raw" / "raw_conditioning.json",
        root / "conditioning-raw" / "raw_conditioning.json",
    };
    for (const auto& candidate : candidates) {
        if (!std::filesystem::is_regular_file(candidate)) continue;
        auto raw = read_json(candidate);
        require(string_value(raw, @"format") ==
                    "turbocider-ltx-raw-conditioning-v1",
                "unsupported LTX raw conditioning format");
        id prompt = raw[@"prompt"];
        require([prompt isKindOfClass:NSString.class],
                "LTX raw conditioning is missing its prompt identity");
        return string_value(raw, @"prompt");
    }
    return std::nullopt;
}

static Conditioning load_conditioning_directory(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::filesystem::path& directory,
        const std::string& requested_prompt,
        bool cache_hit) {
    Conditioning result;
    result.directory = directory;
    result.cache_hit = cache_hit;
    if (result.directory.empty()) return result;
    result.connected = std::filesystem::is_regular_file(
        result.directory / "conditioning.json");
    auto metadata = read_json(result.directory /
        (result.connected ? "conditioning.json" : "raw_conditioning.json"));
    auto format = string_value(metadata, @"format");
    require((result.connected && (format == "turbocider-ltx-conditioning-v1" ||
                                  format == "ltx-mac-conditioning-v1")) ||
            (!result.connected && (format == "turbocider-ltx-raw-conditioning-v1" ||
                                   format == "ltx-mac-raw-conditioning-v1")),
            "unsupported LTX conditioning format");
    if (cache_hit) {
        require(result.connected,
                "LTX conditioning cache must contain connected tensors");
        auto cached = ltx_conditioning_cache_location(
            root, checkpoint, requested_prompt);
        require(string_value(metadata, @"cache_identity") == cached.key,
                "LTX conditioning cache identity mismatch");
    }
    auto bound_prompt = conditioning_prompt(
        root, result.directory, metadata, result.connected);
    /* Unbound or stale fixtures are diagnostics, not a license to ignore the
     * request prompt.  Fall back to the dynamic Gemma path instead. */
    if (!bound_prompt || *bound_prompt != requested_prompt) return {};
    id rows = metadata[@"output_rows"];
    id raw_rows = metadata[@"input_rows"];
    id video_dim = metadata[@"video_dim"];
    id audio_dim = metadata[@"audio_dim"];
    if (!result.connected) {
        NSArray *video_shape = metadata[@"video_shape"];
        NSArray *audio_shape = metadata[@"audio_shape"];
        raw_rows = video_shape.count == 2 ? video_shape[0] : nil;
        video_dim = video_shape.count == 2 ? video_shape[1] : nil;
        audio_dim = audio_shape.count == 2 ? audio_shape[1] : nil;
        require(audio_shape.count == 2 &&
                [audio_shape[0] unsignedIntValue] == [raw_rows unsignedIntValue],
                "LTX raw conditioning row counts differ");
    }
    require((result.connected ? [rows isKindOfClass:NSNumber.class] :
             [metadata[@"expected_processed_rows"] isKindOfClass:NSNumber.class]) &&
            [raw_rows isKindOfClass:NSNumber.class] &&
            [video_dim isKindOfClass:NSNumber.class] &&
            [audio_dim isKindOfClass:NSNumber.class],
            "LTX conditioning metadata is incomplete");
    require([video_dim unsignedIntValue] == kLtxVideoDim &&
            [audio_dim unsignedIntValue] == kLtxAudioDim,
            "LTX conditioning dimensions do not match the native transformer");
    result.raw_rows = [raw_rows unsignedIntValue];
    result.rows = result.connected ? [rows unsignedIntValue] :
        [metadata[@"expected_processed_rows"] unsignedIntValue];
    require(result.raw_rows > 0 && result.raw_rows <= 4096 &&
            result.rows >= result.raw_rows && result.rows <= 4096,
            "LTX conditioning row count is out of range");
    result.video.resize(static_cast<size_t>(result.connected ? result.rows : result.raw_rows) * kLtxVideoDim);
    result.audio.resize(static_cast<size_t>(result.connected ? result.rows : result.raw_rows) * kLtxAudioDim);
    result.mask.resize(result.connected ? result.rows : result.raw_rows);
    const char* video_name = result.connected ? "video_context.bf16" : "raw_video_context.bf16";
    const char* audio_name = result.connected ? "audio_context.bf16" : "raw_audio_context.bf16";
    const char* mask_name = result.connected ? "text_mask.bf16" : "raw_text_mask.bf16";
    read_exact(result.directory / video_name, result.video.data(), result.video.size() * sizeof(uint16_t));
    read_exact(result.directory / audio_name, result.audio.data(), result.audio.size() * sizeof(uint16_t));
    read_exact(result.directory / mask_name, result.mask.data(), result.mask.size() * sizeof(uint16_t));
    if (cache_hit) {
        const auto video_hash = string_value(metadata, @"video_sha256");
        const auto audio_hash = string_value(metadata, @"audio_sha256");
        const auto mask_hash = string_value(metadata, @"mask_sha256");
        require(valid_sha256(video_hash) && valid_sha256(audio_hash) &&
                    valid_sha256(mask_hash),
                "LTX conditioning cache hashes are invalid");
        require(sha256_file(result.directory / video_name) == video_hash &&
                    sha256_file(result.directory / audio_name) == audio_hash &&
                    sha256_file(result.directory / mask_name) == mask_hash,
                "LTX conditioning cache content hash mismatch");
    }
    return result;
}

static Conditioning load_conditioning(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& requested_prompt) {
    auto directory = conditioning_directory(root);
    if (!directory.empty())
        return load_conditioning_directory(
            root, checkpoint, directory, requested_prompt, false);
    auto cached = ltx_conditioning_cache_location(
        root, checkpoint, requested_prompt);
    if (!cached.directory.empty() && std::filesystem::is_regular_file(
            cached.directory / "conditioning.json")) {
        try {
            return load_conditioning_directory(
                root, checkpoint, cached.directory, requested_prompt, true);
        } catch (...) {
            /* A cache entry is an optimization, never a reason to reject a
             * valid generation request. Recompute it from the bound Gemma
             * and connector instead of consuming stale/corrupt tensors. */
            std::error_code ignored;
            std::filesystem::remove_all(
                cached.directory.parent_path(), ignored);
        }
    }
    directory = raw_conditioning_directory(root);
    return load_conditioning_directory(
        root, checkpoint, directory, requested_prompt, false);
}

static void write_ltx_conditioning_cache(
        const std::filesystem::path& root,
        const std::filesystem::path& checkpoint,
        const std::string& prompt,
        const Conditioning& conditioning) {
    if (!conditioning.connected || conditioning.rows == 0 ||
        conditioning.video.empty() || conditioning.audio.empty() ||
        conditioning.mask.empty()) return;
    auto location = ltx_conditioning_cache_location(root, checkpoint, prompt);
    if (location.root.empty() || location.directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(location.root, error);
    require(!error, "cannot create LTX conditioning cache root: " +
                location.root.string());
    const auto final_directory = location.directory.parent_path();
    const auto final_parent = final_directory.parent_path();
    if (std::filesystem::is_regular_file(
            location.directory / "conditioning.json")) return;

    std::string template_text = (location.root / ".tmp-conditioning-XXXXXX").string();
    std::vector<char> mutable_template(template_text.begin(),
                                        template_text.end());
    mutable_template.push_back('\0');
    char* created = ::mkdtemp(mutable_template.data());
    require(created != nullptr,
            "cannot create LTX conditioning cache staging directory: " +
                std::string(std::strerror(errno)));
    const std::filesystem::path staging(created);
    auto cleanup = [&] {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
    };
    try {
        const auto staging_directory = staging / "conditioning";
        std::filesystem::create_directories(staging_directory, error);
        require(!error, "cannot create LTX conditioning cache directory");
        const auto video_path = staging_directory / "video_context.bf16";
        const auto audio_path = staging_directory / "audio_context.bf16";
        const auto mask_path = staging_directory / "text_mask.bf16";
        write_exact(video_path,
                    conditioning.video.data(),
                    conditioning.video.size() * sizeof(uint16_t),
                    "LTX cached video conditioning");
        write_exact(audio_path,
                    conditioning.audio.data(),
                    conditioning.audio.size() * sizeof(uint16_t),
                    "LTX cached audio conditioning");
        write_exact(mask_path,
                    conditioning.mask.data(),
                    conditioning.mask.size() * sizeof(uint16_t),
                    "LTX cached text mask");
        const auto video_hash = sha256_file(video_path);
        const auto audio_hash = sha256_file(audio_path);
        const auto mask_hash = sha256_file(mask_path);
        require(valid_sha256(video_hash) && valid_sha256(audio_hash) &&
                    valid_sha256(mask_hash),
                "cannot hash LTX conditioning cache tensors");
        auto metadata = json(@{
            @"format": @"turbocider-ltx-conditioning-v1",
            @"prompt": @(prompt.c_str()),
            @"checkpoint": @(checkpoint.string().c_str()),
            @"cache_identity": @(location.key.c_str()),
            @"input_rows": @(conditioning.rows),
            @"output_rows": @(conditioning.rows),
            @"register_rows": @128,
            @"video_dim": @4096,
            @"audio_dim": @2048,
            @"warmup": @0,
            @"cache": @YES,
            @"video_sha256": @(video_hash.c_str()),
            @"audio_sha256": @(audio_hash.c_str()),
            @"mask_sha256": @(mask_hash.c_str()),
        });
        std::ofstream metadata_stream(staging_directory / "conditioning.json");
        require(metadata_stream.good(),
                "cannot create LTX conditioning cache metadata");
        metadata_stream << metadata;
        metadata_stream.close();
        require(metadata_stream.good(),
                "cannot write LTX conditioning cache metadata");
        std::filesystem::create_directories(final_parent, error);
        require(!error, "cannot create LTX conditioning cache parent");
        std::filesystem::rename(staging, final_directory, error);
        if (error && !std::filesystem::is_directory(final_directory))
            require(false, "cannot publish LTX conditioning cache: " +
                    error.message());
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

struct Progress {
    const Event& event;
    std::atomic<bool>& cancel;
    std::exception_ptr failure;
    static int receive(const char* phase, int current, int total,
                       void* opaque) noexcept {
        auto& value = *static_cast<Progress*>(opaque);
        try {
            if (value.cancel.load()) return 1;
            value.event(phase ? phase : "ltx", current, total);
            return value.cancel.load() ? 1 : 0;
        } catch (...) {
            value.failure = std::current_exception();
            return 1;
        }
    }
};

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> action)
        : action_(std::move(action)) {}
    ~ScopeExit() noexcept { action_(); }

private:
    std::function<void()> action_;
};

class LtxNativeSession final : public ModelSession {
public:
    explicit LtxNativeSession(const std::filesystem::path& root)
        : root_(std::filesystem::absolute(root)) {
        require(std::filesystem::is_directory(root_),
                "LTX model directory missing: " + root_.string());
        checkpoint_path_ = root_ / "diffusion_models" /
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors";
        upsampler_path_ = root_ / "latent_upscale_models" /
            "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
        video_vae_path_ = root_ / "vae" /
            "ltx-2.5-video-vae-conv-bf16.safetensors";
        gemma_checkpoint_path_ = root_ / "text_encoders" /
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
        gemma_tokenizer_path_ = root_ / "gemma4-12b-ltx-v1" /
            "tokenizer.json";
        shader_path_ = ltx_runtime_resource(root_, "ltx_shaders.metal");
        require(std::filesystem::is_regular_file(shader_path_),
                "LTX Metal shader source is missing; install it beside "
                "libturbocider.dylib or set TURBOCIDER_RESOURCE_DIR");
        video_vae_helper_path_ = ltx_runtime_resource(
            root_, "ltx-video-vae-decode");
        video_vae_finalizer_path_ = ltx_runtime_resource(
            root_, "ltx-video-finalizer");
        require(std::filesystem::is_regular_file(checkpoint_path_),
                "LTX transformer checkpoint is missing");
        require(std::filesystem::is_regular_file(upsampler_path_),
                "LTX latent upsampler checkpoint is missing");
        require(std::filesystem::is_regular_file(video_vae_path_),
                "LTX video VAE checkpoint is missing");
    }
    void unload() override {
        denoiser_.reset(); denoiser_key_.clear(); gemma_encoder_.reset();
        video_vae_.reset(); audio_vae_.reset(); base_vocoder_.reset(); bwe_.reset();
    }

    RunResult generate(const Request& request,
                           const Event& event,
                           std::atomic<bool>& cancel) override {
        const auto request_started = Clock::now();
        require(request.model == "ltx-2.5-distilled",
                "request model differs from LTX session");
        auto plan = make_plan(request);
        require(!request.prompt.empty() && !request.output.empty(),
                "prompt and output are required");
        require(request.output.ends_with(".mp4"), "LTX output must be .mp4");
        const bool image_to_video = request.operation == "video.image";
        require(request.operation == "video.generate" || image_to_video,
                "native LTX candidate supports video.generate or video.image");
        if (!image_to_video) {
            require(request.inputs.empty(),
                    "video.generate does not accept media inputs");
        } else {
            require(request.inputs.size() == 1 &&
                        request.inputs.front().kind == "image" &&
                        request.inputs.front().role == "first_frame",
                    "video.image requires one first_frame image");
            require(std::filesystem::is_regular_file(request.inputs.front().path),
                    "LTX first-frame image is missing: " +
                        request.inputs.front().path);
            require(std::isfinite(request.inputs.front().strength) &&
                        request.inputs.front().strength >= 0.0f &&
                        request.inputs.front().strength <= 1.0f,
                    "LTX first-frame strength must be in [0, 1]");
        }
        require(request.loras.size() <= 1,
                "native LTX supports one manifest-bound transformer LoRA");
        if (!request.loras.empty()) {
            require(request.loras[0].role == "transformer" ||
                    request.loras[0].role == "refiner",
                    "native LTX LoRA role must be transformer or refiner");
            require(request.loras[0].strength > 0.0f &&
                    request.loras[0].strength <= 4.0f,
                    "native LTX LoRA strength must be in (0, 4]");
        }
        if (request.audio) {
            auto audio_assets = inspect_ltx_audio_assets(root_);
            require(audio_assets.assets_verified &&
                        audio_assets.native_supported,
                    "native LTX audio assets are not ready: " +
                        audio_assets.reason);
            audio_checkpoint_path_ = std::move(audio_assets.artifact);
        }
        require(request.execution == "gpu" || request.execution == "gpu_ane" ||
                    request.execution == "auto",
                "native LTX session supports auto, gpu, or explicit gpu_ane");
        const std::string effective_execution =
            request.execution == "auto" ? "gpu" : request.execution;
        checkpoint(cancel);
        LtxCheckpointSelection selection;
        if (!request.loras.empty()) {
            selection = resolve_ltx_checkpoint(
                root_, checkpoint_path_, request.loras.front(), base_hash_cache_,
                lora_hash_cache_, output_hash_cache_);
        } else {
            selection.checkpoint = checkpoint_path_;
        }
        const auto& selected_checkpoint = selection.checkpoint;
        const std::string selected_identity = selection.checkpoint_sha256.empty() ?
            checkpoint_stat_identity(selected_checkpoint) :
            selected_checkpoint.string() + ":sha256=" + selection.checkpoint_sha256;
        auto workload = ltx_workload{};
        char error[1024] = {};
        require(ltx_workload_init(&workload, request.width, request.height,
                                  request.frames, request.fps, error, sizeof(error)),
                error);
        LtxAneConfig ane_config;
        if (effective_execution == "gpu_ane")
            ane_config = resolve_ltx_ane_config(
                request.ane_manifest,
                static_cast<uint32_t>(workload.stage1_video_tokens),
                static_cast<uint32_t>(workload.stage2_video_tokens), 1024u,
                selected_checkpoint);
        require(request.residency == "resident" ||
                request.residency == "component_staged",
                "unsupported LTX residency");
        const bool component_staged = request.residency == "component_staged";
        auto conditioning = load_conditioning(
            root_, selected_checkpoint, request.prompt);
        bool used_dynamic_gemma = false;
        ScopeExit staged_cleanup([this, component_staged] {
            if (!component_staged) return;
            denoiser_.reset();
            denoiser_key_.clear();
            gemma_encoder_.reset();
            video_vae_.reset();
            audio_vae_.reset();
            base_vocoder_.reset();
            bwe_.reset();
            ltx_mlx_video_vae_clear_cache();
            ltx_mlx_audio_vae_clear_cache();
            ltx_mlx_vocoder_clear_cache();
            ltx_mlx_bwe_clear_cache();
        });

        std::vector<uint16_t> stage1_clean_prefix;
        std::vector<uint16_t> stage2_clean_prefix;
        const float first_frame_strength = image_to_video ?
            request.inputs.front().strength : 0.0f;
        if (image_to_video) {
            /* A resident decoder from an earlier request must not overlap the
             * encoder's weights in unified memory. */
            video_vae_.reset();
            ltx_mlx_video_vae_clear_cache();
            event("video_vae_encode", 0, 2);
            std::unique_ptr<ltx_mlx_video_vae,
                            decltype(&ltx_mlx_video_vae_free)> encoder(
                ltx_mlx_video_vae_create_encoder(
                    video_vae_path_.c_str(), error, sizeof(error)),
                ltx_mlx_video_vae_free);
            require(encoder != nullptr, error);
            ScopeExit encoder_cleanup([&encoder] {
                encoder.reset();
                ltx_mlx_video_vae_clear_cache();
            });
            stage1_clean_prefix = encode_ltx_first_frame(
                encoder.get(), request.inputs.front().path,
                workload.stage1_latent_width * 32u,
                workload.stage1_latent_height * 32u,
                workload.stage1_latent_height,
                workload.stage1_latent_width,
                error, sizeof(error));
            event("video_vae_encode", 1, 2);
            stage2_clean_prefix = encode_ltx_first_frame(
                encoder.get(), request.inputs.front().path,
                workload.stage2_latent_width * 32u,
                workload.stage2_latent_height * 32u,
                workload.stage2_latent_height,
                workload.stage2_latent_width,
                error, sizeof(error));
            event("video_vae_encode", 2, 2);
        }

        if (conditioning.directory.empty()) {
            /* Gemma and the 22B Transformer must not be resident together on
             * the 64 GB target.  The connected conditioning is persisted, so
             * a repeated prompt can keep the Transformer hot; a new prompt
             * safely gives up that residency before text encoding. */
            if (denoiser_) {
                denoiser_.reset();
                denoiser_key_.clear();
            }
            require(std::filesystem::is_regular_file(gemma_checkpoint_path_),
                    "LTX Gemma4 checkpoint is missing; provide conditioning or "
                    "install the text encoder checkpoint");
            require(std::filesystem::is_regular_file(gemma_tokenizer_path_),
                    "LTX Gemma4 tokenizer is missing; provide conditioning or "
                    "install the tokenizer");
            if (!gemma_encoder_) {
                event("text_encoder", 0, 1);
                ltx_gemma_encoder_options options{};
                options.checkpoint = gemma_checkpoint_path_.c_str();
                options.tokenizer_json = gemma_tokenizer_path_.c_str();
                options.shader_source = shader_path_.c_str();
                options.max_tokens = 1024u;
                gemma_encoder_.reset(ltx_gemma_encoder_create(
                    &options, error, sizeof(error)));
                require(gemma_encoder_ != nullptr, error);
                event("text_encoder", 1, 1);
            }
            const size_t max_video = static_cast<size_t>(1024u) * kLtxVideoDim;
            const size_t max_audio = static_cast<size_t>(1024u) * kLtxAudioDim;
            std::vector<uint16_t> raw_video(max_video);
            std::vector<uint16_t> raw_audio(max_audio);
            std::vector<uint16_t> raw_mask(1024u);
            uint32_t raw_rows = 0;
            Progress text_progress{event, cancel, {}};
            event("text_encode", 0, 1);
            bool encoded = ltx_gemma_encoder_encode(
                gemma_encoder_.get(), request.prompt.c_str(),
                raw_video.data(), raw_video.size(), raw_audio.data(),
                raw_audio.size(), raw_mask.data(), raw_mask.size(),
                &raw_rows, Progress::receive, &text_progress,
                error, sizeof(error));
            if (text_progress.failure)
                std::rethrow_exception(text_progress.failure);
            checkpoint(cancel);
            require(encoded, error);
            event("text_encode", 1, 1);
            require(raw_rows > 0 && raw_rows <= 1024u,
                    "LTX Gemma4 encoder returned an invalid row count");
            raw_video.resize(static_cast<size_t>(raw_rows) * kLtxVideoDim);
            raw_audio.resize(static_cast<size_t>(raw_rows) * kLtxAudioDim);
            raw_mask.resize(raw_rows);
            conditioning.directory = "<dynamic-gemma>";
            conditioning.raw_rows = raw_rows;
            /* The native connector pads all prompts shorter than 1024 rows. */
            conditioning.rows = 1024u;
            conditioning.video = std::move(raw_video);
            conditioning.audio = std::move(raw_audio);
            conditioning.mask = std::move(raw_mask);
            used_dynamic_gemma = true;
            gemma_encoder_.reset();
        }

        std::string denoiser_key = std::to_string(request.width) + "x" +
            std::to_string(request.height) + "x" +
            std::to_string(request.frames) + "@" +
            std::to_string(request.fps) + ":" + selected_identity + ":" +
            effective_execution + ":" +
            (ane_config.identity.empty() ? "dense" : ane_config.identity) +
            ":residency=" + request.residency +
            ":release_blocks=" +
            ((component_staged && (effective_execution != "gpu_ane" ||
                                   ane_config.release_blocks_final_step)) ?
                "1" : "0");
        const bool release_blocks_final_step = component_staged &&
            (effective_execution != "gpu_ane" ||
             ane_config.release_blocks_final_step);
        const bool denoiser_cache_hit = denoiser_ != nullptr &&
            denoiser_key == denoiser_key_;
        const auto model_load_started = Clock::now();
        if (denoiser_key != denoiser_key_) {
            denoiser_.reset();
            denoiser_key_.clear();
            event("model_load", 0, 1);
            ltx_native_options options{};
            options.checkpoint = selected_checkpoint.c_str();
            options.shader_source = shader_path_.c_str();
            options.width = request.width;
            options.height = request.height;
            options.frames = request.frames;
            options.fps = request.fps;
            options.parallel_av = ane_config.parallel_av ? 1 : 0;
            options.preload_ane_stage2 = ane_config.preload_stage2 ? 1 : 0;
            options.release_full_gpu_mlp =
                ane_config.release_full_gpu_mlp ? 1 : 0;
            options.detach_ane_stage1 = ane_config.detach_stage1 ? 1 : 0;
            options.detach_ane_stage2 = ane_config.detach_stage2 ? 1 : 0;
            options.release_blocks_final_step =
                release_blocks_final_step ? 1 : 0;
            options.ane_mlp_fused_residual =
                ane_config.fused_mlp_residual ? 1 : 0;
            options.ane_mlp_fused_adaln_pack =
                ane_config.fused_mlp_adaln_pack ? 1 : 0;
            options.ane_kv_stage_mask = ane_config.kv_stage_mask;
            options.mlp_directories[0] = ane_config.mlp_stage1.empty() ?
                nullptr : ane_config.mlp_stage1.c_str();
            options.mlp_directories[1] = ane_config.mlp_stage2.empty() ?
                nullptr : ane_config.mlp_stage2.c_str();
            options.v2a_directories[0] = ane_config.v2a_stage1.empty() ?
                nullptr : ane_config.v2a_stage1.c_str();
            options.v2a_directories[1] = ane_config.v2a_stage2.empty() ?
                nullptr : ane_config.v2a_stage2.c_str();
            options.kv_directory = ane_config.kv.empty() ?
                nullptr : ane_config.kv.c_str();
            options.qkv_directories[0] = ane_config.qkv_stage1.empty() ?
                nullptr : ane_config.qkv_stage1.c_str();
            options.qkv_directories[1] = ane_config.qkv_stage2.empty() ?
                nullptr : ane_config.qkv_stage2.c_str();
            Progress progress{event, cancel, {}};
            auto* created = ltx_native_create(&options, Progress::receive,
                                              &progress, error, sizeof(error));
            if (progress.failure) std::rethrow_exception(progress.failure);
            denoiser_.reset(created);
            require(denoiser_ != nullptr, error);
            denoiser_key_ = denoiser_key;
            event("model_load", 1, 1);
        }
        const auto model_ready = Clock::now();

        size_t stage1_video_count = static_cast<size_t>(workload.stage1_video_tokens) *
            kLtxVideoChannels;
        size_t stage2_video_count = static_cast<size_t>(workload.stage2_video_tokens) *
            kLtxVideoChannels;
        size_t audio_count = static_cast<size_t>(workload.audio_tokens) *
            kLtxAudioChannels;
        std::vector<uint16_t> video(stage1_video_count);
        std::vector<uint16_t> audio(audio_count);
        write_ltx_dump_metadata(request.dump, workload);
        const bool used_native_connector = !conditioning.connected;
        if (!conditioning.connected) {
            event("connector", 0, 1);
            auto raw_video = std::move(conditioning.video);
            auto raw_audio = std::move(conditioning.audio);
            auto raw_mask = std::move(conditioning.mask);
            std::vector<uint16_t> connected_video(
                static_cast<size_t>(conditioning.rows) * kLtxVideoDim);
            std::vector<uint16_t> connected_audio(
                static_cast<size_t>(conditioning.rows) * kLtxAudioDim);
            std::vector<uint16_t> connected_mask(conditioning.rows);
            require(ltx_native_connect_conditioning(
                denoiser_.get(), connected_video.data(), connected_video.size(),
                connected_audio.data(), connected_audio.size(),
                connected_mask.data(), connected_mask.size(), conditioning.rows,
                raw_video.data(), raw_video.size(), raw_audio.data(), raw_audio.size(),
                raw_mask.data(), raw_mask.size(), conditioning.raw_rows,
                error, sizeof(error)), error);
            conditioning.video.swap(connected_video);
            conditioning.audio.swap(connected_audio);
            conditioning.mask.swap(connected_mask);
            conditioning.connected = true;
            conditioning.raw_rows = conditioning.rows;
            write_ltx_conditioning_cache(
                root_, selected_checkpoint, request.prompt, conditioning);
            event("connector", 1, 1);
        }
        const std::string conditioning_mode = conditioning.cache_hit ?
            "connected_cache" : (used_dynamic_gemma ? "dynamic_gemma" :
            (used_native_connector ? "native_connector" :
             "connected_artifact"));
        ltx_rng video_rng{}, audio_rng{};
        ltx_rng_seed(&video_rng, request.seed + 10000u, 0u);
        ltx_rng_seed(&audio_rng, request.seed + 2u, 0u);
        ltx_rng_fill_normal_bf16(&video_rng, video.data(), video.size());
        ltx_rng_fill_normal_bf16(&audio_rng, audio.data(), audio.size());
        Progress progress{event, cancel, {}};
        const auto stage1_started = Clock::now();
        bool stage1_ok = ltx_native_run(denoiser_.get(), 1, request.seed,
                                video.data(), video.size(),
                                audio.data(), audio.size(), conditioning.video.data(),
                                conditioning.audio.data(), conditioning.mask.data(),
                                conditioning.rows,
                                stage1_clean_prefix.empty() ? nullptr :
                                    stage1_clean_prefix.data(),
                                first_frame_strength,
                                Progress::receive, &progress, error, sizeof(error));
        if (progress.failure) std::rethrow_exception(progress.failure);
        require(stage1_ok, error);
        dump_ltx_bf16(request.dump, "stage1_video", video.data(), video.size());
        dump_ltx_bf16(request.dump, "stage1_audio", audio.data(), audio.size());
        checkpoint(cancel);
        const auto stage1_finished = Clock::now();
        event("latent_upsample", 0, 1);
        std::vector<uint16_t> upsampled(stage2_video_count);
        require(ltx_native_upsample_stage2(
            denoiser_.get(), upsampler_path_.c_str(), video_vae_path_.c_str(),
            upsampled.data(), upsampled.size(), video.data(), video.size(),
            error, sizeof(error)), error);
        event("latent_upsample", 1, 1);
        video.swap(upsampled);
        dump_ltx_bf16(request.dump, "stage2_input_video", video.data(), video.size());
        const auto upsample_finished = Clock::now();
        bool stage2_ok = ltx_native_run(denoiser_.get(), 2, request.seed,
                                video.data(), video.size(),
                                audio.data(), audio.size(), conditioning.video.data(),
                                conditioning.audio.data(), conditioning.mask.data(),
                                conditioning.rows,
                                stage2_clean_prefix.empty() ? nullptr :
                                    stage2_clean_prefix.data(),
                                first_frame_strength,
                                Progress::receive, &progress, error, sizeof(error));
        if (progress.failure) std::rethrow_exception(progress.failure);
        require(stage2_ok, error);
        dump_ltx_bf16(request.dump, "stage2_video", video.data(), video.size());
        dump_ltx_bf16(request.dump, "stage2_audio", audio.data(), audio.size());
        checkpoint(cancel);
        const auto stage2_finished = Clock::now();
        if (component_staged) {
            denoiser_.reset();
            denoiser_key_.clear();
        }
        event("video_vae", 0, 1);
        const char* exec_value = std::getenv("TURBOCIDER_LTX_EXEC_FINALIZER");
        const bool exec_finalizer = component_staged && !request.audio &&
            exec_value && std::strcmp(exec_value, "1") == 0;
        if (exec_finalizer) {
            /* Only a single-shot CLI or disposable service worker may set
             * this flag. Replacing the process is the lifecycle optimization
             * validated by the clean-process reference path: it destroys Metal/MPSGraph/Core ML/MLX
             * process caches before the convolutional decoder starts. */
            require(std::filesystem::is_regular_file(video_vae_finalizer_path_),
                    "LTX exec finalizer is unavailable");
            video_vae_.reset();
            audio_vae_.reset();
            base_vocoder_.reset();
            bwe_.reset();
            ltx_mlx_video_vae_clear_cache();
            ltx_mlx_audio_vae_clear_cache();
            ltx_mlx_vocoder_clear_cache();
            ltx_mlx_bwe_clear_cache();
            const auto pre_finalizer_seconds = std::chrono::duration<double>(
                Clock::now() - request_started).count();
            const auto pre_finalizer_text =
                std::to_string(pre_finalizer_seconds);
            ::setenv("TURBOCIDER_LTX_PRE_FINALIZER_SECONDS",
                     pre_finalizer_text.c_str(), 1);
            ::setenv("TURBOCIDER_LTX_CONDITIONING_CACHE_HIT",
                     conditioning.cache_hit ? "1" : "0", 1);
            ::setenv("TURBOCIDER_LTX_CONDITIONING_MODE",
                     conditioning_mode.c_str(), 1);
            exec_ltx_video_finalizer(
                video_vae_finalizer_path_, video_vae_path_, video, workload,
                request, effective_execution);
        }
        std::string video_vae_isolation;
        std::vector<uint16_t> pixels;
        if (!request.audio &&
            std::filesystem::is_regular_file(video_vae_helper_path_)) {
            /* The old native runtime proved that a clean MLX process restores
             * the Video VAE from roughly 14 seconds to roughly 2.8 seconds at
             * the 13x14x22 latent shape.  For component-staged requests the
             * Transformer is already released; for resident requests the
             * clean child preserves the hot Transformer Session without
             * mixing its MPSGraph allocator with the MLX decoder. */
            video_vae_.reset();
            ltx_mlx_video_vae_clear_cache();
            pixels = decode_ltx_video_isolated(
                video_vae_helper_path_, video_vae_path_, video, workload,
                cancel);
            video_vae_isolation = "process";
        } else {
            if (!video_vae_) {
                video_vae_.reset(ltx_mlx_video_vae_create(
                    video_vae_path_.c_str(), error, sizeof(error)));
                require(video_vae_ != nullptr, error);
            }
            size_t pixel_count = static_cast<size_t>(3) * workload.frames *
                workload.output_height * workload.output_width;
            pixels.resize(pixel_count);
            require(ltx_mlx_video_vae_decode_tokens_bf16(
                video_vae_.get(), pixels.data(), pixels.size(), video.data(),
                video.size(), 1, workload.latent_frames,
                workload.stage2_latent_height, workload.stage2_latent_width,
                error, sizeof(error)), error);
            video_vae_isolation = "in_process";
        }
        event("video_vae", 1, 1);
        const auto video_vae_finished = Clock::now();
        std::vector<uint8_t> rgb(pixels.size());
        require(ltx_video_bf16_planar_to_rgb24(
                    rgb.data(), rgb.size(), pixels.data(), pixels.size(),
                    workload.frames, workload.output_height,
                    workload.output_width, error, sizeof(error)), error);
        const auto rgb_finished = Clock::now();
        std::filesystem::path video_only = request.output;
        std::filesystem::path audio_wav;
        ScopeExit media_cleanup([&] {
            if (!request.audio) return;
            std::error_code ignored;
            if (!video_only.empty()) std::filesystem::remove(video_only, ignored);
            if (!audio_wav.empty()) std::filesystem::remove(audio_wav, ignored);
        });
        if (request.audio) {
            video_only += std::string(".") + NSUUID.UUID.UUIDString.UTF8String + ".mp4";
        }
        event("export", 0, 1);
        write_video_rgb24(video_only, rgb.data(), workload.frames,
                          workload.output_width, workload.output_height,
                          workload.fps);
        event("export", 1, 1);
        const auto video_export_finished = Clock::now();
        tc::AudioMediaInfo audio_info{};
        if (request.audio) {
            const uint32_t audio_tokens = workload.audio_tokens;
            require(audio_tokens >= 1 && audio_tokens <= UINT32_MAX / 4u,
                    "LTX audio latent token count is invalid");
            const uint32_t mel_time = audio_tokens * 4u - 3u;
            const size_t mel_elements = static_cast<size_t>(2u) * mel_time * 64u;
            const size_t waveform_samples = static_cast<size_t>(mel_time) * 160u;
            const size_t waveform16_elements = waveform_samples * 2u;
            const size_t waveform48_elements = waveform_samples * 3u * 2u;
            std::vector<uint16_t> mel(mel_elements);
            std::vector<float> waveform16(waveform16_elements);
            std::vector<float> waveform48(waveform48_elements);
            if (!audio_vae_) {
                event("audio_vae", 0, 1);
                audio_vae_.reset(ltx_mlx_audio_vae_create(
                    audio_checkpoint_path_.c_str(), error, sizeof(error)));
                require(audio_vae_ != nullptr, error);
                event("audio_vae", 1, 1);
            }
            require(ltx_mlx_audio_vae_decode_bf16(
                audio_vae_.get(), mel.data(), mel.size(), audio.data(), audio.size(),
                1, audio_tokens, error, sizeof(error)), error);
            checkpoint(cancel);
            if (!base_vocoder_) {
                event("base_vocoder", 0, 1);
                base_vocoder_.reset(ltx_mlx_vocoder_create_base(
                    audio_checkpoint_path_.c_str(), error, sizeof(error)));
                require(base_vocoder_ != nullptr, error);
                event("base_vocoder", 1, 1);
            }
            require(ltx_mlx_vocoder_decode_base_bf16(
                base_vocoder_.get(), waveform16.data(), waveform16.size(),
                mel.data(), mel.size(), 1, mel_time, error, sizeof(error)), error);
            checkpoint(cancel);
            if (!bwe_) {
                event("bwe_vocoder", 0, 1);
                bwe_.reset(ltx_mlx_bwe_create(
                    audio_checkpoint_path_.c_str(), error, sizeof(error)));
                require(bwe_ != nullptr, error);
                event("bwe_vocoder", 1, 1);
            }
            require(ltx_mlx_bwe_extend_f32(
                bwe_.get(), waveform48.data(), waveform48.size(),
                waveform16.data(), waveform16.size(), 1,
                static_cast<uint32_t>(waveform_samples), error, sizeof(error)), error);
            checkpoint(cancel);
            audio_wav = video_only;
            audio_wav += ".wav";
            audio_info = write_audio_pcm16_wav(
                audio_wav, waveform48.data(), waveform48_elements / 2u,
                48000, 2);
            event("audio_mux", 0, 1);
            audio_info = mux_video_with_audio(
                video_only, request.output, waveform48.data(),
                waveform48_elements / 2u, 48000, 2);
            event("audio_mux", 1, 1);
        }
        auto info = probe_video(request.output);
        require(info.frames == static_cast<int>(workload.frames),
                "LTX exporter did not preserve the requested frame count");
        const auto request_wall_seconds = std::chrono::duration<double>(
            Clock::now() - request_started).count();
        auto value = @{ @"schema_version": @1,
                  @"model": @(request.model.c_str()),
                  @"operation": @(request.operation.c_str()),
                  @"output": @(request.output.c_str()),
                  @"width": @(info.width),
                  @"height": @(info.height),
                  @"frames": @(info.frames),
                  @"fps": @(info.fps),
                  @"audio": @(request.audio),
                  @"audio_sample_rate": request.audio ?
                      (id)@(audio_info.sample_rate) : (id)[NSNull null],
                  @"audio_channels": request.audio ?
                      (id)@(audio_info.channels) : (id)[NSNull null],
                  @"audio_samples": request.audio ?
                      (id)@(audio_info.samples) : (id)[NSNull null],
                  @"audio_duration_seconds": request.audio ?
                      (id)@(audio_info.duration_seconds) : (id)[NSNull null],
                  @"audio_clipped_samples": request.audio ?
                      (id)@(audio_info.clipped_samples) : (id)[NSNull null],
                  @"first_frame_conditioning": @(image_to_video),
                  @"first_frame_strength": image_to_video ?
                      (id)@(first_frame_strength) : (id)[NSNull null],
                  @"conditioning_cache_hit": @(conditioning.cache_hit),
                  @"conditioning_mode": @(conditioning_mode.c_str()),
                  @"denoiser_cache_hit": @(denoiser_cache_hit),
                  @"timings_seconds": @{
                      @"pre_model_load": @(
                          std::chrono::duration<double>(
                              model_load_started - request_started).count()),
                      @"model_load": @(
                          std::chrono::duration<double>(
                              model_ready - model_load_started).count()),
                      @"conditioning_and_rng": @(
                          std::chrono::duration<double>(
                              stage1_started - model_ready).count()),
                      @"stage1": @(
                          std::chrono::duration<double>(
                              stage1_finished - stage1_started).count()),
                      @"latent_upsample": @(
                          std::chrono::duration<double>(
                              upsample_finished - stage1_finished).count()),
                      @"stage2": @(
                          std::chrono::duration<double>(
                              stage2_finished - upsample_finished).count()),
                      @"video_vae_decode": @(
                          std::chrono::duration<double>(
                              video_vae_finished - stage2_finished).count()),
                      @"rgb_convert": @(
                          std::chrono::duration<double>(
                              rgb_finished - video_vae_finished).count()),
                      @"video_export": @(
                          std::chrono::duration<double>(
                              video_export_finished - rgb_finished).count()),
                      @"request_wall": @(request_wall_seconds),
                  },
                  @"execution": @(effective_execution.c_str()),
                  @"ane_profile": ane_config.identity.empty() ?
                      (id)[NSNull null] : @(ane_config.identity.c_str()),
                  @"video_vae_isolation": @(video_vae_isolation.c_str()),
                  @"plan": to_dictionary(plan),
                  @"lora_fusion": request.loras.empty() ? @"none" :
                      @"sidecar_manifest_verified",
                  @"checkpoint_sha256": selection.checkpoint_sha256.empty() ?
                      (id)[NSNull null] : @(selection.checkpoint_sha256.c_str()),
                  @"validation": request.audio ?
                      @"native_gpu_audio_video_aac_candidate" :
                      (image_to_video ?
                      @"native_gpu_video_only_i2v_clean_prefix_candidate" :
                      (used_dynamic_gemma ?
                      @"native_gpu_video_only_dynamic_gemma_connector_candidate" :
                      (used_native_connector ?
                      @"native_gpu_video_only_connector_verified" :
                      @"native_gpu_video_only_conditioning_verified"))) };
        return native_run_result(value, request, plan);
    }

private:
    std::filesystem::path root_;
    std::filesystem::path checkpoint_path_;
    std::filesystem::path upsampler_path_;
    std::filesystem::path video_vae_path_;
    std::filesystem::path audio_checkpoint_path_;
    std::filesystem::path gemma_checkpoint_path_;
    std::filesystem::path gemma_tokenizer_path_;
    std::filesystem::path shader_path_;
    std::filesystem::path video_vae_helper_path_;
    std::filesystem::path video_vae_finalizer_path_;
    std::string denoiser_key_;
    HashCache base_hash_cache_;
    HashCache lora_hash_cache_;
    HashCache output_hash_cache_;
    std::unique_ptr<ltx_native_denoiser, decltype(&ltx_native_free)> denoiser_{nullptr, ltx_native_free};
    std::unique_ptr<ltx_gemma_encoder, decltype(&ltx_gemma_encoder_free)> gemma_encoder_{nullptr, ltx_gemma_encoder_free};
    std::unique_ptr<ltx_mlx_video_vae, decltype(&ltx_mlx_video_vae_free)> video_vae_{nullptr, ltx_mlx_video_vae_free};
    std::unique_ptr<ltx_mlx_audio_vae, decltype(&ltx_mlx_audio_vae_free)> audio_vae_{nullptr, ltx_mlx_audio_vae_free};
    std::unique_ptr<ltx_mlx_vocoder, decltype(&ltx_mlx_vocoder_free)> base_vocoder_{nullptr, ltx_mlx_vocoder_free};
    std::unique_ptr<ltx_mlx_bwe, decltype(&ltx_mlx_bwe_free)> bwe_{nullptr, ltx_mlx_bwe_free};
};

}  // namespace

std::unique_ptr<ModelSession> create_ltx_native_candidate(
        const std::filesystem::path& root) {
    return std::make_unique<LtxNativeSession>(root);
}

std::string validate_ltx_ane_profile(
        const std::filesystem::path& manifest_path,
        uint32_t width, uint32_t height, uint32_t frames, uint32_t fps) {
    ltx_workload workload{};
    char error[1024] = {};
    require(ltx_workload_init(
                &workload, width, height, frames, fps, error, sizeof(error)),
            error);
    require(workload.stage1_video_tokens <= UINT32_MAX &&
                workload.stage2_video_tokens <= UINT32_MAX,
            "LTX request is too large for ANE row geometry");
    return resolve_ltx_ane_config(
        manifest_path,
        static_cast<uint32_t>(workload.stage1_video_tokens),
        static_cast<uint32_t>(workload.stage2_video_tokens), 1024u).identity;
}

NSDictionary* preflight_ltx_lora(const std::filesystem::path& root,
                                 const LoRAAsset& lora) {
    HashCache base_hash_cache;
    HashCache lora_hash_cache;
    HashCache output_hash_cache;
    auto default_checkpoint = root / "diffusion_models" /
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors";
    auto selection = resolve_ltx_checkpoint(
        root, default_checkpoint, lora, base_hash_cache, lora_hash_cache,
        output_hash_cache);
    return @{@"schema_version": @1,
             @"manifest": @(selection.manifest.c_str()),
             @"checkpoint": @(selection.checkpoint.c_str()),
             @"checkpoint_sha256": @(selection.checkpoint_sha256.c_str()),
             @"lora_sha256": @(selection.lora_sha256.c_str()),
             @"strength": @(selection.strength),
             @"validation": @"premerged_manifest_verified"};
}

NSDictionary* preflight_ltx_audio(const std::filesystem::path& root) {
    auto status = inspect_ltx_audio_assets(root);
    NSArray* blockers = status.assets_verified ? @[
        @"end_to_end_session_audio_parity",
        @"public_executor_validation",
    ] : @[
        @"provenance_verified_audio_weights",
        @"end_to_end_session_audio_parity",
        @"public_executor_validation",
    ];
    return @{
        @"schema_version": @1,
        @"model": @"ltx-2.5-distilled",
        @"status": @(status.status.c_str()),
        @"reason": @(status.reason.c_str()),
        @"assets_verified": @(status.assets_verified),
        @"native_runtime_available": @YES,
        @"native_audio_candidate": @(status.native_supported),
        @"native_audio_supported": @NO,
        @"executor_ready": @NO,
        @"artifact": status.artifact.empty() ?
            (id)[NSNull null] : @(status.artifact.c_str()),
        @"manifest": status.manifest.empty() ?
            (id)[NSNull null] : @(status.manifest.c_str()),
        @"artifact_sha256": status.artifact_sha256.empty() ?
            (id)[NSNull null] : @(status.artifact_sha256.c_str()),
        @"latent_contract": @{
            @"dtype": @"bfloat16",
            @"layout": @"BLC-token-major",
            @"channels": @128,
        },
        @"target_audio": @{
            @"sample_rate": @48000,
            @"channels": @2,
            @"codec": @"AAC",
        },
        @"blocking_components": blockers,
    };
}

}  // namespace tc
