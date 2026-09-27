#include "memory_probe.hpp"

#include "../../core/common.hpp"
#include "../../runtime/memory_execution.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tc {
namespace {

constexpr uint64_t kMaximumProbeSidecarBytes = 16ull << 20;
constexpr NSUInteger kMaximumCheckpointFiles = 4096;
constexpr NSUInteger kMaximumManifestSites = 16384;
constexpr NSUInteger kMaximumManifestInstances = 131072;
constexpr NSUInteger kMaximumProbeStringBytes = 4096;

NSDictionary *dictionary(id value, const char *context) {
    require([value isKindOfClass:NSDictionary.class],
            std::string("memory_probe_invalid: ") + context +
                " must be an object");
    return value;
}

NSArray *array(id value, const char *context) {
    require([value isKindOfClass:NSArray.class],
            std::string("memory_probe_invalid: ") + context +
                " must be an array");
    return value;
}

std::string string_value(NSDictionary *object, NSString *key,
                         const char *context) {
    id value = object[key];
    require([value isKindOfClass:NSString.class],
            std::string("memory_probe_invalid: ") + context + "." +
                key.UTF8String + " must be a string");
    const char *text = [value UTF8String];
    const auto bytes = [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    require(text && bytes <= kMaximumProbeStringBytes &&
                        std::strlen(text) == bytes,
            std::string("memory_probe_invalid: invalid UTF-8 in ") + context +
                "." + key.UTF8String);
    return text;
}

uint64_t uint64_value(NSDictionary *object, NSString *key,
                      const char *context, bool optional = false,
                      uint64_t fallback = 0) {
    id value = object[key];
    if (!value && optional) return fallback;
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                !CFNumberIsFloatType((__bridge CFNumberRef)value),
            std::string("memory_probe_invalid: ") + context + "." +
                key.UTF8String + " must be an integer");
    const double number = [value doubleValue];
    require(std::isfinite(number) && number >= 0.0 &&
                number == std::floor(number) &&
                number <= static_cast<double>(1ull << 53),
            std::string("memory_probe_invalid: invalid integer in ") +
                context + "." + key.UTF8String);
    return [value unsignedLongLongValue];
}

bool bool_value(NSDictionary *object, NSString *key, const char *context,
                bool fallback) {
    id value = object[key];
    if (!value) return fallback;
    require(CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID(),
            std::string("memory_probe_invalid: ") + context + "." +
                key.UTF8String + " must be bool");
    return [value boolValue];
}

MemoryClass memory_class(std::string_view value) {
    if (value == "process_baseline") return MemoryClass::ProcessBaseline;
    if (value == "weights") return MemoryClass::Weights;
    if (value == "activation") return MemoryClass::Activation;
    if (value == "conditioning") return MemoryClass::Conditioning;
    if (value == "refill_slot") return MemoryClass::RefillSlot;
    if (value == "conversion_scratch") return MemoryClass::ConversionScratch;
    if (value == "output") return MemoryClass::Output;
    if (value == "allocator_cache") return MemoryClass::AllocatorCache;
    if (value == "compile_temporary") return MemoryClass::CompileTemporary;
    if (value == "child_process_envelope")
        return MemoryClass::ChildProcessEnvelope;
    if (value == "unknown_external") return MemoryClass::UnknownExternal;
    require(false, "memory_probe_invalid: unknown memory class " +
                       std::string(value));
    return MemoryClass::UnknownExternal;
}

AllocationLifetime lifetime(std::string_view value) {
    if (value == "request") return AllocationLifetime::Request;
    if (value == "stage") return AllocationLifetime::Stage;
    if (value == "block") return AllocationLifetime::Block;
    if (value == "tile") return AllocationLifetime::Tile;
    if (value == "command_buffer") return AllocationLifetime::CommandBuffer;
    if (value == "export") return AllocationLifetime::Export;
    require(false, "memory_probe_invalid: unknown allocation lifetime " +
                       std::string(value));
    return AllocationLifetime::Request;
}

UpperProvenance provenance(std::string_view value) {
    if (value == "exact_metadata") return UpperProvenance::ExactMetadata;
    if (value == "exact_shape_formula")
        return UpperProvenance::ExactShapeFormula;
    if (value == "validated_envelope")
        return UpperProvenance::ValidatedEnvelope;
    if (value == "heuristic_not_executable")
        return UpperProvenance::HeuristicNotExecutable;
    require(false, "memory_probe_invalid: unknown upper provenance " +
                       std::string(value));
    return UpperProvenance::HeuristicNotExecutable;
}

bool path_has_prefix(const std::filesystem::path &path,
                     const std::filesystem::path &root) {
    auto path_it = path.begin();
    for (auto root_it = root.begin(); root_it != root.end();
         ++root_it, ++path_it) {
        if (path_it == path.end() || *path_it != *root_it) return false;
    }
    return true;
}

std::filesystem::path resolve_probe_sidecar(
        const std::filesystem::path &model_root,
        const std::filesystem::path &sidecar) {
    std::error_code error;
    const auto absolute_root = std::filesystem::absolute(model_root, error);
    require(!error,
            "memory_probe_invalid: cannot resolve model root");
    const auto canonical_root = std::filesystem::weakly_canonical(
        absolute_root, error);
    require(!error && std::filesystem::is_directory(canonical_root, error) &&
                !error,
            "memory_probe_invalid: model root is unavailable");
    const auto absolute_sidecar = std::filesystem::absolute(sidecar, error);
    require(!error,
            "memory_probe_invalid: cannot resolve capability sidecar");
    const auto canonical_sidecar = std::filesystem::weakly_canonical(
        absolute_sidecar, error);
    require(!error && path_has_prefix(canonical_sidecar, canonical_root),
            "memory_probe_invalid: capability sidecar escapes model root");
    return canonical_sidecar;
}

std::filesystem::path resolve_checkpoint_file(
        const std::filesystem::path &model_root,
        const std::string &logical_name) {
    const std::filesystem::path logical(logical_name);
    require(!logical.empty() && !logical.is_absolute(),
            "memory_checkpoint_invalid: checkpoint logical path is absolute");
    for (const auto &component : logical)
        require(component != ".." && component != ".",
                "memory_checkpoint_invalid: checkpoint logical path escapes root");
    std::error_code error;
    const auto canonical_root = std::filesystem::weakly_canonical(
        std::filesystem::absolute(model_root, error), error);
    require(!error,
            "memory_checkpoint_invalid: cannot canonicalize model root");
    const auto candidate = std::filesystem::weakly_canonical(
        canonical_root / logical, error);
    require(!error && path_has_prefix(candidate, canonical_root),
            "memory_checkpoint_invalid: checkpoint file escapes model root");
    require(std::filesystem::is_regular_file(candidate, error) && !error,
            "memory_checkpoint_invalid: checkpoint file is missing " +
                logical_name);
    return candidate;
}

std::string shape_bucket(const Request &request) {
    std::ostringstream value;
    value << 'w' << request.width << "_h" << request.height
          << "_f" << request.frames << "_s" << request.steps
          << "_audio" << (request.audio ? 1 : 0)
          << "_input" << request.inputs.size()
          << "_lora" << request.loras.size();
    return value.str();
}

bool same_snapshot(const struct stat &left, const struct stat &right) {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
        left.st_size == right.st_size &&
        left.st_ctimespec.tv_sec == right.st_ctimespec.tv_sec &&
        left.st_ctimespec.tv_nsec == right.st_ctimespec.tv_nsec &&
        left.st_mtimespec.tv_sec == right.st_mtimespec.tv_sec &&
        left.st_mtimespec.tv_nsec == right.st_mtimespec.tv_nsec;
}

struct FileDescriptorGuard {
    int value = -1;
    ~FileDescriptorGuard() { if (value >= 0) close(value); }
};

NSData *read_probe_sidecar(const std::filesystem::path &path) {
    const auto text = path.string();
    const int descriptor = open(text.c_str(), O_RDONLY | O_CLOEXEC);
    require(descriptor >= 0,
            "memory_probe_invalid: capability sidecar cannot be opened");
    FileDescriptorGuard guard{descriptor};
    struct stat before{};
    require(fstat(descriptor, &before) == 0 && S_ISREG(before.st_mode) &&
                before.st_size > 0 &&
                static_cast<uint64_t>(before.st_size) <=
                    kMaximumProbeSidecarBytes,
            "memory_probe_invalid: capability sidecar size is invalid");
    NSMutableData *data = [NSMutableData dataWithLength:
        static_cast<NSUInteger>(before.st_size)];
    require(data != nil,
            "memory_probe_invalid: cannot allocate capability sidecar buffer");
    size_t offset = 0;
    while (offset < data.length) {
        const ssize_t count = read(
            descriptor, static_cast<unsigned char *>(data.mutableBytes) + offset,
            data.length - offset);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0,
                "memory_probe_invalid: capability sidecar read failed");
        offset += static_cast<size_t>(count);
    }
    struct stat after{};
    require(fstat(descriptor, &after) == 0 && same_snapshot(before, after),
            "memory_probe_invalid: capability sidecar changed while reading");
    return data;
}

std::string read_sha256(int descriptor) {
    CC_SHA256_CTX context;
    require(CC_SHA256_Init(&context) == 1,
            "memory_checkpoint_invalid: cannot initialize SHA-256");
    std::array<char, 64 * 1024> buffer{};
    for (;;) {
        const ssize_t count = read(descriptor, buffer.data(), buffer.size());
        if (count == 0) break;
        if (count < 0 && errno == EINTR) continue;
        require(count > 0,
                "memory_checkpoint_invalid: checkpoint read failed");
        require(CC_SHA256_Update(
                    &context, buffer.data(), static_cast<CC_LONG>(count)) == 1,
                "memory_checkpoint_invalid: cannot update SHA-256");
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    require(CC_SHA256_Final(digest, &context) == 1,
            "memory_checkpoint_invalid: cannot finalize SHA-256");
    char output[CC_SHA256_DIGEST_LENGTH * 2 + 1] = {};
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index)
        std::snprintf(output + index * 2, 3, "%02x", digest[index]);
    return output;
}

} // namespace

std::string MemoryCheckpointHashCache::sha256(
        const std::filesystem::path &input,
        MemoryModelRootTrust trust) const {
    std::error_code error;
    const auto path = std::filesystem::canonical(input, error);
    require(!error && std::filesystem::is_regular_file(path),
            "memory_checkpoint_invalid: checkpoint file is unavailable");
    const auto path_text = path.string();
    const int descriptor = open(path_text.c_str(), O_RDONLY | O_CLOEXEC);
    require(descriptor >= 0,
            "memory_checkpoint_invalid: checkpoint file cannot be opened");
    FileDescriptorGuard descriptor_guard{descriptor};
    struct stat before{};
    require(fstat(descriptor, &before) == 0 && S_ISREG(before.st_mode),
            "memory_checkpoint_invalid: checkpoint file metadata is unavailable");
    require(before.st_size > 0,
            "memory_checkpoint_invalid: checkpoint file is empty");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = entries_.find(path);
        if (trust == MemoryModelRootTrust::OwnedImmutable &&
            found != entries_.end() &&
            found->second.device == static_cast<uint64_t>(before.st_dev) &&
            found->second.inode == static_cast<uint64_t>(before.st_ino) &&
            found->second.size_bytes ==
                static_cast<uintmax_t>(before.st_size) &&
            found->second.change_time_seconds == before.st_ctimespec.tv_sec &&
            found->second.change_time_nanoseconds ==
                before.st_ctimespec.tv_nsec &&
            found->second.modify_time_seconds == before.st_mtimespec.tv_sec &&
            found->second.modify_time_nanoseconds ==
                before.st_mtimespec.tv_nsec)
            return found->second.sha256;
    }
    const auto digest = read_sha256(descriptor);
    struct stat after{};
    require(fstat(descriptor, &after) == 0 && same_snapshot(before, after),
            "memory_checkpoint_invalid: checkpoint changed while hashing");
    std::lock_guard<std::mutex> lock(mutex_);
    Entry entry;
    entry.device = static_cast<uint64_t>(after.st_dev);
    entry.inode = static_cast<uint64_t>(after.st_ino);
    entry.size_bytes = static_cast<uintmax_t>(after.st_size);
    entry.change_time_seconds = after.st_ctimespec.tv_sec;
    entry.change_time_nanoseconds = after.st_ctimespec.tv_nsec;
    entry.modify_time_seconds = after.st_mtimespec.tv_sec;
    entry.modify_time_nanoseconds = after.st_mtimespec.tv_nsec;
    entry.sha256 = digest;
    entries_[path] = std::move(entry);
    return digest;
}

std::filesystem::path memory_capability_probe_path(
        const std::filesystem::path &model_root, std::string_view adapter,
        const Request &request, unsigned refill_slots) {
    require(!adapter.empty() && std::all_of(
                adapter.begin(), adapter.end(), [](unsigned char character) {
                    return std::isalnum(character) || character == '_' ||
                           character == '-';
                }),
            "memory_probe_invalid: unsafe adapter id");
    std::ostringstream name;
    name << shape_bucket(request) << "_slots" << refill_slots << ".json";
    return model_root / "memory-capabilities" / std::string(adapter) /
           name.str();
}

MemoryCandidateKey expected_memory_candidate_key(
        const ExecutionPlan &plan, const MemoryDeviceIdentity &device,
        std::string checkpoint_digest) {
    require(plan.memory_policy && plan.memory_policy->enabled,
            "memory_policy_invalid: expected capability key requires policy");
    const auto &policy = *plan.memory_policy;
    require(!policy.adapter_candidate.empty() &&
                !policy.candidate_backend.empty() &&
                !policy.candidate_dtype.empty() &&
                !policy.candidate_model_variant.empty() &&
                !policy.candidate_sampler_mode.empty() &&
                !policy.candidate_tiling_mode.empty(),
            "memory_policy_unsupported: candidate identity is incomplete");
    require(!device.device_family.empty() && !device.runtime_revision.empty(),
            "memory_policy_unsupported: device identity is incomplete");
    return {
        policy.adapter_candidate,
        plan.request.model,
        std::move(checkpoint_digest),
        policy.candidate_backend,
        policy.candidate_dtype,
        policy.candidate_model_variant,
        plan.request.operation,
        shape_bucket(plan.request),
        policy.candidate_sampler_mode,
        policy.refill_slots,
        policy.candidate_tiling_mode,
        device.runtime_revision,
        device.device_family,
    };
}

std::optional<MemoryCapabilityProbe> load_memory_capability_probe(
        const std::filesystem::path &model_root,
        const std::filesystem::path &sidecar, const ExecutionPlan &plan,
        const MemoryDeviceIdentity &device,
        MemoryModelRootTrust trust,
        const MemoryCheckpointHashCache &hash_cache) {
    const auto resolved_sidecar = resolve_probe_sidecar(model_root, sidecar);
    std::error_code file_error;
    const bool sidecar_exists = std::filesystem::exists(
        resolved_sidecar, file_error);
    require(!file_error,
            "memory_probe_invalid: cannot inspect capability sidecar");
    if (!sidecar_exists)
        return std::nullopt;
    require(std::filesystem::is_regular_file(resolved_sidecar, file_error) &&
                !file_error,
            "memory_probe_invalid: capability sidecar is not a regular file");
    NSData *data = read_probe_sidecar(resolved_sidecar);
    NSError *json_error = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&json_error];
    require(parsed != nil && json_error == nil,
            "memory_probe_invalid: malformed capability sidecar");
    NSDictionary *root = dictionary(parsed, "root");
    require(string_value(root, @"schema", "root") ==
                "turbocider.memory_probe.v1",
            "memory_probe_invalid: unsupported probe schema");

    NSDictionary *key_object = dictionary(root[@"key"], "key");
    MemoryCandidateKey key;
    key.adapter = string_value(key_object, @"adapter", "key");
    key.model_id = string_value(key_object, @"model_id", "key");
    key.checkpoint_digest =
        string_value(key_object, @"checkpoint_digest", "key");
    key.backend = string_value(key_object, @"backend", "key");
    key.dtype = string_value(key_object, @"dtype", "key");
    key.model_variant =
        string_value(key_object, @"model_variant", "key");
    key.operation = string_value(key_object, @"operation", "key");
    key.shape_bucket = string_value(key_object, @"shape_bucket", "key");
    key.sampler_mode = string_value(key_object, @"sampler_mode", "key");
    const auto refill_slots = uint64_value(
        key_object, @"refill_slots", "key");
    require(refill_slots >= 1 && refill_slots <= 3,
            "memory_probe_invalid: key.refill_slots must be 1...3");
    key.refill_slots = static_cast<unsigned>(refill_slots);
    key.tiling_mode = string_value(key_object, @"tiling_mode", "key");
    key.runtime_revision =
        string_value(key_object, @"runtime_revision", "key");
    key.device_family =
        string_value(key_object, @"device_family", "key");

    require(plan.memory_policy && plan.memory_policy->enabled,
            "memory_policy_invalid: probe requires enabled policy");
    const auto expected_without_checkpoint = expected_memory_candidate_key(
        plan, device);
    require(key.adapter == expected_without_checkpoint.adapter &&
                key.model_id == expected_without_checkpoint.model_id &&
                key.backend == expected_without_checkpoint.backend &&
                key.dtype == expected_without_checkpoint.dtype &&
                key.model_variant == expected_without_checkpoint.model_variant &&
                key.operation == expected_without_checkpoint.operation &&
                key.shape_bucket == expected_without_checkpoint.shape_bucket &&
                key.sampler_mode == expected_without_checkpoint.sampler_mode &&
                key.refill_slots == expected_without_checkpoint.refill_slots &&
                key.tiling_mode == expected_without_checkpoint.tiling_mode &&
                key.runtime_revision == expected_without_checkpoint.runtime_revision &&
                key.device_family == expected_without_checkpoint.device_family,
            "memory_policy_unsupported: capability sidecar identity mismatch");

    NSArray *file_array = array(root[@"checkpoint_files"],
                                "checkpoint_files");
    require(file_array.count > 0 &&
                file_array.count <= kMaximumCheckpointFiles,
            "memory_checkpoint_invalid: checkpoint file list is empty");
    std::vector<MemoryCheckpointFileIdentity> identities;
    identities.reserve(file_array.count);
    for (id item in file_array) {
        NSDictionary *entry = dictionary(item, "checkpoint_file");
        MemoryCheckpointFileIdentity identity;
        identity.logical_name =
            string_value(entry, @"logical_name", "checkpoint_file");
        identity.size_bytes =
            uint64_value(entry, @"size_bytes", "checkpoint_file");
        identity.sha256 =
            string_value(entry, @"sha256", "checkpoint_file");
        const auto actual_path = resolve_checkpoint_file(
            model_root, identity.logical_name);
        const auto actual_size = std::filesystem::file_size(
            actual_path, file_error);
        require(!file_error && actual_size == identity.size_bytes,
                "memory_checkpoint_invalid: checkpoint size mismatch " +
                    identity.logical_name);
        require(hash_cache.sha256(actual_path, trust) ==
                    identity.sha256,
                "memory_checkpoint_invalid: checkpoint digest mismatch " +
                    identity.logical_name);
        identities.push_back(std::move(identity));
    }
    const auto checkpoint_digest =
        memory_checkpoint_identity_digest(identities);
    require(checkpoint_digest == key.checkpoint_digest,
            "memory_checkpoint_invalid: canonical checkpoint digest mismatch");
    auto expected = expected_memory_candidate_key(
        plan, device, checkpoint_digest);
    require(key == expected,
            "memory_policy_unsupported: complete capability key mismatch");

    NSDictionary *manifest_object = dictionary(root[@"manifest"], "manifest");
    require(string_value(manifest_object, @"schema", "manifest") ==
                "turbocider.memory_manifest.v1",
            "memory_manifest_invalid: unsupported sidecar manifest schema");
    const auto candidate_id =
        string_value(manifest_object, @"candidate_id", "manifest");
    const auto manifest_checkpoint = string_value(
        manifest_object, @"checkpoint_digest", "manifest");
    const auto backend_revision = string_value(
        manifest_object, @"backend_revision", "manifest");
    const auto runtime_revision = string_value(
        manifest_object, @"runtime_revision", "manifest");
    require(candidate_id == key.adapter &&
                manifest_checkpoint == checkpoint_digest &&
                runtime_revision == key.runtime_revision,
            "memory_policy_unsupported: manifest identity mismatch");
    MemoryManifestBuilder builder(
        candidate_id, manifest_checkpoint, backend_revision,
        runtime_revision);
    NSArray *sites = array(manifest_object[@"sites"], "manifest.sites");
    require(sites.count > 0 && sites.count <= kMaximumManifestSites,
            "memory_manifest_invalid: invalid allocation site count");
    for (id item in sites) {
        NSDictionary *entry = dictionary(item, "manifest.site");
        AllocationSiteSpec site;
        site.site_id = string_value(entry, @"site_id", "manifest.site");
        site.component = string_value(entry, @"component", "manifest.site");
        site.stage = string_value(entry, @"stage", "manifest.site");
        site.memory_class = memory_class(
            string_value(entry, @"memory_class", "manifest.site"));
        site.lifetime = lifetime(
            string_value(entry, @"lifetime", "manifest.site"));
        site.provenance = provenance(
            string_value(entry, @"provenance", "manifest.site"));
        site.required = bool_value(entry, @"required", "manifest.site", true);
        site.asynchronous = bool_value(
            entry, @"asynchronous", "manifest.site", false);
        site.aliasable = bool_value(
            entry, @"aliasable", "manifest.site", false);
        site.guard_threshold_bytes = uint64_value(
            entry, @"guard_threshold_bytes", "manifest.site", true, 0);
        builder.add_site(std::move(site));
    }
    NSArray *instances = array(
        manifest_object[@"instances"], "manifest.instances");
    require(instances.count > 0 &&
                instances.count <= kMaximumManifestInstances,
            "memory_manifest_invalid: invalid allocation instance count");
    for (id item in instances) {
        NSDictionary *entry = dictionary(item, "manifest.instance");
        AllocationInstance instance;
        instance.site_id =
            string_value(entry, @"site_id", "manifest.instance");
        const auto instance_id = uint64_value(
            entry, @"instance_id", "manifest.instance");
        require(instance_id <= std::numeric_limits<uint32_t>::max(),
                "memory_manifest_invalid: allocation instance id overflow");
        instance.instance_id = static_cast<uint32_t>(instance_id);
        instance.upper_bytes = uint64_value(
            entry, @"upper_bytes", "manifest.instance");
        instance.live_begin = uint64_value(
            entry, @"live_begin", "manifest.instance");
        instance.live_end = uint64_value(
            entry, @"live_end", "manifest.instance");
        instance.alias_group = uint64_value(
            entry, @"alias_group", "manifest.instance", true, 0);
        instance.last_use_event = string_value(
            entry, @"last_use_event", "manifest.instance");
        builder.add_instance(std::move(instance));
    }
    return MemoryCapabilityProbe{std::move(key), builder.build()};
}

} // namespace tc
