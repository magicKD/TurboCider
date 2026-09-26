#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/streaming/resolved_request.hpp"
#include "../../native/runtime/memory_manifest.hpp"

#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace {

using namespace tc::streaming;

std::string digest(char value) {
    return std::string(64, value);
}

void write_file(const std::filesystem::path &path, std::string_view value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create fixture");
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

bool rejects(std::function<void()> fn, std::string_view text) {
    try {
        fn();
    } catch (const std::exception &error) {
        return std::string(error.what()).find(text) != std::string::npos;
    }
    return false;
}

SourceFileIdentity source_file(std::string id,
                               const std::filesystem::path &path) {
    SourceFileIdentity value;
    value.logical_id = std::move(id);
    value.path = path;
    return value;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::filesystem::path root = argv[1];
    std::filesystem::create_directories(root);
    // Content identity must survive a copy while request binding changes.
    const auto content_path = root / "content.bin";
    const auto copy_path = root / "copy.bin";
    write_file(content_path, "abc");
    write_file(copy_path, "abc");
    const std::string abc_sha =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert(rejects([&] {
        SourceLease::capture_preverified({source_file("weights", content_path)});
    }, "artifact_verification_required"));
    bool query_requires_content = false;
    const auto legacy_query = SourceLease::capture_for_query(
        {source_file("weights", content_path)}, query_requires_content);
    assert(!query_requires_content && !legacy_query->has_verified_content());
    assert(legacy_query->verification_bytes_read() == 0);
    const auto verified = SourceLease::capture_verified(
        {source_file("weights", content_path)});
    assert(verified->has_verified_content());
    // Golden independently encoded with Python struct.pack('>Q', ...) and
    // hashlib: domain H, files L, logical_id S, bytes U, sha256 S.
    assert(verified->artifact_digest() ==
           "a9f38a05906edccaba47bc39980ac8c899ace84eb5d164381f0e7e6d623ec39b");
    assert(verified->file("weights").content_digest == abc_sha);
    assert(verified->verification_bytes_read() == 3);
    assert(verified->verification_cache_hits() == 0);
    const auto adopted = SourceLease::capture_for_query(
        {source_file("weights", content_path)}, query_requires_content);
    assert(query_requires_content && adopted->has_verified_content());
    assert(adopted->artifact_digest() == verified->artifact_digest());
    assert(adopted->verification_bytes_read() == 0);
    bool partial_requires_content = false;
    const auto partial = SourceLease::capture_for_query(
        {source_file("weights", content_path), source_file("zz_unverified", copy_path)},
        partial_requires_content);
    assert(!partial_requires_content && !partial->has_verified_content());
    const auto cached = SourceLease::capture_verified(
        {source_file("weights", content_path)});
    assert(cached->artifact_digest() == verified->artifact_digest());
    assert(cached->verification_bytes_read() == 0);
    assert(cached->verification_cache_hits() == 1);
    const auto ready = SourceLease::capture_preverified(
        {source_file("weights", content_path)});
    assert(ready->artifact_digest() == verified->artifact_digest());
    assert(ready->verification_bytes_read() == 0);
    const auto copied = SourceLease::capture_verified(
        {source_file("weights", copy_path)});
    assert(copied->artifact_digest() == verified->artifact_digest());
    assert(copied->digest() != verified->digest());
    assert(copied->verification_bytes_read() == 3);
    const auto renamed_role = SourceLease::capture_verified(
        {source_file("different-role", copy_path)});
    assert(renamed_role->artifact_digest() != copied->artifact_digest());
    const auto ordered = SourceLease::capture_verified(
        {source_file("b", content_path), source_file("a", copy_path)});
    const auto reordered = SourceLease::capture_verified(
        {source_file("a", content_path), source_file("b", copy_path)});
    assert(ordered->artifact_digest() == reordered->artifact_digest());
    const auto replay = SourceLease::open_and_verify(verified->descriptor());
    assert(!replay->has_verified_content());
    assert(rejects([&] { (void) replay->artifact_digest(); }, "not verified"));
    auto expected = source_file("weights", content_path);
    expected.content_digest = digest('0');
    bool mismatched_requires_content = false;
    assert(rejects([&] { SourceLease::capture_for_query({expected}, mismatched_requires_content); },
                   "content digest mismatch"));
    const auto untrusted = SourceLease::capture({expected});
    assert(!untrusted->has_verified_content());
    assert(rejects([&] { (void) untrusted->artifact_digest(); }, "not verified"));
    assert(rejects([&] { SourceLease::capture_verified({expected}); },
                   "content digest mismatch"));
    expected.content_digest = abc_sha;
    assert(SourceLease::capture_verified({expected})->has_verified_content());
    std::atomic<bool> cancelled{true};
    assert(rejects([&] { SourceLease::capture_verified({expected}, &cancelled); },
                   "cancelled"));
    {
        const auto fd = verified->duplicate_fd("weights");
        assert(::lseek(fd.get(), 2, SEEK_SET) == 2);
        assert(tc::memory_sha256_fd(fd.get(), 3) == abc_sha);
        assert(::lseek(fd.get(), 0, SEEK_CUR) == 2);
        assert(rejects([&] { tc::memory_sha256_fd(fd.get(), 4); }, "truncated"));
        assert(rejects([&] { tc::memory_sha256_fd(fd.get(), 3, &cancelled); },
                       "cancelled"));
    }
    // Same-size mutation with restored mtime must not hit the native cache.
    struct stat original{};
    assert(::stat(content_path.c_str(), &original) == 0);
    write_file(content_path, "abd");
#if defined(__APPLE__)
    const struct timespec original_times[] = {original.st_atimespec, original.st_mtimespec};
#else
    const struct timespec original_times[] = {original.st_atim, original.st_mtim};
#endif
    assert(::utimensat(AT_FDCWD, content_path.c_str(), original_times, 0) == 0);
    assert(rejects([&] {
        SourceLease::capture_preverified({source_file("weights", content_path)});
    }, "artifact_verification_required"));
    assert(rejects([&] {
        SourceLease::capture_for_query({source_file("weights", content_path)}, query_requires_content);
    }, "artifact_verification_required"));
    assert(query_requires_content);
    const auto mutated = SourceLease::capture_verified(
        {source_file("weights", content_path)});
    assert(mutated->artifact_digest() != verified->artifact_digest());
    assert(mutated->verification_bytes_read() == 3);
    assert(mutated->verification_cache_hits() == 0);
    assert(rejects([&] { verified->revalidate_after_drain(); }, "changed"));
    // Cross the bounded read-buffer boundary, checked against the independent
    // portable in-memory SHA implementation (Apple fd hashing is accelerated).
    const auto large_path = root / "large.bin";
    const std::string large(2 * 1024 * 1024 + 17, 'a');
    write_file(large_path, large);
    const auto large_lease = SourceLease::capture_verified(
        {source_file("large", large_path)});
    const auto large_sha = tc::memory_sha256_hex(large);
    assert(large_lease->file("large").content_digest == large_sha);
    // App worker stacks are small: the 1 MiB read buffer must stay on the heap.
    const auto large_fd = large_lease->duplicate_fd("large");
    struct HashWorker {
        int fd;
        uint64_t bytes;
        std::string digest;
        std::exception_ptr error;
    } worker{large_fd.get(), large.size(), {}, {}};
    pthread_attr_t attributes;
    assert(pthread_attr_init(&attributes) == 0);
    assert(pthread_attr_setstacksize(&attributes, 256 * 1024) == 0);
    pthread_t thread;
    assert(pthread_create(&thread, &attributes, [](void *raw) -> void * {
        auto &value = *static_cast<HashWorker *>(raw);
        try { value.digest = tc::memory_sha256_fd(value.fd, value.bytes); }
        catch (...) { value.error = std::current_exception(); }
        return nullptr;
    }, &worker) == 0);
    assert(pthread_attr_destroy(&attributes) == 0);
    assert(pthread_join(thread, nullptr) == 0);
    if (worker.error) std::rethrow_exception(worker.error);
    assert(worker.digest == large_sha);
    assert(rejects([&] { tc::memory_sha256_fd(-1, 3); }, "invalid"));
    const auto first_path = root / "first.safetensors";
    const auto second_path = root / "second.safetensors";
    write_file(first_path, "0123456789abcdef");
    write_file(second_path, "ABCDEFGHIJKLMNOP");

    std::vector<SourceFileIdentity> files{
        source_file("transformer.shard.1", second_path),
        source_file("transformer.shard.0", first_path),
    };
    auto lease = SourceLease::capture(std::move(files));
    const auto descriptor = lease->descriptor();
    assert(descriptor.files.size() == 2);
    assert(descriptor.files[0].logical_id == "transformer.shard.0");
    assert(descriptor.files[1].logical_id == "transformer.shard.1");
    assert(descriptor.source_snapshot_digest.size() == 64);
    assert(descriptor.files[0].bytes == 16);
    assert(descriptor.files[0].ctime_ns != 0);
    assert(descriptor.files[0].stat_identity_complete);
    assert(descriptor.files[0].path ==
           std::filesystem::absolute(first_path).lexically_normal());
    assert(descriptor.files[0].canonical_target_path ==
           std::filesystem::canonical(first_path));

    assert(lease && lease->file_count() == 2);
    assert(lease->generation() != 0);
    assert(lease->digest() == descriptor.source_snapshot_digest);
    lease->revalidate_paths();
    lease->revalidate_open_files();
    {
        auto fd = lease->duplicate_fd("transformer.shard.0");
        char data[17]{};
        assert(::pread(fd.get(), data, 16, 0) == 16);
        assert(std::string(data, 16) == "0123456789abcdef");
    }

    const auto original_mtime = descriptor.files[0].mtime_ns;
    write_file(first_path, "fedcba9876543210");
    struct timespec times[2]{};
    times[0].tv_sec = original_mtime / 1'000'000'000ll;
    times[0].tv_nsec = original_mtime % 1'000'000'000ll;
    times[1] = times[0];
    (void)::utimensat(AT_FDCWD, first_path.c_str(), times, 0);
    assert(rejects([&] { lease->revalidate_paths(); }, "source path"));
    assert(rejects([&] { lease->revalidate_open_files(); }, "source fd"));

    write_file(first_path, "0123456789abcdef");
    auto replace_lease = SourceLease::capture(
        std::vector<SourceFileIdentity>{
            source_file("transformer.shard.0", first_path),
            source_file("transformer.shard.1", second_path)});
    auto replaced = root / "replacement.safetensors";
    write_file(replaced, "xxxxxxxxxxxxxxxx");
    assert(::rename(replaced.c_str(), first_path.c_str()) == 0);
    assert(rejects([&] { replace_lease->revalidate_paths(); }, "source path"));
    assert(rejects([&] { replace_lease->revalidate_open_files(); }, "source fd"));
    {
        auto fd = replace_lease->duplicate_fd("transformer.shard.0");
        char data[17]{};
        assert(::pread(fd.get(), data, 16, 0) == 16);
        assert(std::string(data, 16) == "0123456789abcdef");
    }

    auto second_descriptor = capture_source_lease_descriptor(
        std::vector<SourceFileIdentity>{
            source_file("transformer.shard.0", first_path),
            source_file("transformer.shard.1", second_path)});
    auto second_lease = SourceLease::open_and_verify(second_descriptor);
    assert(second_lease->generation() > replace_lease->generation());
    assert(rejects([&] {
        auto altered = second_descriptor;
        altered.source_snapshot_digest = digest('0');
        (void)SourceLease::open_and_verify(altered);
    }, "digest mismatch"));

    assert(rejects([&] {
        (void)capture_source_lease_descriptor(
            std::vector<SourceFileIdentity>{
                source_file("duplicate", second_path),
                source_file("duplicate", second_path)});
    }, "logical ids"));
    assert(rejects([&] { (void)lease->file("missing"); }, "unknown logical id"));

    const auto empty_path = root / "empty.safetensors";
    write_file(empty_path, "");
    assert(rejects([&] {
        (void)SourceLease::capture(
            std::vector<SourceFileIdentity>{
                source_file("empty", empty_path)});
    }, "source is empty"));

    // Keep the manifest's named symlink and its canonical target separately.
    // Repointing the alias to another same-size file must fail even though the
    // original request fd remains readable.
    const auto target_a = root / "target-a.safetensors";
    const auto target_b = root / "target-b.safetensors";
    const auto alias = root / "model-current.safetensors";
    write_file(target_a, "aaaaaaaaaaaaaaaa");
    write_file(target_b, "bbbbbbbbbbbbbbbb");
    assert(::symlink(target_a.filename().c_str(), alias.c_str()) == 0);
    auto alias_lease = SourceLease::capture(
        std::vector<SourceFileIdentity>{source_file("alias", alias)});
    assert(alias_lease->descriptor().files.front().path ==
           std::filesystem::absolute(alias).lexically_normal());
    assert(alias_lease->descriptor().files.front().canonical_target_path ==
           std::filesystem::canonical(target_a));
    assert(::unlink(alias.c_str()) == 0);
    assert(::symlink(target_b.filename().c_str(), alias.c_str()) == 0);
    assert(rejects([&] { alias_lease->revalidate_paths(); },
                   "canonical target"));
    alias_lease->revalidate_open_files();
    {
        auto fd = alias_lease->duplicate_fd("alias");
        char data[17]{};
        assert(::pread(fd.get(), data, 16, 0) == 16);
        assert(std::string(data, 16) == "aaaaaaaaaaaaaaaa");
    }

    // Exercise the shared value probe/snapshot implementation with the same
    // lease.  This verifies that model adapters can reuse the common source
    // and lifecycle contract without allocating GPU state.
    PresetSourceIdentity source{
        "test-bf16", "safetensors-bf16", digest('a'),
        std::string(second_lease->digest())};
    PresetRuntimeIdentity runtime{
        "test-build", "runtime-v1", "adapter-v1", "reader-v1",
        "kernel-v1", "allocator-v1"};
    PresetWorkload workload;
    workload.model = "test-model";
    workload.operation = "image.generate";
    workload.execution = "gpu";
    workload.device_class = "test-gpu";
    workload.execution_container = "embedded_app";
    workload.width = 16;
    workload.height = 16;
    workload.steps = 1;
    workload.feature_digest = digest('b');
    workload.conditioning_revision = "conditioning-v1";
    workload.vae_policy_revision = "vae-v1";
    auto probe = std::make_shared<ValueModelStreamingProbe>(
        ValueModelStreamingProbe::Values{
            "test-model", source, workload, runtime, "components-v1", second_lease});
    assert(probe->source_lease() == second_lease.get());
    Descriptor model_descriptor;
    model_descriptor.model = "test-model";
    Layout layout;
    layout.digest = digest('c');
    layout.materializations_complete = true;
    auto snapshot = std::make_shared<ValueModelStreamingSnapshot>(
        ValueModelStreamingSnapshot::Values{
            "test-model", source, runtime, std::move(model_descriptor),
            std::move(layout), "components-v1", second_lease});
    snapshot->revalidate_source();
    assert(snapshot->source_lease() == second_lease.get());

    auto proof = SourceLease::capture_verified(second_lease->descriptor().files);
    auto portable = source;
    portable.identity_version = 2;
    portable.source_snapshot_digest.clear();
    portable.artifact_manifest_digest = std::string(proof->artifact_digest());
    ValueModelStreamingProbe verified_probe({
        "test-model", portable, workload, runtime, "components-v1", proof});
    assert(verified_probe.source_lease() == proof.get());
    assert(rejects([&] {
        ValueModelStreamingProbe unverified({
            "test-model", portable, workload, runtime, "components-v1", second_lease});
    }, "source lease digest differs"));
    portable.source_snapshot_digest = std::string(proof->digest());
    assert(rejects([&] {
        ValueModelStreamingProbe mixed({
            "test-model", portable, workload, runtime, "components-v1", proof});
    }, "source lease digest differs"));

    std::cout << "PASS source lease/value probe: single-fd capture, canonical ordering, "
                 "verified content, copy identity, cache invalidation, cancellation, "
                 "fd duplication, generation, same-size mutation, path/alias replace, "
                 "empty/digest rejection and snapshot revalidation\n";
}
