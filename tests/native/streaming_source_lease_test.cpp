#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/streaming/resolved_request.hpp"

#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
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

    std::cout << "PASS source lease/value probe: single-fd capture, canonical ordering, "
                 "fd duplication, generation, same-size mutation, path/alias replace, "
                 "empty/digest rejection and snapshot revalidation\n";
}
