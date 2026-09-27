// Host-only exact-entry tests. Inputs are disposable synthetic checkpoints;
// successful validation is cancelled before the first GPU construction.
#include "ltx_streaming_descriptor.hpp"
extern "C" {
#include "ltx_native.h"
}
#include <array>
#include <cassert>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

static int cancel_description(const char *phase, int, int, void *user) {
    auto &calls = *static_cast<unsigned *>(user);
    ++calls;
    assert(std::strcmp(phase, "ltx_describe_block") == 0);
    return 1;
}
static int cancel_after_validation(const char *phase, int, int, void *user) {
    auto &calls = *static_cast<unsigned *>(user);
    ++calls;
    if (std::strcmp(phase, "ltx_describe_block") == 0) return 0;
    assert(std::strcmp(phase, "ltx_streaming_validated") == 0);
    return 1;
}
static size_t fd_count() {
    DIR *directory = opendir("/dev/fd"); assert(directory);
    size_t count = 0;
    while (auto *entry = readdir(directory))
        if (entry->d_name[0] != '.') ++count;
    closedir(directory);
    return count;
}
static void is_stale(const tc::ltx::StreamingMetadata &metadata) {
    try { metadata.check_unchanged(); }
    catch (const std::invalid_argument &e) {
        assert(std::strstr(e.what(), "checkpoint_changed")); return;
    }
    assert(false);
}

struct Fixture {
    const char *path;
    const tc::ltx::StreamingMetadata &metadata;
    std::array<uint32_t, 47> blocks{};
    std::array<tc_stream_group_v1, 47> groups{};
    std::array<uint64_t, 2> capacities{};
    ltx_native_options options{};
    tc_stream_stage_plan_v1 plan{};
    ltx_native_streaming_options_v1 base{};
    ltx_native_streaming_options_v2 exact{};
    ltx_native_streaming_options_v3 stage{};
    unsigned calls = 0;
    char error[1024]{};

    Fixture(const char *p, const tc::ltx::StreamingMetadata &m) : path(p), metadata(m) { reset(); }
    void reset() {
        options = {}; options.checkpoint = path; options.fps = 24;
        options.width = 64; options.height = 64; options.frames = 9;
        options.shader_source = "must-not-reach-GPU-in-host-test.metal";
        const auto &b = metadata.block(0);
        capacities.fill(b.gpu_bytes + b.cpu_bytes);
        for (uint32_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = i + 1;
            groups[i] = {i, i % 2, 1, &blocks[i], capacities[0]};
        }
        plan = {sizeof(plan), TC_STREAM_SLOT_ABI_V1, 0, 0, 2, 1, 2, 11, 1,
                capacities.data(), uint32_t(groups.size()), groups.data()};
        base = {sizeof(base), 1, 1, &plan};
        exact = {sizeof(exact), 2, base, &metadata.header(), &metadata.mapping()};
        stage = {sizeof(stage), 3, base, &metadata.header(),
                 &metadata.mapping(), 0, 0};
    }
    void reject(unsigned version, const char *message, unsigned expected_calls = 0) {
        calls = 0; error[0] = 0;
        auto *ctx = reinterpret_cast<ltx_native_denoiser *>(uintptr_t(1));
        exact.base = base;
        const int ok = version == 1
            ? ltx_native_create_streamed_v1(&options, &base, &ctx, cancel_description, &calls, error, sizeof(error))
            : ltx_native_create_streamed_v2(&options, &exact, &ctx, cancel_description, &calls, error, sizeof(error));
        if (ok || ctx || calls != expected_calls || !std::strstr(error, message)) {
            std::cerr << "v" << version << " expected=" << message << " actual=" << error
                      << " callbacks=" << calls << '\n';
            assert(false);
        }
        assert(fcntl(metadata.mapping().descriptor, F_GETFD) >= 0);
    }
    void reject_v3(const char *message, unsigned expected_calls = 0) {
        calls = 0; error[0] = 0;
        auto *ctx = reinterpret_cast<ltx_native_denoiser *>(uintptr_t(1));
        stage.base = base;
        const int ok = ltx_native_create_streamed_v3(
            &options, &stage, &ctx, cancel_description, &calls,
            error, sizeof(error));
        if (ok || ctx || calls != expected_calls ||
            !std::strstr(error, message)) {
            std::cerr << "v3 expected=" << message << " actual=" << error
                      << " callbacks=" << calls << '\n';
            assert(false);
        }
        assert(fcntl(metadata.mapping().descriptor, F_GETFD) >= 0);
    }
    template<class Mutate> void invalid_both(Mutate mutate, const char *message = "invalid") {
        reset(); mutate(); reject(1, message); reject(2, message); reset();
    }
};

int main(int argc, char **argv) {
    assert(argc == 3); // Runner owns both files; never accept real checkpoints.
    try {
        tc::ltx::StreamingMetadata metadata(argv[1]);
        Fixture f(argv[1], metadata);
        f.invalid_both([&] { f.base.struct_size--; });
        f.invalid_both([&] { f.base.version++; });
        f.invalid_both([&] { f.base.plan = nullptr; });
        f.invalid_both([&] { f.plan.struct_size--; });
        f.invalid_both([&] { f.plan.version++; });
        f.invalid_both([&] { f.plan.request_generation = 0; });
        f.invalid_both([&] { f.plan.slot_count = 0; });
        f.invalid_both([&] { f.plan.slot_count = 4; });
        f.invalid_both([&] { f.plan.groups = nullptr; });
        f.invalid_both([&] { f.plan.slot_capacity_bytes = nullptr; });
        f.invalid_both([&] { f.plan.io_workers = 0; });
        f.invalid_both([&] { f.plan.io_workers = 3; });
        f.invalid_both([&] { f.plan.prefetch_distance = 2; });
        f.invalid_both([&] { f.base.resident_prefix_blocks = 0; });
        f.invalid_both([&] { f.base.resident_prefix_blocks = 48; });
        f.invalid_both([&] { f.plan.group_count = UINT32_MAX; });
        f.invalid_both([&] { f.plan.pass_count = 1; }, "wrong pass count");
        f.invalid_both([&] { f.groups[0].blocks = nullptr; });
        f.invalid_both([&] { f.groups[0].block_count = 2; });
        f.invalid_both([&] { f.groups[0].slot = UINT32_MAX; });
        f.invalid_both([&] { f.groups[0].group = 1; });
        f.invalid_both([&] { f.blocks[0] = 48; });
        f.invalid_both([&] { f.groups[0].content_bytes = 0; });
        f.invalid_both([&] { f.groups[0].content_bytes = f.capacities[0] + 1; });
        f.invalid_both([&] { f.capacities[0] = 0; });
        f.invalid_both([&] { f.capacities.fill(UINT64_MAX); });
        f.invalid_both([&] { f.options.memory_budget_bytes = 1; });
        f.invalid_both([&] { f.options.max_refill_slots = 1; });
        f.invalid_both([&] { f.options.sol_stage1 = 1; });
        f.invalid_both([&] { f.options.mlp_directories[0] = "unused"; });
        f.invalid_both([&] { f.options.release_blocks_final_step = 1; });

        // V3 is reserved for one exact executor per denoising stage.  It must
        // not accept the historical 11-pass single-stage plan, nor allow the
        // local stage index and global schedule offset to drift apart.
        f.reset();
        f.reject_v3("wrong pass count");
        f.reset(); f.stage.struct_size--;
        f.reject_v3("stage metadata ABI");
        f.reset(); f.stage.version++;
        f.reject_v3("stage metadata ABI");
        f.reset(); f.stage.reserved = 1;
        f.reject_v3("stage metadata ABI");
        f.reset(); f.stage.metadata_header = nullptr;
        f.reject_v3("stage metadata ABI");
        f.reset(); f.plan.pass_count = 8; f.plan.stage = 1;
        f.reject_v3("wrong pass count");
        f.reset(); f.plan.pass_count = 3; f.plan.stage = 1;
        f.reject_v3("wrong pass count");
        f.reset(); f.plan.pass_count = 3; f.plan.stage = 1;
        f.stage.schedule_pass_begin = 7;
        f.reject_v3("wrong pass count");
        f.reset(); f.plan.pass_count = 8;
        f.reject_v3("cancelled", 1);
        f.reset(); f.plan.pass_count = 3; f.plan.stage = 1;
        f.stage.schedule_pass_begin = 8;
        f.reject_v3("cancelled", 1);
        f.reset();

        assert(!ltx_native_create_streamed_v1(&f.options, &f.base, nullptr,
                                             cancel_description, &f.calls, nullptr, 0));
        assert(!ltx_native_create_streamed_v2(&f.options, &f.exact, nullptr,
                                             cancel_description, &f.calls, nullptr, 1024));
        assert(!ltx_native_create_streamed_v3(&f.options, &f.stage, nullptr,
                                             cancel_description, &f.calls, nullptr, 1024));
        auto *ctx = reinterpret_cast<ltx_native_denoiser *>(uintptr_t(1));
        assert(!ltx_native_create_streamed_v2(nullptr, &f.exact, &ctx,
                                             nullptr, nullptr, f.error, sizeof(f.error)) && !ctx);
        f.exact.struct_size--; f.reject(2, "snapshot ABI"); f.reset();
        f.exact.version++; f.reject(2, "snapshot ABI"); f.reset();
        f.exact.metadata_header = nullptr; f.reject(2, "snapshot ABI"); f.reset();
        auto mapping = metadata.mapping();
        mapping.bytes--; f.exact.metadata_mapping = &mapping;
        f.reject(2, "mapping size mismatch"); f.reset();
        auto header = metadata.header(); header.snapshot.valid = 0;
        f.exact.metadata_header = &header; f.reject(2, "metadata snapshot"); f.reset();
        header = metadata.header(); header.header_size = UINT64_MAX;
        f.exact.metadata_header = &header; f.reject(2, "metadata snapshot"); f.reset();

        const int other = open(argv[2], O_RDONLY | O_CLOEXEC); assert(other >= 0);
        struct stat first_status{}, other_status{};
        assert(fstat(metadata.mapping().descriptor, &first_status) == 0 && fstat(other, &other_status) == 0);
        assert(first_status.st_size == other_status.st_size && first_status.st_ino != other_status.st_ino);
        mapping = metadata.mapping(); mapping.descriptor = other;
        f.exact.metadata_mapping = &mapping;
        f.reject(2, "checkpoint_changed"); f.reset();
        f.options.checkpoint = argv[2]; f.reject(2, "checkpoint_changed"); f.reset();
        ltx_st_mapping duplicate{};
        assert(!ltx_st_map_fd(&metadata.header(), other, &duplicate, f.error, sizeof(f.error)));
        assert(!duplicate.descriptor_open && !duplicate.address && fcntl(other, F_GETFD) >= 0);
        close(other);
        assert(ltx_st_map_fd(&metadata.header(), metadata.mapping().descriptor, &duplicate, f.error, sizeof(f.error)));
        assert(duplicate.descriptor != metadata.mapping().descriptor);
        assert((fcntl(duplicate.descriptor, F_GETFD) & FD_CLOEXEC) != 0);
        ltx_st_map_close(&duplicate);
        metadata.check_unchanged();

        // Both paths reach the pre-GPU cancellation callback. Repeated borrowed
        // failure must neither free the caller header nor leak duplicate fds.
        f.reject(1, "cancelled", 1); f.reject(2, "cancelled", 1);
        for (unsigned version = 1; version <= 2; ++version) {
            f.calls = 0; f.error[0] = 0; ctx = nullptr;
            const int ok = version == 1
                ? ltx_native_create_streamed_v1(&f.options, &f.base, &ctx,
                    cancel_after_validation, &f.calls, f.error, sizeof(f.error))
                : ltx_native_create_streamed_v2(&f.options, &f.exact, &ctx,
                    cancel_after_validation, &f.calls, f.error, sizeof(f.error));
            assert(!ok && !ctx && f.calls == 49 && std::strstr(f.error, "cancelled"));
            metadata.check_unchanged();
        }
        // A structurally valid but incorrect capacity must fail the actual
        // metadata comparison, before the validated callback/GPU allocation.
        f.capacities[0]++;
        f.calls = 0; f.error[0] = 0;
        assert(!ltx_native_create_streamed_v2(&f.options, &f.exact, &ctx,
            cancel_after_validation, &f.calls, f.error, sizeof(f.error)));
        assert(!ctx && f.calls == 48 && std::strstr(f.error, "capacity mismatch"));
        f.reset();
        const auto before = fd_count();
        for (unsigned i = 0; i < 16; ++i) f.reject(2, "cancelled", 1);
        assert(fd_count() == before);
        metadata.check_unchanged();
        std::cout << "PASS v1/v2 legacy and v3 split-stage ABI/plan rejection; full metadata validation/capacity mismatch; borrowed fd/header, same-size identity, repeated cancellation cleanup\n";

        // Same inode/size, different payload and explicit mtime (no timing sleeps).
        int writable = open(argv[1], O_RDWR | O_CLOEXEC); assert(writable >= 0);
        unsigned char byte = 0;
        assert(pread(writable, &byte, 1, first_status.st_size - 1) == 1); byte ^= 1;
        assert(pwrite(writable, &byte, 1, first_status.st_size - 1) == 1);
        timespec times[2] = {first_status.st_atimespec, first_status.st_mtimespec};
        times[1].tv_sec += 2;
        assert(futimens(writable, times) == 0); close(writable);
        f.reject(2, "checkpoint_changed"); is_stale(metadata);
        assert(!ltx_st_map_fd(&metadata.header(), metadata.mapping().descriptor, &duplicate, f.error, sizeof(f.error)));
        {
            tc::ltx::StreamingMetadata refreshed(argv[1]); Fixture truncated(argv[1], refreshed);
            writable = open(argv[1], O_RDWR | O_CLOEXEC); assert(writable >= 0);
            assert(ftruncate(writable, first_status.st_size - 1) == 0);
            truncated.reject(2, "checkpoint_changed"); is_stale(refreshed);
            assert(ftruncate(writable, first_status.st_size) == 0); close(writable);
        }
        {
            tc::ltx::StreamingMetadata refreshed(argv[1]); Fixture replaced(argv[1], refreshed);
            assert(rename(argv[2], argv[1]) == 0);
            replaced.reject(2, "checkpoint_changed"); is_stale(refreshed);
        }
        std::cout << "PASS same-size in-place mutation, truncate, and path replacement reject stale snapshots before GPU construction\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
