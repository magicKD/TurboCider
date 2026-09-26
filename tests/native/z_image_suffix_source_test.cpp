#include "models/z_image/streaming_descriptor.hpp"
#include "runtime/memory_manifest.hpp"
#include <cassert>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
using namespace tc;
using namespace tc::streaming;

SourceFileIdentity identity(const char *path) {
    SourceFileIdentity f; f.logical_id = "transformer"; f.path = path; return f;
}
size_t open_fds() {
    DIR *dir = opendir("/dev/fd"); assert(dir);
    size_t count = 0;
    while (auto *entry = readdir(dir)) if (entry->d_name[0] != '.') ++count;
    closedir(dir); return count;
}
template<class F> void rejected(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    assert(failed);
}
int main(int argc, char **argv) {
    if (argc == 2) {
        auto verified = SourceLease::capture_verified({identity(argv[1])});
        z_image::StreamingMetadata metadata(verified);
        std::atomic<bool> cancelled{false};
        const auto before = open_fds();
        auto source = metadata.materialize_gpu_suffix({512, 512, 64, 9}, 5120, cancelled);
        const auto &plan = source->plan();
        {
            auto parent = source->duplicate_fd(0), derived = source->duplicate_fd(1);
            std::vector<char> original(5120 * 2), packed(original.size());
            for (const auto &record : plan.packing)
                for (uint64_t row : {0u, 1234u, 3839u}) {
                    const auto input_offset = record.source.offset + (row * 10240 + 5120) * 2;
                    const auto output_offset = record.destination_offset + row * 5120 * 2;
                    assert(pread(parent.get(), original.data(), original.size(), input_offset) == ssize_t(original.size()));
                    assert(pread(derived.get(), packed.data(), packed.size(), output_offset) == ssize_t(packed.size()));
                    assert(original == packed);
                }
        }
        source->check_unchanged();
        std::cout << "{\"scope\":\"Real checkpoint derived-file validation; no GPU/Core ML execution\","
                  << "\"first_gpu_channel\":5120,\"branches\":32,\"sampled_rows_per_branch\":3,"
                  << "\"parent_sha256\":\"" << verified->file("transformer").content_digest << "\","
                  << "\"parent_verification_bytes\":" << verified->verification_bytes_read() << ','
                  << "\"recipe_sha256\":\"" << plan.recipe_digest << "\","
                  << "\"derived_sha256\":\"" << source->content_digest() << "\","
                  << "\"pack_read_bytes\":" << plan.setup_read_bytes << ','
                  << "\"pack_write_bytes\":" << plan.setup_write_bytes << ','
                  << "\"derived_verification_read_bytes\":" << source->verification_read_bytes()
                  << ",\"sampled_rows_equal\":true}" << '\n';
        source.reset();
        assert(open_fds() == before);
        return 0;
    }
    assert(argc == 3);
    const z_image::StreamingWorkload work{512, 512, 64, 9};
    std::atomic<bool> cancelled{false};
    {
        z_image::StreamingMetadata unverified(argv[1]);
        rejected([&] { unverified.materialize_gpu_suffix(work, 10239, cancelled); });
    }
    auto parent = SourceLease::capture_verified({identity(argv[1])});
    z_image::StreamingMetadata metadata(parent);
    const auto plan = metadata.describe_gpu_suffix(work, 10239);
    const auto baseline_fds = open_fds();
    // Cancellation during packing and before/after hashing must not publish a
    // result or retain a temporary descriptor, and must preserve Cancelled.
    for (int point : {0, 1, 2, 3}) {
        bool caught = false;
        try {
            metadata.materialize_gpu_suffix(work, 10239, cancelled,
                [&](const std::string &phase, int index, int) {
                    if ((point == 0 && phase == "pack_z_image_suffix" && index == 0) ||
                        (point == 1 && phase == "pack_z_image_suffix" && index == 1) ||
                        (point == 2 && phase == "verify_z_image_suffix" && index == 0) ||
                        (point == 3 && phase == "verify_z_image_suffix" && index == 1)) cancelled = true;
                });
        } catch (const Cancelled &) { caught = true; }
        assert(caught); cancelled = false;
        assert(open_fds() == baseline_fds);
    }
    rejected([&] {
        metadata.materialize_gpu_suffix(work, 10239, cancelled,
            [](const std::string &, int, int) { throw std::runtime_error("event failure"); });
    });
    assert(open_fds() == baseline_fds);
    auto source = metadata.materialize_gpu_suffix(work, 10239, cancelled);
    assert(source->plan().recipe_digest == plan.recipe_digest);
    assert(source->verification_read_bytes() == 32ull * 3840 * 2);
    const auto hash = source->content_digest();
    {
        auto fd = source->duplicate_fd(1);
        assert((fcntl(fd.get(), F_GETFL) & O_ACCMODE) == O_RDONLY);
        assert(fcntl(fd.get(), F_GETFD) & FD_CLOEXEC);
        struct stat st{}; assert(!fstat(fd.get(), &st));
        assert(st.st_nlink == 0 && st.st_size == 32 * 3840 * 2);
        unsigned char value = 1;
        assert(pwrite(fd.get(), &value, 1, 0) == -1 && errno == EBADF);
        std::vector<uint16_t> actual(32 * 3840);
        assert(pread(fd.get(), actual.data(), actual.size() * 2, 0) == ssize_t(actual.size() * 2));
        for (unsigned branch = 0; branch < 32; ++branch)
            for (unsigned row = 0; row < 3840; ++row) {
                const uint16_t expected = row == 0 || row == 1024 || row == 3839 ? 0x3f00 + branch : 0;
                assert(actual[branch * 3840 + row] == expected);
            }
        assert(memory_sha256_fd(fd.get(), st.st_size) == hash);
        auto original = source->duplicate_fd(0);
        assert((fcntl(original.get(), F_GETFL) & O_ACCMODE) == O_RDONLY);
        rejected([&] { source->duplicate_fd(2); });
    }
    source->check_unchanged();
    source.reset();
    assert(open_fds() == baseline_fds);
    // Independently generated sparse copy: different inode/path, same bytes.
    // Verify both parent contents natively; JSON/digests from callers are not proof.
    auto copy = [&] {
        auto copied = SourceLease::capture_verified({identity(argv[2])});
        z_image::StreamingMetadata copy_metadata(copied);
        return copy_metadata.materialize_gpu_suffix(work, 10239, cancelled);
    }();
    // The returned source must retain its parent after metadata/lease owners exit.
    copy->check_unchanged();
    assert(copy->plan().recipe_digest == plan.recipe_digest);
    assert(copy->content_digest() == hash);
    assert(copy->plan().descriptor.checkpoint_identity == plan.descriptor.checkpoint_identity);
    const auto with_copy_fds = open_fds();
    // Mutate the original after an actual branch was packed. Parent generation
    // validation must reject the whole result, even if output bytes still match.
    rejected([&] {
        metadata.materialize_gpu_suffix(work, 10239, cancelled,
            [&](const std::string &phase, int index, int) {
                if (phase == "pack_z_image_suffix" && index == 1) {
                    const int fd = open(argv[1], O_WRONLY); assert(fd >= 0);
                    struct stat st{}; assert(!fstat(fd, &st));
                    const char byte = 42; assert(pwrite(fd, &byte, 1, st.st_size - 1) == 1); close(fd);
                }
            });
    });
    assert(open_fds() == with_copy_fds);
    // A live derived owner also detects replacement of its named parent.
    const std::string renamed = std::string(argv[2]) + ".moved";
    assert(!rename(argv[2], renamed.c_str()));
    rejected([&] { copy->check_unchanged(); });
    rejected([&] { copy->duplicate_fd(1); });
    copy.reset();
    assert(open_fds() == baseline_fds);
    std::cout << "PASS verified suffix source: byte oracle, readonly/unlinked fd, hash, cancellation/event cleanup, cross-install identity, parent mutation/replacement\n";
}
