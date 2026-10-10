#pragma once

#include "../core/common.hpp"
#include "../runtime/streaming/source_lease.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/clonefile.h>
#endif

namespace tc {

// The pinned upstream GGUF reader opens O_RDWR and uses a writable MAP_SHARED.
// Even a read changes ctime on Darwin. Never give it the original artifact:
// clone compressed bytes from a held read-only fd (APFS CoW), with a bounded
// copy fallback elsewhere. This is not a dense sidecar or content proof.
class GgufReadSnapshot final {
  public:
    explicit GgufReadSnapshot(const std::filesystem::path &source, bool force_copy = false) {
        try {
            streaming::SourceFileIdentity identity;
            identity.logical_id = "gguf-compat-source";
            identity.path = source;
            lease_ = streaming::SourceLease::capture({std::move(identity)});
            auto input = lease_->duplicate_fd("gguf-compat-source");
            std::string name = (std::filesystem::temp_directory_path() / "turbocider-gguf-read-XXXXXX").string();
            const auto *directory = ::mkdtemp(name.data());
            require(directory != nullptr, "cannot create private GGUF compatibility directory");
            directory_ = directory;
            path_ = directory_ / "weights.gguf";
#if defined(__APPLE__)
            cloned_ = !force_copy && ::fclonefileat(input.get(), AT_FDCWD, path_.c_str(), CLONE_NOOWNERCOPY) == 0;
#else
            (void)force_copy;
#endif
            if (!cloned_) copy(input.get(), lease_->file("gguf-compat-source").bytes);
            // A read-only source may yield a read-only clone. Only this distinct
            // owner-private temporary inode may be made writable for upstream.
            streaming::OwnedSourceFd output(::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
            require(bool(output), "cannot open private GGUF snapshot");
            struct stat status{};
            require(::fstat(output.get(), &status) == 0 && S_ISREG(status.st_mode) &&
                        status.st_uid == ::geteuid() && status.st_size > 0,
                    "invalid private GGUF snapshot");
            const auto &original = lease_->file("gguf-compat-source");
            require(uint64_t(status.st_size) == original.bytes &&
                        (uint64_t(status.st_dev) != original.device || uint64_t(status.st_ino) != original.inode),
                    "GGUF snapshot must not alias the source inode");
            require(::fchmod(output.get(), 0600) == 0, "cannot protect private GGUF snapshot");
            lease_->revalidate_after_drain();
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~GgufReadSnapshot() { cleanup(); }
    GgufReadSnapshot(const GgufReadSnapshot &) = delete;
    GgufReadSnapshot &operator=(const GgufReadSnapshot &) = delete;
    const std::filesystem::path &path() const noexcept { return path_; }
    bool cloned() const noexcept { return cloned_; }
    void check_unchanged() const { lease_->revalidate_after_drain(); }

  private:
    void copy(int input, uint64_t bytes) {
        streaming::OwnedSourceFd output(::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
        require(bool(output), "cannot create bounded-copy GGUF snapshot");
        std::vector<unsigned char> buffer(64 * 1024);
        uint64_t offset = 0;
        while (offset < bytes) {
            const auto amount = size_t(std::min<uint64_t>(buffer.size(), bytes - offset));
            ssize_t count;
            do { count = ::pread(input, buffer.data(), amount, off_t(offset)); } while (count < 0 && errno == EINTR);
            require(count > 0 && size_t(count) <= amount, "short GGUF snapshot source read");
            size_t written = 0;
            while (written < size_t(count)) {
                ssize_t result;
                do { result = ::write(output.get(), buffer.data() + written, size_t(count) - written); }
                while (result < 0 && errno == EINTR);
                require(result > 0 && size_t(result) <= size_t(count) - written, "short GGUF snapshot write");
                written += size_t(result);
            }
            offset += uint64_t(count);
        }
    }
    void cleanup() noexcept {
        std::error_code error;
        if (!path_.empty()) std::filesystem::remove(path_, error);
        if (!directory_.empty()) std::filesystem::remove(directory_, error);
    }
    std::shared_ptr<const streaming::SourceLease> lease_;
    std::filesystem::path directory_, path_;
    bool cloned_ = false;
};

} // namespace tc
