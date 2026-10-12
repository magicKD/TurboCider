#pragma once

#include <mlx/io.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ios>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>

namespace tc {

namespace mx = mlx::core;

class MlxOwnedFd final {
    int fd_ = -1;

  public:
    explicit MlxOwnedFd(int fd = -1) noexcept : fd_(fd) {}
    MlxOwnedFd(const MlxOwnedFd &) = delete;
    MlxOwnedFd &operator=(const MlxOwnedFd &) = delete;
    MlxOwnedFd(MlxOwnedFd &&other) noexcept : fd_(other.release()) {}
    MlxOwnedFd &operator=(MlxOwnedFd &&other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = other.release();
        }
        return *this;
    }
    ~MlxOwnedFd() {
        if (fd_ >= 0) ::close(fd_);
    }
    int get() const noexcept { return fd_; }
    int release() noexcept {
        const int value = fd_;
        fd_ = -1;
        return value;
    }
    explicit operator bool() const noexcept { return fd_ >= 0; }
};

// Reader for MLX safetensors backed by a request-scoped SourceLease fd.
// MLX may perform lazy offset reads, so the duplicate fd is owned by this
// reader until all arrays loaded through it have been materialized.
class MlxLeaseFdReader final : public mx::io::Reader {
    int fd_ = -1;
    std::string label_;
    mutable std::mutex cursor_mutex_;
    off_t cursor_ = 0;

    [[noreturn]] void fail(const char *operation) const {
        throw std::runtime_error(
            std::string("streaming source ") + operation + " failed for " +
            label_ + ": " + std::strerror(errno));
    }

  public:
    MlxLeaseFdReader(MlxOwnedFd owned_fd, std::string label)
        : fd_(owned_fd.release()), label_(std::move(label)) {
        if (fd_ < 0)
            throw std::invalid_argument(
                "streaming source reader descriptor is unavailable");
    }
    ~MlxLeaseFdReader() override {
        if (fd_ >= 0) ::close(fd_);
    }

    bool is_open() const override { return fd_ >= 0; }
    bool good() const override { return is_open(); }

    size_t tell() override {
        std::lock_guard lock(cursor_mutex_);
        return static_cast<size_t>(cursor_);
    }

    void seek(int64_t offset,
              std::ios_base::seekdir direction = std::ios_base::beg) override {
        std::lock_guard lock(cursor_mutex_);
        off_t base = 0;
        if (direction == std::ios_base::cur) base = cursor_;
        else if (direction == std::ios_base::end) {
            struct stat info{};
            if (::fstat(fd_, &info) < 0) fail("stat");
            base = info.st_size;
        } else if (direction != std::ios_base::beg)
            throw std::invalid_argument("streaming source invalid seek direction");
        if (base < 0 || offset < -base ||
            (offset > 0 && base > std::numeric_limits<off_t>::max() - offset))
            throw std::overflow_error("streaming source seek offset overflows");
        cursor_ = base + offset;
    }

    void read(char *destination, size_t bytes) override {
        std::lock_guard lock(cursor_mutex_);
        // dup() shares the kernel file position with every reader of the
        // lease. Keep a private logical cursor so repeated/concurrent header
        // loads always start at zero without disturbing another lazy reader.
        if (bytes > static_cast<size_t>(std::numeric_limits<off_t>::max() - cursor_))
            throw std::overflow_error("streaming source read range overflows");
        read(destination, bytes, static_cast<size_t>(cursor_));
        cursor_ += static_cast<off_t>(bytes);
    }

    void read(char *destination, size_t bytes, size_t offset) override {
        if (offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
            throw std::overflow_error("streaming source read offset overflows");
        size_t done = 0;
        while (done < bytes) {
            if (offset > static_cast<size_t>(
                    std::numeric_limits<off_t>::max()) - done)
                throw std::overflow_error(
                    "streaming source read range overflows");
            const size_t chunk = std::min(
                bytes - done,
                static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t count = ::pread(
                fd_, destination + done, chunk,
                static_cast<off_t>(offset + done));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) fail(count == 0 ? "short pread" : "pread");
            done += static_cast<size_t>(count);
        }
    }

    std::string label() const override { return "leased file " + label_; }
};

} // namespace tc
