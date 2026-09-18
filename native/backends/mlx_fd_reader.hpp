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

    [[noreturn]] void fail(const char *operation) const {
        throw std::runtime_error(
            std::string("streaming source ") + operation + " failed for " +
            label_ + ": " + std::strerror(errno));
    }

    static int whence(std::ios_base::seekdir direction) {
        if (direction == std::ios_base::beg) return SEEK_SET;
        if (direction == std::ios_base::end) return SEEK_END;
        return SEEK_CUR;
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
        const off_t value = ::lseek(fd_, 0, SEEK_CUR);
        if (value < 0) fail("tell");
        return static_cast<size_t>(value);
    }

    void seek(int64_t offset,
              std::ios_base::seekdir direction = std::ios_base::beg) override {
        std::lock_guard lock(cursor_mutex_);
        if (::lseek(fd_, static_cast<off_t>(offset),
                    whence(direction)) < 0)
            fail("seek");
    }

    void read(char *destination, size_t bytes) override {
        std::lock_guard lock(cursor_mutex_);
        size_t done = 0;
        while (done < bytes) {
            const size_t chunk = std::min(
                bytes - done,
                static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t count = ::read(fd_, destination + done, chunk);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) fail(count == 0 ? "short read" : "read");
            done += static_cast<size_t>(count);
        }
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
