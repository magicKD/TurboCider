#pragma once
#include "../../backends/mlx.hpp"
#include <cerrno>
#include <fcntl.h>
#include <system_error>
#include <unistd.h>

namespace tc::qwen21 {
// Request-owned backing. Each denoise layer reads only its own compact KV.
// The unnamed file disappears on close, including cancellation and failure.
class StreamedPrefixCache {
public:
    explicit StreamedPrefixCache(std::atomic<bool> &cancelled) : cancelled_(cancelled) {
        const auto name = (std::filesystem::temp_directory_path() / "TurboCider-Qwen-KV-XXXXXX").string();
        std::vector<char> path(name.begin(), name.end());
        path.push_back('\0');
        fd_ = mkstemp(path.data());
        if (fd_ < 0) fail("cannot create Qwen prefix backing");
        if (unlink(path.data()) != 0 || fcntl(fd_, F_SETFD, FD_CLOEXEC) != 0 || fcntl(fd_, F_NOCACHE, 1) != 0) {
            const int error = errno;
            close(fd_);
            throw std::system_error(error, std::generic_category(), "cannot configure Qwen prefix backing");
        }
    }
    ~StreamedPrefixCache() { close(fd_); }
    StreamedPrefixCache(const StreamedPrefixCache &) = delete;
    StreamedPrefixCache &operator=(const StreamedPrefixCache &) = delete;

    size_t layers() const { return layout_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t read_bytes() const { return read_bytes_; }
    uint64_t reads() const { return reads_; }
    uint64_t max_layer_bytes() const { return max_layer_bytes_; }
    void clear() { layout_.clear(); bytes_ = 0; }

    void append(const Tensor &key, const Tensor &value) {
        require(key.dtype() == mx::bfloat16 && value.dtype() == key.dtype() && key.shape() == value.shape() && key.size() > 0,
                "streamed Qwen prefix geometry changed");
        require(key.is_available() && value.is_available() && key.flags().row_contiguous && value.flags().row_contiguous,
                "streamed Qwen prefix requires evaluated contiguous K/V");
        const auto stride = key.nbytes();
        write(key.data<uint8_t>(), stride, bytes_);
        write(value.data<uint8_t>(), stride, bytes_ + stride);
        layout_.push_back({key.shape(), bytes_, stride});
        bytes_ += 2 * stride;
        max_layer_bytes_ = std::max(max_layer_bytes_, uint64_t(2) * stride);
    }

    std::pair<Tensor, Tensor> load(size_t layer) {
        const auto &entry = layout_.at(layer);
        auto load = [&](uint64_t offset) {
            Tensor value(mx::allocator::malloc(entry.stride), entry.shape, mx::bfloat16);
            require(value.buffer().ptr() != nullptr, "cannot allocate Qwen prefix refill slot");
            size_t done = 0;
            while (done < entry.stride) {
                checkpoint(cancelled_);
                const auto count = pread(fd_, value.data<uint8_t>() + done, entry.stride - done, off_t(offset + done));
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) fail("cannot read Qwen prefix backing");
                require(count > 0, "truncated Qwen prefix backing");
                done += size_t(count);
            }
            return value;
        };
        auto key = load(entry.offset);
        auto value = load(entry.offset + entry.stride);
        read_bytes_ += 2 * entry.stride;
        ++reads_;
        return {std::move(key), std::move(value)};
    }

private:
    struct Layer { mx::Shape shape; uint64_t offset; size_t stride; };
    std::atomic<bool> &cancelled_;
    int fd_ = -1;
    std::vector<Layer> layout_;
    uint64_t bytes_ = 0, read_bytes_ = 0, reads_ = 0, max_layer_bytes_ = 0;

    [[noreturn]] static void fail(const char *message) {
        throw std::system_error(errno, std::generic_category(), message);
    }
    void write(const uint8_t *bytes, size_t size, uint64_t offset) {
        size_t done = 0;
        while (done < size) {
            checkpoint(cancelled_);
            const auto count = pwrite(fd_, bytes + done, size - done, off_t(offset + done));
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) fail("cannot write Qwen prefix backing");
            require(count > 0, "short Qwen prefix backing write");
            done += size_t(count);
        }
    }
};
} // namespace tc::qwen21
