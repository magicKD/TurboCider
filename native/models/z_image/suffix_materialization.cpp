#include "suffix_materialization.hpp"
#include "../../core/common.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tc::z_image {
namespace {
constexpr uint64_t scratch_limit = 4ull << 20;
uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > uint64_t(std::numeric_limits<off_t>::max()) / b)
        throw std::invalid_argument("Z-Image suffix geometry overflow");
    return a * b;
}
void range(uint64_t offset, uint64_t bytes) {
    const auto limit = uint64_t(std::numeric_limits<off_t>::max());
    if (offset > limit || bytes > limit - offset)
        throw std::invalid_argument("Z-Image suffix file range overflow");
}
void checkpoint(const std::atomic<bool> &cancelled) {
    if (cancelled.load(std::memory_order_acquire))
        throw tc::Cancelled();
}
}

SuffixGeometry suffix_geometry(uint64_t h, uint64_t m, uint64_t a, uint64_t e) {
    if (!h || !m || a >= m || (e != 1 && e != 2))
        throw std::invalid_argument("invalid Z-Image suffix geometry");
    const auto row = multiply(m, e);
    if (row > scratch_limit)
        throw std::invalid_argument("Z-Image MLP row exceeds suffix packing limit");
    const auto suffix = multiply(m - a, e);
    return {h, m, a, e, multiply(multiply(a, h), e),
            multiply(h, suffix), row, multiply(a, e), suffix,
            multiply(h, row), multiply(h, suffix)};
}

void pack_suffix_rows(int source, uint64_t source_offset,
                      int destination, uint64_t destination_offset,
                      const SuffixGeometry &supplied,
                      const std::atomic<bool> &cancelled,
                      SuffixPackMetrics &metrics) {
    // Recompute derived fields rather than trusting caller-populated sizes.
    const auto g = suffix_geometry(supplied.hidden, supplied.intermediate,
                                  supplied.first_gpu_channel, supplied.element_bytes);
    range(source_offset, g.down_source_bytes);
    range(destination_offset, g.down_suffix_bytes);
    if (metrics.read_bytes > UINT64_MAX - g.down_source_bytes ||
        metrics.write_bytes > UINT64_MAX - g.down_suffix_bytes)
        throw std::invalid_argument("Z-Image suffix accounting overflow");
    checkpoint(cancelled);
    struct stat src{}, dst{};
    if (::fstat(source, &src) || ::fstat(destination, &dst))
        throw std::system_error(errno, std::generic_category(), "Z-Image suffix fstat");
    if (!S_ISREG(src.st_mode) || !S_ISREG(dst.st_mode) ||
        (src.st_dev == dst.st_dev && src.st_ino == dst.st_ino))
        throw std::invalid_argument("Z-Image suffix requires distinct regular files");
    if (src.st_size < 0 || source_offset + g.down_source_bytes > uint64_t(src.st_size))
        throw std::runtime_error("Z-Image suffix source truncated");
    const uint64_t batch_rows = scratch_limit / g.down_row_bytes;
    std::vector<char> buffer(std::min(g.hidden, batch_rows) * g.down_row_bytes);
    for (uint64_t row = 0; row < g.hidden;) {
        const auto count = std::min(batch_rows, g.hidden - row);
        const auto bytes = count * g.down_row_bytes;
        uint64_t done = 0;
        while (done < bytes) {
            checkpoint(cancelled);
            const auto n = ::pread(source, buffer.data() + done, size_t(bytes - done),
                                   off_t(source_offset + row * g.down_row_bytes + done));
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) throw std::system_error(errno, std::generic_category(), "Z-Image suffix read");
            if (!n) throw std::runtime_error("Z-Image suffix source truncated");
            done += uint64_t(n);
            metrics.read_bytes += uint64_t(n);
        }
        for (uint64_t j = 0; j < count; ++j)
            std::memmove(buffer.data() + j * g.down_suffix_row_bytes,
                         buffer.data() + j * g.down_row_bytes + g.down_skip_bytes,
                         size_t(g.down_suffix_row_bytes));
        done = 0;
        const auto output_bytes = count * g.down_suffix_row_bytes;
        while (done < output_bytes) {
            checkpoint(cancelled);
            const auto n = ::pwrite(destination, buffer.data() + done, size_t(output_bytes - done),
                                    off_t(destination_offset + row * g.down_suffix_row_bytes + done));
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) throw std::system_error(errno, std::generic_category(), "Z-Image suffix write");
            if (!n) throw std::runtime_error("Z-Image suffix write made no progress");
            done += uint64_t(n);
            metrics.write_bytes += uint64_t(n);
        }
        row += count;
    }
    checkpoint(cancelled);
}
} // namespace tc::z_image
