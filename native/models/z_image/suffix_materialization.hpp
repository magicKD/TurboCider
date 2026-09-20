#pragma once

#include <atomic>
#include <cstdint>

namespace tc::z_image {

// Metadata-only geometry, shared by planning and execution. BF16 uses two
// bytes; one-byte elements preserve the existing legacy ConvRot packer.
// This is not artifact verification or authority to execute a hybrid route.
struct SuffixGeometry final {
    uint64_t hidden, intermediate, first_gpu_channel, element_bytes;
    uint64_t up_skip_bytes, up_suffix_bytes;
    uint64_t down_row_bytes, down_skip_bytes, down_suffix_row_bytes;
    uint64_t down_source_bytes, down_suffix_bytes;
};

SuffixGeometry suffix_geometry(uint64_t hidden, uint64_t intermediate,
                               uint64_t first_gpu_channel,
                               uint64_t element_bytes);

struct SuffixPackMetrics final {
    uint64_t read_bytes = 0, write_bytes = 0;
};

// Packs [hidden, intermediate][:, first_gpu_channel:] using <= 4 MiB
// scratch and held descriptors only. Caller owns a private destination and
// must not publish it on failure. No GPU operations or path reopening.
// Cancellation preserves the native tc::Cancelled exception contract.
// Metrics count actual I/O, including successful partial I/O before failure.
void pack_suffix_rows(int source_fd, uint64_t source_offset,
                      int destination_fd, uint64_t destination_offset,
                      const SuffixGeometry &, const std::atomic<bool> &cancelled,
                      SuffixPackMetrics &);

} // namespace tc::z_image
