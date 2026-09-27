#pragma once
#include "../../runtime/streaming/layout.hpp"
#include "../../runtime/streaming/source_lease.hpp"
extern "C" {
#include "ltx_streaming_layout.h"
}
#include <memory>

namespace tc::ltx {

struct StreamingWorkload {
    uint32_t width = 0, height = 0, frames = 0, fps = 0, text_rows = 0;
    bool parallel_av = false, batch_audio_commands = false, video_attention_batch = false;
    // Explicitly distinguish the native test's fabricated inputs from a real
    // connected context. Neither choice grants public execution qualification.
    std::string conditioning = "connected";
    // Public request-scoped execution releases the complete Stage-1
    // transformer backing before the spatial upsampler and materializes a
    // fresh Stage-2 executor afterwards.  Private/legacy exact streaming keeps
    // the historical single 11-pass stage so its ABI and performance remain
    // unchanged.
    bool split_stages = false;
};

// Owns the header, fd and all borrowed C field metadata. No GPU allocations,
// no full-weight reads, and no whole-checkpoint mmap. Not a content-hash trust
// authority: the artifact is explicitly a mutable-file snapshot identity.
// Keep this owner alive until all users of header/mapping/blocks have joined.
class StreamingMetadata {
public:
    explicit StreamingMetadata(const std::string &checkpoint);
    StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id);
    ~StreamingMetadata();
    StreamingMetadata(const StreamingMetadata &) = delete;
    StreamingMetadata &operator=(const StreamingMetadata &) = delete;
    streaming::Descriptor describe(const StreamingWorkload &) const;
    void check_unchanged() const;
    const ltx_st_header &header() const;
    const ltx_st_mapping &mapping() const;
    const ltx_stream_block_layout &block(uint32_t) const;
    uint64_t quant_metadata_read_bytes() const;
    const std::shared_ptr<const streaming::SourceLease> &source_lease() const noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::ltx
