#pragma once

#include "../../runtime/streaming/layout.hpp"

extern "C" {
#include "h3_dit_schedule.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_weights.h"
}

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tc::h3 {

/*
 * Metadata needed to describe one H3 C/Metal DiT request.  This type is
 * deliberately independent of h3_params: it describes the layout and pass
 * identity, but never selects kernels, sampler shortcuts, or quantized cache
 * behavior.  active_blocks is the legacy uniform layer policy input; the
 * descriptor expands it into an explicit block-id sequence.
 */
struct StreamingWorkload {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t frames = 0;
    uint32_t fps = H3_FPS;
    uint32_t text_rows = 0;
    uint32_t steps = 0;
    uint32_t active_blocks = H3_DIT_BLOCKS;
    bool audio = true;
    bool token_reduction = false;
    bool first_block_cache = false;
};

/*
 * Owns all safetensors headers for one transformer directory.  It performs no
 * payload reads, mmap, GPU allocation, or model construction.  The snapshot
 * identity is metadata-based (device/inode/size/times plus the sorted shard
 * set), so it is intentionally not a content-hash trust authority.
 */
class StreamingMetadata {
  public:
    explicit StreamingMetadata(const std::string &transformer_directory);
    ~StreamingMetadata();

    StreamingMetadata(const StreamingMetadata &) = delete;
    StreamingMetadata &operator=(const StreamingMetadata &) = delete;

    streaming::Descriptor describe(const StreamingWorkload &) const;
    void check_unchanged() const;

    uint32_t block_count() const noexcept { return H3_DIT_BLOCKS; }
    size_t shard_count() const noexcept;
    uint64_t snapshot_identity() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

/*
 * The first execution projection is intentionally plan-only.  It accepts
 * exactly the current H3 candidate contract: uniform active blocks, one
 * streamed matrix group per block, and K=2/G=1.  It does not claim that the
 * legacy cross-forward prefetch/fusion sequence is already represented by the
 * generic executor.
 */
class StreamingPlanView {
  public:
    StreamingPlanView(const std::string &transformer_directory,
                      const StreamingConfig &config,
                      const StreamingWorkload &workload);
    ~StreamingPlanView() = default;

    StreamingPlanView(const StreamingPlanView &) = delete;
    StreamingPlanView &operator=(const StreamingPlanView &) = delete;

    const streaming::Descriptor &descriptor() const noexcept {
        return descriptor_;
    }
    const streaming::Layout &layout() const noexcept { return layout_; }
    const StreamingMetadata &metadata() const noexcept { return metadata_; }

  private:
    StreamingMetadata metadata_;
    streaming::Descriptor descriptor_;
    streaming::Layout layout_;
};

} // namespace tc::h3
