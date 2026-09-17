#pragma once

#include "../../runtime/streaming/layout.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace tc::flux2 {

struct StreamingWorkload {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t caption_tokens = 0;
    uint32_t reference_tokens = 0;
    uint32_t steps = 0;
};

// Metadata-only view of a Diffusers sharded BF16 FLUX.2 transformer.  It
// reads config/index JSON and each safetensors header, but never reads tensor
// payloads, constructs MLX arrays, allocates Metal buffers, or starts workers.
class StreamingMetadata {
  public:
    StreamingMetadata(const std::string &transformer_directory,
                      const std::string &model_id);
    ~StreamingMetadata();

    StreamingMetadata(const StreamingMetadata &) = delete;
    StreamingMetadata &operator=(const StreamingMetadata &) = delete;

    streaming::Descriptor describe(const StreamingWorkload &) const;
    void check_unchanged() const;

    uint32_t dual_block_count() const noexcept;
    uint32_t single_block_count() const noexcept;
    uint32_t dual_fields_per_block() const noexcept { return 16; }
    uint32_t single_fields_per_block() const noexcept { return 4; }
    uint64_t dual_block_bytes() const noexcept;
    uint64_t single_block_bytes() const noexcept;
    uint64_t fixed_bytes() const noexcept;
    const std::string &snapshot_identity() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

// First execution-shaped projection is intentionally restricted to Klein 9B:
// its current eager GPU path establishes a synchronization boundary after
// every dual and single block.  Klein 4B keeps its compiled multi-block graph
// and remains fail-closed until address/lifetime experiments prove a slot-safe
// binding strategy without adding per-block waits to the default path.
class StreamingPlanView {
  public:
    StreamingPlanView(const std::string &transformer_directory,
                      const std::string &model_id,
                      const StreamingConfig &config,
                      const StreamingWorkload &workload);

    StreamingPlanView(const StreamingPlanView &) = delete;
    StreamingPlanView &operator=(const StreamingPlanView &) = delete;

    const StreamingMetadata &metadata() const noexcept { return metadata_; }
    const streaming::Descriptor &descriptor() const noexcept {
        return descriptor_;
    }
    const streaming::Layout &layout() const noexcept { return layout_; }

  private:
    StreamingMetadata metadata_;
    streaming::Descriptor descriptor_;
    streaming::Layout layout_;
};

} // namespace tc::flux2
