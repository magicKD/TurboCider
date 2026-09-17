#pragma once

#include "../../runtime/streaming/layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tc::z_image {

struct StreamingWorkload {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t caption_rows = 0;
    uint32_t steps = 0;
};

// Header-only view of the single-file Comfy BF16 transformer.  Construction
// reads the safetensors prefix and JSON header only.  It never reads tensor
// payloads, creates MLX arrays, allocates Metal buffers, or starts workers.
class StreamingMetadata {
  public:
    explicit StreamingMetadata(const std::string &checkpoint);
    ~StreamingMetadata();

    StreamingMetadata(const StreamingMetadata &) = delete;
    StreamingMetadata &operator=(const StreamingMetadata &) = delete;

    streaming::Descriptor describe(const StreamingWorkload &) const;
    void check_unchanged() const;

    uint32_t block_count() const noexcept { return 30; }
    uint32_t tensors_per_block() const noexcept { return 13; }
    uint64_t block_bytes() const noexcept;
    uint64_t fixed_bytes() const noexcept;
    const std::string &snapshot_identity() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

// Private metadata/plan shadow for the existing ZImageWeightStream contract.
// It intentionally exposes no execution bridge yet.
class StreamingPlanView {
  public:
    StreamingPlanView(const std::string &checkpoint,
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

} // namespace tc::z_image
