#pragma once

#include "ltx_native.h"
#include "ltx_streaming_descriptor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tc::ltx {

// Request-owned projection from the generic immutable layout to the native C
// execution ABI. The native executor copies the plan arrays during create,
// while metadata_header/mapping remain borrowed until streaming_destroy.
// Consequently this object must outlive the exact native handle.
class StreamingPlanView {
  public:
    StreamingPlanView(const std::string &checkpoint,
                      const StreamingConfig &config,
                      const StreamingWorkload &workload,
                      uint64_t request_generation,
                      uint32_t stage_index = 0);
    StreamingPlanView(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string checkpoint_logical_id,
        const StreamingConfig &config,
        const StreamingWorkload &workload,
        uint64_t request_generation,
        uint32_t stage_index = 0);
    ~StreamingPlanView() = default;

    StreamingPlanView(const StreamingPlanView &) = delete;
    StreamingPlanView &operator=(const StreamingPlanView &) = delete;
    StreamingPlanView(StreamingPlanView &&) = delete;
    StreamingPlanView &operator=(StreamingPlanView &&) = delete;

    const ltx_native_streaming_options_v2 &native_options() const noexcept {
        return native_options_;
    }
    const ltx_native_streaming_options_v3 &native_stage_options() const noexcept {
        return native_stage_options_;
    }
    const streaming::Descriptor &descriptor() const noexcept {
        return descriptor_;
    }
    const streaming::Layout &layout() const noexcept { return layout_; }
    const StreamingMetadata &metadata() const noexcept { return metadata_; }
    const tc_stream_stage_plan_v1 &c_plan() const noexcept { return c_plan_; }
    uint32_t resident_prefix_blocks() const noexcept {
        return native_options_.base.resident_prefix_blocks;
    }
    uint32_t stage_index() const noexcept { return stage_index_; }
    uint32_t schedule_pass_begin() const noexcept {
        return native_stage_options_.schedule_pass_begin;
    }
    bool split_stages() const noexcept { return layout_.stages.size() > 1; }

  private:
    void initialize(const StreamingConfig &, const StreamingWorkload &,
                    uint64_t request_generation, uint32_t stage_index);

    StreamingMetadata metadata_;
    streaming::Descriptor descriptor_;
    streaming::Layout layout_;
    std::vector<uint64_t> slot_capacities_;
    std::vector<tc_stream_group_v1> groups_;
    tc_stream_stage_plan_v1 c_plan_{};
    ltx_native_streaming_options_v2 native_options_{};
    ltx_native_streaming_options_v3 native_stage_options_{};
    uint32_t stage_index_ = 0;
};

} // namespace tc::ltx
