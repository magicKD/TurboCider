#include "ltx_streaming_plan.hpp"

#include <stdexcept>

namespace tc::ltx {
namespace {

void require_plan(bool value, const std::string &reason) {
    if (!value)
        throw std::invalid_argument("ltx_streaming_plan_invalid: " + reason);
}

} // namespace

StreamingPlanView::StreamingPlanView(
        const std::string &checkpoint,
        const StreamingConfig &config,
        const StreamingWorkload &workload,
        uint64_t request_generation,
        uint32_t stage_index)
    : metadata_(checkpoint) {
    initialize(config, workload, request_generation, stage_index);
}

StreamingPlanView::StreamingPlanView(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string checkpoint_logical_id,
        const StreamingConfig &config,
        const StreamingWorkload &workload,
        uint64_t request_generation,
        uint32_t stage_index)
    : metadata_(std::move(lease), std::move(checkpoint_logical_id)) {
    initialize(config, workload, request_generation, stage_index);
}

void StreamingPlanView::initialize(
        const StreamingConfig &config,
        const StreamingWorkload &workload,
        uint64_t request_generation,
        uint32_t stage_index) {
    require_plan(request_generation != 0,
                 "request generation must be nonzero");
    descriptor_ = metadata_.describe(workload);
    layout_ = streaming::compile_layout(config, descriptor_);
    require_plan(layout_.materializations_complete,
                 "source/materialization/pass metadata is incomplete");
    require_plan(layout_.stages.size() == (workload.split_stages ? 2u : 1u),
                 "exact LTX compiled stage count is invalid");
    require_plan(stage_index < layout_.stages.size(),
                 "exact LTX stage index is invalid");
    stage_index_ = stage_index;
    const auto &stage = layout_.stages[stage_index];
    const std::string expected_id = workload.split_stages ?
        (stage_index == 0 ? "ltx-stage1-denoiser" :
                            "ltx-stage2-denoiser") : "denoiser";
    require_plan(stage.id == expected_id && !stage.resident,
                 "exact LTX requires a streamed denoiser stage");
    const uint32_t expected_passes = workload.split_stages ?
        (stage_index == 0 ? 8u : 3u) : 11u;
    require_plan(stage.prefix > 0 && stage.group_size == 1 &&
                     stage.slot_count > 0 &&
                     stage.pass_count == expected_passes,
                 "unsupported prefix/group/slot/pass layout");
    require_plan(stage.pools.size() == 1 &&
                     stage.pools.front().id == 0 &&
                     stage.pools.front().slots.size() == stage.slot_count,
                 "exact LTX requires one homogeneous slot pool");
    require_plan(stage.groups.size() ==
                     descriptor_.stages[stage_index].blocks.size() -
                         stage.prefix,
                 "compiled suffix group count mismatch");
    slot_capacities_.reserve(stage.slot_count);
    for (const auto &slot : stage.pools.front().slots) {
        require_plan(slot.capacity_bytes != 0,
                     "compiled slot capacity is zero");
        slot_capacities_.push_back(slot.capacity_bytes);
    }
    groups_.reserve(stage.groups.size());
    for (size_t index = 0; index < stage.groups.size(); ++index) {
        const auto &group = stage.groups[index];
        require_plan(group.id == index && group.pool == 0 &&
                         group.slot == index % stage.slot_count &&
                         group.blocks.size() == 1 &&
                         group.blocks.front() == stage.prefix + index &&
                         group.bytes != 0,
                     "compiled group/binding is not exact LTX G=1");
        groups_.push_back({group.id, group.slot,
                           static_cast<uint32_t>(group.blocks.size()),
                           group.blocks.data(), group.bytes});
    }
    c_plan_ = {sizeof(c_plan_), TC_STREAM_SLOT_ABI_V1,
               stage_index, 0u, stage.slot_count, stage.distance,
               stage.workers,
               stage.pass_count, request_generation,
               slot_capacities_.data(), static_cast<uint32_t>(groups_.size()),
               groups_.data()};
    native_options_.struct_size = sizeof(native_options_);
    native_options_.version = 2u;
    native_options_.base = {sizeof(native_options_.base), 1u,
                            stage.prefix, &c_plan_};
    native_options_.metadata_header = &metadata_.header();
    native_options_.metadata_mapping = &metadata_.mapping();
    uint32_t schedule_pass_begin = 0;
    for (uint32_t index = 0; index < stage_index; ++index)
        schedule_pass_begin += layout_.stages[index].pass_count;
    native_stage_options_.struct_size = sizeof(native_stage_options_);
    native_stage_options_.version = 3u;
    native_stage_options_.base = native_options_.base;
    native_stage_options_.metadata_header = &metadata_.header();
    native_stage_options_.metadata_mapping = &metadata_.mapping();
    native_stage_options_.schedule_pass_begin = schedule_pass_begin;
    metadata_.check_unchanged();
}

} // namespace tc::ltx
