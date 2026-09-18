#include "public_result.hpp"

#include "../../core/common.hpp"
#include "actual_receipt.hpp"

namespace tc::streaming {
namespace {

const char *pass_transition_name(PassTransition value) {
    switch (value) {
    case PassTransition::reload: return "reload";
    case PassTransition::carry_first_group: return "carry_first_group";
    }
    return "unknown";
}

const char *multi_pool_policy_name(MultiPoolPolicy value) {
    switch (value) {
    case MultiPoolPolicy::serial: return "serial";
    case MultiPoolPolicy::retain_all: return "retain_all";
    }
    return "unknown";
}

void actual_check(bool value, const char *detail) {
    require(value,
            std::string("streaming_actual_plan_mismatch: ") + detail);
}

bool same_streaming_runtime(
        const StreamingRuntimeMetrics &left,
        const StreamingRuntimeMetrics &right) {
    return left.implementation == right.implementation &&
           left.layout_digest == right.layout_digest &&
           left.stage == right.stage &&
           left.resident_prefix_blocks == right.resident_prefix_blocks &&
           left.block_group_size == right.block_group_size &&
           left.slot_count == right.slot_count &&
           left.prefetch_distance == right.prefetch_distance &&
           left.io_workers == right.io_workers &&
           left.group_count == right.group_count &&
           left.pass_count == right.pass_count &&
           left.startup_policy == right.startup_policy &&
           left.pass_transition == right.pass_transition &&
           left.retention == right.retention &&
           left.reader_revision == right.reader_revision &&
           left.weight_format == right.weight_format &&
           left.kernel_revision == right.kernel_revision &&
           left.conditioning_recipe == right.conditioning_recipe &&
           left.upsample_boundary == right.upsample_boundary &&
           left.component_policy_revision ==
               right.component_policy_revision &&
           left.multi_pool_policy == right.multi_pool_policy &&
           left.pool_count == right.pool_count &&
           left.slot_bundle_count == right.slot_bundle_count &&
           left.refill_worker_count == right.refill_worker_count &&
           left.source_lease_verified == right.source_lease_verified &&
           left.drained == right.drained &&
           left.receipt_schema_version == right.receipt_schema_version &&
           left.receipt_fills == right.receipt_fills &&
           left.receipt_groups_submitted == right.receipt_groups_submitted &&
           left.receipt_logical_read_bytes ==
               right.receipt_logical_read_bytes &&
           left.receipt_reader_fences_issued ==
               right.receipt_reader_fences_issued &&
           left.receipt_reader_fences_completed ==
               right.receipt_reader_fences_completed &&
           left.receipt_source_generation ==
               right.receipt_source_generation &&
           left.receipt_event_digest == right.receipt_event_digest &&
           left.receipt_digest == right.receipt_digest &&
           left.receipt_verifier_revision ==
               right.receipt_verifier_revision;
}

void verify_stage_runtime(
        const StageLayout &stage, uint32_t stage_index,
        const StreamingPresetRecord &record,
        const ModelStreamingSnapshot &snapshot,
        StreamingRuntimeMetrics &actual, bool single_stage) {
    actual_check(!stage.resident, "authorized stage is resident");
    uint64_t expected_slot_bundles = 0;
    for (const auto &pool : stage.pools)
        expected_slot_bundles += pool.slots.size();

    actual_check(!actual.implementation.empty(),
                 "missing implementation identity");
    actual_check(actual.stage == stage.id, "stage identity differs");
    actual_check(actual.layout_digest == snapshot.layout().digest,
                 "layout digest differs");
    actual_check(actual.resident_prefix_blocks == stage.prefix,
                 "resident prefix differs");
    actual_check(actual.block_group_size == stage.group_size,
                 "block group size differs");
    actual_check(actual.slot_count == stage.slot_count,
                 "slot count differs");
    actual_check(actual.prefetch_distance == stage.distance,
                 "prefetch distance differs");
    actual_check(actual.io_workers == stage.workers,
                 "I/O worker count differs");
    actual_check(actual.group_count == stage.groups.size(),
                 "group count differs");
    actual_check(actual.pass_count == stage.pass_count,
                 "pass count differs");
    actual_check(actual.pass_transition ==
                     pass_transition_name(stage.pass_transition),
                 "pass transition differs");
    actual_check(actual.multi_pool_policy ==
                     multi_pool_policy_name(stage.multi_pool_policy),
                 "multi-pool policy differs");
    if (single_stage) {
        actual_check(actual.pass_transition == record.plan.pass_transition,
                     "preset pass transition differs");
        actual_check(actual.multi_pool_policy ==
                         record.plan.multi_pool_policy,
                     "preset multi-pool policy differs");
    }
    actual_check(actual.component_policy_revision ==
                     record.plan.component_policy_revision &&
                     actual.component_policy_revision ==
                         snapshot.component_policy_revision(),
                 "component policy differs");
    actual_check(actual.retention == "request",
                 "public retention is not request-scoped");
    actual_check(actual.pool_count == stage.pools.size(),
                 "pool count differs");
    actual_check(actual.slot_bundle_count == expected_slot_bundles,
                 "slot bundle count differs");
    actual_check(actual.refill_worker_count == stage.workers,
                 "refill worker count differs");
    actual_check(actual.drained, "adapter did not report a completed drain");
    (void)stage_index;
}

void attach_stage_receipt_summary(
        StreamingRuntimeMetrics &actual,
        const ActualStageReceipt &stage_receipt,
        const ActualExecutionReceipt &execution_receipt) {
    actual.source_lease_verified = true;
    actual.receipt_schema_version = execution_receipt.schema_version;
    actual.receipt_fills = stage_receipt.fills;
    actual.receipt_groups_submitted = stage_receipt.groups_submitted;
    actual.receipt_logical_read_bytes = stage_receipt.logical_read_bytes;
    actual.receipt_reader_fences_issued =
        stage_receipt.reader_fences_issued;
    actual.receipt_reader_fences_completed =
        stage_receipt.reader_fences_completed;
    actual.receipt_source_generation = stage_receipt.source_generation;
    actual.receipt_event_digest = stage_receipt.event_digest;
    actual.receipt_digest = execution_receipt.canonical_digest;
    actual.receipt_verifier_revision =
        std::string(actual_receipt_verifier_revision);
}

} // namespace

void verify_and_attach_public_streaming_result(
        const ResolvedRequestExecution &execution, RunResult &result) {
    actual_check(execution.selection.authority != nullptr,
                 "missing public authority");
    actual_check(execution.model_snapshot != nullptr,
                 "missing model snapshot");

    const auto &record = execution.selection.record;
    const auto &layout = execution.model_snapshot->layout();
    actual_check(layout.materializations_complete,
                 "authorized layout has incomplete source metadata");
    actual_check(layout.digest == record.plan.layout_digest,
                 "authorized layout differs from preset");
    actual_check(!layout.stages.empty(), "authorized layout has no stages");
    if (result.streaming_stages.empty()) {
        actual_check(layout.stages.size() == 1 &&
                         result.streaming_runtime.has_value(),
                     "adapter did not report every streaming stage");
        result.streaming_stages.push_back(
            {0, *result.streaming_runtime});
    }
    actual_check(result.streaming_stages.size() == layout.stages.size(),
                 "streaming stage count differs");
    for (uint32_t index = 0; index < layout.stages.size(); ++index) {
        auto &reported = result.streaming_stages[index];
        actual_check(reported.stage_index == index,
                     "streaming stage index differs");
        verify_stage_runtime(
            layout.stages[index], index, record,
            *execution.model_snapshot, reported.runtime,
            layout.stages.size() == 1);
    }
    if (result.streaming_runtime) {
        actual_check(layout.stages.size() == 1,
                     "legacy streaming summary is ambiguous for multi-stage");
        actual_check(same_streaming_runtime(
                         *result.streaming_runtime,
                         result.streaming_stages.front().runtime),
                     "legacy streaming summary differs");
    }
    const auto *probe_lease = execution.probe->source_lease();
    const auto *snapshot_lease = execution.model_snapshot->source_lease();
    actual_check(probe_lease != nullptr && snapshot_lease != nullptr,
                 "missing source lease");
    actual_check(probe_lease == snapshot_lease &&
                     probe_lease->generation() != 0,
                 "source lease generation differs");
    actual_check(result.streaming_receipt != nullptr,
                 "adapter did not return an actual execution receipt");
    verify_actual_execution_receipt(
        layout, result.streaming_stages.front().runtime.implementation,
        record.plan.component_policy_revision,
        snapshot_lease->generation(), *result.streaming_receipt);
    actual_check(result.streaming_receipt->stages.size() ==
                     layout.stages.size(),
                 "receipt stage count differs");
    for (uint32_t index = 0; index < layout.stages.size(); ++index) {
        const auto &stage = layout.stages[index];
        const auto &stage_receipt = result.streaming_receipt->stages[index];
        auto &actual = result.streaming_stages[index].runtime;
        actual_check(stage_receipt.completed_passes == stage.pass_count &&
                         stage_receipt.completed_groups ==
                             stage.pass_count * stage.groups.size() &&
                         stage_receipt.fills == actual.group_count *
                             actual.pass_count &&
                         stage_receipt.groups_submitted ==
                             actual.group_count * actual.pass_count &&
                         stage_receipt.reader_fences_issued ==
                             stage_receipt.reader_fences_completed &&
                         stage_receipt.drain_completed,
                     "receipt summary differs");
        attach_stage_receipt_summary(
            actual, stage_receipt, *result.streaming_receipt);
    }
    if (result.streaming_receipt->schema_version ==
            actual_receipt_schema_v3) {
        actual_check(result.streaming_receipt->boundaries.size() + 1 ==
                         layout.stages.size(),
                     "receipt boundary count differs");
        if (result.streaming_boundaries.empty()) {
            for (const auto &boundary : result.streaming_receipt->boundaries) {
                const auto &from = layout.stages[boundary.from_stage_index];
                const auto &to = layout.stages[boundary.to_stage_index];
                result.streaming_boundaries.push_back({
                    boundary.boundary_index, boundary.id, from.id, to.id,
                    boundary.source_stage_drained,
                    boundary.source_stage_backing_released,
                    boundary.live_slot_bytes_before,
                    boundary.live_slot_bytes_after,
                    boundary.pending_readers_before,
                    boundary.pending_readers_after,
                    boundary.released_slot_bytes,
                    boundary.event_digest});
            }
        }
        actual_check(result.streaming_boundaries.size() ==
                         result.streaming_receipt->boundaries.size(),
                     "runtime boundary count differs");
        for (uint32_t index = 0;
             index < result.streaming_boundaries.size(); ++index) {
            const auto &runtime_boundary = result.streaming_boundaries[index];
            const auto &receipt_boundary =
                result.streaming_receipt->boundaries[index];
            actual_check(runtime_boundary.boundary_index == index &&
                             runtime_boundary.id == receipt_boundary.id &&
                             runtime_boundary.source_stage_drained ==
                                 receipt_boundary.source_stage_drained &&
                             runtime_boundary.source_stage_backing_released ==
                                 receipt_boundary.source_stage_backing_released &&
                             runtime_boundary.pending_readers_after == 0 &&
                             runtime_boundary.event_digest ==
                                 receipt_boundary.event_digest,
                         "runtime boundary differs");
        }
    } else {
        actual_check(layout.stages.size() == 1 &&
                         result.streaming_boundaries.empty(),
                     "legacy receipt has multi-stage runtime");
    }
    snapshot_lease->revalidate_after_drain();
    if (result.streaming_runtime)
        *result.streaming_runtime =
            result.streaming_stages.front().runtime;

    PublicStreamingSelectionMetrics metrics;
    metrics.target_request_memory_bytes =
        *execution.selection.exact_selector.target_request_memory_bytes;
    metrics.calibrated_request_bytes =
        record.calibration.calibrated_request_bytes;
    metrics.preset_id = record.id;
    metrics.preset_revision = record.revision;
    metrics.catalog_revision = record.catalog_revision;
    metrics.record_digest = record.canonical_record_digest;
    metrics.resolution_digest = execution.selection.resolution_digest;
    metrics.source_digest = streaming_source_identity_digest(
        execution.probe->source_identity());
    metrics.workload_digest = streaming_workload_identity_digest(
        execution.probe->workload_identity());
    metrics.runtime_digest = streaming_runtime_identity_digest(
        execution.probe->runtime_identity());
    metrics.device_digest = streaming_device_identity_digest(
        execution.selection.device);
    metrics.authorized_layout_digest = record.plan.layout_digest;
    metrics.actual_layout_digest = layout.digest;
    metrics.component_policy_revision =
        record.plan.component_policy_revision;
    metrics.execution_container = record.workload.execution_container;
    metrics.memory_scope = record.calibration.scope;
    metrics.receipt_schema_version =
        result.streaming_receipt->schema_version;
    metrics.receipt_source_generation =
        result.streaming_receipt->stages.front().source_generation;
    metrics.receipt_digest = result.streaming_receipt->canonical_digest;
    metrics.receipt_verifier_revision =
        std::string(actual_receipt_verifier_revision);
    metrics.actual_plan_verified = true;
    result.public_streaming = std::move(metrics);
}

} // namespace tc::streaming
