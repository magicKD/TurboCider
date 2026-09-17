#include "public_result.hpp"

#include "../../core/common.hpp"

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

} // namespace

void verify_and_attach_public_streaming_result(
        const ResolvedRequestExecution &execution, RunResult &result) {
    actual_check(execution.selection.authority != nullptr,
                 "missing public authority");
    actual_check(result.streaming_runtime.has_value(),
                 "adapter did not report streaming execution");
    actual_check(execution.model_snapshot != nullptr,
                 "missing model snapshot");

    const auto &record = execution.selection.record;
    const auto &layout = execution.model_snapshot->layout();
    const auto &actual = *result.streaming_runtime;
    actual_check(layout.materializations_complete,
                 "authorized layout has incomplete source metadata");
    actual_check(layout.digest == record.plan.layout_digest,
                 "authorized layout differs from preset");
    actual_check(layout.stages.size() == 1,
                 "public v1 requires one streamed stage");
    const auto &stage = layout.stages.front();
    actual_check(!stage.resident, "authorized stage is resident");

    uint64_t expected_slot_bundles = 0;
    for (const auto &pool : stage.pools)
        expected_slot_bundles += pool.slots.size();

    actual_check(actual.implementation.size() > 0,
                 "missing implementation identity");
    actual_check(actual.stage == stage.id, "stage identity differs");
    actual_check(actual.layout_digest == layout.digest,
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
                     pass_transition_name(stage.pass_transition) &&
                     actual.pass_transition == record.plan.pass_transition,
                 "pass transition differs");
    actual_check(actual.multi_pool_policy ==
                     multi_pool_policy_name(stage.multi_pool_policy) &&
                     actual.multi_pool_policy == record.plan.multi_pool_policy,
                 "multi-pool policy differs");
    actual_check(actual.component_policy_revision ==
                     record.plan.component_policy_revision &&
                     actual.component_policy_revision ==
                         execution.model_snapshot
                             ->component_policy_revision(),
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
    const auto *probe_lease = execution.probe->source_lease();
    const auto *snapshot_lease = execution.model_snapshot->source_lease();
    actual_check(probe_lease != nullptr && snapshot_lease != nullptr,
                 "missing source lease");
    actual_check(probe_lease == snapshot_lease &&
                     probe_lease->generation() != 0,
                 "source lease generation differs");
    snapshot_lease->revalidate_after_drain();
    actual_check(actual.source_lease_verified,
                 "adapter did not use the revalidated source lease");

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
    metrics.actual_layout_digest = actual.layout_digest;
    metrics.component_policy_revision =
        record.plan.component_policy_revision;
    metrics.execution_container = record.workload.execution_container;
    metrics.memory_scope = record.calibration.scope;
    metrics.actual_plan_verified = true;
    result.public_streaming = std::move(metrics);
}

} // namespace tc::streaming
