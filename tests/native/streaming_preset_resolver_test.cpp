#include "../../native/runtime/streaming/preset_resolver.hpp"
#include "../../native/runtime/streaming/public_runtime.hpp"
#include "../../native/runtime/streaming/public_result.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace tc;
using namespace tc::streaming;

namespace {

std::string digest(char value) {
    return std::string(64, value);
}

StreamingConfig config(uint32_t prefix = 14) {
    StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    StreamingStageConfig stage;
    stage.residency = "streamed";
    stage.block_group_size = 1;
    stage.slot_count = 2;
    stage.resident_prefix_blocks = prefix;
    stage.prefetch_distance = 0;
    stage.io_workers = 1;
    value.stages.emplace("denoiser", std::move(stage));
    return value;
}

PresetSourceIdentity source() {
    return {"original-bf16", "safetensors-bf16", digest('a'), digest('b')};
}

PresetRuntimeIdentity runtime() {
    return {"build-test", "runtime-v1", "adapter-v1",
            "reader-v1", "kernel-v1", "allocator-v1"};
}

PresetWorkload workload() {
    PresetWorkload value;
    value.model = "z-image-turbo";
    value.operation = "image.generate";
    value.execution = "gpu";
    value.device_class = "Apple Test GPU/16GiB";
    value.execution_container = "embedded_app";
    value.width = 512;
    value.height = 512;
    value.frames = 1;
    value.steps = 9;
    value.batch = 1;
    value.dynamic_text = true;
    value.conditioning_revision = "zimage-conditioning-v1";
    value.vae_policy_revision = "zimage-vae-v1";
    value.feature_digest = digest('c');
    value.token_shapes.push_back(
        {"qwen3", "tokenizer-v1", "template-v1", 16, 32, 32});
    return value;
}

StreamingPresetRecord record(const char *id, uint32_t rank,
                             uint64_t calibrated, uint64_t reads,
                             char layout = 'd') {
    StreamingPresetRecord value;
    value.id = id;
    value.revision = 1;
    value.catalog_revision = "test-r1";
    value.source = source();
    value.workload = workload();
    value.runtime = runtime();
    value.device = {16 * gib, 16 * gib};
    value.plan.canonical_config = config();
    value.plan.layout_digest = digest(layout);
    value.plan.component_policy_revision = "zimage-components-v1";
    value.plan.pass_transition = "reload";
    value.plan.multi_pool_policy = "serial";
    value.calibration.complete = true;
    value.calibration.calibrated_request_bytes = calibrated;
    value.calibration.scope = "execution_process_tree_v1";
    value.calibration.estimator_revision = "observed-tree-max-v1";
    value.calibration.calibration_id = "calibration-test-v1";
    value.calibration.execution_container = "embedded_app";
    value.calibration.evidence_digest = digest('e');
    value.calibration.confirmation_sample_count = 13;
    value.calibration.maximum_sample_gap_ns = 20'000'000;
    value.performance.rank = rank;
    value.performance.logical_read_bytes = reads;
    value.performance.profile_id = "performance-test-v1";
    value.performance.comparison_kind = "strategy_tradeoff";
    value.performance.confidence_status = "pass";
    value.performance.evidence_digest = digest('f');
    value.release.channel = "public-experimental";
    value.release.reviewed_commit = "test-review-commit";
    value.release.review_digest = digest('1');
    return finalize_streaming_preset_record(std::move(value));
}

PresetResolveQuery query(uint64_t target, bool exact_identity = false) {
    PresetResolveQuery value;
    value.source = source();
    value.workload = workload();
    value.runtime = runtime();
    value.target_request_memory_bytes = target;
    value.physical_memory_bytes = 16 * gib;
    value.require_exact_identity = exact_identity;
    return value;
}

StreamingSelector selector(uint64_t target) {
    StreamingSelector value;
    value.schema_version = 2;
    value.enabled = true;
    value.selection = "memory_tier";
    value.retention = "request";
    value.target_request_memory_bytes = target;
    return value;
}

StreamingDeviceIdentity device() {
    return {"Apple Test GPU", "Apple Test GPU/16GiB", "test-os", 16 * gib};
}

class Probe final : public ModelStreamingProbe {
  public:
    PresetSourceIdentity source_value = source();
    PresetWorkload workload_value = workload();
    PresetRuntimeIdentity runtime_value = runtime();

    std::string_view model_id() const noexcept override {
        return workload_value.model;
    }
    const PresetSourceIdentity &source_identity() const noexcept override {
        return source_value;
    }
    const PresetWorkload &workload_identity() const noexcept override {
        return workload_value;
    }
    const PresetRuntimeIdentity &runtime_identity() const noexcept override {
        return runtime_value;
    }
    std::string_view component_policy_revision() const noexcept override {
        return "zimage-components-v1";
    }
};

class Snapshot final : public ModelStreamingSnapshot {
  public:
    PresetSourceIdentity source_value = source();
    PresetRuntimeIdentity runtime_value = runtime();
    Descriptor descriptor_value;
    Layout layout_value;
    std::string component = "zimage-components-v1";
    mutable uint32_t source_revalidations = 0;
    mutable bool source_valid = true;

    explicit Snapshot(std::string layout_digest) {
        descriptor_value.model = "z-image-turbo";
        layout_value.digest = std::move(layout_digest);
        layout_value.materializations_complete = true;
        StageLayout stage;
        stage.id = "denoiser";
        stage.prefix = 14;
        stage.group_size = 1;
        stage.slot_count = 2;
        stage.distance = 0;
        stage.workers = 1;
        stage.pass_count = 9;
        stage.pass_transition = PassTransition::reload;
        stage.multi_pool_policy = MultiPoolPolicy::serial;
        stage.groups.resize(16);
        PoolLayout pool;
        pool.id = 0;
        pool.layout_class = "z-image-bf16-main-block-v1";
        pool.slots.resize(2);
        stage.pools.push_back(std::move(pool));
        layout_value.stages.push_back(std::move(stage));
    }
    std::string_view model_id() const noexcept override {
        return descriptor_value.model;
    }
    const PresetSourceIdentity &source_identity() const noexcept override {
        return source_value;
    }
    const PresetRuntimeIdentity &runtime_identity() const noexcept override {
        return runtime_value;
    }
    const Descriptor &descriptor() const noexcept override {
        return descriptor_value;
    }
    const Layout &layout() const noexcept override {
        return layout_value;
    }
    std::string_view component_policy_revision() const noexcept override {
        return component;
    }
    void revalidate_source() const override {
        ++source_revalidations;
        if (!source_valid)
            throw std::runtime_error("artifact_changed");
    }
};

class FixedCatalogProvider final : public StreamingCatalogProvider {
  public:
    StreamingPresetCatalog value;
    mutable uint32_t snapshots = 0;

    explicit FixedCatalogProvider(StreamingPresetCatalog catalog)
        : value(std::move(catalog)) {}
    std::shared_ptr<const StreamingPresetCatalog>
    snapshot() const override {
        ++snapshots;
        return std::make_shared<const StreamingPresetCatalog>(value);
    }
};

class PublicSession final : public ModelSession {
  public:
    mutable uint32_t probe_calls = 0;
    mutable uint32_t compile_calls = 0;
    mutable std::shared_ptr<Snapshot> snapshot;

    std::shared_ptr<const ModelStreamingProbe> probe_public_streaming(
            const PublicResolveInput &) const override {
        ++probe_calls;
        return std::make_shared<Probe>();
    }
    std::shared_ptr<const ModelStreamingSnapshot> compile_public_streaming(
            std::shared_ptr<const ModelStreamingProbe>,
            const StreamingPresetRecord &selected) const override {
        ++compile_calls;
        snapshot = std::make_shared<Snapshot>(
            selected.plan.layout_digest);
        return snapshot;
    }
    RunResult generate(const Request &, const Event &,
                       std::atomic<bool> &) override {
        throw std::runtime_error("fake session does not generate");
    }
    void unload() override {}
};

Request public_request(uint64_t target = 10 * gib) {
    Request value;
    value.model = "z-image-turbo";
    value.operation = "image.generate";
    value.execution = "gpu";
    value.width = 512;
    value.height = 512;
    value.frames = 1;
    value.steps = 9;
    value.audio = false;
    value.dynamic_text = true;
    value.streaming_selector = selector(target);
    value.streaming_selector_requested = value.streaming_selector;
    return value;
}

template <class Function>
void rejects(Function &&function, const char *message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(message) != std::string::npos;
    }
    assert(rejected);
}

} // namespace

int main() {
    assert(streaming_target_margin_bytes(8 * gib) == 858993460ull);
    assert(streaming_target_margin_bytes(10 * gib) == gib);
    assert(supported_streaming_target(12 * gib));
    assert(!supported_streaming_target(14 * gib));

    auto slow = record("slower-small", 2, 7 * gib, 100);
    const auto stable_digest = slow.canonical_record_digest;
    assert(stable_digest.size() == 64);
    assert(streaming_preset_record_digest(slow) == stable_digest);
    auto changed = slow;
    changed.plan.canonical_config = config(12);
    assert(streaming_preset_record_digest(changed) != stable_digest);

    StreamingPresetCatalog catalog{"test-r1", {
        slow,
        record("fast-fit", 1, 8 * gib, 200),
        record("fast-too-large", 0, 10 * gib, 50),
    }};
    auto ten = resolve_streaming_preset(query(10 * gib), catalog);
    assert(ten.selected && ten.selected->id == "fast-fit");
    auto twelve = resolve_streaming_preset(query(12 * gib), catalog);
    assert(twelve.selected && twelve.selected->id == "fast-too-large");

    auto exact_query = query(10 * gib, true);
    exact_query.preset_id = "slower-small";
    exact_query.preset_revision = 1;
    exact_query.catalog_revision = "test-r1";
    auto exact = resolve_streaming_preset(exact_query, catalog);
    assert(exact.selected && exact.selected->id == "slower-small");

    exact_query.catalog_revision = "old-r0";
    assert(resolve_streaming_preset(exact_query, catalog).rejection_code ==
           "streaming_resolution_stale");

    assert(resolve_streaming_preset(query(14 * gib), catalog).rejection_code ==
           "unsupported_memory_target");

    auto wrong_memory = query(10 * gib);
    wrong_memory.physical_memory_bytes = 12 * gib;
    assert(resolve_streaming_preset(wrong_memory, catalog).rejection_code ==
           "unvalidated_device");

    auto wrong_source = query(10 * gib, true);
    wrong_source.source.source_snapshot_digest = digest('9');
    assert(resolve_streaming_preset(wrong_source, catalog).rejection_code ==
           "artifact_verification_required");

    auto revoked = record("revoked", 0, 7 * gib, 0);
    revoked.release.revoked = true;
    revoked.release.channel = "revoked";
    revoked = finalize_streaming_preset_record(std::move(revoked));
    StreamingPresetCatalog revoked_catalog{"test-r1", {revoked}};
    assert(resolve_streaming_preset(query(10 * gib), revoked_catalog)
               .rejection_code == "preset_not_public");

    auto invalid = record("invalid", 0, 7 * gib, 0);
    invalid.calibration.complete = false;
    invalid = finalize_streaming_preset_record(std::move(invalid));
    StreamingPresetCatalog invalid_catalog{"test-r1", {invalid}};
    rejects([&] { resolve_streaming_preset(query(10 * gib), invalid_catalog); },
            "calibration is incomplete");

    Probe probe;
    auto selected = PublicPresetResolver::select(
        selector(10 * gib), probe, device(), catalog);
    assert(selected.record.id == "fast-fit");
    Snapshot snapshot(selected.record.plan.layout_digest);
    auto authorized = PublicPresetResolver::authorize(
        selected, probe, snapshot, device());
    assert(authorized.exact_selector.selection == "preset");
    assert(authorized.exact_selector.preset_id == "fast-fit");
    assert(authorized.resolution_digest.size() == 64);
    assert(authorized.authority->matches(
        authorized.record, snapshot, authorized.device));

    auto replay = authorized.exact_selector;
    replay.expected_resolution_digest = digest('0');
    auto stale = PublicPresetResolver::select(replay, probe, device(), catalog);
    rejects([&] {
        PublicPresetResolver::authorize(stale, probe, snapshot, device());
    }, "streaming_resolution_stale");

    Snapshot mismatched(digest('8'));
    rejects([&] {
        PublicPresetResolver::authorize(selected, probe, mismatched, device());
    }, "streaming_actual_plan_mismatch");

    auto result_probe = std::make_shared<Probe>();
    auto result_selected = PublicPresetResolver::select(
        selector(10 * gib), *result_probe, device(), catalog);
    auto result_snapshot = std::make_shared<Snapshot>(
        result_selected.record.plan.layout_digest);
    auto result_authorized = PublicPresetResolver::authorize(
        result_selected, *result_probe, *result_snapshot, device());
    Request result_request;
    result_request.model = "z-image-turbo";
    result_request.streaming = result_selected.record.plan.canonical_config;
    ResolvedRequestExecution execution{
        std::move(result_request), std::move(result_authorized),
        result_probe, result_snapshot, digest('2')};
    RunResult run;
    StreamingRuntimeMetrics actual;
    actual.implementation = "generic_stage_executor_v1";
    actual.layout_digest = result_selected.record.plan.layout_digest;
    actual.stage = "denoiser";
    actual.resident_prefix_blocks = 14;
    actual.block_group_size = 1;
    actual.slot_count = 2;
    actual.prefetch_distance = 0;
    actual.io_workers = 1;
    actual.group_count = 16;
    actual.pass_count = 9;
    actual.pass_transition = "reload";
    actual.retention = "request";
    actual.component_policy_revision = "zimage-components-v1";
    actual.multi_pool_policy = "serial";
    actual.pool_count = 1;
    actual.slot_bundle_count = 2;
    actual.refill_worker_count = 1;
    actual.source_lease_verified = true;
    actual.drained = true;
    run.streaming_runtime = actual;
    verify_and_attach_public_streaming_result(execution, run);
    assert(run.public_streaming &&
           run.public_streaming->actual_plan_verified);
    assert(run.public_streaming->preset_id == "fast-fit");
    assert(run.public_streaming->authorized_layout_digest ==
           run.public_streaming->actual_layout_digest);

    auto wrong_actual = run;
    wrong_actual.public_streaming.reset();
    wrong_actual.streaming_runtime->slot_count = 3;
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_plan_mismatch");
    wrong_actual = run;
    wrong_actual.public_streaming.reset();
    wrong_actual.streaming_runtime->source_lease_verified = false;
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_plan_mismatch");
    wrong_actual = run;
    wrong_actual.public_streaming.reset();
    wrong_actual.streaming_runtime->drained = false;
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_plan_mismatch");

    FixedCatalogProvider provider(catalog);
    PublicSession public_session;
    PublicStreamingCoordinator coordinator(
        public_session, "z-image-turbo", "embedded_app", provider);
    auto public_input = public_request();
    auto preflight = coordinator.preflight(public_input);
    auto coordinated = coordinator.resolve_normalized(
        std::move(public_input), device(), std::move(preflight));
    assert(coordinated && public_session.probe_calls == 1 &&
           public_session.compile_calls == 1);
    assert(provider.snapshots == 1);
    assert(!coordinated->request.streaming_selector &&
           coordinated->request.streaming.active());
    assert(coordinated->selection.exact_selector.preset_id == "fast-fit");
    coordinator.revalidate(*coordinated, device());
    assert(provider.snapshots == 2);
    assert(public_session.snapshot &&
           public_session.snapshot->source_revalidations == 1);

    public_session.snapshot->source_valid = false;
    rejects([&] { coordinator.revalidate(*coordinated, device()); },
            "artifact_changed");
    public_session.snapshot->source_valid = true;

    FixedCatalogProvider pinned_provider(catalog);
    PublicSession pinned_session;
    PublicStreamingCoordinator pinned_coordinator(
        pinned_session, "z-image-turbo", "embedded_app", pinned_provider);
    auto pinned_request = public_request();
    auto pinned_preflight = pinned_coordinator.preflight(pinned_request);
    pinned_provider.value.records.clear();
    auto pinned_execution = pinned_coordinator.resolve_normalized(
        std::move(pinned_request), device(), std::move(pinned_preflight));
    assert(pinned_execution &&
           pinned_execution->selection.record.id == "fast-fit" &&
           pinned_provider.snapshots == 1);
    rejects([&] {
        pinned_coordinator.revalidate(*pinned_execution, device());
    }, "catalog_has_no_public_records");
    assert(pinned_provider.snapshots == 2);

    FixedCatalogProvider mismatch_provider(catalog);
    PublicSession mismatch_session;
    PublicStreamingCoordinator mismatch_coordinator(
        mismatch_session, "z-image-turbo", "embedded_app",
        mismatch_provider);
    auto changed_request = public_request();
    auto changed_preflight =
        mismatch_coordinator.preflight(changed_request);
    ++changed_request.steps;
    rejects([&] {
        mismatch_coordinator.resolve_normalized(
            std::move(changed_request), device(),
            std::move(changed_preflight));
    }, "streaming_preflight_mismatch");
    assert(mismatch_provider.snapshots == 1 &&
           mismatch_session.probe_calls == 0 &&
           mismatch_session.compile_calls == 0);

    FixedCatalogProvider prompt_provider(catalog);
    PublicSession prompt_session;
    PublicStreamingCoordinator prompt_coordinator(
        prompt_session, "z-image-turbo", "embedded_app", prompt_provider);
    auto prompt_request = public_request();
    auto prompt_preflight = prompt_coordinator.preflight(prompt_request);
    prompt_request.prompt = "changed after preflight";
    rejects([&] {
        prompt_coordinator.resolve_normalized(
            std::move(prompt_request), device(),
            std::move(prompt_preflight));
    }, "streaming_preflight_mismatch");
    assert(prompt_session.probe_calls == 0 &&
           prompt_session.compile_calls == 0);

    FixedCatalogProvider empty_provider({"empty-test", {}});
    PublicSession unopened_session;
    PublicStreamingCoordinator empty_coordinator(
        unopened_session, "z-image-turbo", "embedded_app", empty_provider);
    rejects([&] { empty_coordinator.preflight(public_request()); },
            "catalog_has_no_public_records");
    assert(empty_provider.snapshots == 1);
    assert(unopened_session.probe_calls == 0 &&
           unopened_session.compile_calls == 0);

    auto wrong_model_request = public_request();
    wrong_model_request.model = "flux2-klein-9b";
    rejects([&] { coordinator.preflight(wrong_model_request); },
            "streaming_engine_model_mismatch");

    const auto production_snapshot_a =
        production_streaming_catalog_provider().snapshot();
    const auto production_snapshot_b =
        production_streaming_catalog_provider().snapshot();
    assert(production_snapshot_a && production_snapshot_b &&
           production_snapshot_a == production_snapshot_b);
    assert(production_streaming_preset_catalog().records.empty());
    assert(resolve_streaming_preset(
        query(10 * gib), production_streaming_preset_catalog()).rejection_code ==
        "catalog_has_no_public_records");
    std::cout << "PASS public preset runtime/result: canonical identity, "
                 "deterministic rank, coordinator/provider, internal "
                 "authority, source revalidation, actual-plan/drain and "
                 "device/calibration/revocation fail-closed\n";
}
