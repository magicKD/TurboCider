#include "../../native/runtime/streaming/preset_resolver.hpp"
#include "../../native/runtime/streaming/public_runtime.hpp"
#include "../../native/runtime/streaming/public_result.hpp"
#include "../../native/runtime/streaming/actual_receipt.hpp"
#include "../../native/runtime/streaming/run_context.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace tc;
using namespace tc::streaming;

namespace {

std::shared_ptr<const SourceLease> test_lease;

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
    assert(test_lease);
    return {"original-bf16", "safetensors-bf16", digest('a'),
            std::string(test_lease->digest())};
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
    std::shared_ptr<const SourceLease> lease_value = test_lease;

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
    const SourceLease *source_lease() const noexcept override {
        return lease_value.get();
    }
};

StageLayout context_stage(std::string id);

class Snapshot final : public ModelStreamingSnapshot {
  public:
    PresetSourceIdentity source_value = source();
    PresetRuntimeIdentity runtime_value = runtime();
    Descriptor descriptor_value;
    Layout layout_value;
    std::string component = "zimage-components-v1";
    std::shared_ptr<const SourceLease> lease_value = test_lease;
    mutable uint32_t source_revalidations = 0;
    mutable bool source_valid = true;

    explicit Snapshot(std::string layout_digest, bool multi_stage = false) {
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
        PoolLayout pool;
        pool.id = 0;
        pool.layout_class = "z-image-bf16-main-block-v1";
        pool.slots.resize(2);
        for (auto &slot : pool.slots) slot.capacity_bytes = 8;
        stage.pools.push_back(std::move(pool));
        for (uint32_t group = 0; group < 16; ++group)
            stage.groups.push_back(
                {group, 0, group % 2, {group}, {8}, 8});
        layout_value.stages.push_back(std::move(stage));
        if (multi_stage)
            layout_value.stages.push_back(context_stage("upsampler"));
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
    const SourceLease *source_lease() const noexcept override {
        return lease_value.get();
    }
    void revalidate_source() const override {
        ++source_revalidations;
        if (!source_valid)
            throw std::runtime_error("artifact_changed");
    }
};

class ImmediateContextAdapter final : public ModelSlotAdapter {
  public:
    uint64_t sequence = 0;
    bool fail_drain = false;

    void create_pool(const PoolLayout &) override {}

    static int fill(void *, const tc_stream_slot_ticket_v1 *,
                    const std::atomic<bool> *cancel, uint64_t *bytes) {
        if (cancel->load(std::memory_order_acquire)) return -1;
        *bytes = 8;
        return 0;
    }

    FillJob make_fill_job(
            const Group &, const tc_stream_slot_ticket_v1 &ticket) override {
        return {ticket, this, &fill};
    }

    void encode_prefix(uint32_t) override {}
    void prepare_group(const Group &, const tc_stream_slot_ticket_v1 &) override {}

    ReaderSet encode_group(
            const Group &, const tc_stream_slot_ticket_v1 &,
            CompletionMailbox &) override {
        ReaderSet result;
        result.count = 1;
        result.fences[0] = {1, ++sequence};
        result.already_complete = true;
        return result;
    }

    bool drain() noexcept override { return !fail_drain; }
    void destroy_pool() noexcept override {}
};

StageLayout context_stage(std::string id) {
    StageLayout stage;
    stage.id = std::move(id);
    stage.group_size = 1;
    stage.slot_count = 1;
    stage.workers = 1;
    stage.pass_count = 1;
    PoolLayout pool;
    pool.id = 0;
    pool.layout_class = stage.id + "-class";
    pool.slots.push_back({{8}, 8});
    stage.pools.push_back(std::move(pool));
    stage.groups.push_back({0, 0, 0, {0}, {8}, 8});
    return stage;
}

StreamingRuntimeMetrics context_runtime(
        const StageLayout &stage, std::string_view layout_digest) {
    StreamingRuntimeMetrics runtime;
    runtime.implementation = "generic_stage_executor_v2";
    runtime.layout_digest = std::string(layout_digest);
    runtime.stage = stage.id;
    runtime.resident_prefix_blocks = stage.prefix;
    runtime.block_group_size = stage.group_size;
    runtime.slot_count = stage.slot_count;
    runtime.prefetch_distance = stage.distance;
    runtime.io_workers = stage.workers;
    runtime.group_count = static_cast<uint32_t>(stage.groups.size());
    runtime.pass_count = stage.pass_count;
    runtime.pass_transition = "reload";
    runtime.retention = "request";
    runtime.component_policy_revision = "zimage-components-v1";
    runtime.multi_pool_policy = "serial";
    runtime.pool_count = static_cast<uint32_t>(stage.pools.size());
    runtime.slot_bundle_count =
        stage.slot_count * static_cast<uint32_t>(stage.pools.size());
    runtime.refill_worker_count = stage.workers;
    runtime.drained = true;
    return runtime;
}

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

std::shared_ptr<const ActualExecutionReceipt> actual_receipt(
        const Layout &layout, uint64_t source_generation) {
    assert(layout.stages.size() == 1);
    const auto &stage = layout.stages.front();
    constexpr uint64_t request_generation = 71;
    constexpr const char *implementation = "generic_stage_executor_v2";
    ActualReceiptRecorder recorder(
        stage, 0, request_generation,
        {layout.digest, implementation, source_generation});
    uint64_t content_generation = 0;
    uint64_t fence_sequence = 0;
    for (uint32_t pass = 0; pass < stage.pass_count; ++pass) {
        recorder.pool_selected(pass, 0);
        for (uint32_t group = 0; group < stage.groups.size(); ++group) {
            const auto &planned = stage.groups[group];
            tc_stream_slot_ticket_v1 ticket{
                sizeof(ticket), TC_STREAM_SLOT_ABI_V1,
                planned.pool, planned.slot, request_generation,
                ++content_generation, {0, pass, pass, group}};
            recorder.fill_submitted(ticket);
            tc_stream_completion_v1 fill{};
            fill.struct_size = sizeof(fill);
            fill.version = TC_STREAM_SLOT_ABI_V1;
            fill.kind = TC_STREAM_FILL_COMPLETE;
            fill.ticket = ticket;
            fill.bytes = planned.bytes;
            recorder.fill_completed(fill);
            const tc_stream_reader_fence_v1 fence{
                1, ++fence_sequence};
            recorder.readers_issued(ticket, {&fence, 1});
            tc_stream_completion_v1 reader{};
            reader.struct_size = sizeof(reader);
            reader.version = TC_STREAM_SLOT_ABI_V1;
            reader.kind = TC_STREAM_READER_COMPLETE;
            reader.ticket = ticket;
            reader.fence = fence;
            recorder.reader_completed(reader);
        }
        recorder.pass_completed(pass, std::nullopt);
    }
    recorder.drained();
    const auto stage_receipt = recorder.finish();
    return std::make_shared<const ActualExecutionReceipt>(
        make_actual_execution_receipt(
            implementation, layout.digest, "zimage-components-v1",
            std::vector<ActualStageReceipt>{*stage_receipt}));
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
    const auto fixture_root = std::filesystem::temp_directory_path() /
        ("tc-streaming-preset-resolver-" + std::to_string(::getpid()));
    std::filesystem::create_directories(fixture_root);
    const auto fixture_path = fixture_root / "model.safetensors";
    {
        std::ofstream out(fixture_path, std::ios::binary);
        out << "resolver-source-fixture";
    }
    SourceFileIdentity fixture_source;
    fixture_source.logical_id = "model";
    fixture_source.path = fixture_path;
    test_lease = SourceLease::capture(
        std::vector<SourceFileIdentity>{std::move(fixture_source)});

    assert(streaming_target_margin_bytes(8 * gib) == 858993460ull);
    assert(streaming_target_margin_bytes(10 * gib) == gib);
    assert(supported_streaming_target(12 * gib));
    assert(!supported_streaming_target(14 * gib));

    auto slow = record("slower-small", 2, 7 * gib, 100);
    const auto stable_digest = slow.canonical_record_digest;
    assert(stable_digest.size() == 64);
    assert(streaming_preset_record_digest(slow) == stable_digest);
    auto canonical_fixture = record("canonical-fixture", 2, 7 * gib, 100);
    canonical_fixture.source.source_snapshot_digest = digest('b');
    assert(streaming_preset_record_digest(canonical_fixture) ==
           "13b5797176d713924b9c16857acbf0f7a35313d2426a22fdca38c488b3ea3df1");
    auto changed = slow;
    changed.plan.canonical_config = config(12);
    assert(streaming_preset_record_digest(changed) != stable_digest);

    StreamingPresetCatalog catalog{"test-r1", {
        slow,
        record("fast-fit", 1, 8 * gib, 200),
        record("fast-too-large", 0, 10 * gib, 50),
    }};
    // Discovery cannot authorize incomplete or differently tokenized requests.
    auto basic = query(10 * gib, true);
    auto options_request = public_request();
    options_request.fps = 24;
    basic.workload = basic_streaming_workload(options_request,
        basic.workload.device_class, basic.workload.execution_container);
    assert(basic.workload.fps == 0);
    options_request.operation = "video.generate";
    assert(basic_streaming_workload(options_request, "device", "embedded_app").fps == 24);
    basic.source = {};
    basic.runtime = {};
    basic.workload.conditioning_revision.clear();
    basic.workload.vae_policy_revision.clear();
    basic.workload.feature_digest.clear();
    basic.workload.token_shapes.clear();
    const auto candidate = find_streaming_preset_candidate(basic, catalog);
    assert(candidate.candidate && candidate.candidate->id == "fast-fit");
    assert(resolve_streaming_preset(basic, catalog).rejection_code ==
           "unvalidated_workload");
    auto wrong_tokens = query(10 * gib, true);
    ++wrong_tokens.workload.token_shapes.front().valid_rows;
    assert(find_streaming_preset_candidate(wrong_tokens, catalog).candidate);
    assert(resolve_streaming_preset(wrong_tokens, catalog).rejection_code ==
           "unvalidated_workload");
    basic.workload.width += 16;
    assert(!find_streaming_preset_candidate(basic, catalog).candidate);
    basic.workload.width -= 16;
    basic.workload.execution_container = "cli_worker";
    assert(!find_streaming_preset_candidate(basic, catalog).candidate);
    basic.workload.execution_container = "embedded_app";
    basic.physical_memory_bytes = 4 * gib;
    assert(!find_streaming_preset_candidate(basic, catalog).candidate);
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

    Probe missing_lease_probe;
    missing_lease_probe.lease_value.reset();
    rejects([&] {
        (void)PublicPresetResolver::select(
            selector(10 * gib), missing_lease_probe, device(), catalog);
    }, "streaming_source_lease_required");

    auto different_generation = SourceLease::open_and_verify(
        test_lease->descriptor());
    assert(different_generation->digest() == test_lease->digest());
    assert(different_generation->generation() != test_lease->generation());
    Snapshot split_lease_snapshot(selected.record.plan.layout_digest);
    split_lease_snapshot.lease_value = different_generation;
    rejects([&] {
        (void)PublicPresetResolver::authorize(
            selected, probe, split_lease_snapshot, device());
    }, "streaming_source_lease_mismatch");

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
    actual.implementation = "generic_stage_executor_v2";
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
    actual.source_lease_verified = false;
    actual.drained = true;
    run.streaming_runtime = actual;
    run.streaming_receipt = actual_receipt(
        result_snapshot->layout_value, test_lease->generation());
    verify_and_attach_public_streaming_result(execution, run);
    assert(run.public_streaming &&
           run.public_streaming->actual_plan_verified);
    assert(run.streaming_runtime->source_lease_verified &&
           run.streaming_runtime->receipt_schema_version == 2 &&
           run.streaming_runtime->receipt_fills == 144 &&
           run.public_streaming->receipt_digest.size() == 64);
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
    wrong_actual.streaming_receipt.reset();
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_plan_mismatch");
    wrong_actual = run;
    wrong_actual.public_streaming.reset();
    auto damaged_receipt = std::make_shared<ActualExecutionReceipt>(
        *run.streaming_receipt);
    damaged_receipt->stages.front().groups.front().fill_count = 0;
    wrong_actual.streaming_receipt = std::move(damaged_receipt);
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_receipt_mismatch");
    wrong_actual = run;
    wrong_actual.public_streaming.reset();
    wrong_actual.streaming_runtime->drained = false;
    rejects([&] {
        verify_and_attach_public_streaming_result(execution, wrong_actual);
    }, "streaming_actual_plan_mismatch");

    auto context_execution =
        std::make_shared<const ResolvedRequestExecution>(execution);
    std::atomic<bool> context_cancel{false};
    {
        PublicStreamingRunContext context(
            context_execution, context_cancel);
        rejects([&] {
            context.attach_stage(
                0, std::make_shared<ImmediateContextAdapter>(),
                "generic_stage_executor_v2");
        }, "streaming_context_not_attachable");
        assert(context.phase() == PublicRunPhase::created);
    }
    {
        PublicStreamingRunContext context(
            context_execution, context_cancel);
        context.mark_gpu_revalidated();
        auto adapter = std::make_shared<ImmediateContextAdapter>();
        auto &stage = context.attach_stage(
            0, adapter, "generic_stage_executor_v2");
        rejects([&] {
            context.attach_stage(
                0, std::make_shared<ImmediateContextAdapter>(),
                "generic_stage_executor_v2");
        }, "streaming_context_duplicate_stage");
        const auto &planned = context.execution().model_snapshot->layout()
                                  .stages.front();
        for (uint32_t pass = 0; pass < planned.pass_count; ++pass)
            stage.run_pass(pass, pass, context_cancel);
        context.finish_stage(0);
        const auto receipt = context.seal_receipt(
            "generic_stage_executor_v2", "zimage-components-v1");
        assert(receipt && receipt->schema_version == actual_receipt_schema_v2);
        context.revalidate_source_after_drain();
        rejects([&] { context.revalidate_source_after_drain(); },
                "streaming_context_source_phase");
        context.complete();
        assert(context.phase() == PublicRunPhase::completed);
    }
    {
        PublicStreamingRunContext context(
            context_execution, context_cancel);
        context.mark_gpu_revalidated();
        auto adapter = std::make_shared<ImmediateContextAdapter>();
        adapter->fail_drain = true;
        auto &stage = context.attach_stage(
            0, adapter, "generic_stage_executor_v2");
        rejects([&] { stage.run_pass(0, 0, context_cancel); },
                "streaming_pass_drain_failed");
        context.quarantine("injected-drain-failure");
        assert(context.quarantined() &&
               context.quarantine_reason() == "injected-drain-failure");
    }
    {
        auto multi_snapshot = std::make_shared<Snapshot>(
            result_selected.record.plan.layout_digest, true);
        auto multi_authorized = PublicPresetResolver::authorize(
            result_selected, *result_probe, *multi_snapshot, device());
        Request multi_request;
        multi_request.model = "z-image-turbo";
        multi_request.streaming =
            result_selected.record.plan.canonical_config;
        auto multi_execution =
            std::make_shared<const ResolvedRequestExecution>(
                ResolvedRequestExecution{
                    std::move(multi_request), std::move(multi_authorized),
                    result_probe, multi_snapshot, digest('2')});
        PublicStreamingRunContext context(multi_execution, context_cancel);
        context.mark_gpu_revalidated();
        auto first_adapter = std::make_shared<ImmediateContextAdapter>();
        auto &first = context.attach_stage(
            0, first_adapter, "generic_stage_executor_v2");
        const auto &first_layout = multi_snapshot->layout_value.stages[0];
        for (uint32_t pass = 0; pass < first_layout.pass_count; ++pass)
            first.run_pass(pass, pass, context_cancel);
        rejects([&] {
            ActualBoundaryReceipt premature;
            context.record_boundary(std::move(premature));
        }, "streaming_context_boundary_phase");
        context.finish_stage(0);
        rejects([&] {
            context.attach_stage(
                1, std::make_shared<ImmediateContextAdapter>(),
                "generic_stage_executor_v2");
        }, "streaming_context_not_attachable");
        ActualBoundaryReceipt boundary;
        boundary.boundary_index = 0;
        boundary.id = "denoiser-to-upsampler";
        boundary.from_stage_index = 0;
        boundary.to_stage_index = 1;
        boundary.source_generation = test_lease->generation();
        boundary.last_reader_sequence = first_adapter->sequence;
        boundary.completed_reader_sequence = first_adapter->sequence;
        boundary.live_slot_bytes_before = 16;
        boundary.released_slot_bytes = 16;
        boundary.source_stage_drained = true;
        boundary.source_stage_backing_released = true;
        boundary.event_digest = actual_boundary_event_digest(boundary);
        boundary.canonical_digest =
            actual_boundary_canonical_digest(boundary);
        auto pending_boundary = boundary;
        pending_boundary.pending_readers_after = 1;
        pending_boundary.event_digest =
            actual_boundary_event_digest(pending_boundary);
        pending_boundary.canonical_digest =
            actual_boundary_canonical_digest(pending_boundary);
        rejects([&] {
            context.record_boundary(std::move(pending_boundary));
        }, "boundary_pending_readers");
        context.record_boundary(boundary);
        auto second_adapter = std::make_shared<ImmediateContextAdapter>();
        auto &second = context.attach_stage(
            1, second_adapter, "generic_stage_executor_v2");
        const auto &second_layout = multi_snapshot->layout_value.stages[1];
        second.run_pass(0, 0, context_cancel);
        context.finish_stage(1);
        context.drain_all();
        const auto receipt = context.seal_receipt(
            "generic_stage_executor_v2", "zimage-components-v1");
        assert(receipt && receipt->schema_version == actual_receipt_schema_v3 &&
               receipt->stages.size() == 2 &&
               receipt->boundaries.size() == 1);
        context.revalidate_source_after_drain();
        context.complete();

        RunResult multi_run;
        multi_run.streaming_stages.push_back(
            {0, context_runtime(first_layout, multi_snapshot->layout_value.digest)});
        multi_run.streaming_stages.push_back(
            {1, context_runtime(second_layout, multi_snapshot->layout_value.digest)});
        multi_run.streaming_receipt = receipt;
        verify_and_attach_public_streaming_result(
            *multi_execution, multi_run);
        assert(!multi_run.streaming_runtime &&
               multi_run.streaming_stages.size() == 2 &&
               multi_run.streaming_boundaries.size() == 1 &&
               multi_run.streaming_boundaries.front().pending_readers_after == 0 &&
               multi_run.public_streaming &&
               multi_run.public_streaming->actual_plan_verified);
        auto ambiguous_multi = multi_run;
        ambiguous_multi.public_streaming.reset();
        ambiguous_multi.streaming_runtime =
            ambiguous_multi.streaming_stages.front().runtime;
        rejects([&] {
            verify_and_attach_public_streaming_result(
                *multi_execution, ambiguous_multi);
        }, "legacy streaming summary is ambiguous for multi-stage");

        PublicStreamingRunContext incomplete(
            multi_execution, context_cancel);
        incomplete.mark_gpu_revalidated();
        auto incomplete_first_adapter =
            std::make_shared<ImmediateContextAdapter>();
        auto &incomplete_first = incomplete.attach_stage(
            0, incomplete_first_adapter, "generic_stage_executor_v2");
        for (uint32_t pass = 0; pass < first_layout.pass_count; ++pass)
            incomplete_first.run_pass(pass, pass, context_cancel);
        incomplete.finish_stage(0);
        incomplete.record_boundary(boundary);
        incomplete.attach_stage(
            1, std::make_shared<ImmediateContextAdapter>(),
            "generic_stage_executor_v2");
        incomplete.drain_all();
        rejects([&] {
            (void)incomplete.seal_receipt(
                "generic_stage_executor_v2", "zimage-components-v1");
        }, "streaming_context_stage_receipt_missing");
        incomplete.quarantine("missing-stage-receipt");
        assert(incomplete.quarantined());
    }

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
    std::filesystem::remove_all(fixture_root);
}
