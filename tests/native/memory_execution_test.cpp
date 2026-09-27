#include "memory_execution.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace tc;

namespace {

constexpr const char *checkpoint =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char *evidence =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

AllocationSiteSpec site(std::string id, MemoryClass memory_class,
                        bool aliasable = false) {
    AllocationSiteSpec result;
    result.site_id = std::move(id);
    result.component = "synthetic";
    result.stage = "test";
    result.memory_class = memory_class;
    result.lifetime = AllocationLifetime::Stage;
    result.provenance = UpperProvenance::ExactShapeFormula;
    result.aliasable = aliasable;
    return result;
}

MemoryManifest manifest() {
    MemoryManifest result;
    result.candidate_id = "synthetic_streamed_v1";
    result.checkpoint_digest = checkpoint;
    result.backend_revision = "synthetic-backend-v1";
    result.runtime_revision = memory_runtime_revision();
    result.sites = {
        site("synthetic.weights", MemoryClass::Weights, true),
        site("synthetic.activation", MemoryClass::Activation),
    };
    result.instances = {
        {"synthetic.weights", 0, 40, 1, 3, 7, "weights.complete"},
        {"synthetic.weights", 1, 30, 3, 4, 7, "weights.complete"},
        {"synthetic.activation", 0, 20, 2, 3, 0,
         "activation.complete"},
    };
    return result;
}

MemoryCandidateKey key() {
    return {
        "synthetic_streamed_v1", "synthetic", checkpoint,
        "c_metal", "bf16", "test", "video.generate",
        "64x64x1.s1.audio0.input-none.lora-none", "gpu_euler",
        2, "none", memory_runtime_revision(), "AppleGPU-Test",
    };
}

ExecutionPlan plan(uint64_t budget = 75) {
    Request request;
    request.model = "synthetic";
    request.operation = "video.generate";
    request.memory_constrained.enabled = true;
    request.memory_constrained.specified_fields = MemoryFieldEnabled |
        MemoryFieldLimit;
    EffectiveMemoryPolicy policy;
    policy.enabled = true;
    policy.route_available = true;
    policy.adapter_candidate = "synthetic_streamed_v1";
    policy.refill_slots = 2;
    policy.effective_budget_bytes = budget;
    policy.estimate_fits = true;
    policy.digest = "preview-policy";
    return {request, {"synthetic", {}, true}, {}, policy};
}

MemoryCapabilityRecord certified_record(uint64_t maximum_upper = 75) {
    MemoryCapabilityRecord record;
    record.key = key();
    record.level = MemoryCapabilityLevel::L3Certified;
    record.state = MemoryCertificationState::Certified;
    record.manifest_digest = manifest().digest();
    record.evidence_digest = evidence;
    record.verified_refill_slots = {2};
    record.framework_upper_bytes = 5;
    record.framework_provenance = "synthetic-envelope-v1";
    record.maximum_validated_upper_bytes = maximum_upper;
    record.release_enabled = true;
    return record;
}

class ProbeSession final : public ModelSession {
  public:
    explicit ProbeSession(std::optional<MemoryCapabilityProbe> probe)
        : probe_(std::move(probe)) {}

    std::optional<MemoryCapabilityProbe> probe_memory_capability(
        const ExecutionPlan &, const MemoryDeviceIdentity &) const override {
        ++probe_calls;
        return probe_;
    }
    RunResult generate(const Request &, const Event &,
                       std::atomic<bool> &) override {
        return {};
    }
    void unload() override {}

    mutable unsigned probe_calls = 0;

  private:
    std::optional<MemoryCapabilityProbe> probe_;
};

class BindingSession final : public ModelSession {
  public:
    void set_memory_admission(MemoryAdmission *value) override {
        events.push_back(value ? "admission.bind" : "admission.clear");
        admission = value;
    }
    void bind_memory_context(MemoryExecutionContext *value) override {
        events.push_back("context.bind");
        assert(admission != nullptr);
        context = value;
        if (throw_on_bind)
            throw std::runtime_error("synthetic bind failure");
    }
    void unbind_memory_context() noexcept override {
        events.push_back("context.unbind");
        context = nullptr;
    }
    MemoryDrainResult drain_memory_completions(
            MemoryExecutionContext &value,
            std::chrono::milliseconds) override {
        events.push_back("context.drain");
        assert(context == &value);
        assert(admission == &value.admission());
        return {};
    }
    RunResult generate(const Request &, const Event &,
                       std::atomic<bool> &) override {
        return {};
    }
    void unload() override {}

    MemoryAdmission *admission = nullptr;
    MemoryExecutionContext *context = nullptr;
    bool throw_on_bind = false;
    std::vector<std::string> events;
};

template <typename Function>
void expect_rejected(Function function, const char *needle) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(needle) != std::string::npos;
    }
    assert(rejected);
}

MemoryDeviceIdentity device() {
    return {"AppleGPU-Test", memory_runtime_revision()};
}

std::unique_ptr<MemoryExecutionContext> execution_context(
        ProcessMemoryObserver process_observer = {},
        SwapActivityObservation swap_baseline = {},
        SwapActivityObserver swap_observer = {}) {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    auto result = std::make_unique<MemoryExecutionContext>(
        *value.memory_policy, std::move(compiled), baseline,
        std::move(process_observer), std::move(swap_baseline),
        std::move(swap_observer));
    result->begin_running();
    return result;
}

std::vector<MemoryScheduleBindingSpec> strict_schedule() {
    MemoryScheduleBindingSpec begin;
    begin.key.stage = TC_MEMORY_STAGE_ADMISSION;
    begin.key.action = TC_MEMORY_ACTION_END;
    MemoryScheduleBindingSpec compute;
    compute.key.stage = TC_MEMORY_STAGE_DENOISER;
    compute.key.action = TC_MEMORY_ACTION_COMPUTE;
    compute.key.branch = TC_MEMORY_BRANCH_VIDEO;
    compute.key.step = 0;
    compute.key.block = 0;
    compute.expected_slot = 0;
    compute.epoch = 2;
    MemoryScheduleBindingSpec terminal;
    terminal.key.stage = TC_MEMORY_STAGE_TERMINAL_DRAIN;
    terminal.key.action = TC_MEMORY_ACTION_END;
    terminal.epoch = 4;
    terminal.required_flags = TC_MEMORY_EVENT_SAFE_POINT |
        TC_MEMORY_EVENT_GPU_DRAINED | TC_MEMORY_EVENT_TERMINAL;
    terminal.checkpoint_after = true;
    return {begin, compute, terminal};
}

std::unique_ptr<MemoryExecutionContext> scheduled_execution_context(
        ProcessMemoryObserver observer = {}) {
    auto value = plan();
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    auto record = certified_record();
    record.require_explicit_schedule = true;
    record.schedule = strict_schedule();
    registry.add(record);
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    assert(compiled.require_explicit_epoch);
    assert(compiled.require_explicit_schedule);
    assert(compiled.schedule_bindings.size() == 3);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    if (!observer) {
        observer = [] { return ProcessMemoryObservation{
            true, 10, 1024, 512, "synthetic"}; };
    }
    auto result = std::make_unique<MemoryExecutionContext>(
        *value.memory_policy, std::move(compiled), baseline,
        std::move(observer));
    result->begin_running();
    return result;
}

tc_memory_schedule_event_v1 schedule_event(
        uint32_t stage, uint32_t action,
        uint32_t flags = 0, uint32_t slot = TC_MEMORY_INDEX_NONE,
        uint32_t step = TC_MEMORY_INDEX_NONE,
        uint32_t block = TC_MEMORY_INDEX_NONE,
        uint32_t branch = TC_MEMORY_BRANCH_COMMON) {
    tc_memory_schedule_event_v1 event{};
    event.struct_size = sizeof(event);
    event.version = TC_MEMORY_SCHEDULE_EVENT_VERSION_1;
    event.stage = stage;
    event.action = action;
    event.step = step;
    event.block = block;
    event.tile = TC_MEMORY_INDEX_NONE;
    event.branch = branch;
    event.slot = slot;
    event.flags = flags;
    return event;
}

void test_production_catalog_is_empty_and_missing_probe_fails_closed() {
    assert(production_memory_capability_registry().size() == 0);
    ProbeSession session(std::nullopt);
    auto value = plan();
    MemoryCapabilityRegistry registry;
    expect_rejected(
        [&] { (void)preflight_memory_capability(
                  session, value, device(), registry); },
        "capability probe is unavailable");
    assert(session.probe_calls == 1);
    assert(!value.memory_policy->execution_supported);
}

void test_exact_match_authorizes_manifest_plan() {
    auto probe = MemoryCapabilityProbe{key(), manifest()};
    ProbeSession session(probe);
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    assert(compiled.complete && compiled.fits_budget);
    assert(compiled.planned_peak_bytes == 75);
    assert(value.memory_policy->execution_supported);
    assert(value.memory_policy->release_stable);
    assert(value.memory_policy->planned_upper_bytes == 75);
    assert(value.memory_policy->framework_upper_bytes == 5);
    assert(value.memory_policy->estimate_provenance ==
           "manifest_live_interval_v1");
    assert(value.memory_policy->manifest_digest == manifest().digest());
    assert(value.memory_policy->digest.size() == 64);
}

void test_identity_slot_and_release_gates_are_exact() {
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto mismatched_key = key();
    mismatched_key.refill_slots = 3;
    ProbeSession mismatched(
        MemoryCapabilityProbe{mismatched_key, manifest()});
    auto value = plan();
    expect_rejected(
        [&] { (void)preflight_memory_capability(
                  mismatched, value, device(), registry); },
        "refill-slot mismatch");

    ProbeSession exact(MemoryCapabilityProbe{key(), manifest()});
    auto wrong_device = device();
    wrong_device.device_family = "AppleGPU-Other";
    expect_rejected(
        [&] { (void)preflight_memory_capability(
                  exact, value, wrong_device, registry); },
        "device family mismatch");

    MemoryCapabilityRegistry experimental;
    auto record = certified_record();
    record.level = MemoryCapabilityLevel::EnvelopeValidated;
    record.state = MemoryCertificationState::ExperimentalGuarded;
    record.evidence_digest.clear();
    record.release_enabled = false;
    experimental.add(record);
    expect_rejected(
        [&] { (void)preflight_memory_capability(
                  exact, value, device(), experimental); },
        "manifest or release identity mismatch");
    auto accepted = preflight_memory_capability(
        exact, value, device(), experimental, true);
    assert(accepted.record.state ==
           MemoryCertificationState::ExperimentalGuarded);
}

void test_budget_and_validated_upper_are_independent_gates() {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto too_small = plan(74);
    auto resolution = preflight_memory_capability(
        session, too_small, device(), registry);
    expect_rejected(
        [&] { (void)authorize_memory_capability(
                  too_small, resolution, 10); },
        "planned peak 75 exceeds budget 74");
    assert(!too_small.memory_policy->execution_supported);

    MemoryCapabilityRegistry narrow_registry;
    narrow_registry.add(certified_record(74));
    auto narrow = plan();
    auto narrow_resolution = preflight_memory_capability(
        session, narrow, device(), narrow_registry);
    expect_rejected(
        [&] { (void)authorize_memory_capability(
                  narrow, narrow_resolution, 10); },
        "exceeds the validated capability upper");
    assert(!narrow.memory_policy->execution_supported);
}

void test_execution_context_owns_admission_scheduler_and_finish_gate() {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    MemoryExecutionContext context(
        *value.memory_policy, std::move(compiled), baseline);
    context.begin_running();
    assert(context.metrics().initial_process_footprint_bytes == 10);
    assert(context.metrics().framework_upper_bytes == 5);
    assert(context.metrics().planned_process_upper_bytes == 75);
    assert(context.metrics().allocation_ceiling_bytes == 70);
    context.enter_epoch(2, "synthetic.activation");
    {
        auto stage = context.scheduler().begin_stage("synthetic");
        auto reservation = context.try_reserve_site(
            "synthetic.activation", MemoryClass::Activation, 20);
        assert(reservation);
        StorageId id{9, 101, 20, 3};
        auto lease = reservation->commit(id);
        stage.adopt(std::move(lease));
        assert(context.admission().snapshot().storage_bytes == 20);
    }
    assert(context.admission().snapshot().storage_bytes == 0);
    context.begin_draining();
    context.finish_success();
    assert(context.finished());
    assert(!context.tainted());
    assert(context.state() == MemoryExecutionState::Succeeded);
}

void test_execution_state_transitions_are_explicit_and_terminal() {
    auto context = execution_context();
    assert(context->state() == MemoryExecutionState::Running);
    expect_rejected(
        [&] { context->begin_running(); },
        "only start after admission");
    context->begin_draining();
    assert(context->state() == MemoryExecutionState::Draining);
    context->finish_success();
    assert(context->state() == MemoryExecutionState::Succeeded);
    const auto metrics = context->metrics();
    assert(metrics.trace_event_count >= 4);
    assert(!metrics.trace_overflowed);
    const auto trace = context->drain_trace();
    assert(!trace.empty());
    assert(trace.back().kind == MemoryTraceEventKind::State);
    expect_rejected(
        [&] { context->begin_running(); },
        "only start after admission");
    expect_rejected(
        [&] { context->begin_draining(); },
        "only drain after running");
}

void test_terminal_report_is_idempotent_and_bounded() {
    auto context = execution_context();
    context->begin_draining();
    context->finish_success();
    const auto first = context->take_report();
    assert(first.metrics.execution_state == "succeeded");
    assert(!first.trace.empty());
    const auto second = context->take_report();
    assert(second.metrics.execution_state == first.metrics.execution_state);
    assert(second.trace.size() == first.trace.size());
    assert(second.trace.front().sequence == first.trace.front().sequence);
    const auto drained = context->drain_trace();
    assert(drained.size() == first.trace.size());
}

void test_certified_context_requires_explicit_epoch_before_allocation() {
    auto context = execution_context();
    expect_rejected(
        [&] { (void)context->try_reserve_site(
                  "synthetic.activation", MemoryClass::Activation, 20); },
        "explicit epoch transition is required");
    assert(context->current_epoch() == 0);
    assert(context->metrics().automatic_epoch_transition_count == 0);
    context->enter_epoch(2, "explicit.activation");
    auto reservation = context->try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(reservation);
    reservation->cancel();
    auto metrics = context->metrics();
    assert(metrics.explicit_epoch_transition_count == 1);
    assert(metrics.automatic_epoch_transition_count == 0);
    assert(metrics.current_epoch == 2);
    assert(metrics.planned_peak_epoch == 2);
    context->begin_draining();
    context->finish_success();
}

void test_noncertified_compatibility_plan_records_automatic_epoch() {
    auto value = plan();
    value.memory_policy->planned_upper_bytes = 75;
    value.memory_policy->framework_upper_bytes = 5;
    auto compiled = compile_memory_plan(manifest(), 10, 5, 75, false);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    MemoryExecutionContext context(
        *value.memory_policy, std::move(compiled), baseline);
    context.begin_running();
    const auto hooks = context.make_schedule_hooks();
    assert(hooks.user == nullptr);
    assert(hooks.emit == nullptr);
    auto reservation = context.try_reserve_site(
        "synthetic.weights", MemoryClass::Weights, 35);
    assert(reservation);
    assert(context.current_epoch() == 1);
    reservation->cancel();
    const auto metrics = context.metrics();
    assert(metrics.explicit_epoch_transition_count == 0);
    assert(metrics.automatic_epoch_transition_count == 1);
    assert(metrics.schedule_cursor_state == "disabled");
    context.begin_draining();
    context.finish_success();
}

void test_explicit_schedule_drives_epoch_and_terminal_gate() {
    unsigned observation_count = 0;
    auto context = scheduled_execution_context([&] {
        ++observation_count;
        return ProcessMemoryObservation{
            true, 10, 1024, 512, "synthetic"};
    });
    auto hooks = context->make_schedule_hooks();
    char error[256] = {};
    auto begin = schedule_event(
        TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END);
    assert(hooks.emit(hooks.user, &begin, error, sizeof(error)) == 1);
    auto compute = schedule_event(
        TC_MEMORY_STAGE_DENOISER, TC_MEMORY_ACTION_COMPUTE,
        0, 0, 0, 0, TC_MEMORY_BRANCH_VIDEO);
    assert(hooks.emit(hooks.user, &compute, error, sizeof(error)) == 1);
    assert(context->current_epoch() == 2);
    auto reservation = context->try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(reservation);
    reservation->cancel();
    context->emit_terminal_schedule_event();
    context->begin_draining();
    context->finish_success();
    const auto metrics = context->metrics();
    assert(metrics.schedule_event_count == 3);
    assert(metrics.schedule_event_attempted_count == 3);
    assert(metrics.schedule_event_rejected_count == 0);
    assert(metrics.schedule_expected_event_count == 3);
    assert(metrics.schedule_mismatch_count == 0);
    assert(metrics.schedule_next_sequence == 3);
    assert(metrics.schedule_cursor_state == "complete");
    assert(metrics.schedule_first_failure.empty());
    assert(metrics.schedule_cursor_complete);
    assert(metrics.automatic_epoch_transition_count == 0);
    assert(metrics.explicit_epoch_transition_count == 2);
    assert(observation_count == 1);
}

void test_schedule_callback_fails_closed_on_order_and_flags() {
    {
        auto context = scheduled_execution_context();
        auto hooks = context->make_schedule_hooks();
        char error[256] = {};
        auto compute = schedule_event(
            TC_MEMORY_STAGE_DENOISER, TC_MEMORY_ACTION_COMPUTE,
            0, 0, 0, 0, TC_MEMORY_BRANCH_VIDEO);
        assert(hooks.emit(hooks.user, &compute, error, sizeof(error)) == 0);
        assert(std::string(error).find("out of order") != std::string::npos);
        const auto first = context->metrics();
        assert(first.schedule_event_attempted_count == 1);
        assert(first.schedule_event_rejected_count == 1);
        assert(first.schedule_mismatch_count == 1);
        assert(first.schedule_cursor_state == "poisoned");
        assert(first.schedule_first_failure.find("out of order") !=
               std::string::npos);
        char second_error[256] = {};
        assert(hooks.emit(
                   hooks.user, &compute, second_error,
                   sizeof(second_error)) == 0);
        assert(std::string(second_error).find("cursor is poisoned") !=
               std::string::npos);
        const auto second = context->metrics();
        assert(second.schedule_event_attempted_count == 2);
        assert(second.schedule_event_rejected_count == 1);
        assert(second.schedule_first_failure == first.schedule_first_failure);
        context->finish_failure(error);
    }
    {
        auto context = scheduled_execution_context();
        auto hooks = context->make_schedule_hooks();
        char error[256] = {};
        auto begin = schedule_event(
            TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END,
            TC_MEMORY_EVENT_SAFE_POINT);
        assert(hooks.emit(hooks.user, &begin, error, sizeof(error)) == 0);
        assert(std::string(error).find("flags differ") != std::string::npos);
        assert(context->metrics().schedule_cursor_state == "poisoned");
        context->finish_failure(error);
    }
}

void test_schedule_callback_rejects_non_owner_and_reentrant_emit() {
    {
        auto context = scheduled_execution_context();
        auto hooks = context->make_schedule_hooks();
        std::string failure;
        std::thread worker([&] {
            char error[256] = {};
            auto begin = schedule_event(
                TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END);
            assert(hooks.emit(
                       hooks.user, &begin, error, sizeof(error)) == 0);
            failure = error;
        });
        worker.join();
        assert(failure.find("request owner") != std::string::npos);
        const auto metrics = context->metrics();
        assert(metrics.schedule_event_attempted_count == 1);
        assert(metrics.schedule_event_rejected_count == 1);
        assert(metrics.schedule_cursor_state == "poisoned");
        context->finish_failure(failure);
    }
    {
        MemoryExecutionContext *raw = nullptr;
        tc_memory_schedule_hooks_v1 hooks{};
        int nested_status = 1;
        std::string nested_failure;
        auto context = scheduled_execution_context([&] {
            if (raw) {
                char error[256] = {};
                auto terminal = schedule_event(
                    TC_MEMORY_STAGE_TERMINAL_DRAIN,
                    TC_MEMORY_ACTION_END,
                    TC_MEMORY_EVENT_SAFE_POINT |
                        TC_MEMORY_EVENT_GPU_DRAINED |
                        TC_MEMORY_EVENT_TERMINAL);
                nested_status = hooks.emit(
                    hooks.user, &terminal, error, sizeof(error));
                nested_failure = error;
                raw = nullptr;
            }
            return ProcessMemoryObservation{
                true, 10, 1024, 512, "synthetic"};
        });
        raw = context.get();
        hooks = context->make_schedule_hooks();
        char error[256] = {};
        auto begin = schedule_event(
            TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END);
        assert(hooks.emit(hooks.user, &begin, error, sizeof(error)) == 1);
        auto compute = schedule_event(
            TC_MEMORY_STAGE_DENOISER, TC_MEMORY_ACTION_COMPUTE,
            0, 0, 0, 0, TC_MEMORY_BRANCH_VIDEO);
        assert(hooks.emit(hooks.user, &compute, error, sizeof(error)) == 1);
        expect_rejected(
            [&] { context->emit_terminal_schedule_event(); },
            "poisoned during event processing");
        assert(nested_status == 0);
        assert(nested_failure.find("reentrant") != std::string::npos);
        const auto metrics = context->metrics();
        assert(metrics.schedule_event_attempted_count == 4);
        assert(metrics.schedule_event_count == 2);
        assert(metrics.schedule_event_rejected_count == 1);
        assert(metrics.schedule_cursor_state == "poisoned");
        assert(metrics.schedule_first_failure.find("reentrant") !=
               std::string::npos);
        context->finish_failure(metrics.schedule_first_failure);
    }
}

void test_finish_success_rejects_incomplete_explicit_schedule() {
    auto context = scheduled_execution_context();
    auto hooks = context->make_schedule_hooks();
    char error[256] = {};
    auto begin = schedule_event(
        TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END);
    assert(hooks.emit(hooks.user, &begin, error, sizeof(error)) == 1);
    context->begin_draining();
    expect_rejected(
        [&] { context->finish_success(); },
        "did not consume the complete schedule");
    context->finish_failure("expected incomplete schedule");
    const auto metrics = context->metrics();
    assert(metrics.schedule_event_count == 1);
    assert(metrics.schedule_event_attempted_count == 1);
    assert(metrics.schedule_event_rejected_count == 1);
    assert(metrics.schedule_expected_event_count == 3);
    assert(metrics.schedule_cursor_state == "poisoned");
    assert(!metrics.schedule_cursor_complete);
}

void test_swap_metrics_are_terminal_and_report_does_not_reprobe() {
    const SwapActivityObservation baseline{
        true, 3, 7, 100, "synthetic_swap"};
    unsigned calls = 0;
    auto context = execution_context(
        [] { return ProcessMemoryObservation{
            true, 10, 1024, 512, "synthetic"}; },
        baseline, [&] {
            ++calls;
            return SwapActivityObservation{
                true, 4, 7, 90, "synthetic_swap"};
        });
    context->checkpoint("swap_stable");
    context->begin_draining();
    context->finish_success();
    assert(calls == 2);
    const auto first = context->take_report();
    assert(first.metrics.swap_observation_available);
    assert(!first.metrics.swap_counter_invalid);
    assert(!first.metrics.swap_activity_detected);
    assert(first.metrics.swap_sample_count == 3);
    assert(first.metrics.swapins_begin == 3);
    assert(first.metrics.swapins_end == 4);
    assert(first.metrics.swapins_delta == 1);
    assert(first.metrics.swapouts_begin == 7);
    assert(first.metrics.swapouts_end == 7);
    assert(first.metrics.swapouts_delta == 0);
    assert(first.metrics.compressed_pages_begin == 100);
    assert(first.metrics.compressed_pages_end == 90);
    assert(first.metrics.swap_observation_source == "synthetic_swap");
    const auto second = context->take_report();
    assert(second.metrics.swap_sample_count == first.metrics.swap_sample_count);
    assert(calls == 2);
}

void test_swapout_delta_fails_closed_and_is_preserved_in_report() {
    const SwapActivityObservation baseline{
        true, 10, 20, 100, "synthetic_swap"};
    auto context = execution_context(
        [] { return ProcessMemoryObservation{
            true, 10, 1024, 512, "synthetic"}; },
        baseline, [] {
            return SwapActivityObservation{
                true, 11, 21, 99, "synthetic_swap"};
        });
    std::string failure;
    try {
        context->checkpoint("denoise.stage");
    } catch (const std::exception &error) {
        failure = error.what();
    }
    assert(failure.find("memory_swap_activity_detected") !=
           std::string::npos);
    context->finish_failure(failure);
    const auto report = context->take_report();
    assert(report.metrics.execution_state == "failed");
    assert(report.metrics.failure_disposition == "clean");
    assert(report.metrics.swap_observation_available);
    assert(report.metrics.swap_activity_detected);
    assert(!report.metrics.swap_counter_invalid);
    assert(report.metrics.swapouts_delta == 1);
    assert(report.metrics.swap_first_observed_phase == "denoise.stage");
    assert(report.metrics.watchdog_critical);
    assert(report.metrics.watchdog_failure_reason.find(
               "memory_swap_activity_detected") != std::string::npos);
}

void test_swap_counter_regression_is_unreliable_and_fails_closed() {
    const SwapActivityObservation baseline{
        true, 10, 20, 100, "synthetic_swap"};
    auto context = execution_context(
        [] { return ProcessMemoryObservation{
            true, 10, 1024, 512, "synthetic"}; },
        baseline, [] {
            return SwapActivityObservation{
                true, 9, 19, 99, "synthetic_swap"};
        });
    std::string failure;
    try {
        context->checkpoint("counter_regression");
    } catch (const std::exception &error) {
        failure = error.what();
    }
    assert(failure.find("counter regressed") != std::string::npos);
    context->finish_failure(failure);
    const auto metrics = context->take_report().metrics;
    assert(!metrics.swap_observation_available);
    assert(metrics.swap_counter_invalid);
    assert(!metrics.swap_activity_detected);
    assert(metrics.swapouts_delta == 0);
    assert(metrics.swap_first_observed_phase == "counter_regression");
}

void test_swap_observer_requires_available_baseline() {
    expect_rejected(
        [&] {
            (void)execution_context(
                {}, {}, [] { return SwapActivityObservation{}; });
        },
        "swap baseline is unavailable");
}

void test_execution_context_rejects_unmanifested_or_mismatched_sites() {
    auto context = execution_context();
    auto generic_stage = context->scheduler().begin_stage("generic-bypass");
    expect_rejected(
        [&] { (void)generic_stage.reserve(
                  MemoryClass::Activation, 1, "synthetic.activation"); },
        "unmanifested allocation is forbidden");
    context->enter_epoch(2, "activation");
    auto known = context->try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(known);
    known->cancel();
    expect_rejected(
        [&] { (void)context->try_reserve_site(
                  "synthetic.missing", MemoryClass::Activation, 1); },
        "runtime allocation site is absent");
    expect_rejected(
        [&] { (void)context->try_reserve_site(
                  "synthetic.activation", MemoryClass::Weights, 1); },
        "runtime allocation class differs");
    expect_rejected(
        [&] { (void)context->try_reserve_site(
                  "synthetic.activation", MemoryClass::Activation, 21); },
        "exceeds manifest site upper");
    context->begin_draining();
    context->finish_success();
}

void test_stage_site_token_uses_owning_execution_context() {
    auto context = execution_context();
    context->enter_epoch(2, "stage-site-token");
    auto stage = context->scheduler().begin_stage("stage-site-token");
    auto token = stage.reserve_site(
        *context, "synthetic.activation", MemoryClass::Activation, 20);
    assert(token && token->site() == "synthetic.activation");
    auto lease = stage.commit_site(
        std::move(*token), StorageId{9, 601, 20, 8}, 20);
    stage.adopt(std::move(lease));
    assert(context->admission().snapshot().site_active_count == 1);
    stage = {};
    context->begin_draining();
    context->finish_success();

    auto other = execution_context();
    auto foreign_stage = other->scheduler().begin_stage("foreign");
    expect_rejected(
        [&] { (void)foreign_stage.reserve_site(
                  *execution_context(), "synthetic.activation",
                  MemoryClass::Activation, 20); },
        "another execution context");
    other->finish_failure("expected foreign-context rejection");
}

void test_execution_context_enforces_site_instance_and_epoch_lifetime() {
    auto context = execution_context();
    context->enter_epoch(1, "weights-first");
    auto first = context->try_reserve_site(
        "synthetic.weights", MemoryClass::Weights, 35);
    assert(first);
    assert(context->current_epoch() == 1);
    auto first_lease = first->commit(StorageId{9, 501, 35, 7});
    expect_rejected(
        [&] { (void)context->try_reserve_site(
                  "synthetic.weights", MemoryClass::Weights, 30); },
        "no manifest allocation instance");
    expect_rejected(
        [&] { context->enter_epoch(3, "weights-still-live"); },
        "outlived manifest interval");
    first_lease.release();
    context->enter_epoch(3, "weights-second");
    auto second = context->try_reserve_site(
        "synthetic.weights", MemoryClass::Weights, 30);
    assert(second);
    auto second_lease = second->commit(StorageId{9, 502, 30, 7});
    second_lease.release();
    context->begin_draining();
    context->finish_success();
}

void test_execution_context_rejects_framework_upper_overage() {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    MemoryExecutionContext context(
        *value.memory_policy, std::move(compiled), baseline,
        [] { return ProcessMemoryObservation{
            true, 16, 1024, 512, "synthetic"}; });
    context.begin_running();
    expect_rejected(
        [&] { context.checkpoint("framework_overage"); },
        "framework upper");
    assert(context.tainted());
    assert(!context.metrics().observed_within_budget);
    assert(context.metrics().watchdog_critical);
    assert(context.metrics().watchdog_failure_reason.find(
               "framework upper") != std::string::npos);
    context.finish_failure("expected framework failure");
    assert(context.failure_disposition() ==
           MemoryFailureDisposition::Clean);
}

void test_execution_context_rejects_pending_completion_on_success() {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    ProcessMemoryObservation baseline{true, 10, 1024, 512, "synthetic"};
    MemoryExecutionContext context(
        *value.memory_policy, std::move(compiled), baseline);
    context.begin_running();
    context.enter_epoch(2, "pending.activation");
    auto stage = context.scheduler().begin_stage("pending");
    auto reservation = context.try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(reservation);
    stage.adopt(reservation->commit(StorageId{9, 102, 20, 3}));
    const auto pending = stage.retire_all();
    assert(pending.size() == 1);
    context.begin_draining();
    expect_rejected(
        [&] { context.finish_success(); },
        "retained pending GPU releases");
    assert(context.scheduler().complete(pending.front()));
    context.finish_success();
    assert(context.finished());
}

void test_execution_context_drains_mailbox_before_success() {
    auto context = execution_context();
    context->enter_epoch(2, "mailbox.activation");
    auto stage = context->scheduler().begin_stage("mailbox");
    auto reservation = context->try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(reservation);
    stage.adopt(reservation->commit(StorageId{9, 103, 20, 4}));
    const auto pending = stage.retire_all();
    assert(pending.size() == 1);
    const auto token = context->scheduler().expect_completion(
        pending.front(), 9, 4, 1, 0);
    assert(context->scheduler().post_completion(token));
    const auto drained = context->drain_completion_mailbox();
    assert(drained.ok() && drained.consumed == 1);
    context->begin_draining();
    context->finish_success();
    assert(context->finished() && !context->tainted());
}

void test_finish_success_rejects_active_and_cached_storage() {
    {
        auto context = execution_context();
        context->enter_epoch(2, "active.activation");
        auto stage = context->scheduler().begin_stage("active-storage");
        auto reservation = context->try_reserve_site(
            "synthetic.activation", MemoryClass::Activation, 20);
        assert(reservation);
        stage.adopt(reservation->commit(StorageId{9, 104, 20, 4}));
        context->begin_draining();
        expect_rejected(
            [&] { context->finish_success(); },
            "successful request retained storage");
        stage = {};
        context->finish_success();
    }
    {
        auto context = execution_context();
        context->enter_epoch(2, "cached-activation");
        auto reservation = context->try_reserve_site(
            "synthetic.activation", MemoryClass::Activation, 20);
        assert(reservation);
        StorageId cached_id{9, 105, 20, 4};
        auto lease = reservation->commit(cached_id);
        lease.cache();
        context->begin_draining();
        expect_rejected(
            [&] { context->finish_success(); },
            "successful request retained storage");
        assert(context->admission().ledger().drop_cached(cached_id));
        context->finish_success();
    }
}

void test_execution_context_pressure_checkpoint_taints_on_system_reserve() {
    ProbeSession session(MemoryCapabilityProbe{key(), manifest()});
    MemoryCapabilityRegistry registry;
    registry.add(certified_record());
    auto value = plan();
    value.memory_policy->system_reserve_bytes = 100;
    auto resolution = preflight_memory_capability(
        session, value, device(), registry);
    auto compiled = authorize_memory_capability(value, resolution, 10);
    ProcessMemoryObservation baseline{true, 10, 1024, 500, "synthetic"};
    MemoryExecutionContext context(
        *value.memory_policy, std::move(compiled), baseline,
        [] { return ProcessMemoryObservation{
            true, 10, 1024, 99, "synthetic"}; });
    context.begin_running();
    expect_rejected(
        [&] { context.checkpoint("system_pressure"); },
        "pressure controller entered critical");
    assert(context.tainted());
    const auto metrics = context.metrics();
    assert(metrics.final_system_available_bytes == 99);
    assert(metrics.minimum_system_available_bytes == 99);
    assert(metrics.pressure_state == "critical");
    assert(metrics.pressure_transitions == 1);
    assert(metrics.watchdog_sample_count == 1);
    assert(metrics.watchdog_critical);
    context.finish_failure("expected pressure failure");
    assert(context.finished());
}

void test_context_binding_is_ordered_and_disabled_is_inert() {
    BindingSession session;
    {
        ScopedMemoryExecutionBinding disabled(session, nullptr);
        assert(session.events.empty());
    }
    auto context = execution_context();
    {
        ScopedMemoryExecutionBinding binding(session, context.get());
        assert(session.admission == &context->admission());
        assert(session.context == context.get());
        assert((session.events == std::vector<std::string>{
            "admission.bind", "context.bind"}));
        const auto drained = session.drain_memory_completions(
            *context, std::chrono::milliseconds(1));
        assert(drained.completed && drained.pending_after == 0);
    }
    assert(session.admission == nullptr);
    assert(session.context == nullptr);
    assert((session.events == std::vector<std::string>{
        "admission.bind", "context.bind", "context.drain",
        "context.unbind", "admission.clear"}));
    context->begin_draining();
    context->finish_success();
}

void test_context_binding_failure_clears_partial_state() {
    BindingSession session;
    session.throw_on_bind = true;
    auto context = execution_context();
    expect_rejected(
        [&] { ScopedMemoryExecutionBinding binding(session, context.get()); },
        "synthetic bind failure");
    assert(session.admission == nullptr);
    assert(session.context == nullptr);
    assert((session.events == std::vector<std::string>{
        "admission.bind", "context.bind", "context.unbind",
        "admission.clear"}));
    context->finish_failure("expected bind failure");
    assert(context->failure_disposition() ==
           MemoryFailureDisposition::Clean);
}

void test_failure_disposition_distinguishes_cleanup_states() {
    {
        auto context = execution_context();
        assert(context->finalize_failure("clean failure") ==
               MemoryFailureDisposition::Clean);
        const auto metrics = context->metrics();
        assert(metrics.tainted);
        assert(!metrics.worker_quarantined);
        assert(metrics.failure_disposition == "clean");
        assert(metrics.failure_reason == "clean failure");
    }
    {
        auto context = execution_context();
        context->enter_epoch(2, "failure.reservation");
        auto reservation = context->try_reserve_site(
            "synthetic.activation", MemoryClass::Activation, 20);
        assert(reservation);
        assert(context->finalize_failure("reservation retained") ==
               MemoryFailureDisposition::QuarantineWorker);
        const auto metrics = context->metrics();
        assert(metrics.worker_quarantined);
        assert(metrics.failure_disposition == "quarantine_worker");
        reservation->cancel();
    }
    {
        auto context = execution_context();
        context->enter_epoch(2, "failure.pending");
        auto stage = context->scheduler().begin_stage("pending");
        auto reservation = context->try_reserve_site(
            "synthetic.activation", MemoryClass::Activation, 20);
        assert(reservation);
        stage.adopt(reservation->commit(StorageId{9, 404, 20, 5}));
        const auto pending = stage.retire_all();
        assert(pending.size() == 1);
        assert(context->finalize_failure("pending completion") ==
               MemoryFailureDisposition::NeedsGpuDrain);
        assert(context->state() == MemoryExecutionState::Draining);
        const auto metrics = context->metrics();
        assert(!metrics.worker_quarantined);
        assert(metrics.failure_disposition == "needs_gpu_drain");
        assert(context->scheduler().complete(pending.front()));
    }
}

void test_failure_cleanup_can_demote_needs_drain_to_clean() {
    auto context = execution_context();
    context->enter_epoch(2, "failure-drain.activation");
    auto stage = context->scheduler().begin_stage("failure-drain");
    auto reservation = context->try_reserve_site(
        "synthetic.activation", MemoryClass::Activation, 20);
    assert(reservation);
    stage.adopt(reservation->commit(StorageId{9, 405, 20, 6}));
    const auto pending = stage.retire_all();
    assert(context->finalize_failure("primary failure") ==
           MemoryFailureDisposition::NeedsGpuDrain);
    const auto token = context->scheduler().expect_completion(
        pending.front(), 9, 6, 2, 1);
    assert(context->scheduler().post_completion(token));
    assert(context->complete_failure_cleanup(true) ==
           MemoryFailureDisposition::Clean);
    assert(context->state() == MemoryExecutionState::Failed);
    assert(context->metrics().cleanup_failure.empty());
}

} // namespace

int main() {
    test_production_catalog_is_empty_and_missing_probe_fails_closed();
    test_exact_match_authorizes_manifest_plan();
    test_identity_slot_and_release_gates_are_exact();
    test_budget_and_validated_upper_are_independent_gates();
    test_execution_context_owns_admission_scheduler_and_finish_gate();
    test_execution_state_transitions_are_explicit_and_terminal();
    test_terminal_report_is_idempotent_and_bounded();
    test_certified_context_requires_explicit_epoch_before_allocation();
    test_noncertified_compatibility_plan_records_automatic_epoch();
    test_explicit_schedule_drives_epoch_and_terminal_gate();
    test_schedule_callback_fails_closed_on_order_and_flags();
    test_schedule_callback_rejects_non_owner_and_reentrant_emit();
    test_finish_success_rejects_incomplete_explicit_schedule();
    test_swap_metrics_are_terminal_and_report_does_not_reprobe();
    test_swapout_delta_fails_closed_and_is_preserved_in_report();
    test_swap_counter_regression_is_unreliable_and_fails_closed();
    test_swap_observer_requires_available_baseline();
    test_execution_context_rejects_unmanifested_or_mismatched_sites();
    test_stage_site_token_uses_owning_execution_context();
    test_execution_context_enforces_site_instance_and_epoch_lifetime();
    test_execution_context_rejects_framework_upper_overage();
    test_execution_context_rejects_pending_completion_on_success();
    test_execution_context_drains_mailbox_before_success();
    test_finish_success_rejects_active_and_cached_storage();
    test_execution_context_pressure_checkpoint_taints_on_system_reserve();
    test_context_binding_is_ordered_and_disabled_is_inert();
    test_context_binding_failure_clears_partial_state();
    test_failure_disposition_distinguishes_cleanup_states();
    test_failure_cleanup_can_demote_needs_drain_to_clean();
    std::cout << "memory execution tests passed\n";
    return 0;
}
