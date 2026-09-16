#include "memory_execution.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <thread>
#include <utility>

namespace tc {
namespace {

bool contains_unsigned(const std::vector<unsigned> &values, unsigned value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool contains_string(const std::vector<std::string> &values,
                     const std::string &value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

const char *memory_failure_disposition_name(
        MemoryFailureDisposition disposition) {
    switch (disposition) {
    case MemoryFailureDisposition::None: return "none";
    case MemoryFailureDisposition::Clean: return "clean";
    case MemoryFailureDisposition::NeedsGpuDrain: return "needs_gpu_drain";
    case MemoryFailureDisposition::QuarantineWorker:
        return "quarantine_worker";
    }
    return "quarantine_worker";
}

const char *memory_execution_state_name(MemoryExecutionState state) {
    switch (state) {
    case MemoryExecutionState::Admitted: return "admitted";
    case MemoryExecutionState::Running: return "running";
    case MemoryExecutionState::Draining: return "draining";
    case MemoryExecutionState::Succeeded: return "succeeded";
    case MemoryExecutionState::Failed: return "failed";
    case MemoryExecutionState::Quarantined: return "quarantined";
    }
    return "quarantined";
}

const char *memory_schedule_cursor_state_name(
        MemoryScheduleCursorState state) {
    switch (state) {
    case MemoryScheduleCursorState::Disabled: return "disabled";
    case MemoryScheduleCursorState::Healthy: return "healthy";
    case MemoryScheduleCursorState::Processing: return "processing";
    case MemoryScheduleCursorState::Poisoned: return "poisoned";
    case MemoryScheduleCursorState::Complete: return "complete";
    }
    return "poisoned";
}

MemoryExecutionContext::MemoryExecutionContext(
        EffectiveMemoryPolicy policy, CompiledMemoryPlan compiled_plan,
        const ProcessMemoryObservation &baseline,
        ProcessMemoryObserver next_observer,
        SwapActivityObservation swap_baseline,
        SwapActivityObserver swap_observer)
    : policy_(std::move(policy)), compiled_plan_(std::move(compiled_plan)),
      swap_baseline_(std::move(swap_baseline)),
      swap_last_(swap_baseline_),
      swap_observer_(std::move(swap_observer)) {
    require(policy_.enabled,
            "memory_policy_invalid: execution context requires an enabled policy");
    require(baseline.available,
            "memory_observation_unreliable: execution context baseline is unavailable");
    require(policy_.planned_upper_bytes >= baseline.process_footprint_bytes,
            "memory_budget_too_small: planned upper is below process baseline");
    require(compiled_plan_.complete && compiled_plan_.fits_budget,
            "memory_policy_unsupported: execution context requires an authorized plan");
    require(compiled_plan_.planned_peak_bytes == policy_.planned_upper_bytes,
            "memory_lifetime_violation: policy and compiled plan upper differ");
    require(compiled_plan_.framework_upper_bytes ==
                policy_.framework_upper_bytes &&
                policy_.framework_upper_bytes > 0,
            "memory_lifetime_violation: framework upper contract differs");
    require(!compiled_plan_.epochs.empty() &&
                compiled_plan_.epochs.front().id == 0,
            "memory_plan_invalid: execution plan has no initial epoch");
    require(!compiled_plan_.require_explicit_schedule ||
                !compiled_plan_.schedule_bindings.empty(),
            "memory_schedule_invalid: explicit execution schedule is absent");
    schedule_cursor_state_ = compiled_plan_.schedule_bindings.empty() ?
        MemoryScheduleCursorState::Disabled :
        MemoryScheduleCursorState::Healthy;
    if (swap_observer_) {
        require(swap_baseline_.available &&
                    !swap_baseline_.source.empty(),
                "memory_observation_unreliable: swap baseline is unavailable");
        swap_sample_count_ = 1;
    }
    current_epoch_ = compiled_plan_.epochs.front().id;
    const uint64_t planned_increment =
        policy_.planned_upper_bytes - baseline.process_footprint_bytes;
    auto observer = [baseline, first = true,
                     next = std::move(next_observer)]() mutable {
        if (first) {
            first = false;
            return baseline;
        }
        if (next) return next();
        return observe_process_memory();
    };
    admission_ = std::make_unique<MemoryAdmission>(
        policy_.effective_budget_bytes, planned_increment,
        policy_.framework_upper_bytes, policy_.system_reserve_bytes,
        std::move(observer));
    admission_->ledger().set_site_constraints_required(true);
    trace_ = std::make_unique<MemoryTraceBuffer>(4096);
    scheduler_ = std::make_unique<MemoryStageScheduler>(
        admission_->ledger(), trace_.get());
    watchdog_ = std::make_unique<MemoryWatchdog>(MemoryWatchdogConfig{
        policy_.effective_budget_bytes, policy_.framework_upper_bytes,
        policy_.system_reserve_bytes, 1024});
    trace_->record(MemoryTraceEventKind::State, "admitted", {}, current_epoch_);
}

MemoryExecutionContext::~MemoryExecutionContext() noexcept {
    if (!finished())
        (void)finalize_failure(
            "execution context left scope before finish_success");
}

std::optional<MemorySiteToken> MemoryStageLease::reserve_site(
        MemoryExecutionContext &context, std::string site_id,
        MemoryClass memory_class, uint64_t upper_bytes) {
    require(scheduler_ != nullptr,
            "memory_scheduler_invalid: inactive stage reservation");
    require(&context.scheduler() == scheduler_,
            "memory_lifetime_violation: stage belongs to another execution context");
    require(!site_id.empty(),
            "memory_manifest_invalid: runtime allocation site is empty");
    auto reservation = context.try_reserve_site(
        site_id, memory_class, upper_bytes);
    if (!reservation) return std::nullopt;
    return MemorySiteToken(
        std::move(site_id), memory_class,
        MemoryAllocationTxn(std::move(*reservation), upper_bytes));
}

void MemoryExecutionContext::begin_running() {
    require(state_ == MemoryExecutionState::Admitted,
            "memory_lifetime_violation: execution can only start after admission");
    state_ = MemoryExecutionState::Running;
    owner_thread_ = std::this_thread::get_id();
    trace_->record(MemoryTraceEventKind::State, "running", {}, current_epoch_);
}

void MemoryExecutionContext::begin_draining() {
    require(state_ == MemoryExecutionState::Running,
            "memory_lifetime_violation: execution can only drain after running");
    state_ = MemoryExecutionState::Draining;
    trace_->record(MemoryTraceEventKind::State, "draining", {}, current_epoch_);
}

std::optional<MemoryReservation> MemoryExecutionContext::try_reserve_site(
        std::string_view site_id, MemoryClass memory_class,
        uint64_t upper_bytes) {
    require(state_ == MemoryExecutionState::Running,
            "memory_lifetime_violation: allocation outside running execution");
    require(!site_id.empty() && upper_bytes > 0,
            "memory_manifest_invalid: invalid runtime allocation site");
    const auto found = std::lower_bound(
        compiled_plan_.allocation_sites.begin(),
        compiled_plan_.allocation_sites.end(), site_id,
        [](const auto &site, std::string_view value) {
            return site.site_id < value;
        });
    require(found != compiled_plan_.allocation_sites.end() &&
                found->site_id == site_id,
            "memory_estimate_unknown: runtime allocation site is absent from manifest: " +
                std::string(site_id));
    require(found->memory_class == memory_class,
            "memory_lifetime_violation: runtime allocation class differs from manifest: " +
                std::string(site_id));
    require(found->maximum_instance_upper_bytes > 0 &&
                upper_bytes <= found->maximum_instance_upper_bytes,
            "memory_lifetime_violation: runtime allocation exceeds manifest site upper: " +
                std::string(site_id));
    MemorySiteReservationConstraint constraint;
    constraint.site_id = std::string(site_id);
    constraint.maximum_live_instances = found->instance_count;
    constraint.aggregate_upper_bytes =
        found->aggregate_instance_upper_bytes;
    const auto first = std::lower_bound(
        compiled_plan_.allocation_instances.begin(),
        compiled_plan_.allocation_instances.end(), site_id,
        [](const auto &instance, std::string_view value) {
            return instance.site_id < value;
        });
    bool current_epoch_matches = false;
    std::optional<uint64_t> next_epoch;
    for (auto instance = first;
         instance != compiled_plan_.allocation_instances.end() &&
             instance->site_id == site_id;
         ++instance) {
        require(instance->memory_class == memory_class,
                "memory_plan_invalid: compiled instance class differs");
        constraint.instances.push_back({
            instance->instance_id, instance->upper_bytes,
            instance->live_begin, instance->live_end,
            instance->alias_group});
        if (upper_bytes > instance->upper_bytes) continue;
        if (current_epoch_ >= instance->live_begin &&
            current_epoch_ < instance->live_end) {
            current_epoch_matches = true;
        } else if (current_epoch_ < instance->live_begin &&
                   (!next_epoch || instance->live_begin < *next_epoch)) {
            next_epoch = instance->live_begin;
        }
    }
    require(!constraint.instances.empty(),
            "memory_estimate_unknown: runtime allocation site has no compiled instances: " +
                std::string(site_id));
    if (!current_epoch_matches && next_epoch) {
        require(!compiled_plan_.require_explicit_epoch,
                "memory_lifetime_violation: explicit epoch transition is required before allocation site: " +
                    std::string(site_id));
        enter_epoch_impl(
            *next_epoch,
            std::string("automatic allocation: ") + std::string(site_id),
            true);
    }
    constraint.epoch = current_epoch_;
    auto reservation = admission_->try_reserve_site(
        memory_class, upper_bytes, constraint);
    const auto snapshot = admission_->snapshot();
    trace_->record(MemoryTraceEventKind::Reserve, {}, site_id, current_epoch_,
                   upper_bytes, 0, snapshot.committed_bytes,
                   snapshot.reserved_bytes, snapshot.pending_release_bytes);
    return reservation;
}

void MemoryExecutionContext::enter_epoch(
        uint64_t epoch, std::string_view event) {
    enter_epoch_impl(epoch, event, false);
}

tc_memory_schedule_hooks_v1
MemoryExecutionContext::make_schedule_hooks() noexcept {
    tc_memory_schedule_hooks_v1 hooks{};
    hooks.struct_size = sizeof(hooks);
    hooks.version = TC_MEMORY_SCHEDULE_HOOKS_VERSION_1;
    if (!uses_explicit_schedule()) return hooks;
    hooks.user = this;
    hooks.emit = &MemoryExecutionContext::schedule_emit_callback;
    return hooks;
}

int MemoryExecutionContext::schedule_emit_callback(
        void *user, const tc_memory_schedule_event_v1 *event,
        char *error, size_t error_size) noexcept {
    try {
        require(user != nullptr && event != nullptr,
                "memory_schedule_invalid: schedule callback input is null");
        static_cast<MemoryExecutionContext *>(user)->emit_schedule_event(
            *event);
        return 1;
    } catch (const std::exception &failure) {
        if (error && error_size)
            std::snprintf(error, error_size, "%s", failure.what());
    } catch (...) {
        if (error && error_size)
            std::snprintf(error, error_size, "%s",
                          "memory_lifetime_violation: unknown schedule callback failure");
    }
    return 0;
}

void MemoryExecutionContext::emit_schedule_event(
        const tc_memory_schedule_event_v1 &event) {
    ++schedule_attempted_count_;
    try {
        require(state_ == MemoryExecutionState::Running,
                "memory_lifetime_violation: schedule event outside running execution");
        require(owner_thread_ == std::this_thread::get_id(),
                "memory_lifetime_violation: schedule event was not emitted by the request owner");
        require(schedule_cursor_state_ != MemoryScheduleCursorState::Disabled,
                "memory_schedule_invalid: schedule event emitted without a compiled schedule");
        require(schedule_cursor_state_ != MemoryScheduleCursorState::Poisoned,
                "memory_lifetime_violation: schedule cursor is poisoned: " +
                    schedule_first_failure_);
        require(schedule_cursor_state_ != MemoryScheduleCursorState::Processing,
                "memory_lifetime_violation: schedule callback is reentrant");
        require(schedule_cursor_state_ != MemoryScheduleCursorState::Complete,
                "memory_lifetime_violation: schedule emitted an unexpected extra event");
        require(schedule_cursor_state_ == MemoryScheduleCursorState::Healthy,
                "memory_lifetime_violation: schedule cursor is not healthy");
        schedule_cursor_state_ = MemoryScheduleCursorState::Processing;

        require(event.version == TC_MEMORY_SCHEDULE_EVENT_VERSION_1 &&
                    event.struct_size >= sizeof(tc_memory_schedule_event_v1),
                "memory_schedule_invalid: unsupported schedule event ABI");
        require(schedule_cursor_ < compiled_plan_.schedule_bindings.size(),
                "memory_lifetime_violation: schedule emitted an unexpected extra event");
        const auto &binding =
            compiled_plan_.schedule_bindings[schedule_cursor_];
        require(binding.sequence == schedule_cursor_,
                "memory_plan_invalid: compiled schedule sequence is not contiguous");
        const auto key = memory_schedule_event_key(event);
        require(key == binding.key,
                "memory_lifetime_violation: schedule event is out of order: expected " +
                    memory_schedule_event_name(binding.key) + " but received " +
                    memory_schedule_event_name(key));
        require(event.slot == binding.expected_slot,
                "memory_lifetime_violation: schedule event slot differs from plan");
        require(event.flags == binding.required_flags,
                "memory_lifetime_violation: schedule event flags differ from plan");

        const std::string name = memory_schedule_event_name(key);
        if (event.flags & TC_MEMORY_EVENT_SAFE_POINT) {
            const auto drain = drain_completion_mailbox();
            require(drain.ok(),
                    "memory_lifetime_violation: schedule safe-point completion drain failed: " +
                        drain.failure);
            if (event.flags & TC_MEMORY_EVENT_GPU_DRAINED) {
                const bool drained = scheduler_->pending_count() == 0 &&
                    scheduler_->outstanding_completion_count() == 0 &&
                    scheduler_->completion_mailbox_count() == 0 &&
                    !scheduler_->completion_mailbox_overflowed();
                require(drained,
                        "memory_lifetime_violation: schedule claimed GPU drain with pending completion state");
            }
        }
        enter_epoch_impl(binding.epoch, name, false);
        if (binding.checkpoint_after) checkpoint(name);
        require(schedule_cursor_state_ == MemoryScheduleCursorState::Processing,
                "memory_lifetime_violation: schedule cursor was poisoned during event processing");
        trace_->record(
            MemoryTraceEventKind::Schedule, name, {}, current_epoch_, 0, 0,
            0, 0, 0, event.stage,
            event.slot == TC_MEMORY_INDEX_NONE ? 0 : event.slot,
            static_cast<int32_t>(event.action));
        ++schedule_cursor_;
        schedule_cursor_state_ =
            schedule_cursor_ == compiled_plan_.schedule_bindings.size() ?
                MemoryScheduleCursorState::Complete :
                MemoryScheduleCursorState::Healthy;
    } catch (const std::exception &failure) {
        poison_schedule(failure.what());
        throw;
    } catch (...) {
        poison_schedule(
            "memory_lifetime_violation: unknown schedule event failure");
        throw;
    }
}

void MemoryExecutionContext::emit_terminal_schedule_event() {
    auto reject = [&](const std::string &reason) {
        poison_schedule(reason);
        require(false, reason);
    };
    if (schedule_cursor_ >= compiled_plan_.schedule_bindings.size())
        reject("memory_lifetime_violation: terminal schedule binding is absent");
    const auto &binding = compiled_plan_.schedule_bindings[schedule_cursor_];
    const uint32_t terminal_flags =
        TC_MEMORY_EVENT_SAFE_POINT | TC_MEMORY_EVENT_GPU_DRAINED |
        TC_MEMORY_EVENT_TERMINAL;
    if (binding.key.stage != TC_MEMORY_STAGE_TERMINAL_DRAIN ||
        binding.key.action != TC_MEMORY_ACTION_END ||
        (binding.required_flags & terminal_flags) != terminal_flags)
        reject("memory_lifetime_violation: next schedule binding is not a terminal drain");
    tc_memory_schedule_event_v1 event{};
    event.struct_size = sizeof(event);
    event.version = TC_MEMORY_SCHEDULE_EVENT_VERSION_1;
    event.stage = binding.key.stage;
    event.action = binding.key.action;
    event.step = binding.key.step;
    event.block = binding.key.block;
    event.tile = binding.key.tile;
    event.branch = binding.key.branch;
    event.slot = binding.expected_slot;
    event.flags = binding.required_flags;
    emit_schedule_event(event);
}

void MemoryExecutionContext::poison_schedule(std::string reason) noexcept {
    if (schedule_cursor_state_ == MemoryScheduleCursorState::Poisoned)
        return;
    schedule_cursor_state_ = MemoryScheduleCursorState::Poisoned;
    ++schedule_rejected_count_;
    ++schedule_mismatch_count_;
    schedule_first_failure_ = std::move(reason);
    tainted_ = true;
    if (trace_)
        trace_->record(MemoryTraceEventKind::Schedule,
                       schedule_first_failure_, {}, current_epoch_,
                       0, 0, 0, 0, 0, 0, 0, -1);
}

void MemoryExecutionContext::enter_epoch_impl(
        uint64_t epoch, std::string_view event, bool automatic) {
    require(state_ == MemoryExecutionState::Running,
            "memory_lifetime_violation: epoch transition outside running execution");
    require(epoch >= current_epoch_,
            "memory_lifetime_violation: execution epoch moved backwards");
    const auto found = std::lower_bound(
        compiled_plan_.epochs.begin(), compiled_plan_.epochs.end(), epoch,
        [](const auto &item, uint64_t value) { return item.id < value; });
    require(found != compiled_plan_.epochs.end() && found->id == epoch,
            "memory_lifetime_violation: execution epoch is absent from plan" +
                (event.empty() ? std::string{} :
                 std::string(" at ") + std::string(event)));
    admission_->validate_site_epoch(epoch);
    if (epoch != current_epoch_) {
        if (automatic)
            ++automatic_epoch_transition_count_;
        else
            ++explicit_epoch_transition_count_;
    }
    current_epoch_ = epoch;
    trace_->record(MemoryTraceEventKind::Epoch, event, {}, current_epoch_);
}

void MemoryExecutionContext::sample_swap_activity(
        std::string_view phase, bool fail_closed) {
    if (!swap_observer_) return;
    ++swap_sample_count_;
    SwapActivityObservation observation;
    try {
        observation = swap_observer_();
    } catch (...) {
        observation.available = false;
    }
    auto reject = [&](const std::string &reason) {
        swap_counter_invalid_ = true;
        if (swap_first_observed_phase_.empty())
            swap_first_observed_phase_ = std::string(phase);
        watchdog_->mark_failure(reason, phase);
        trace_->record(MemoryTraceEventKind::Pressure, phase,
                       "system.swap", current_epoch_, 0, 0, 0, 0, 0,
                       0, 0, 1);
        if (fail_closed) require(false, reason);
    };
    if (!observation.available || observation.source.empty()) {
        reject("memory_observation_unreliable: swap activity sample is unavailable at " +
               std::string(phase));
        return;
    }
    if (observation.source != swap_baseline_.source ||
        observation.swapins < swap_last_.swapins ||
        observation.swapouts < swap_last_.swapouts) {
        reject("memory_observation_unreliable: swap counter regressed or changed source at " +
               std::string(phase));
        return;
    }
    swap_last_ = observation;
    if (observation.swapouts > swap_baseline_.swapouts) {
        swap_activity_detected_ = true;
        if (swap_first_observed_phase_.empty())
            swap_first_observed_phase_ = std::string(phase);
        const std::string reason =
            "memory_swap_activity_detected: system swapouts increased during constrained execution at " +
            std::string(phase);
        watchdog_->mark_failure(reason, phase);
        trace_->record(
            MemoryTraceEventKind::Pressure, phase, "system.swap",
            current_epoch_, observation.swapouts - swap_baseline_.swapouts,
            observation.swapins - swap_baseline_.swapins, 0, 0, 0,
            0, 0, 2);
        if (fail_closed) require(false, reason);
    }
}

void MemoryExecutionContext::sample_swap_activity_noexcept(
        std::string_view phase) noexcept {
    try {
        sample_swap_activity(phase, false);
    } catch (...) {
        swap_counter_invalid_ = true;
    }
}

void MemoryExecutionContext::checkpoint(const std::string &phase) {
    require(state_ == MemoryExecutionState::Running ||
                state_ == MemoryExecutionState::Draining,
            "memory_lifetime_violation: checkpoint outside active execution");
    try {
        try {
            admission_->checkpoint(phase);
        } catch (const std::exception &error) {
            watchdog_->mark_failure(error.what(), phase);
            throw;
        } catch (...) {
            watchdog_->mark_failure(
                "memory_observation_unreliable: unknown checkpoint failure",
                phase);
            throw;
        }
        sample_swap_activity(phase, true);
        const auto snapshot = admission_->snapshot();
        const auto metrics = admission_->metrics();
        MemoryPressureSample sample;
        sample.committed_bytes = snapshot.known_bytes;
        sample.reserved_bytes = snapshot.reserved_bytes;
        sample.budget_bytes = policy_.effective_budget_bytes;
        sample.process_footprint_bytes =
            metrics.final_process_footprint_bytes;
        sample.system_available_bytes =
            metrics.final_system_available_bytes;
        sample.system_reserve_bytes = policy_.system_reserve_bytes;
        sample.observed_over_budget = !metrics.observed_within_budget;
        /* MemoryAdmission already consumed the observation used below.  The
         * watchdog is owner-polled and records the same boundary without a
         * second process probe. */
        watchdog_->sample(
            ProcessMemoryObservation{
                true, metrics.final_process_footprint_bytes, 0,
                metrics.final_system_available_bytes,
                metrics.observation_source},
            snapshot, phase);
        const auto watchdog_status = watchdog_->status();
        sample.observed_over_budget = sample.observed_over_budget ||
            watchdog_status.critical;
        const auto decision = scheduler_->observe(sample);
        trace_->record(
            MemoryTraceEventKind::Checkpoint, phase, {}, current_epoch_, 0, 0,
            snapshot.committed_bytes, snapshot.reserved_bytes,
            snapshot.pending_release_bytes);
        if (watchdog_status.critical || decision.state != MemoryPressureState::Normal)
            trace_->record(MemoryTraceEventKind::Pressure, phase, {},
                           current_epoch_, 0, 0, snapshot.committed_bytes,
                           snapshot.reserved_bytes,
                           snapshot.pending_release_bytes);
        if (decision.abort_request) {
            tainted_ = true;
            require(false,
                    "memory_pressure_abort: pressure controller entered critical at " +
                        phase);
        }
    } catch (...) {
        tainted_ = true;
        throw;
    }
}

MemoryCompletionDrainResult
MemoryExecutionContext::drain_completion_mailbox() noexcept {
    if (state_ == MemoryExecutionState::Succeeded ||
        state_ == MemoryExecutionState::Quarantined) {
        return {0, 0, 0, false,
                "memory_lifetime_violation: completion drain after terminal execution"};
    }
    if (!scheduler_) {
        return {0, 0, 0, false,
                "memory_lifetime_violation: scheduler is unavailable"};
    }
    return scheduler_->drain_completion_mailbox();
}

void MemoryExecutionContext::finish_success() {
    require(state_ == MemoryExecutionState::Draining,
            "memory_lifetime_violation: successful execution must finish from draining state");
    if (uses_explicit_schedule() &&
        schedule_cursor_state_ != MemoryScheduleCursorState::Complete) {
        const std::string reason =
            "memory_lifetime_violation: successful request did not consume the complete schedule";
        poison_schedule(reason);
        require(false, reason);
    }
    sample_swap_activity("finish_success", true);
    const auto snapshot = admission_->snapshot();
    require(snapshot.reservation_count == 0,
            "memory_lifetime_violation: successful request retained reservations");
    require(snapshot.pending_release_count == 0 && scheduler_->pending_count() == 0,
            "memory_lifetime_violation: successful request retained pending GPU releases");
    require(scheduler_->outstanding_completion_count() == 0 &&
                scheduler_->completion_mailbox_count() == 0 &&
                !scheduler_->completion_mailbox_overflowed(),
            "memory_lifetime_violation: successful request retained completion state");
    require(snapshot.storage_count == 0 && snapshot.cached_bytes == 0,
            "memory_lifetime_violation: successful request retained storage");
    require(snapshot.site_allocation_count == 0,
            "memory_lifetime_violation: successful request retained site allocations");
    require(snapshot.unknown_bytes == 0 && snapshot.peak_unknown_bytes == 0,
            "memory_lifetime_violation: successful request used unknown memory");
    failure_disposition_ = MemoryFailureDisposition::None;
    state_ = MemoryExecutionState::Succeeded;
    trace_->record(MemoryTraceEventKind::State, "succeeded", {}, current_epoch_);
}

MemoryFailureDisposition MemoryExecutionContext::finalize_failure(
        std::string reason) noexcept {
    if (finished()) return failure_disposition_;
    tainted_ = true;
    failure_reason_ = std::move(reason);
    sample_swap_activity_noexcept("failure");
    failure_disposition_ = MemoryFailureDisposition::QuarantineWorker;
    try {
        const auto snapshot = admission_->snapshot();
        if (snapshot.pending_release_count || scheduler_->pending_count() ||
            snapshot.storage_count ||
            snapshot.site_active_count || snapshot.site_pending_count ||
            snapshot.site_cached_count ||
            scheduler_->outstanding_completion_count() ||
            scheduler_->completion_mailbox_count()) {
            failure_disposition_ = MemoryFailureDisposition::NeedsGpuDrain;
        } else if (!snapshot.reservation_count &&
                   !snapshot.site_allocation_count &&
                   !snapshot.unknown_bytes) {
            failure_disposition_ = MemoryFailureDisposition::Clean;
        }
    } catch (...) {
        /* If accounting itself cannot be inspected, reuse is unsafe. */
        failure_disposition_ = MemoryFailureDisposition::QuarantineWorker;
    }
    if (failure_disposition_ == MemoryFailureDisposition::QuarantineWorker)
        state_ = MemoryExecutionState::Quarantined;
    else if (failure_disposition_ == MemoryFailureDisposition::NeedsGpuDrain)
        state_ = MemoryExecutionState::Draining;
    else
        state_ = MemoryExecutionState::Failed;
    trace_->record(MemoryTraceEventKind::Failure, failure_reason_, {},
                   current_epoch_);
    return failure_disposition_;
}

MemoryFailureDisposition MemoryExecutionContext::complete_failure_cleanup(
        bool drain_completed, std::string cleanup_failure) noexcept {
    if (!cleanup_failure.empty()) cleanup_failure_ = std::move(cleanup_failure);
    sample_swap_activity_noexcept("failure_cleanup");
    if (failure_disposition_ != MemoryFailureDisposition::NeedsGpuDrain)
        return failure_disposition_;
    if (!drain_completed) {
        failure_disposition_ = MemoryFailureDisposition::QuarantineWorker;
        state_ = MemoryExecutionState::Quarantined;
        return failure_disposition_;
    }
    try {
        const auto mailbox = scheduler_->drain_completion_mailbox();
        if (!mailbox.ok() && cleanup_failure_.empty())
            cleanup_failure_ = mailbox.failure;
        const auto snapshot = admission_->snapshot();
        const bool clean = mailbox.ok() &&
            snapshot.reservation_count == 0 &&
            snapshot.storage_count == 0 &&
            snapshot.pending_release_count == 0 &&
            snapshot.unknown_bytes == 0 &&
            snapshot.site_allocation_count == 0 &&
            scheduler_->pending_count() == 0 &&
            scheduler_->outstanding_completion_count() == 0 &&
            scheduler_->completion_mailbox_count() == 0 &&
            !scheduler_->completion_mailbox_overflowed();
        failure_disposition_ = clean ? MemoryFailureDisposition::Clean :
                                       MemoryFailureDisposition::QuarantineWorker;
        state_ = clean ? MemoryExecutionState::Failed :
                         MemoryExecutionState::Quarantined;
        trace_->record(
            MemoryTraceEventKind::State,
            clean ? "failed" : "quarantined", {}, current_epoch_);
    } catch (...) {
        failure_disposition_ = MemoryFailureDisposition::QuarantineWorker;
        state_ = MemoryExecutionState::Quarantined;
        trace_->record(MemoryTraceEventKind::State, "quarantined", {},
                       current_epoch_);
    }
    return failure_disposition_;
}

void MemoryExecutionContext::finish_failure(std::string reason) noexcept {
    (void)finalize_failure(std::move(reason));
}

MemoryAdmissionMetrics MemoryExecutionContext::metrics() const {
    auto result = admission_->metrics();
    result.tainted = tainted_;
    result.worker_quarantined =
        failure_disposition_ == MemoryFailureDisposition::QuarantineWorker;
    result.failure_disposition =
        memory_failure_disposition_name(failure_disposition_);
    result.failure_reason = failure_reason_;
    result.cleanup_failure = cleanup_failure_;
    result.pressure_transitions = scheduler_->transition_count();
    result.pressure_state =
        memory_pressure_state_name(scheduler_->pressure_state());
    result.execution_state = memory_execution_state_name(state_);
    const auto watchdog_status = watchdog_->status();
    result.watchdog_sample_count = watchdog_status.sample_count;
    result.watchdog_dropped_samples = watchdog_status.dropped_samples;
    result.watchdog_critical = watchdog_status.critical;
    result.watchdog_failure_reason = watchdog_status.first_failure_reason;
    const auto trace_status = trace_->status();
    result.trace_event_count = trace_status.event_count;
    result.trace_dropped_events = trace_status.dropped_events;
    result.trace_overflowed = trace_status.overflowed;
    result.explicit_epoch_transition_count =
        explicit_epoch_transition_count_;
    result.automatic_epoch_transition_count =
        automatic_epoch_transition_count_;
    result.current_epoch = current_epoch_;
    result.planned_peak_epoch = compiled_plan_.peak_epoch;
    result.schedule_event_count = schedule_cursor_;
    result.schedule_event_attempted_count = schedule_attempted_count_;
    result.schedule_event_rejected_count = schedule_rejected_count_;
    result.schedule_expected_event_count =
        compiled_plan_.schedule_bindings.size();
    result.schedule_mismatch_count = schedule_mismatch_count_;
    result.schedule_next_sequence = schedule_cursor_;
    result.schedule_cursor_state =
        memory_schedule_cursor_state_name(schedule_cursor_state_);
    result.schedule_first_failure = schedule_first_failure_;
    result.schedule_cursor_complete =
        schedule_cursor_state_ == MemoryScheduleCursorState::Complete &&
        schedule_cursor_ == compiled_plan_.schedule_bindings.size();
    result.swap_observation_available = bool(swap_observer_) &&
        swap_baseline_.available && swap_last_.available &&
        !swap_counter_invalid_;
    result.swap_counter_invalid = swap_counter_invalid_;
    result.swap_activity_detected = swap_activity_detected_;
    result.swap_sample_count = swap_sample_count_;
    result.swapins_begin = swap_baseline_.swapins;
    result.swapins_end = swap_last_.swapins;
    result.swapins_delta = swap_last_.swapins >= swap_baseline_.swapins ?
        swap_last_.swapins - swap_baseline_.swapins : 0;
    result.swapouts_begin = swap_baseline_.swapouts;
    result.swapouts_end = swap_last_.swapouts;
    result.swapouts_delta = swap_last_.swapouts >= swap_baseline_.swapouts ?
        swap_last_.swapouts - swap_baseline_.swapouts : 0;
    result.compressed_pages_begin = swap_baseline_.compressed_pages;
    result.compressed_pages_end = swap_last_.compressed_pages;
    result.swap_first_observed_phase = swap_first_observed_phase_;
    result.swap_observation_source = swap_baseline_.source;
    return result;
}

std::vector<MemoryTraceEvent> MemoryExecutionContext::drain_trace() {
    if (terminal_report_) return terminal_report_->trace;
    auto result = trace_ ? trace_->drain() : std::vector<MemoryTraceEvent>{};
    if (finished()) {
        MemoryExecutionReport report;
        report.metrics = metrics();
        report.trace = result;
        terminal_report_ = std::move(report);
    }
    return result;
}

MemoryExecutionReport MemoryExecutionContext::take_report() {
    require(finished(),
            "memory_lifetime_violation: execution report requires terminal state");
    if (!terminal_report_) {
        MemoryExecutionReport report;
        report.metrics = metrics();
        report.trace = trace_ ? trace_->drain() : std::vector<MemoryTraceEvent>{};
        terminal_report_ = std::move(report);
    }
    return *terminal_report_;
}

ScopedMemoryExecutionBinding::ScopedMemoryExecutionBinding(
        ModelSession &session, MemoryExecutionContext *context) {
    if (!context) return;
    session.set_memory_admission(&context->admission());
    try {
        session.bind_memory_context(context);
        session_ = &session;
    } catch (...) {
        try {
            session.unbind_memory_context();
        } catch (...) {
        }
        try {
            session.set_memory_admission(nullptr);
        } catch (...) {
        }
        throw;
    }
}

ScopedMemoryExecutionBinding::~ScopedMemoryExecutionBinding() noexcept {
    if (!session_) return;
    try {
        session_->unbind_memory_context();
    } catch (...) {
    }
    try {
        session_->set_memory_admission(nullptr);
    } catch (...) {
    }
}

const MemoryCapabilityRegistry &production_memory_capability_registry() {
    static const MemoryCapabilityRegistry registry;
    return registry;
}

const char *memory_runtime_revision() {
    return "memory-runtime-v3";
}

MemoryCapabilityResolution preflight_memory_capability(
        const ModelSession &session, const ExecutionPlan &plan,
        const MemoryDeviceIdentity &device,
        const MemoryCapabilityRegistry &registry, bool allow_experimental) {
    require(plan.memory_policy && plan.memory_policy->enabled,
            "memory_policy_invalid: capability preflight requires an enabled policy");
    const auto &policy = *plan.memory_policy;
    require(policy.route_available && !policy.adapter_candidate.empty(),
            "memory_policy_unsupported: " + policy.reason);
    require(!device.device_family.empty() && !device.runtime_revision.empty(),
            "memory_policy_unsupported: memory device identity is unavailable");

    auto probe = session.probe_memory_capability(plan, device);
    require(probe.has_value(),
            "memory_policy_unsupported: exact checkpoint capability probe is unavailable for " +
                policy.adapter_candidate);
    probe->manifest.validate();
    const auto manifest_digest = probe->manifest.digest();
    const auto &key = probe->key;
    require(key.adapter == policy.adapter_candidate,
            "memory_policy_unsupported: capability adapter mismatch");
    if (!policy.candidate_backend.empty())
        require(key.backend == policy.candidate_backend,
                "memory_policy_unsupported: capability backend mismatch");
    if (!policy.candidate_dtype.empty())
        require(key.dtype == policy.candidate_dtype,
                "memory_policy_unsupported: capability dtype mismatch");
    if (!policy.candidate_model_variant.empty())
        require(key.model_variant == policy.candidate_model_variant,
                "memory_policy_unsupported: capability model variant mismatch");
    if (!policy.candidate_sampler_mode.empty())
        require(key.sampler_mode == policy.candidate_sampler_mode,
                "memory_policy_unsupported: capability sampler mismatch");
    if (!policy.candidate_tiling_mode.empty())
        require(key.tiling_mode == policy.candidate_tiling_mode,
                "memory_policy_unsupported: capability tiling mismatch");
    require(key.model_id == plan.request.model &&
                key.operation == plan.request.operation,
            "memory_policy_unsupported: capability request identity mismatch");
    require(key.refill_slots == policy.refill_slots,
            "memory_policy_unsupported: capability refill-slot mismatch");
    require(key.checkpoint_digest == probe->manifest.checkpoint_digest,
            "memory_policy_unsupported: capability checkpoint digest mismatch");
    require(probe->manifest.candidate_id == key.adapter,
            "memory_policy_unsupported: manifest candidate mismatch");
    require(probe->manifest.runtime_revision == key.runtime_revision &&
                key.runtime_revision == device.runtime_revision,
            "memory_policy_unsupported: capability runtime revision mismatch");
    require(key.device_family == device.device_family,
            "memory_policy_unsupported: capability device family mismatch");
    require(!key.backend.empty() && !key.dtype.empty() &&
                !key.model_variant.empty() && !key.shape_bucket.empty() &&
                !key.sampler_mode.empty() && !key.tiling_mode.empty(),
            "memory_policy_unsupported: capability candidate key is incomplete");

    auto match = registry.lookup(key, manifest_digest, allow_experimental);
    require(match.matched && match.record.has_value(),
            "memory_policy_unsupported: " + match.reason);
    require(contains_unsigned(match.record->verified_refill_slots,
                              policy.refill_slots),
            "memory_policy_unsupported: refill-slot revision is not verified");
    if (key.tiling_mode != "none")
        require(contains_string(match.record->verified_tiling_modes,
                                key.tiling_mode),
                "memory_policy_unsupported: tiling revision is not verified");
    return {std::move(*probe), std::move(*match.record)};
}

CompiledMemoryPlan authorize_memory_capability(
        ExecutionPlan &plan, const MemoryCapabilityResolution &resolution,
        uint64_t process_baseline_bytes) {
    require(plan.memory_policy && plan.memory_policy->enabled,
            "memory_policy_invalid: capability authorization requires an enabled policy");
    MemoryPlanCompileOptions compile_options;
    compile_options.require_explicit_epoch =
        resolution.record.require_explicit_epoch;
    compile_options.require_explicit_schedule =
        resolution.record.require_explicit_schedule;
    compile_options.schedule = resolution.record.schedule;
    auto compiled = compile_memory_plan(
        resolution.probe.manifest, process_baseline_bytes,
        resolution.record.framework_upper_bytes,
        plan.memory_policy->effective_budget_bytes,
        compile_options);
    auto &policy = *plan.memory_policy;
    policy.planned_upper_bytes = compiled.planned_peak_bytes;
    policy.estimate_fits = compiled.complete && compiled.fits_budget;
    policy.estimate_provenance = "manifest_live_interval_v1";
    policy.manifest_digest = compiled.manifest_digest;
    policy.evidence_digest = resolution.record.evidence_digest;
    policy.framework_upper_bytes =
        resolution.record.framework_upper_bytes;
    if (!compiled.fits_budget) {
        policy.execution_supported = false;
        policy.admission_state = "rejected";
        policy.reason = compiled.rejection_reason;
        require(false, "memory_budget_too_small: " + compiled.rejection_reason);
    }
    require(compiled.planned_peak_bytes <=
                resolution.record.maximum_validated_upper_bytes,
            "memory_policy_unsupported: compiled plan exceeds the validated capability upper");
    policy.execution_supported = true;
    policy.capability_level = resolution.record.level;
    policy.certification_state = resolution.record.state;
    policy.release_stable =
        resolution.record.state == MemoryCertificationState::Certified &&
        resolution.record.release_enabled;
    policy.admission_state = "capability_verified";
    policy.reason = "exact checkpoint capability manifest matched";
    std::ostringstream digest;
    digest << policy.digest << "|candidate=" << resolution.probe.key.digest()
           << "|manifest=" << compiled.manifest_digest
           << "|plan=" << compiled.plan_digest;
    policy.digest = memory_sha256_hex(digest.str());
    require_memory_constrained_execution_supported(policy);
    return compiled;
}

} // namespace tc
