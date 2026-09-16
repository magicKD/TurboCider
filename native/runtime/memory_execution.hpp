#pragma once

#include "memory_plan.hpp"
#include "memory_scheduler.hpp"
#include "memory_trace.hpp"
#include "memory_watchdog.hpp"
#include "session.hpp"

#include <memory>
#include <string_view>
#include <thread>

namespace tc {

struct MemoryCapabilityResolution {
    MemoryCapabilityProbe probe;
    MemoryCapabilityRecord record;
};

enum class MemoryFailureDisposition : uint8_t {
    None,
    Clean,
    NeedsGpuDrain,
    QuarantineWorker,
};

const char *memory_failure_disposition_name(MemoryFailureDisposition);

enum class MemoryExecutionState : uint8_t {
    Admitted,
    Running,
    Draining,
    Succeeded,
    Failed,
    Quarantined,
};

const char *memory_execution_state_name(MemoryExecutionState);

enum class MemoryScheduleCursorState : uint8_t {
    Disabled,
    Healthy,
    Processing,
    Poisoned,
    Complete,
};

const char *memory_schedule_cursor_state_name(MemoryScheduleCursorState);

struct MemoryExecutionReport {
    MemoryAdmissionMetrics metrics;
    std::vector<MemoryTraceEvent> trace;
};

class MemoryExecutionContext {
  public:
    MemoryExecutionContext(EffectiveMemoryPolicy policy,
                           CompiledMemoryPlan compiled_plan,
                           const ProcessMemoryObservation &baseline,
                           ProcessMemoryObserver observer = {},
                           SwapActivityObservation swap_baseline = {},
                           SwapActivityObserver swap_observer = {});
    ~MemoryExecutionContext() noexcept;
    MemoryExecutionContext(const MemoryExecutionContext &) = delete;
    MemoryExecutionContext &operator=(const MemoryExecutionContext &) = delete;

    MemoryAdmission &admission() { return *admission_; }
    MemoryStageScheduler &scheduler() { return *scheduler_; }
    const EffectiveMemoryPolicy &policy() const { return policy_; }
    const CompiledMemoryPlan &compiled_plan() const { return compiled_plan_; }
    void begin_running();
    void begin_draining();
    std::optional<MemoryReservation> try_reserve_site(
        std::string_view site_id, MemoryClass memory_class,
        uint64_t upper_bytes);
    void enter_epoch(uint64_t epoch, std::string_view event = {});
    uint64_t current_epoch() const { return current_epoch_; }
    bool uses_explicit_schedule() const {
        return compiled_plan_.require_explicit_schedule ||
               !compiled_plan_.schedule_bindings.empty();
    }
    tc_memory_schedule_hooks_v1 make_schedule_hooks() noexcept;
    void emit_schedule_event(const tc_memory_schedule_event_v1 &event);
    void emit_terminal_schedule_event();
    void checkpoint(const std::string &phase);
    MemoryCompletionDrainResult drain_completion_mailbox() noexcept;
    std::vector<MemoryTraceEvent> drain_trace();
    /* Terminal, owner-thread snapshot. The first call drains the bounded
     * trace into an immutable cache; later calls return the same report and
     * never re-probe process memory or mutate scheduler state. */
    MemoryExecutionReport take_report();
    void finish_success();
    MemoryFailureDisposition finalize_failure(std::string reason) noexcept;
    MemoryFailureDisposition complete_failure_cleanup(
        bool drain_completed, std::string cleanup_failure = {}) noexcept;
    void finish_failure(std::string reason) noexcept;
    bool tainted() const { return tainted_; }
    bool finished() const {
        return state_ == MemoryExecutionState::Succeeded ||
               state_ == MemoryExecutionState::Failed ||
               state_ == MemoryExecutionState::Quarantined;
    }
    MemoryExecutionState state() const { return state_; }
    const std::string &failure_reason() const { return failure_reason_; }
    MemoryFailureDisposition failure_disposition() const {
        return failure_disposition_;
    }
    MemoryAdmissionMetrics metrics() const;

  private:
    static int schedule_emit_callback(
        void *user, const tc_memory_schedule_event_v1 *event,
        char *error, size_t error_size) noexcept;
    void enter_epoch_impl(uint64_t epoch, std::string_view event,
                          bool automatic);
    void poison_schedule(std::string reason) noexcept;
    void sample_swap_activity(std::string_view phase, bool fail_closed);
    void sample_swap_activity_noexcept(std::string_view phase) noexcept;

    EffectiveMemoryPolicy policy_;
    CompiledMemoryPlan compiled_plan_;
    std::unique_ptr<MemoryAdmission> admission_;
    std::unique_ptr<MemoryTraceBuffer> trace_;
    std::unique_ptr<MemoryStageScheduler> scheduler_;
    std::unique_ptr<MemoryWatchdog> watchdog_;
    uint64_t current_epoch_ = 0;
    uint64_t explicit_epoch_transition_count_ = 0;
    uint64_t automatic_epoch_transition_count_ = 0;
    uint64_t schedule_cursor_ = 0;
    uint64_t schedule_attempted_count_ = 0;
    uint64_t schedule_rejected_count_ = 0;
    uint64_t schedule_mismatch_count_ = 0;
    MemoryScheduleCursorState schedule_cursor_state_ =
        MemoryScheduleCursorState::Disabled;
    std::string schedule_first_failure_;
    std::thread::id owner_thread_;
    bool tainted_ = false;
    MemoryExecutionState state_ = MemoryExecutionState::Admitted;
    MemoryFailureDisposition failure_disposition_ =
        MemoryFailureDisposition::None;
    std::string failure_reason_;
    std::string cleanup_failure_;
    SwapActivityObservation swap_baseline_;
    SwapActivityObservation swap_last_;
    SwapActivityObserver swap_observer_;
    uint64_t swap_sample_count_ = 0;
    bool swap_counter_invalid_ = false;
    bool swap_activity_detected_ = false;
    std::string swap_first_observed_phase_;
    std::optional<MemoryExecutionReport> terminal_report_;
};

/* Request-scoped compatibility bridge.  Disabled requests pass no context
 * and therefore perform no session hook calls. */
class ScopedMemoryExecutionBinding {
  public:
    ScopedMemoryExecutionBinding(ModelSession &, MemoryExecutionContext *);
    ~ScopedMemoryExecutionBinding() noexcept;
    ScopedMemoryExecutionBinding(const ScopedMemoryExecutionBinding &) = delete;
    ScopedMemoryExecutionBinding &operator=(
        const ScopedMemoryExecutionBinding &) = delete;

  private:
    ModelSession *session_ = nullptr;
};

/* Release builds start with an immutable, empty catalog.  Certified records
 * are added by reviewed source/build artifacts, never by request fields or
 * ordinary environment variables. */
const MemoryCapabilityRegistry &production_memory_capability_registry();

const char *memory_runtime_revision();

MemoryCapabilityResolution preflight_memory_capability(
    const ModelSession &, const ExecutionPlan &, const MemoryDeviceIdentity &,
    const MemoryCapabilityRegistry &, bool allow_experimental = false);

CompiledMemoryPlan authorize_memory_capability(
    ExecutionPlan &, const MemoryCapabilityResolution &,
    uint64_t process_baseline_bytes);

} // namespace tc
