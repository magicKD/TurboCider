#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tc {

enum class MemoryClass : uint8_t {
    ProcessBaseline,
    Weights,
    Activation,
    Conditioning,
    RefillSlot,
    ConversionScratch,
    Output,
    AllocatorCache,
    CompileTemporary,
    ChildProcessEnvelope,
    UnknownExternal,
};

struct StorageId {
    uint64_t allocator_domain = 0;
    uint64_t handle = 0;
    uint64_t capacity = 0;
    uint64_t session_generation = 0;
};

struct MemorySnapshot {
    uint64_t budget_bytes = 0;
    uint64_t process_baseline_bytes = 0;
    uint64_t storage_bytes = 0;
    uint64_t known_bytes = 0;
    uint64_t committed_bytes = 0;
    uint64_t active_bytes = 0;
    uint64_t pending_release_bytes = 0;
    uint64_t cached_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t peak_committed_bytes = 0;
    uint64_t unknown_bytes = 0;
    uint64_t peak_unknown_bytes = 0;
    uint64_t storage_count = 0;
    uint64_t reservation_count = 0;
    uint64_t pending_release_count = 0;
    uint64_t site_allocation_count = 0;
    uint64_t site_reserved_count = 0;
    uint64_t site_active_count = 0;
    uint64_t site_pending_count = 0;
    uint64_t site_cached_count = 0;
};

struct ProcessMemoryObservation {
    bool available = false;
    uint64_t process_footprint_bytes = 0;
    uint64_t physical_memory_bytes = 0;
    uint64_t system_available_bytes = 0;
    std::string source;
};

/* System-wide monotonically increasing VM counters.  They are used as a
 * conservative request-level acceptance signal: a positive swapout delta
 * invalidates a constrained request, but is not attributed to this process
 * or treated as additional capacity. */
struct SwapActivityObservation {
    bool available = false;
    uint64_t swapins = 0;
    uint64_t swapouts = 0;
    uint64_t compressed_pages = 0;
    std::string source;
};

struct MemoryAdmissionMetrics {
    uint64_t budget_bytes = 0;
    uint64_t planned_increment_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t planned_process_upper_bytes = 0;
    uint64_t allocation_ceiling_bytes = 0;
    uint64_t ledger_budget_bytes = 0;
    uint64_t ledger_process_baseline_bytes = 0;
    uint64_t ledger_storage_bytes = 0;
    uint64_t ledger_known_bytes = 0;
    uint64_t ledger_committed_bytes = 0;
    uint64_t ledger_active_bytes = 0;
    uint64_t ledger_reserved_bytes = 0;
    uint64_t ledger_pending_release_bytes = 0;
    uint64_t ledger_cached_bytes = 0;
    uint64_t ledger_unknown_bytes = 0;
    uint64_t ledger_peak_unknown_bytes = 0;
    uint64_t initial_process_footprint_bytes = 0;
    uint64_t peak_process_footprint_bytes = 0;
    uint64_t final_process_footprint_bytes = 0;
    uint64_t unattributed_process_footprint_bytes = 0;
    uint64_t peak_unattributed_process_footprint_bytes = 0;
    uint64_t peak_observed_over_budget_bytes = 0;
    uint64_t system_available_bytes_at_admission = 0;
    uint64_t final_system_available_bytes = 0;
    uint64_t minimum_system_available_bytes = 0;
    uint64_t ledger_peak_committed_bytes = 0;
    uint64_t ledger_storage_count = 0;
    uint64_t ledger_site_allocation_count = 0;
    uint64_t ledger_site_reserved_count = 0;
    uint64_t ledger_site_active_count = 0;
    uint64_t ledger_site_pending_count = 0;
    uint64_t ledger_site_cached_count = 0;
    uint64_t observation_count = 0;
    bool observed_within_budget = true;
    bool tainted = false;
    bool worker_quarantined = false;
    uint64_t pressure_transitions = 0;
    std::string pressure_state = "normal";
    std::string execution_state = "unavailable";
    uint64_t watchdog_sample_count = 0;
    uint64_t watchdog_dropped_samples = 0;
    bool watchdog_critical = false;
    std::string watchdog_failure_reason;
    uint64_t trace_event_count = 0;
    uint64_t trace_dropped_events = 0;
    bool trace_overflowed = false;
    uint64_t explicit_epoch_transition_count = 0;
    uint64_t automatic_epoch_transition_count = 0;
    uint64_t current_epoch = 0;
    uint64_t planned_peak_epoch = 0;
    uint64_t schedule_event_count = 0;
    uint64_t schedule_event_attempted_count = 0;
    uint64_t schedule_event_rejected_count = 0;
    uint64_t schedule_expected_event_count = 0;
    uint64_t schedule_mismatch_count = 0;
    uint64_t schedule_next_sequence = 0;
    std::string schedule_cursor_state = "disabled";
    std::string schedule_first_failure;
    bool schedule_cursor_complete = false;
    bool swap_observation_available = false;
    bool swap_counter_invalid = false;
    bool swap_activity_detected = false;
    uint64_t swap_sample_count = 0;
    uint64_t swapins_begin = 0;
    uint64_t swapins_end = 0;
    uint64_t swapins_delta = 0;
    uint64_t swapouts_begin = 0;
    uint64_t swapouts_end = 0;
    uint64_t swapouts_delta = 0;
    uint64_t compressed_pages_begin = 0;
    uint64_t compressed_pages_end = 0;
    std::string swap_first_observed_phase;
    std::string swap_observation_source;
    std::string failure_disposition = "none";
    std::string failure_reason;
    std::string cleanup_failure;
    std::string last_phase;
    std::string observation_source;
};

struct PendingReleaseId {
    uint64_t value = 0;
    explicit operator bool() const { return value != 0; }
};

struct MemorySiteInstanceConstraint {
    uint32_t instance_id = 0;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    uint64_t alias_group = 0;
};

struct MemorySiteReservationConstraint {
    std::string site_id;
    uint64_t epoch = 0;
    uint32_t maximum_live_instances = 0;
    uint64_t aggregate_upper_bytes = 0;
    std::vector<MemorySiteInstanceConstraint> instances;
};

namespace detail {
struct MemoryLedgerState;
struct StorageKey;
struct SiteKey;
} // namespace detail

class StorageLease;

class MemoryReservation {
  public:
    MemoryReservation() = default;
    MemoryReservation(MemoryReservation &&) noexcept;
    MemoryReservation &operator=(MemoryReservation &&) noexcept;
    MemoryReservation(const MemoryReservation &) = delete;
    MemoryReservation &operator=(const MemoryReservation &) = delete;
    ~MemoryReservation() noexcept;

    explicit operator bool() const { return reservation_id_ != 0; }
    uint64_t reserved_bytes() const { return reserved_bytes_; }
    StorageLease commit(const StorageId &storage);
    void cancel() noexcept;

  private:
    friend class MemoryLedger;
    MemoryReservation(std::shared_ptr<detail::MemoryLedgerState>, uint64_t,
                      uint64_t);

    std::shared_ptr<detail::MemoryLedgerState> state_;
    uint64_t reservation_id_ = 0;
    uint64_t reserved_bytes_ = 0;
};

class StorageLease {
  public:
    StorageLease();
    StorageLease(StorageLease &&) noexcept;
    StorageLease &operator=(StorageLease &&) noexcept;
    StorageLease(const StorageLease &) = delete;
    StorageLease &operator=(const StorageLease &) = delete;
    ~StorageLease() noexcept;

    explicit operator bool() const { return active_; }
    uint64_t capacity() const { return capacity_; }
    PendingReleaseId retire();
    void cache();
    void release() noexcept;

  private:
    friend class MemoryReservation;
    friend class MemoryLedger;
    StorageLease(std::shared_ptr<detail::MemoryLedgerState>,
                 const detail::StorageKey &, uint64_t,
                 std::unique_ptr<detail::SiteKey> = {});

    std::shared_ptr<detail::MemoryLedgerState> state_;
    std::unique_ptr<detail::StorageKey> key_;
    std::unique_ptr<detail::SiteKey> site_key_;
    uint64_t capacity_ = 0;
    bool active_ = false;
};

class MemoryLedger {
  public:
    explicit MemoryLedger(uint64_t budget_bytes,
                          uint64_t process_baseline_bytes = 0);

    std::optional<MemoryReservation> try_reserve(
        MemoryClass memory_class, uint64_t upper_bytes,
        std::string tag = {});
    void set_site_constraints_required(bool required);
    std::optional<MemoryReservation> try_reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint);
    void validate_site_epoch(uint64_t epoch) const;
    void set_process_baseline(uint64_t baseline_bytes);
    std::optional<StorageLease> reactivate_cached(const StorageId &storage);
    bool drop_cached(const StorageId &storage) noexcept;
    bool complete_pending(PendingReleaseId pending,
                          bool retain_as_cache = false) noexcept;
    MemorySnapshot snapshot() const;

  private:
    std::shared_ptr<detail::MemoryLedgerState> state_;
};

ProcessMemoryObservation observe_process_memory();
using ProcessMemoryObserver =
    std::function<ProcessMemoryObservation()>;
SwapActivityObservation observe_swap_activity();
using SwapActivityObserver =
    std::function<SwapActivityObservation()>;

class MemoryAdmission {
  public:
    MemoryAdmission(uint64_t budget_bytes, uint64_t planned_increment_bytes,
                    uint64_t system_reserve_bytes);
    MemoryAdmission(uint64_t budget_bytes, uint64_t planned_increment_bytes,
                    uint64_t system_reserve_bytes,
                    ProcessMemoryObserver observer);
    MemoryAdmission(uint64_t budget_bytes, uint64_t planned_increment_bytes,
                    uint64_t framework_upper_bytes,
                    uint64_t system_reserve_bytes,
                    ProcessMemoryObserver observer);
    MemoryAdmission(const MemoryAdmission &) = delete;
    MemoryAdmission &operator=(const MemoryAdmission &) = delete;

    void checkpoint(const std::string &phase);
    MemoryAdmissionMetrics metrics() const;
    MemoryLedger &ledger() { return *ledger_; }
    /* The current runtime serializes GPU jobs, so the request envelope is an
     * admission ceiling rather than a second ledger reservation. Concrete
     * allocations are charged directly against a ledger whose budget is the
     * smaller planned upper bound. This avoids charging root+child twice. */
    std::optional<MemoryReservation> try_reserve(
        MemoryClass memory_class, uint64_t upper_bytes,
        std::string tag = {});
    std::optional<MemoryReservation> try_reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint);
    void validate_site_epoch(uint64_t epoch) const;
    MemorySnapshot snapshot() const { return ledger_->snapshot(); }

  private:
    mutable std::mutex observation_mutex_;
    std::unique_ptr<MemoryLedger> ledger_;
    ProcessMemoryObserver observer_;
    MemoryAdmissionMetrics metrics_;
    bool framework_upper_enforced_ = false;
};

const char *memory_class_name(MemoryClass memory_class);

} // namespace tc
