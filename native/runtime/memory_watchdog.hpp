#pragma once

#include "memory_accounting.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace tc {

/*
 * The watchdog is deliberately owner-thread polled.  It consumes the
 * ProcessMemoryObservation already taken by MemoryAdmission::checkpoint()
 * instead of starting a background thread that could race Metal teardown or
 * allocate while the process is under pressure.  Reservation-before-
 * allocation remains the hard cap; this class only detects un-attributed or
 * instantaneous pressure at safe points.
 */
struct MemoryWatchdogConfig {
    uint64_t budget_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    size_t ring_capacity = 64;
};

struct MemoryWatchdogSample {
    uint64_t sequence = 0;
    uint64_t monotonic_ns = 0;
    uint64_t process_footprint_bytes = 0;
    uint64_t system_available_bytes = 0;
    uint64_t known_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t unattributed_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    uint64_t phase_id = 0;
    bool over_budget = false;
    bool over_framework = false;
    bool under_system_reserve = false;
};

struct MemoryWatchdogStatus {
    uint64_t sample_count = 0;
    uint64_t dropped_samples = 0;
    bool critical = false;
    bool last_over_budget = false;
    bool last_over_framework = false;
    bool last_under_system_reserve = false;
    uint64_t last_phase_id = 0;
    std::string first_failure_reason;
};

class MemoryWatchdog {
  public:
    explicit MemoryWatchdog(MemoryWatchdogConfig config);
    MemoryWatchdog(const MemoryWatchdog &) = delete;
    MemoryWatchdog &operator=(const MemoryWatchdog &) = delete;

    void sample(const ProcessMemoryObservation &observation,
                const MemorySnapshot &snapshot, std::string_view phase);
    void mark_failure(std::string_view reason, std::string_view phase);
    MemoryWatchdogStatus status() const;
    std::vector<MemoryWatchdogSample> drain_samples();

  private:
    static uint64_t phase_id(std::string_view phase);
    static uint64_t monotonic_ns();
    void record_failure_locked(std::string reason);

    MemoryWatchdogConfig config_;
    mutable std::mutex mutex_;
    std::vector<MemoryWatchdogSample> ring_;
    size_t head_ = 0;
    size_t size_ = 0;
    MemoryWatchdogStatus status_;
};

} // namespace tc
