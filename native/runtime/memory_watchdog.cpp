#include "memory_watchdog.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace tc {
namespace {

uint64_t checked_sub(uint64_t larger, uint64_t smaller) {
    return larger > smaller ? larger - smaller : 0;
}

} // namespace

MemoryWatchdog::MemoryWatchdog(MemoryWatchdogConfig config)
    : config_(config), ring_(std::max<size_t>(config.ring_capacity, 1)) {
    require(config_.budget_bytes > 0,
            "memory_watchdog_invalid: budget is zero");
    require(config_.framework_upper_bytes <= config_.budget_bytes,
            "memory_watchdog_invalid: framework upper exceeds budget");
}

uint64_t MemoryWatchdog::phase_id(std::string_view phase) {
    /* FNV-1a is sufficient for a trace label; the string itself remains in
     * the normal result/trace path, while the bounded ring stores only a
     * fixed-width value. */
    uint64_t value = 1469598103934665603ULL;
    for (const unsigned char byte : phase) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

uint64_t MemoryWatchdog::monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void MemoryWatchdog::record_failure_locked(std::string reason) {
    status_.critical = true;
    if (status_.first_failure_reason.empty())
        status_.first_failure_reason = std::move(reason);
}

void MemoryWatchdog::sample(const ProcessMemoryObservation &observation,
                            const MemorySnapshot &snapshot,
                            std::string_view phase) {
    require(observation.available,
            "memory_observation_unreliable: watchdog sample unavailable");
    MemoryWatchdogSample value;
    value.monotonic_ns = monotonic_ns();
    value.process_footprint_bytes = observation.process_footprint_bytes;
    value.system_available_bytes = observation.system_available_bytes;
    value.known_bytes = snapshot.known_bytes;
    value.reserved_bytes = snapshot.reserved_bytes;
    value.unattributed_bytes = checked_sub(
        observation.process_footprint_bytes, snapshot.known_bytes);
    value.budget_bytes = config_.budget_bytes;
    value.framework_upper_bytes = config_.framework_upper_bytes;
    value.system_reserve_bytes = config_.system_reserve_bytes;
    value.phase_id = phase_id(phase);
    value.over_budget = observation.process_footprint_bytes > config_.budget_bytes;
    value.over_framework = value.unattributed_bytes >
        config_.framework_upper_bytes;
    value.under_system_reserve = config_.system_reserve_bytes > 0 &&
        observation.system_available_bytes < config_.system_reserve_bytes;

    std::lock_guard<std::mutex> lock(mutex_);
    value.sequence = ++status_.sample_count;
    if (size_ == ring_.size()) {
        head_ = (head_ + 1) % ring_.size();
        --size_;
        ++status_.dropped_samples;
        record_failure_locked(
            "memory_watchdog_overflow: bounded sample ring overwrote an older sample");
    }
    const size_t tail = (head_ + size_) % ring_.size();
    ring_[tail] = value;
    ++size_;
    status_.last_over_budget = value.over_budget;
    status_.last_over_framework = value.over_framework;
    status_.last_under_system_reserve = value.under_system_reserve;
    status_.last_phase_id = value.phase_id;
    if (value.over_budget)
        record_failure_locked(
            "memory_watchdog_critical: process footprint exceeds budget");
    else if (value.over_framework)
        record_failure_locked(
            "memory_watchdog_critical: unattributed footprint exceeds framework upper");
    else if (value.under_system_reserve)
        record_failure_locked(
            "memory_watchdog_critical: system available memory is below reserve");
}

void MemoryWatchdog::mark_failure(std::string_view reason,
                                  std::string_view phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.last_phase_id = phase_id(phase);
    record_failure_locked(std::string(reason));
}

MemoryWatchdogStatus MemoryWatchdog::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

std::vector<MemoryWatchdogSample> MemoryWatchdog::drain_samples() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MemoryWatchdogSample> result;
    result.reserve(size_);
    for (size_t index = 0; index < size_; ++index)
        result.push_back(ring_[(head_ + index) % ring_.size()]);
    head_ = 0;
    size_ = 0;
    return result;
}

} // namespace tc
