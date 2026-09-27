#include "memory_watchdog.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace tc;

namespace {

ProcessMemoryObservation observation(uint64_t footprint,
                                     uint64_t available = 1000) {
    return {true, footprint, 2000, available, "synthetic"};
}

MemorySnapshot snapshot(uint64_t known, uint64_t reserved = 0) {
    MemorySnapshot result;
    result.known_bytes = known;
    result.reserved_bytes = reserved;
    return result;
}

void test_normal_sample_is_bounded_and_drainable() {
    MemoryWatchdog watchdog({100, 20, 30, 4});
    watchdog.sample(observation(50), snapshot(40, 5), "denoise");
    const auto status = watchdog.status();
    assert(status.sample_count == 1);
    assert(!status.critical && status.dropped_samples == 0);
    const auto values = watchdog.drain_samples();
    assert(values.size() == 1);
    assert(values.front().sequence == 1);
    assert(values.front().process_footprint_bytes == 50);
    assert(values.front().known_bytes == 40);
    assert(values.front().reserved_bytes == 5);
    assert(values.front().unattributed_bytes == 10);
    assert(!values.front().over_budget);
    assert(!values.front().over_framework);
    assert(!values.front().under_system_reserve);
    assert(watchdog.drain_samples().empty());
}

void test_pressure_is_sticky_and_preserves_first_reason() {
    MemoryWatchdog watchdog({100, 20, 30, 8});
    watchdog.sample(observation(70), snapshot(40), "framework");
    auto status = watchdog.status();
    assert(status.critical && status.last_over_framework);
    assert(status.first_failure_reason.find("framework upper") !=
           std::string::npos);
    const auto first = status.first_failure_reason;
    watchdog.sample(observation(101), snapshot(100), "budget");
    status = watchdog.status();
    assert(status.critical && status.last_over_budget);
    assert(status.first_failure_reason == first);
    watchdog.sample(observation(50, 29), snapshot(40), "reserve");
    status = watchdog.status();
    assert(status.critical && status.last_under_system_reserve);
    assert(status.first_failure_reason == first);
}

void test_ring_overflow_is_a_sticky_failure() {
    MemoryWatchdog watchdog({100, 20, 0, 2});
    watchdog.sample(observation(10), snapshot(10), "one");
    watchdog.sample(observation(20), snapshot(20), "two");
    watchdog.sample(observation(30), snapshot(30), "three");
    const auto status = watchdog.status();
    assert(status.sample_count == 3);
    assert(status.dropped_samples == 1);
    assert(status.critical);
    assert(status.first_failure_reason.find("overflow") != std::string::npos);
    const auto values = watchdog.drain_samples();
    assert(values.size() == 2);
    assert(values[0].sequence == 2);
    assert(values[1].sequence == 3);
}

void test_explicit_failure_is_sticky_without_a_sample() {
    MemoryWatchdog watchdog({100, 20, 0, 2});
    watchdog.mark_failure("synthetic watchdog failure", "fault");
    const auto status = watchdog.status();
    assert(status.critical);
    assert(status.sample_count == 0);
    assert(status.first_failure_reason == "synthetic watchdog failure");
    assert(status.last_phase_id != 0);
}

} // namespace

int main() {
    test_normal_sample_is_bounded_and_drainable();
    test_pressure_is_sticky_and_preserves_first_reason();
    test_ring_overflow_is_a_sticky_failure();
    test_explicit_failure_is_sticky_without_a_sample();
    std::cout << "memory watchdog tests passed\n";
    return 0;
}
