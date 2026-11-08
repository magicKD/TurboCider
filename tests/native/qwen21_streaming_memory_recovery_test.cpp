#include "../../native/models/qwen21/streaming_memory_recovery.hpp"

#include <cassert>
#include <iostream>
#include <limits>

namespace {
using Recovery = tc::qwen21::StreamingMemoryRecovery;
using Action = Recovery::Action;
using Denial = tc::ane::MemoryDenial;
using Pressure = tc::ane::StreamingMemoryPressure;
using namespace std::chrono_literals;
constexpr uint64_t MiB = uint64_t(1) << 20;
constexpr uint64_t GiB = uint64_t(1) << 30;
const Recovery::TimePoint origin{std::chrono::seconds(10)};

tc::ane::StreamingMemoryObservation healthy() {
    return {{true, 24 * GiB, 512 * MiB, 8 * GiB, 256 * MiB, 2 * GiB}, 4 * GiB, Pressure::Normal};
}

void initial_health_and_screenshot_recovery() {
    Recovery immediate(512 * MiB, origin);
    const auto ready = immediate.evaluate(healthy(), origin);
    assert(ready.action == Action::Ready && ready.decision.allowed());
    assert(!immediate.waited() && immediate.elapsed_seconds(origin) == 0.0);
    assert(immediate.started() == origin);

    // The user's real screenshot has enough credited pages but warning pressure.
    auto screenshot = healthy();
    screenshot.memory.process_bytes = 207 * MiB;
    screenshot.memory.free_bytes = 1489 * MiB;
    screenshot.memory.inactive_bytes = 4290 * MiB;
    screenshot.file_backed_non_speculative_bytes = 7385 * MiB;
    screenshot.pressure = Pressure::Warning;
    Recovery warning(512 * MiB, origin);
    const auto warned = warning.evaluate(screenshot, origin);
    assert(warned.action == Action::Ready && warned.decision.allowed());
    assert(!warning.waited());

    // Critical pressure still blocks allocation until it recovers.
    screenshot.pressure = Pressure::Critical;
    Recovery recovering(512 * MiB, origin);
    const auto waiting = recovering.evaluate(screenshot, origin);
    assert(waiting.action == Action::Wait && waiting.decision.denial == Denial::SystemPressure);
    assert(recovering.waited());
    assert(recovering.evaluate(screenshot, origin + 15s).action == Action::Wait);
    // Historical complete-image evidence needed about 41.7 s to recover.
    screenshot.pressure = Pressure::Normal;
    assert(recovering.evaluate(screenshot, origin + 42s).action == Action::Wait);
    assert(recovering.evaluate(screenshot, origin + 42499ms).action == Action::Wait);
    assert(recovering.evaluate(screenshot, origin + 42500ms).action == Action::Ready);
    assert(recovering.elapsed_seconds(origin + 42500ms) == 42.5);
}

void critical_resets_stability() {
    auto observation = healthy();
    observation.pressure = Pressure::Critical;
    Recovery recovering(0, origin);
    assert(recovering.evaluate(observation, origin).action == Action::Wait);
    observation.pressure = Pressure::Normal;
    assert(recovering.evaluate(observation, origin + 10s).action == Action::Wait);
    observation.pressure = Pressure::Critical;
    assert(recovering.evaluate(observation, origin + 10400ms).action == Action::Wait);
    observation.pressure = Pressure::Warning;
    assert(recovering.evaluate(observation, origin + 10900ms).action == Action::Wait);
    assert(recovering.evaluate(observation, origin + 11399ms).action == Action::Wait);
    assert(recovering.evaluate(observation, origin + 11400ms).action == Action::Ready);
}

void reserve_and_full_growth_remain_required() {
    auto observation = healthy();
    observation.pressure = Pressure::Warning;
    observation.memory.free_bytes = GiB;
    observation.file_backed_non_speculative_bytes = 3 * GiB - 1;
    Recovery recovering(512 * MiB, origin);
    const auto reserve = recovering.evaluate(observation, origin);
    assert(reserve.action == Action::Wait && reserve.decision.denial == Denial::SystemReserve);
    observation.file_backed_non_speculative_bytes = 3 * GiB + 512 * MiB - 1;
    const auto growth = recovering.evaluate(observation, origin + 1s);
    assert(growth.action == Action::Wait && growth.decision.denial == Denial::GrowthLimit);
    assert(growth.decision.headroom_bytes == 512 * MiB - 1);
    observation.file_backed_non_speculative_bytes = 3 * GiB + 512 * MiB;
    assert(recovering.evaluate(observation, origin + 2s).action == Action::Wait);
    assert(recovering.evaluate(observation, origin + 2500ms).action == Action::Ready);

    // A zero-growth phase is separately admissible at the full reserve.
    observation.file_backed_non_speculative_bytes = 3 * GiB;
    Recovery no_growth(0, origin);
    assert(no_growth.evaluate(observation, origin).action == Action::Ready);
}

void timeout_is_bounded_and_stability_cannot_extend_it() {
    auto observation = healthy();
    observation.pressure = Pressure::Critical;
    Recovery stuck(0, origin);
    assert(stuck.evaluate(observation, origin).action == Action::Wait);
    assert(stuck.evaluate(observation, origin + 119999ms).action == Action::Wait);
    assert(stuck.evaluate(observation, origin + 120s).action == Action::TimedOut);
    observation.pressure = Pressure::Normal;
    assert(stuck.evaluate(observation, origin + 121s).action == Action::TimedOut);

    observation.pressure = Pressure::Critical;
    Recovery too_late(0, origin);
    assert(too_late.evaluate(observation, origin).action == Action::Wait);
    observation.pressure = Pressure::Normal;
    assert(too_late.evaluate(observation, origin + 119750ms).action == Action::Wait);
    assert(too_late.evaluate(observation, origin + 120s).action == Action::TimedOut);

    observation.pressure = Pressure::Critical;
    Recovery at_boundary(0, origin);
    assert(at_boundary.evaluate(observation, origin).action == Action::Wait);
    observation.pressure = Pressure::Warning;
    assert(at_boundary.evaluate(observation, origin + 119500ms).action == Action::Wait);
    assert(at_boundary.evaluate(observation, origin + 120s).action == Action::Ready);
}

void unavailable_and_invalid_fail_fast() {
    assert(tc::ane::streaming_memory_pressure_from_flags(1) == Pressure::Normal);
    assert(tc::ane::streaming_memory_pressure_from_flags(2) == Pressure::Warning);
    assert(tc::ane::streaming_memory_pressure_from_flags(4) == Pressure::Critical);
    for (uint32_t unknown : {0u, 3u, 5u, 6u, 7u, std::numeric_limits<uint32_t>::max()}) {
        auto observation = healthy();
        observation.pressure = tc::ane::streaming_memory_pressure_from_flags(unknown);
        Recovery recovering(0, origin);
        const auto result = recovering.evaluate(observation, origin);
        assert(result.action == Action::Denied && result.decision.denial == Denial::Unavailable);
    }
    for (int missing = 0; missing != 3; ++missing) {
        auto observation = healthy();
        if (missing == 0) observation.memory.available = false;
        if (missing == 1) observation.pressure.reset();
        if (missing == 2) observation.file_backed_non_speculative_bytes.reset();
        Recovery recovering(0, origin);
        const auto result = recovering.evaluate(observation, origin);
        assert(result.action == Action::Denied && result.decision.denial == Denial::Unavailable);
        assert(!recovering.waited());
    }
    for (int invalid = 0; invalid != 6; ++invalid) {
        for (const auto pressure : {Pressure::Normal, Pressure::Warning, Pressure::Critical}) {
            auto observation = healthy();
            constexpr auto max = std::numeric_limits<uint64_t>::max();
            if (invalid == 0) observation.memory.physical_bytes = 0;
            if (invalid == 1) observation.memory.process_bytes = max;
            if (invalid == 2) observation.memory.mlx_active_bytes = max;
            if (invalid == 3) observation.memory.free_bytes = max;
            if (invalid == 4) observation.memory.inactive_bytes = max;
            if (invalid == 5) observation.file_backed_non_speculative_bytes = max;
            observation.pressure = pressure;
            Recovery recovering(0, origin);
            const auto result = recovering.evaluate(observation, origin);
            assert(result.action == Action::Denied && result.decision.denial == Denial::Invalid);
        }
    }
    for (bool mlx : {false, true}) {
        auto observation = healthy();
        if (mlx) observation.memory.mlx_active_bytes = 20 * GiB + 1;
        else observation.memory.process_bytes = 20 * GiB + 1;
        Recovery recovering(0, origin);
        const auto result = recovering.evaluate(observation, origin);
        assert(result.action == Action::Denied);
        assert(result.decision.denial == (mlx ? Denial::MlxReserve : Denial::ProcessReserve));
    }
}
} // namespace

int main() {
    initial_health_and_screenshot_recovery();
    critical_resets_stability();
    reserve_and_full_growth_remain_required();
    timeout_is_bounded_and_stability_cannot_extend_it();
    unavailable_and_invalid_fail_fast();
    std::cout << "PASS streaming memory recovery: screenshot, delayed recovery, stability, growth, deadline and fail-closed observations\n";
}
