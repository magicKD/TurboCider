#include "../../native/backends/ane_memory.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace tc::ane;
using Policy = StreamingAllocationAdmission;
using Pressure = StreamingMemoryPressure;
constexpr uint64_t MiB = uint64_t(1) << 20;
constexpr uint64_t GiB = uint64_t(1) << 30;
constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
const MemoryLimits limits{4 * GiB, 24 * GiB};

StreamingMemoryObservation healthy() {
    return {{true, 24 * GiB, 512 * MiB, 8 * GiB, 256 * MiB, 2 * GiB},
        4 * GiB, Pressure::Normal};
}

void check_policy(const StreamingMemoryObservation &observation, uint64_t growth,
                  MemoryDenial expected_guarded) {
    const auto original = admit_streaming_memory(observation, limits, growth);
    assert(original.denial == expected_guarded);
    const auto guarded = admit_required_streaming_memory(observation, limits, growth, Policy::Guarded);
    assert(guarded.denial == original.denial && guarded.headroom_bytes == original.headroom_bytes);
    const auto managed = admit_required_streaming_memory(observation, limits, growth, Policy::SystemManaged);
    assert(managed.allowed() && managed.headroom_bytes == 0);
    // A required allocation does not alter optional caching's admission policy.
    const auto after = admit_streaming_memory(observation, limits, growth);
    assert(after.denial == original.denial && after.headroom_bytes == original.headroom_bytes);
}

void required_allocations_delegate_capacity_to_the_system() {
    auto observation = healthy();
    check_policy(observation, 512 * MiB, MemoryDenial::None);
    observation.pressure = Pressure::Warning;
    check_policy(observation, 512 * MiB, MemoryDenial::None);
    observation.pressure = Pressure::Critical;
    check_policy(observation, 512 * MiB, MemoryDenial::SystemPressure);

    observation = healthy();
    observation.memory.free_bytes = GiB;
    observation.file_backed_non_speculative_bytes = 2 * GiB;
    check_policy(observation, 0, MemoryDenial::SystemReserve);
    observation.pressure = Pressure::Warning;
    check_policy(observation, 512 * MiB, MemoryDenial::SystemReserve);
    observation.file_backed_non_speculative_bytes = 3 * GiB + 512 * MiB - 1;
    check_policy(observation, 512 * MiB, MemoryDenial::GrowthLimit);

    observation = healthy();
    observation.memory.process_bytes = 20 * GiB + 1;
    check_policy(observation, 0, MemoryDenial::ProcessReserve);
    observation = healthy();
    observation.memory.mlx_active_bytes = 20 * GiB + 1;
    check_policy(observation, 0, MemoryDenial::MlxReserve);
    observation = healthy();
    check_policy(observation, maximum, MemoryDenial::GrowthLimit);

    for (int missing = 0; missing != 3; ++missing) {
        observation = healthy();
        if (missing == 0) observation.memory.available = false;
        if (missing == 1) observation.pressure.reset();
        if (missing == 2) observation.file_backed_non_speculative_bytes.reset();
        check_policy(observation, maximum, MemoryDenial::Unavailable);
    }
    for (uint32_t unknown : {0u, 3u, 5u, 6u, 7u, std::numeric_limits<uint32_t>::max()}) {
        observation = healthy();
        observation.pressure = streaming_memory_pressure_from_flags(unknown);
        check_policy(observation, maximum, MemoryDenial::Unavailable);
    }
    for (int invalid = 0; invalid != 6; ++invalid) {
        for (auto pressure : {Pressure::Normal, Pressure::Warning, Pressure::Critical}) {
            observation = healthy();
            observation.pressure = pressure;
            if (invalid == 0) observation.memory.physical_bytes = 0;
            if (invalid == 1) observation.memory.process_bytes = maximum;
            if (invalid == 2) observation.memory.mlx_active_bytes = maximum;
            if (invalid == 3) observation.memory.free_bytes = maximum;
            if (invalid == 4) observation.memory.inactive_bytes = maximum;
            if (invalid == 5) observation.file_backed_non_speculative_bytes = maximum;
            check_policy(observation, maximum, MemoryDenial::Invalid);
        }
    }
    const auto managed = admit_required_streaming_memory({}, {maximum, 0}, maximum, Policy::SystemManaged);
    assert(managed.allowed() && managed.headroom_bytes == 0);
    const auto invalid_policy = admit_required_streaming_memory(healthy(), limits, 0,
        static_cast<Policy>(99));
    assert(!invalid_policy.allowed() && invalid_policy.denial == MemoryDenial::Invalid);
}

void optional_cache_budget_is_still_guarded() {
    auto observation = healthy();
    observation.pressure = Pressure::Critical;
    assert(admit_streaming_memory(observation, limits, 0).denial == MemoryDenial::SystemPressure);
    observation.pressure.reset();
    assert(admit_streaming_memory(observation, limits, 0).denial == MemoryDenial::Unavailable);
    const auto memory = healthy().memory;
    const MemoryLimits optional_limits{4 * GiB, 2 * GiB};
    assert(admit_memory(memory, optional_limits, 2 * GiB + 1, 0).denial == MemoryDenial::OptionalLimit);
    assert(admit_memory(memory, optional_limits, GiB, GiB + 1).denial == MemoryDenial::GrowthLimit);
    assert(admit_memory(memory, optional_limits, GiB, GiB).allowed());
}

void required_runtime_check_has_no_heuristic_wait_or_allocation() {
    // These are accounting inputs only. No allocator or model is linked here;
    // passing their maximum values must not attempt an enormous allocation.
    require_streaming_memory_headroom(maximum, maximum, "system-managed CPU probe", Policy::SystemManaged);
    for (bool explicit_policy : {false, true}) {
        bool rejected = false;
        try {
            if (explicit_policy)
                require_streaming_memory_headroom(maximum, maximum, "guarded CPU probe", Policy::Guarded);
            else
                require_streaming_memory_headroom(maximum, maximum, "guarded CPU probe");
        } catch (const std::invalid_argument &error) {
            rejected = true;
            const std::string reason = error.what();
#if defined(__APPLE__)
            assert(reason.find("observation_invalid") != std::string::npos);
#else
            assert(reason.find("observation_unavailable") != std::string::npos);
#endif
        }
        assert(rejected);
    }
    bool invalid_policy_rejected = false;
    try {
        require_streaming_memory_headroom(maximum, maximum, "invalid policy CPU probe",
            static_cast<Policy>(99));
    } catch (const std::invalid_argument &) {
        invalid_policy_rejected = true;
    }
    assert(invalid_policy_rejected);
}
} // namespace

int main() {
    required_allocations_delegate_capacity_to_the_system();
    optional_cache_budget_is_still_guarded();
    required_runtime_check_has_no_heuristic_wait_or_allocation();
    std::cout << "PASS streaming allocation: system-managed required weights, guarded optional caching and fail-closed policy\n";
}
