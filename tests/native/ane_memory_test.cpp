#include "../../native/backends/ane_memory.hpp"

#include <array>
#include <cassert>
#include <iostream>

int main() {
    using namespace tc::ane;
    constexpr uint64_t GiB = uint64_t(1) << 30;
    constexpr uint64_t MiB = uint64_t(1) << 20;
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    assert(!plan_packed_conversion(0, 512, 16));
    assert(!plan_packed_conversion(max, 512, 16));
    assert(!plan_packed_conversion(512, max, 16));
    assert(plan_packed_conversion(3, 512, 16)->workers == 1);
    assert(plan_packed_conversion(512, 256, 0)->workers == 1);
    const auto conversion = plan_packed_conversion(10240, 3840, 128);
    assert(conversion && conversion->groups == 640 && conversion->workers == 8);
    assert(conversion->scratch_upper_bytes == 16384);
    assert(free_page_bytes(64, 16384) == MiB);
    assert(!free_page_bytes(1, 0) && !free_page_bytes(max, 16384));
    const MemoryLimits limits;
    const MemoryObservation normal{true, 64 * GiB, 24 * GiB, 32 * GiB, 18 * GiB};
    assert(admit_memory(normal, limits, 0, 2 * GiB).allowed());
    assert(admit_memory(normal, limits, 0, 2 * GiB + 1).denial == MemoryDenial::GrowthLimit);
    assert(admit_memory(normal, limits, GiB, GiB).allowed());
    assert(admit_memory(normal, limits, 2 * GiB + 1, 0).denial == MemoryDenial::OptionalLimit);
    auto sample = normal;
    sample.available = false;
    assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::Unavailable);
    for (int field = 0; field < 4; ++field) {
        sample = normal;
        if (field == 0) sample.physical_bytes = 0;
        if (field == 1) sample.process_bytes = max;
        if (field == 2) sample.mlx_active_bytes = max;
        if (field == 3) sample.free_bytes = max;
        assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::Invalid);
    }
    sample = normal;
    sample.free_bytes = 4 * GiB + 128 * MiB;
    assert(admit_memory(sample, limits, GiB, 128 * MiB).allowed());
    assert(admit_memory(sample, limits, GiB, 128 * MiB + 1).denial == MemoryDenial::GrowthLimit);
    sample.free_bytes = 4 * GiB - 1;
    assert(admit_memory(sample, limits, GiB, 0).denial == MemoryDenial::SystemReserve);
    // Half of inactive is credited, but not double-counted with free pages.
    sample = normal;
    sample.free_bytes = GiB;
    sample.inactive_bytes = 16 * GiB;
    assert(admit_memory(sample, limits, 0, 2 * GiB).allowed());
    sample.inactive_bytes = 2 * GiB;
    assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::SystemReserve);
    sample.inactive_bytes = max;
    assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::Invalid);
    sample = normal;
    sample.process_bytes = 60 * GiB + 1;
    assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::ProcessReserve);
    sample = normal;
    sample.mlx_active_bytes = 60 * GiB + 1;
    assert(admit_memory(sample, limits, 0, 0).denial == MemoryDenial::MlxReserve);

    const auto one = plan_host_scratch(1792, 4096, 12288, true, 0, 0);
    assert(one && one->output_bytes == 14 * MiB && one->hidden_bytes == 42 * MiB);
    assert(one->retained_bytes == 56 * MiB && one->new_payload_bytes == 56 * MiB);
    const auto three = plan_host_scratch(3 * 1792, 4096, 12288, true,
                                         one->output_bytes, one->hidden_bytes);
    assert(three && three->retained_bytes == 168 * MiB && three->new_payload_bytes == 168 * MiB);
    // Old vectors (56 MiB) coexist with new vectors (168 MiB), so net
    // growth (112 MiB) is not the correct admission for this allocation.
    sample = normal;
    sample.free_bytes = 4 * GiB + 128 * MiB;
    assert(admit_memory(sample, limits, one->retained_bytes, 112 * MiB).allowed());
    assert(!admit_memory(sample, limits, one->retained_bytes, three->new_payload_bytes).allowed());
    const auto base = plan_host_scratch(1792, 4096, 12288, false,
                                        three->output_bytes, three->hidden_bytes);
    assert(base && base->hidden_bytes == 0 && base->new_payload_bytes == 0 &&
           base->retained_bytes == three->retained_bytes);
    for (const auto args : {std::array<uint64_t, 3>{0, 64, 96}, {32, 0, 96},
                            {32, 64, 0}, {max, 64, 96}, {32, max, 96}, {32, 64, max}})
        assert(!plan_host_scratch(args[0], args[1], args[2], true, 0, 0));
    assert(!plan_host_scratch(1, 1, 1, true, max, 1));
    const auto actual = observe_runtime_memory(12345);
#if defined(__APPLE__)
    assert(actual.available && actual.physical_bytes && actual.process_bytes &&
           actual.free_bytes <= actual.physical_bytes && actual.mlx_active_bytes == 12345);
    std::cout << "Mach free_bytes=" << actual.free_bytes
              << " inactive_bytes=" << actual.inactive_bytes
              << " process_bytes=" << actual.process_bytes << '\n';
#else
    assert(!actual.available);
#endif
    std::cout << "PASS runtime ANE memory: admission, multi-chunk scratch and Mach observation\n";
}
