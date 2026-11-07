#include "ane_memory.hpp"
#include <stdexcept>

#if defined(__APPLE__)
#include <cstddef>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace tc::ane {
namespace {
MemoryObservation observe_memory(uint64_t mlx_active_bytes, StreamingMemoryObservation *streaming) {
    MemoryObservation result;
#if defined(__APPLE__)
    task_vm_info_data_t task{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&task), &count) != KERN_SUCCESS ||
        count < TASK_VM_INFO_REV1_COUNT) return result;
    uint64_t physical = 0;
    size_t size = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &size, nullptr, 0) ||
        size != sizeof(physical) || !physical) return result;
    const auto host = mach_host_self();
    if (host == MACH_PORT_NULL) return result;
    vm_size_t page_size = 0;
    vm_statistics64_data_t statistics{};
    count = HOST_VM_INFO64_COUNT;
    const auto page_status = host_page_size(host, &page_size);
    const auto stats_status = host_statistics64(host, HOST_VM_INFO64,
                                               reinterpret_cast<host_info64_t>(&statistics), &count);
    mach_port_deallocate(mach_task_self(), host);
    // Older kernels can return a shorter structure than the build SDK defines.
    // Only require the fields consumed below, measured in Mach integer units.
    constexpr auto required_count =
        (offsetof(vm_statistics64_data_t, inactive_count) + sizeof(statistics.inactive_count)) /
        sizeof(integer_t);
    if (page_status != KERN_SUCCESS || stats_status != KERN_SUCCESS || count < required_count)
        return result;
    const auto free = free_page_bytes(statistics.free_count, page_size);
    const auto inactive = free_page_bytes(statistics.inactive_count, page_size);
    if (!free || !inactive) return result;
    result = {true, physical, task.phys_footprint, *free, mlx_active_bytes, *inactive};
    if (streaming) {
        constexpr auto external_count = (offsetof(vm_statistics64_data_t, external_page_count) +
            sizeof(statistics.external_page_count)) / sizeof(integer_t);
        if (count >= external_count) {
            // XNU includes speculative pages in free_count. Deduct them
            // from the file-backed pool before computing cache credit.
            const auto speculative = std::min(statistics.speculative_count, statistics.external_page_count);
            streaming->file_backed_non_speculative_bytes =
                free_page_bytes(statistics.external_page_count - speculative, page_size);
        }
        uint32_t pressure = 0;
        size = sizeof(pressure);
        // The Darwin sysctl publishes NOTE_MEMORYSTATUS dispatch flags, not
        // XNU's internal pressure enum. Normal is DISPATCH_MEMORYPRESSURE_NORMAL.
        if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &size, nullptr, 0) == 0 &&
            size == sizeof(pressure))
            streaming->pressure_normal = pressure == DISPATCH_MEMORYPRESSURE_NORMAL;
    }
#else
    (void)mlx_active_bytes;
    (void)streaming;
#endif
    return result;
}
} // namespace

MemoryObservation observe_runtime_memory(uint64_t mlx_active_bytes) {
    return observe_memory(mlx_active_bytes, nullptr);
}

StreamingMemoryObservation observe_streaming_memory(uint64_t mlx_active_bytes) {
    StreamingMemoryObservation result;
    result.memory = observe_memory(mlx_active_bytes, &result);
    return result;
}

void require_streaming_memory_headroom(uint64_t mlx_active_bytes, uint64_t growth_bytes,
                                       const char *context) {
    const auto observed = observe_streaming_memory(mlx_active_bytes);
    const auto admission = admit_streaming_memory(observed, {uint64_t(4) << 30, observed.memory.physical_bytes},
                                                  growth_bytes);
    if (admission.allowed()) return;
    throw std::invalid_argument(std::string(context) + ": " + memory_denial_reason(admission.denial, observed.memory) +
        "; non_speculative_file_mib=" + std::to_string(observed.file_backed_non_speculative_bytes.value_or(0) >> 20) +
        "; pressure=" + (observed.pressure_normal ? (*observed.pressure_normal ? "normal" : "elevated") : "unavailable"));
}

} // namespace tc::ane
