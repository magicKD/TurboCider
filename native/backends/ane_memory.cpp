#include "ane_memory.hpp"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace tc::ane {

MemoryObservation observe_runtime_memory(uint64_t mlx_active_bytes) {
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
    if (page_status != KERN_SUCCESS || stats_status != KERN_SUCCESS || count != HOST_VM_INFO64_COUNT)
        return result;
    const auto free = free_page_bytes(statistics.free_count, page_size);
    const auto inactive = free_page_bytes(statistics.inactive_count, page_size);
    if (!free || !inactive) return result;
    result = {true, physical, task.phys_footprint, *free, mlx_active_bytes, *inactive};
#else
    (void)mlx_active_bytes;
#endif
    return result;
}

} // namespace tc::ane
