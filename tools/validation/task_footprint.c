/* Self-process accounting only; never interprets this as ANE-exclusive RAM. */
#include <mach/mach.h>
#include <stdint.h>

int tc_task_footprint(uint64_t *values) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                    (task_info_t)&info, &count);
    if (status != KERN_SUCCESS || count < TASK_VM_INFO_REV1_COUNT) return -1;
    values[0] = info.phys_footprint;
    values[1] = info.resident_size;
    values[2] = count >= TASK_VM_INFO_REV3_COUNT ? info.ledger_phys_footprint_peak : 0;
    return 0;
}

int tc_host_vm(uint64_t *values) {
    vm_statistics64_data_t info = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t status = host_statistics64(host, HOST_VM_INFO64,
                                             (host_info64_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (status != KERN_SUCCESS) return -1;
    values[0] = info.free_count;
    values[1] = info.compressor_page_count;
    values[2] = info.pageins;
    values[3] = info.pageouts;
    values[4] = info.compressions;
    values[5] = info.decompressions;
    values[6] = info.swapins;
    values[7] = info.swapouts;
    return 0;
}
