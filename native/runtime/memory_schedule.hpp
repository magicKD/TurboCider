#pragma once

#include "../core/memory_schedule_c.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tc {

struct MemoryScheduleEventKey {
    uint32_t stage = 0;
    uint32_t action = 0;
    uint32_t step = TC_MEMORY_INDEX_NONE;
    uint32_t block = TC_MEMORY_INDEX_NONE;
    uint32_t tile = TC_MEMORY_INDEX_NONE;
    uint32_t branch = TC_MEMORY_BRANCH_COMMON;

    bool operator==(const MemoryScheduleEventKey &) const = default;
    bool operator<(const MemoryScheduleEventKey &) const;
    std::string canonical() const;
};

struct MemoryScheduleBindingSpec {
    MemoryScheduleEventKey key;
    uint32_t expected_slot = TC_MEMORY_INDEX_NONE;
    uint32_t required_flags = 0;
    uint64_t epoch = 0;
    bool checkpoint_after = false;
};

struct CompiledScheduleBinding {
    uint64_t sequence = 0;
    MemoryScheduleEventKey key;
    uint32_t expected_slot = TC_MEMORY_INDEX_NONE;
    uint32_t required_flags = 0;
    uint64_t epoch = 0;
    bool checkpoint_after = false;
};

struct CompiledMemorySchedule {
    std::string schema = "turbocider.memory_schedule.v1";
    std::string revision;
    std::vector<CompiledScheduleBinding> bindings;
};

MemoryScheduleEventKey memory_schedule_event_key(
    const tc_memory_schedule_event_v1 &event);
std::string memory_schedule_event_name(const MemoryScheduleEventKey &key);
CompiledMemorySchedule compile_memory_schedule(
    const std::vector<MemoryScheduleBindingSpec> &bindings,
    const std::vector<uint64_t> &valid_epochs);

} // namespace tc
