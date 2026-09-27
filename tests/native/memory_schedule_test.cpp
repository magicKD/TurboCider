#include "memory_schedule.hpp"

#include <cassert>
#include <iostream>

using namespace tc;

namespace {

MemoryScheduleBindingSpec binding(
        uint32_t stage, uint32_t action, uint64_t epoch,
        uint32_t flags = 0, uint32_t slot = TC_MEMORY_INDEX_NONE,
        bool checkpoint = false) {
    MemoryScheduleBindingSpec result;
    result.key.stage = stage;
    result.key.action = action;
    result.key.branch = TC_MEMORY_BRANCH_COMMON;
    result.expected_slot = slot;
    result.required_flags = flags;
    result.epoch = epoch;
    result.checkpoint_after = checkpoint;
    return result;
}

std::vector<MemoryScheduleBindingSpec> schedule() {
    return {
        binding(TC_MEMORY_STAGE_ADMISSION, TC_MEMORY_ACTION_END, 0),
        binding(TC_MEMORY_STAGE_DENOISER, TC_MEMORY_ACTION_COMPUTE,
                2, 0, 0),
        binding(TC_MEMORY_STAGE_TERMINAL_DRAIN, TC_MEMORY_ACTION_END,
                4, TC_MEMORY_EVENT_SAFE_POINT |
                       TC_MEMORY_EVENT_GPU_DRAINED |
                       TC_MEMORY_EVENT_TERMINAL,
                TC_MEMORY_INDEX_NONE, true),
    };
}

template <typename Function>
void expect_rejected(Function function, const char *needle) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(needle) != std::string::npos;
    }
    assert(rejected);
}

void test_compile_is_deterministic_and_binds_fields() {
    const auto first = compile_memory_schedule(schedule(), {0, 1, 2, 3, 4});
    const auto second = compile_memory_schedule(schedule(), {0, 1, 2, 3, 4});
    assert(first.schema == "turbocider.memory_schedule.v1");
    assert(first.revision.size() == 64);
    assert(first.revision == second.revision);
    assert(first.bindings.size() == 3);
    assert(first.bindings[0].sequence == 0);
    assert(first.bindings[1].sequence == 1);
    assert(first.bindings[1].expected_slot == 0);
    assert(first.bindings[2].sequence == 2);
    assert(first.bindings[2].checkpoint_after);

    auto changed = schedule();
    changed[1].expected_slot = 1;
    assert(compile_memory_schedule(changed, {0, 1, 2, 3, 4}).revision !=
           first.revision);
}

void test_compile_rejects_invalid_sequences() {
    auto value = schedule();
    value[1].epoch = 5;
    expect_rejected(
        [&] { (void)compile_memory_schedule(value, {0, 1, 2, 3, 4}); },
        "unknown epoch");

    value = schedule();
    value[1].epoch = 3;
    value[2].epoch = 2;
    expect_rejected(
        [&] { (void)compile_memory_schedule(value, {0, 1, 2, 3, 4}); },
        "move backwards");

    value = schedule();
    value[1] = value[0];
    expect_rejected(
        [&] { (void)compile_memory_schedule(value, {0, 1, 2, 3, 4}); },
        "duplicate semantic event");

    value = schedule();
    value.back().required_flags = TC_MEMORY_EVENT_TERMINAL;
    expect_rejected(
        [&] { (void)compile_memory_schedule(value, {0, 1, 2, 3, 4}); },
        "terminal event must be a drained safe point");

    value = schedule();
    value.back().required_flags &= ~TC_MEMORY_EVENT_TERMINAL;
    expect_rejected(
        [&] { (void)compile_memory_schedule(value, {0, 1, 2, 3, 4}); },
        "no terminal event");
}

void test_c_event_conversion_and_name() {
    tc_memory_schedule_event_v1 event{};
    event.struct_size = sizeof(event);
    event.version = TC_MEMORY_SCHEDULE_EVENT_VERSION_1;
    event.stage = TC_MEMORY_STAGE_DENOISER;
    event.action = TC_MEMORY_ACTION_COMPUTE;
    event.step = 3;
    event.block = 7;
    event.tile = TC_MEMORY_INDEX_NONE;
    event.branch = TC_MEMORY_BRANCH_VIDEO;
    event.slot = 1;
    const auto key = memory_schedule_event_key(event);
    assert(key.step == 3 && key.block == 7);
    assert(memory_schedule_event_name(key) ==
           "denoiser.compute.step3.block7.branch3");
}

} // namespace

int main() {
    test_compile_is_deterministic_and_binds_fields();
    test_compile_rejects_invalid_sequences();
    test_c_event_conversion_and_name();
    std::cout << "memory schedule tests passed\n";
    return 0;
}
