#include "memory_trace.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace tc;

namespace {

void test_trace_preserves_order_and_fields() {
    MemoryTraceBuffer trace(4);
    trace.record(MemoryTraceEventKind::State, "running", {}, 0);
    trace.record(MemoryTraceEventKind::Reserve, "denoise",
                 "h3.dit.activation.core_input", 2, 100, 0,
                 40, 100, 0, 3, 1);
    auto events = trace.drain();
    assert(events.size() == 2);
    assert(events[0].sequence == 1);
    assert(events[0].kind == MemoryTraceEventKind::State);
    assert(events[1].sequence == 2);
    assert(events[1].kind == MemoryTraceEventKind::Reserve);
    assert(events[1].phase_id != 0 && events[1].site_id != 0);
    assert(events[1].epoch == 2);
    assert(events[1].upper_bytes == 100);
    assert(events[1].committed_bytes == 40);
    assert(events[1].reserved_bytes == 100);
    assert(events[1].stage_id == 3 && events[1].slot_id == 1);
    assert(trace.drain().empty());
}

void test_trace_overflow_keeps_newest_events() {
    MemoryTraceBuffer trace(2);
    trace.record(MemoryTraceEventKind::State, "one", {}, 0);
    trace.record(MemoryTraceEventKind::State, "two", {}, 0);
    trace.record(MemoryTraceEventKind::State, "three", {}, 0);
    const auto status = trace.status();
    assert(status.event_count == 3);
    assert(status.dropped_events == 1);
    assert(status.overflowed);
    const auto events = trace.drain();
    assert(events.size() == 2);
    assert(events[0].sequence == 2);
    assert(events[1].sequence == 3);
}

void test_event_names_are_stable() {
    assert(std::string(memory_trace_event_kind_name(
               MemoryTraceEventKind::Checkpoint)) == "checkpoint");
    assert(std::string(memory_trace_event_kind_name(
               MemoryTraceEventKind::Failure)) == "failure");
}

} // namespace

int main() {
    test_trace_preserves_order_and_fields();
    test_trace_overflow_keeps_newest_events();
    test_event_names_are_stable();
    std::cout << "memory trace tests passed\n";
    return 0;
}
