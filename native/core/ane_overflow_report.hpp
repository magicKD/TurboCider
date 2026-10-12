#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tc::ane {
struct OverflowEvent {
    int layer = -1, rows = 0;
    uint64_t runtime_call_begin = 0, runtime_call_count = 0, retries = 0;
    float headroom_before = 1, headroom_after = 1;
    bool completed = false;
};
// Fixed-size session prefix: reporting never allocates on the FFN completion
// path and never retains weights, outputs or adapters. This is aggregated
// per-launch host telemetry, not a chunk index or physical engine trace.
struct OverflowReport {
    static constexpr size_t capacity = 32;
    std::array<OverflowEvent, capacity> events{};
    size_t size = 0;
    uint64_t dropped = 0;
    void record(OverflowEvent event) noexcept {
        if (!event.retries) return;
        if (size < capacity) events[size++] = event;
        else ++dropped;
    }
};
} // namespace tc::ane
