#include "memory_trace.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace tc {

const char *memory_trace_event_kind_name(MemoryTraceEventKind kind) {
    switch (kind) {
    case MemoryTraceEventKind::State: return "state";
    case MemoryTraceEventKind::Epoch: return "epoch";
    case MemoryTraceEventKind::Reserve: return "reserve";
    case MemoryTraceEventKind::Commit: return "commit";
    case MemoryTraceEventKind::Retire: return "retire";
    case MemoryTraceEventKind::Completion: return "completion";
    case MemoryTraceEventKind::Schedule: return "schedule";
    case MemoryTraceEventKind::Checkpoint: return "checkpoint";
    case MemoryTraceEventKind::Pressure: return "pressure";
    case MemoryTraceEventKind::Failure: return "failure";
    }
    return "failure";
}

MemoryTraceBuffer::MemoryTraceBuffer(size_t capacity)
    : ring_(capacity ? capacity : 1) {}

uint64_t MemoryTraceBuffer::hash(std::string_view value) {
    uint64_t result = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        result ^= byte;
        result *= 1099511628211ULL;
    }
    return result;
}

uint64_t MemoryTraceBuffer::monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void MemoryTraceBuffer::record(MemoryTraceEventKind kind,
                               std::string_view phase,
                               std::string_view site, uint64_t epoch,
                               uint64_t upper_bytes, uint64_t actual_bytes,
                               uint64_t committed_bytes,
                               uint64_t reserved_bytes,
                               uint64_t pending_bytes, uint32_t stage_id,
                               uint32_t slot_id, int32_t status) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (drained_) return;
        MemoryTraceEvent event;
        event.sequence = ++status_.event_count;
        event.monotonic_ns = monotonic_ns();
        event.kind = kind;
        event.phase_id = hash(phase);
        event.site_id = hash(site);
        event.epoch = epoch;
        event.upper_bytes = upper_bytes;
        event.actual_bytes = actual_bytes;
        event.committed_bytes = committed_bytes;
        event.reserved_bytes = reserved_bytes;
        event.pending_bytes = pending_bytes;
        event.stage_id = stage_id;
        event.slot_id = slot_id;
        event.status = status;
        if (size_ == ring_.size()) {
            head_ = (head_ + 1) % ring_.size();
            --size_;
            ++status_.dropped_events;
            status_.overflowed = true;
        }
        const size_t tail = (head_ + size_) % ring_.size();
        ring_[tail] = event;
        ++size_;
    } catch (...) {
        /* Trace is diagnostic only; never turn an allocation or callback
         * failure into a different exception from a noexcept path. */
    }
}

MemoryTraceStatus MemoryTraceBuffer::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

std::vector<MemoryTraceEvent> MemoryTraceBuffer::drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!size_) return {};
    /* Transfer the preallocated backing instead of copying it at successful
     * request teardown.  A copy here would create an unobserved second trace
     * allocation after the final memory checkpoint. */
    if (head_)
        std::rotate(ring_.begin(), ring_.begin() + head_, ring_.end());
    std::vector<MemoryTraceEvent> result = std::move(ring_);
    result.resize(size_);
    drained_ = true;
    head_ = 0;
    size_ = 0;
    return result;
}

} // namespace tc
